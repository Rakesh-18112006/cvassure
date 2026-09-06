# cvassure

**Trustworthy computer-vision integrity assurance for data, models and inference
outputs in multi-contributor pipelines.**
SIH 2026 — Problem Statement 26228.

When training data arrives from many contributors and a model arrives from a
vendor, three questions have to be answerable before anything goes into
service:

1. Has anyone tampered with the training data, and if so, **who sent it**?
2. Is this the model we were given, or has it been swapped or edited?
3. Is the record of what the model actually answered trustworthy?

cvassure answers all three, offline, and — just as importantly — states
plainly which of them it cannot answer at the level of access it was given.

---

## The two rules this project is built on

**Plain English in every output.** Every reason string the system produces is
readable by someone with no machine-learning background, and this is enforced
in code, not by convention: `Finding` rejects a reason containing jargon or
lacking a real number, and the test suite checks every detector's output
against that rule. Not *"anomalous embedding detected"* but:

> This image looks nothing like the other 400 images labelled 'truck' — it is
> 11 times further away than a typical truck photo, where even the most
> unusual genuine one only reaches 1.8 times.

**Measurement before features.** A detector that is not measured is invisible.
Every number in `results/` was produced by poisoning data with a seeded script
that recorded exactly what it touched, then running the detectors blind
against it. `cvassure/detect/` **cannot import** `cvassure/attack/` — there is
a test that reads the source and fails the build if anybody adds such an
import.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .
make bootstrap
make verify
```

Then the single command shown to an evaluator:

```bash
.venv/bin/cvassure audit --dataset data/contributed --model models/vendor.onnx \
  --access-tier 1 --receipts logs/inference.jsonl --out results/
```

which prints, in large plain text:

```
VERDICT: QUARANTINE RECOMMENDED
1 of 5 contributors clean.  3 contributors quarantined (C2, C3, C4).  1 to review.
Model: SUBSTITUTED OR EDITED.
Inference log: 2 records tampered (#347, #348).
```

and writes `results/report.html` — one self-contained file, all CSS inline,
all images base64-embedded, no web fonts, no scripts from anywhere. It opens
identically on a laptop with no network connection.

---

## Air-gapped by construction

The audit never contacts the network. That is a claim, so it is tested:
`--offline-assert` replaces the socket layer with one that raises, and
`make verify` runs a full audit under it. If any code path — ours or a
library's — reached for the network, the run would fail loudly rather than
quietly succeed on a machine that happened to have wifi.

The image encoder is **bundled in the repository**
(`cvassure/assets/encoder_resnet18.pt`, 43 MB). Nothing is downloaded at audit
time. If that file is ever missing, the system falls back to a handcrafted
descriptor computed directly from the pixels rather than reaching for a
download — weaker, and the report says so.

---

## What it checks

### Training data (PS 2.2.1)

| check | what it finds |
| --- | --- |
| `near_duplicate` | the same photograph submitted many times, rotated, cropped or re-lit |
| `ood` | images that do not belong with the rest of their class |
| `label_noise` | images whose label disagrees with their nearest look-alikes |
| `trigger_freq` | pasted-in markers and faint patterns laid over the whole image |
| `spectral_signature` | *[tier 2]* groups that move together inside the model |
| `contributor` | **aggregation** — which source is responsible |

The last one is the point. A list of nine hundred suspect image IDs is not a
decision; *"quarantine everything from Contributor D"* is. Each contributor
gets a Beta-Binomial posterior over their true bad rate, so a contributor who
sent three images and had one flagged gets a wide interval rather than an
accusation, and the disposition rule reads the interval rather than the point
estimate.

### Model (PS 2.2.2)

| check | tier | what it finds |
| --- | --- | --- |
| `fingerprint` | **0** | a swapped model, from its answers alone |
| `weight_digest` | 1 | edited weights, layer by layer |
| `weight_stats` | 1 | a lopsided final layer, with no reference to compare against |
| `trigger_recon` | 2 | a hidden trigger, reconstructed as a picture |

`fingerprint` is the one worth pointing at: it detects substitution at
black-box access. Ask the model 200 fixed questions and compare the replies to
the ones recorded when it was accepted. The same model asked the same question
replies identically to the last decimal place; measured against itself the
difference is around 1e-15, against a different model 0.06. No retraining
happens anywhere in the audit — PS 2.2.6 forbids it.

### Inference provenance (PS 2.2.3)

Every inference gets a receipt binding the input hash, the perceptual hash,
the model digest, the preprocessing and inference config hashes, the output,
a timestamp, a nonce, and the hash of the previous receipt — Ed25519 signed
with keys generated locally that never leave the machine.

`verify_log` names six distinct failures: `SIGNATURE_INVALID`, `CHAIN_BROKEN`,
`SEQUENCE_GAP`, `SEQUENCE_REPLAY`, `NONCE_REUSE`, `TIMESTAMP_REGRESSION`. This
is cryptography rather than statistics — it either matches or it does not — so
the tamper-detection table reads 100% on every row, and anything less is a bug
to fix, not a limitation to report.

### Distribution shift (PS 2.2.4)

Saying "the data has shifted" is nearly useless: data always shifts. So the
shift is *attributed* to physical causes and the remainder is reported:

```
Total shift: 0.34
  Brightness/lighting  62%   normal
  Camera noise         19%   normal
  Terrain colour       11%   normal
  UNEXPLAINED           8%   <- flagged
```

then classified DRIFT or SUSPICIOUS by a rule that is **printed in the report**
so the judgement can be argued with: drift is gradual, global, low-frequency
and spread across contributors; manipulation is localised, high-frequency and
concentrated in one source or one class.

---

## Access tiers, honestly

| tier | what we are given |
| --- | --- |
| 0 | the model's answers only |
| 1 | its weights as well |
| 2 | what it computes part-way through |

The tier is **declared by the user**, and the handle refuses anything above it
*even when the file format would allow it*. Hand this system a black-box model
and you get a clean statement of what could not be checked, not a crash and
not an invented number:

```
NO   trigger_recon        reconstructs a hidden trigger to prove a backdoor
                          unavailable: needs tier 2 — internals readable
```

Every detector degrades this way, and there is a test that runs the whole
suite at tier 0 and asserts zero crashes.

---

## Measurement

```bash
make results     # sweep attacks x poison rates x tiers x seeds, then aggregate
```

produces `results/RESULTS.md` and `results/COVERAGE.md`, both generated from
measured numbers — the coverage statement is derived from what actually ran
and what it actually scored, never hand-written, so it cannot drift out of
sync with reality.

Measured over 270 cells (6 attacks × 5 poison rates × 3 access tiers × 3
seeds) on SYNTH-10, plus 12 tampering attempts, graded on the **median** cell
rather than the best one:

| what an attacker did | typical TPR@1%FPR | range | false alarms | status |
| --- | --- | --- | --- | --- |
| The same photograph submitted many times | 1.00 | 0.45 – 1.00 | 3.0% | strong |
| Images from a completely different source | 1.00 | 0.73 – 1.00 | 1.3% | strong |
| One contributor mislabels systematically | 1.00 | 0.01 – 1.00 | 1.0% | strong |
| A marker pasted onto the image | 0.93 | 0.67 – 1.00 | 1.7% | strong |
| The picture is fine, the label is wrong | 0.93 | 0.83 – 1.00 | 1.0% | strong |
| **A faint pattern laid over the whole image** | **0.20** | 0.00 – 0.67 | 1.1% | **unsupported** |
| Any tampering with the inference log | 1.00 | proof, not an estimate | 0.0% | strong |

That last-but-one row is the point of publishing this table. A blended trigger
at 8% strength has no localised artefact to find and no label to disagree
with, and we catch about a fifth of it at a 1% false-alarm budget. Saying so
is more useful than a table with no weak rows in it.

All four tables (detection, contributor risk, tamper detection, runtime) land
in `results/tables/` as both CSV and Markdown, and the seven figures in
`results/plots/` as 300 DPI PNG and SVG.

Four metrics, in plain words:

- **AUROC** — pick one poisoned image and one clean image at random; AUROC is
  the probability the poisoned one scored higher. 0.5 is a coin flip.
- **TPR at 1% FPR** — the number that matters operationally: if you will
  tolerate one false alarm per hundred clean images, what fraction of the real
  poison do you still catch? A high AUROC can hide a tool that buries an
  analyst in 6,000 false alarms. This number cannot.
- **ECE** — like a weather forecaster. If the system says 0.9 a hundred times,
  about ninety should really be poisoned. Under 0.05 is good.
- **Tamper detection rate** — not statistical at all. 100%, or it is a bug.

Every headline number carries a bootstrap 95% confidence interval, and every
swept cell is run at three seeds so the tables carry a ±.

The scoring harness was built and sanity-checked *before* any detector
existed: feed it random scores and AUROC must come out at 0.5, feed it perfect
scores and it must come out at 1.0. Both are tests.

---

Every clause of the problem statement is mapped to the code that implements it
and the test or measured number that shows it works, in
[`docs/PS_COVERAGE.md`](docs/PS_COVERAGE.md). The report format itself is
published as JSON Schema in [`docs/ASSURANCE_SCHEMA.md`](docs/ASSURANCE_SCHEMA.md)
(regenerate with `cvassure schema`).

## Layout

```
cvassure/
  core/       schemas.py (the Finding contract), hashing.py, offline.py
  ingest/     coco.py, yolo.py, folder.py, models.py, contributors.py
  attack/     make_poison.py, image_ops.py, model_attacks.py, receipt_attacks.py
  detect/     near_duplicate, ood, label_noise, trigger_freq, spectral_signature,
              contributor, fingerprint, weight_digest, weight_stats,
              trigger_recon, shift, embed
  provenance/ receipts.py, verify.py
  report/     build.py (coverage matrix), html.py, audit.py
  score/      metrics.py, calibrate.py, evaluate.py, plots.py, tables.py
  datasets/   synth.py, toy_model.py
  pipeline.py cli.py demo.py
configs/attacks/   eleven ready-made attack configurations
experiments/       run_all.py, aggregate.py
tests/             302 tests
```

## Datasets

`SYNTH-10` is generated from a seed, ships with the repository and needs no
download, which is what keeps `make reproduce` working on an air-gapped
machine. CIFAR-10, GTSRB and a VisDrone/DOTA subset are used when placed under
`data/` in advance; the results run never downloads anything.

## The demo

```bash
make demo
```

Five steps in four minutes, laptop in airplane mode: a clean run that stays
green, a contaminated intake where the report names the contributor, a swapped
model caught at black-box access, one edited digit in the inference log caught
and named, and finally the coverage table — including the rows where we do
badly.

That last step is not a weakness. It is usually the part that decides it.
