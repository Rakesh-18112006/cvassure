# PS 26228 — clause-by-clause coverage

Every row names the code that implements the clause and the test or measured
result that shows it works. Written to be checked, not believed: if a row
claims something, the named file is where you look to disprove it.

Status vocabulary is deliberately narrow — **Met** means implemented *and*
covered by a test or a measured number; **Partial** means implemented but not
fully measured, or measured and weak; **Not met** means what it says.

---

## 2.2.1 Training-Data Integrity

| Requirement | Where | Evidence | Status |
| --- | --- | --- | --- |
| Trigger injection | `detect/trigger_freq.py` | 0.93 typical TPR@1%FPR (`results/COVERAGE.md`) | **Met** |
| Label flipping | `detect/label_noise.py` | 0.93 typical TPR@1%FPR | **Met** |
| Systematic mislabelling | `detect/label_noise.py` + `detect/contributor.py` | **1.00** typical TPR@1%FPR, swept as its own class | **Met** |
| Near-duplicate flooding | `detect/near_duplicate.py` | 1.00 typical TPR@1%FPR | **Met** |
| Out-of-distribution insertion | `detect/ood.py` | 1.00 typical TPR@1%FPR | **Met** |
| Aggregate sample evidence to **source** level | `detect/contributor.py` | Beta-Binomial posterior per source; `test_contributor_names_the_bad_actor` | **Met** |
| ...by **contributor** | `dimension="contributor_id"` | contributor table in every report | **Met** |
| ...by **batch** | `dimension="batch_id"` | `test_risk_can_be_aggregated_by_batch_as_well_as_contributor` | **Met** |
| Not flagging samples in isolation | aggregation runs after every detector | a sample flagged by three checks counts once — `test_a_sample_flagged_by_three_detectors_counts_once` | **Met** |

Blended (whole-image) triggers are the weak row: **0.20** typical TPR@1%FPR,
reported as `unsupported` rather than hidden. See Known limitations below.

## 2.2.2 Model Integrity

| Requirement | Where | Evidence | Status |
| --- | --- | --- | --- |
| Behavioural fingerprinting | `detect/fingerprint.py` | 200 seeded probes; catches substitution at **tier 0** | **Met** |
| Trigger search / reconstruction | `detect/trigger_recon.py` | Neural Cleanse, MAD anomaly index, saves the reconstructed patch | **Met** |
| Parameter statistics | `detect/weight_stats.py`, `detect/weight_digest.py` | per-layer digest + tail/dominance statistics | **Met** |
| Activation statistics | `detect/spectral_signature.py` | tier-2 spectral separation | **Met** |
| Comparison against a reference battery | `detect/model.py::enrol` | fixed seeded probe set + enrolled weight digest | **Met** |
| Methods appropriate to access available | `ingest/models.py` tier gate | `test_every_tier_runs_clean` | **Met** |
| State **access assumptions** | `Finding.access_tier`, report column "access assumed" | every model finding | **Met** |
| State **confidence** | `Finding.confidence` | 1.0 for a hash mismatch, 0.45 for a shape argument — `test_model_findings_state_confidence_and_limits` | **Met** |
| State **limitations** | `Finding.limitations` | printed as its own report column | **Met** |
| No retraining for baseline assessment | none of `detect/` calls `.backward()` on model weights | `trigger_recon` optimises a mask, not the model | **Met** |

## 2.2.3 Inference Provenance and Output Integrity

| Requirement | Where | Evidence | Status |
| --- | --- | --- | --- |
| Bind input image | `input_sha256` + `input_phash` | `provenance/receipts.py` | **Met** |
| Bind model identifier / weight digest | `model_weight_digest` | " | **Met** |
| Bind preprocessing config | `preproc_config_hash` | " | **Met** |
| Bind inference config | `inference_config_hash` | " | **Met** |
| Bind resulting output | `output_sha256` + signature over all of it | " | **Met** |
| Detect post-hoc **alteration** | `SIGNATURE_INVALID`, `CHAIN_BROKEN` | 100% detected, correct code named | **Met** |
| Detect **substitution** | signature + chain | " | **Met** |
| Detect **replay** | `SEQUENCE_REPLAY`, `NONCE_REUSE` | " | **Met** |
| Hashes | SHA-256, canonical JSON | `test_key_order_does_not_change_bytes` | **Met** |
| Signatures | Ed25519, keys generated locally | `provenance/receipts.py::init_keys` | **Met** |
| Sequence control | `seq`, gap + replay detection | `SEQUENCE_GAP` | **Met** |
| Timestamp control | `TIMESTAMP_REGRESSION` | " | **Met** |
| Nonce control | 16 random bytes, reuse detected | `NONCE_REUSE` | **Met** |

Measured: **12/12 tampering attempts detected, 12/12 with the correct failure
mode named**, plus a clean-log control (`results/tables/table3_tamper.md`).

## 2.2.4 Distribution-Shift and Anomaly Assessment

| Requirement | Where | Evidence | Status |
| --- | --- | --- | --- |
| Detect material deviation from a declared reference | `detect/shift.py` | MMD with a permutation test + per-dimension KS | **Met** |
| Terrain / season / sensor / illumination / acquisition | attribution to brightness, colour, camera noise, sharpness | `_attribute()` | **Met** |
| Characterise the observed shift | attribution breakdown with an unexplained remainder | report section | **Met** |
| Distinguish drift from manipulation | `CLASSIFICATION_RULE`, printed in the report | `test_shift_calls_fog_drift_not_an_attack` | **Met** |
| Provide a **calibrated** risk or confidence score | `score/calibrate.py` isotonic, fitted on a held-out split | ECE 0.277 → **0.079** | **Partial** — see below |

**Partial, and why.** Calibration needs labelled outcomes. In the results run
we have them, and the reported ECE is genuine. In a *live* audit there is no
ground truth, so the shift score ships as a ranked severity plus the
permutation test's p-value in `evidence`, not as a calibrated probability. The
report says which of the two it is showing rather than implying a probability
it cannot support.

## 2.2.5 Analyst-Facing Assurance and Governance

| Requirement | Where | Status |
| --- | --- | --- |
| Human-readable reason on every flag | `Finding.reason`, jargon rejected at construction | **Met** |
| Supporting evidence | `Finding.evidence`, `Finding.artefacts` | **Met** |
| Confidence **or** severity | `Finding.severity` (always) + `Finding.confidence` (model checks) | **Met** |
| Affected asset | `Finding.asset_ref` + `asset_type` | **Met** |
| Recommended disposition | `Finding.disposition` ∈ accept / review / quarantine | **Met** |
| Tamper-evident audit trail | `report/audit.py`, hash-chained, `cvassure audit-verify` | **Met** |
| Explicitly declare unsupported attack classes | `results/COVERAGE.md`, generated from measurements | **Met** |

## 2.2.6 Constraints

| Requirement | Where | Status |
| --- | --- | --- |
| Fully offline / air-gapped | `core/offline.py`, `--offline-assert` in `make verify` | **Met** |
| No cloud services or external APIs | encoder weights bundled in `cvassure/assets/` | **Met** |
| Ingest COCO | `ingest/coco.py` | **Met** |
| Ingest YOLO | `ingest/yolo.py` | **Met** |
| ONNX models | `ingest/models.py::OnnxModel` | **Met** |
| TorchScript models | `TorchScriptModel` via `torch.jit.load` | **Met** |
| PyTorch models (pickled `nn.Module`) | same class, falls through to `torch.load` | **Met** |
| Baseline assessment without retraining | no optimiser touches model weights in `detect/` | **Met** |
| White-box methods fall back gracefully | tier gate + `Finding.unavailable` | **Met** |
| ...or clearly report unavailability | plain-English `unavailable_reason` on every blocked check | **Met** |

A bare `state_dict` is refused with an explanation rather than run: weights
without an architecture cannot be executed, and saying so is the honest
outcome.

## 2.3 Expected Solution — submission deliverables

| Deliverable | Where | Status |
| --- | --- | --- |
| Source code | this repository | **Met** |
| Architecture and setup notes | `README.md` (Quick start, Layout, Access tiers) | **Met** |
| **Assurance-report schema** | `docs/ASSURANCE_SCHEMA.md` + two JSON Schema files, generated by `cvassure schema` | **Met** |
| Reproducible audit log | `results/audit_log.jsonl`, hash-chained; `cvassure audit-verify` | **Met** |
| Coverage statement | `results/COVERAGE.md` + this file | **Met** |
| Reproducible poisoning / backdoor / substitution / tampering scenarios | `cvassure/attack/`, 11 seeded configs | **Met** |
| Model-agnostic | no architecture assumptions; ONNX + TorchScript + PyTorch | **Met** |

---

## Known limitations

Stated here and in every generated report, because a clean result on an
unsupported row is not evidence of anything.

1. **Blended, whole-image triggers — 0.20 typical TPR@1%FPR.** There is no
   localised artefact to find and no label to disagree with. Graded
   `unsupported`. A tier-2 spectral check is the intended next step.
2. **Calibration needs labels.** A live audit reports ranked severity and a
   permutation p-value, not a calibrated probability. Only the results run
   reports ECE.
3. **A `state_dict` cannot be executed.** Weight-level checks could in
   principle run on one; today the file is refused with an explanation instead.
4. **No contributor or batch metadata means no source-level assessment.** The
   report says so rather than guessing.
5. **An attacker who knows the detector set can design around it.** A clean
   report means we found nothing, not that there is nothing.
6. **Measured on SYNTH-10**, a procedural dataset that ships with the repo so
   results reproduce air-gapped. CIFAR-10, GTSRB and an aerial subset are
   supported and should be re-measured before any operational claim.
