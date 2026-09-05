# Assurance report schema

Version 1.0.0. Generated from `cvassure.core.schemas.Finding` by `cvassure schema` — not maintained by hand, so it cannot drift away from the data it describes.

Machine-readable: [`assurance-finding.schema.json`](assurance-finding.schema.json) and [`assurance-run.schema.json`](assurance-run.schema.json) (JSON Schema draft 2020-12).

## A finding

Every check returns a list of these and nothing else. `findings.jsonl` in any results directory is one finding per line.

| field | type | required | meaning |
| --- | --- | --- | --- |
| `finding_id` | `string` | yes | Unique id for this finding. Stable across a run, not across runs. |
| `asset_ref` | `string` | yes | What this finding is about: an image path, a contributor id, 'model', or a receipt reference. |
| `asset_type` | `sample` \| `contributor` \| `model` \| `receipt` \| `dataset` | yes | Which kind of thing asset_ref names. |
| `attack_class` | `clean` \| `badnets_patch` \| `blended_trigger` \| `label_flip` \| `systematic_mislabel` \| `near_duplicate_flood` \| `ood_insertion` \| `model_substitute` \| `model_perturb` \| `model_backdoor` \| `receipt_alter` \| `receipt_replay` \| `receipt_delete` \| `receipt_reorder` \| `distribution_shift` | yes | The attack this finding would be evidence of, if true. |
| `detector_id` | `string` | yes | Which check produced it. |
| `access_tier` | `0` \| `1` \| `2` | yes | The access level the audit was granted: 0 answers only, 1 plus weights, 2 plus internals. |
| `raw_score` | `number` | yes | How suspicious the asset is, 0..1, higher is worse. A ranking, not a probability. |
| `calibrated_score` | `number`, `null` | no | raw_score mapped to an actual probability of being poisoned, when a calibration split was available. Null during a live audit, where there is no ground truth. |
| `confidence` | `number`, `null` | no | How sure we are of this finding, 0..1 — a different question from raw_score. A hash mismatch is confidence 1.0; a statistical shape argument is around 0.45 however alarming the score. |
| `severity` | `low` \| `medium` \| `high` \| `critical` | yes | Graded from the score, for triage. |
| `disposition` | `accept` \| `review` \| `quarantine` | yes | The recommended action. |
| `reason` | `string` | yes | One human-readable sentence containing real numbers. Enforced: a finding whose reason contains jargon or no number is rejected at construction. |
| `evidence` | `object` | no | Detector-specific supporting values behind the reason. |
| `artefacts` | `array` | no | Paths to images produced as evidence, such as a saliency heatmap or a reconstructed trigger. |
| `limitations` | `array` | no | What this particular assessment could not establish. |
| `unavailable_reason` | `string`, `null` | no | Set when the check could not run at all. Such a finding is a status message, not a claim: its raw_score is always 0 and must not be scored. |

### Two fields people get the wrong way round

`raw_score` is **how suspicious the asset is**. `confidence` is **how sure we are of the finding**. They are independent: a weight-digest mismatch is `raw_score` 1.0 and `confidence` 1.0 because a hash either matches or it does not, while a lopsided final layer may be `raw_score` 0.8 and `confidence` 0.45 because the shape is suggestive and nothing more.

### Findings that are not claims

A finding with `unavailable_reason` set is a status message — the check could not run at this access tier, or the input did not support it. Its `raw_score` is always 0 and it must be excluded from scoring. Treating one as a clean result is the single easiest way to misread this format.

## A run

`assurance-run.schema.json` wraps the findings with what they were derived from: input digests, the declared access tier, the traffic-light verdict, the generated coverage table, and the state of the audit log. Enough to re-check the verdict without re-running the audit.
