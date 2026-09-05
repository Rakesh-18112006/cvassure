"""``cvassure ingest inspect`` — say what we loaded and what we cannot check.

This is the command a judge is most likely to run against a black-box model to
see whether the system crashes. It must not.
"""

from __future__ import annotations

from cvassure.ingest.dataset import load_dataset
from cvassure.ingest.models import TIER_NAMES, load_model

# Which detectors need which tier. Kept here so one edit updates both the
# inspect output and the report's coverage matrix.
TIER_REQUIREMENTS: dict[str, tuple[int, str]] = {
    "near_duplicate": (0, "finds the same photo submitted many times over"),
    "ood": (0, "finds images that do not belong with the rest of their class"),
    "label_noise": (0, "finds images whose label disagrees with their neighbours"),
    "trigger_freq": (0, "finds pasted-in patches and faint overlaid patterns"),
    "contributor": (0, "scores each contributor's share of suspect data"),
    "fingerprint": (0, "detects a swapped model from its answers alone"),
    "shift": (0, "measures how far the new data has drifted from the reference"),
    "weight_digest": (1, "detects edited weights, layer by layer"),
    "weight_stats": (1, "detects backdoor-shaped anomalies in the final layer"),
    "spectral_signature": (2, "separates poisoned samples inside the model's own view"),
    "trigger_recon": (2, "reconstructs a hidden trigger to prove a backdoor"),
}


def run(
    *,
    dataset: str,
    model: str | None = None,
    access_tier: int = 0,
    fmt: str = "auto",
    contributors: str | None = None,
    contributor_from_path: str | None = None,
) -> int:
    ds = load_dataset(
        dataset,
        fmt=fmt,
        contributors=contributors,
        contributor_from_path=contributor_from_path,
    )
    s = ds.summary()

    print("DATASET")
    print(f"  path            {s['root']}")
    print(f"  format          {s['format'].upper()}")
    print(f"  images          {s['n_samples']}  ({s['n_labelled']} labelled)")
    print(f"  classes         {len(s['classes'])}  {', '.join(s['classes'][:8])}"
          + (" …" if len(s["classes"]) > 8 else ""))
    print(f"  contributors    {s['n_contributors']}  "
          f"(attribution available for {s['contributor_coverage']} images)")

    print()
    print("MODEL")
    handle = None
    if model is None:
        print("  none supplied — model-integrity checks will be skipped")
    else:
        try:
            handle = load_model(model, access_tier)
            d = handle.describe()
            print(f"  path            {d['path']}")
            print(f"  format          {d['kind']}")
            print(f"  file digest     {d['file_digest'][:32]}…")
            print(f"  access granted  tier {access_tier} — {TIER_NAMES[access_tier]}")
        except Exception as exc:  # a judge's model must never crash the tool
            print(f"  could not load this model: {exc}")
            print("  the audit will continue with the data checks only")

    print()
    print("CHECKS AVAILABLE AT THIS ACCESS LEVEL")
    available, unavailable = [], []
    for name, (needed, what) in TIER_REQUIREMENTS.items():
        needs_model = name in {
            "fingerprint", "weight_digest", "weight_stats",
            "spectral_signature", "trigger_recon",
        }
        if needs_model and handle is None:
            unavailable.append((name, what, "no model was supplied"))
        elif needed > access_tier:
            unavailable.append(
                (name, what, f"needs tier {needed} — {TIER_NAMES[needed]}")
            )
        elif handle is not None and needs_model and not handle.can(needed):
            unavailable.append(
                (name, what, f"this model was supplied as {handle.kind}")
            )
        else:
            available.append((name, what))

    for name, what in available:
        print(f"  YES  {name:<20} {what}")
    for name, what, why in unavailable:
        print(f"  NO   {name:<20} {what}")
        print(f"       {'':<20} unavailable: {why}")

    if s["limitations"]:
        print()
        print("LIMITATIONS")
        for note in s["limitations"]:
            print(f"  - {note}")

    print()
    print(f"{len(available)} of {len(TIER_REQUIREMENTS)} checks can run on this input.")
    return 0
