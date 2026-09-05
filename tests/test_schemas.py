import dataclasses

import pytest

from cvassure.core.hashing import canonical_json
from cvassure.core.schemas import (
    Finding,
    PlainEnglishError,
    Sample,
    check_plain_english,
    disposition_for,
    severity_for,
)

GOOD_REASON = (
    "This image looks nothing like the other 400 images labelled 'truck' — it is "
    "11 times further away than a typical truck photo."
)


def make(**kw) -> Finding:
    base = dict(
        asset_ref="img_0001",
        asset_type="sample",
        attack_class="ood_insertion",
        detector_id="ood",
        access_tier=0,
        raw_score=0.93,
        severity="critical",
        disposition="quarantine",
        reason=GOOD_REASON,
    )
    base.update(kw)
    return Finding(**base)


def test_finding_is_frozen_and_has_an_id():
    f = make()
    assert len(f.finding_id) == 32
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.raw_score = 0.1  # type: ignore[misc]


def test_round_trips_through_dict():
    f = make()
    assert Finding.from_dict(f.to_dict()) == f


def test_score_range_is_enforced():
    with pytest.raises(ValueError):
        make(raw_score=1.4)
    with pytest.raises(ValueError):
        make(raw_score=-0.01)


@pytest.mark.parametrize(
    "field,value",
    [("asset_type", "blob"), ("severity", "spicy"), ("disposition", "maybe"), ("access_tier", 3)],
)
def test_enum_fields_are_enforced(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


def test_jargon_is_rejected():
    with pytest.raises(PlainEnglishError):
        make(reason="Anomalous embedding detected at distance 11.")


def test_reason_must_contain_a_number():
    with pytest.raises(PlainEnglishError):
        make(reason="This picture seems wrong compared to the others.")


def test_unavailable_finding_needs_no_number():
    f = Finding.unavailable(
        asset_ref="model",
        asset_type="model",
        attack_class="model_backdoor",
        detector_id="trigger_recon",
        access_tier=0,
        reason="We could not run this check because we were only given the model's "
        "answers, not its internals.",
        unavailable_reason="requires access tier 2",
    )
    assert f.is_unavailable
    assert f.raw_score == 0.0


def test_calibrated_score_wins_when_present():
    f = make(raw_score=0.9)
    assert f.score == pytest.approx(0.9)
    assert f.with_calibration(0.4).score == pytest.approx(0.4)


def test_findings_serialise_canonically():
    f = make()
    assert canonical_json(f.to_dict()) == canonical_json(dict(f.to_dict()))


def test_grading_helpers_are_monotone():
    assert severity_for(0.95) == "critical"
    assert severity_for(0.75) == "high"
    assert severity_for(0.5) == "medium"
    assert severity_for(0.1) == "low"
    assert disposition_for(0.85) == "quarantine"
    assert disposition_for(0.6) == "review"
    assert disposition_for(0.2) == "accept"


def test_sample_shape():
    s = Sample(sample_id="a", image_path="/tmp/a.png", label="truck", contributor_id="C1")
    assert s.to_dict()["contributor_id"] == "C1"
    assert s.batch_id is None


def test_check_plain_english_accepts_the_reference_sentence():
    check_plain_english(GOOD_REASON)
