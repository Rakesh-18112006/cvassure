import copy
import json

import pytest

from cvassure.cli import main
from cvassure.provenance.receipts import (
    ReceiptChain,
    init_keys,
    load_private_key,
    load_public_key,
    read_receipts,
    write_receipts,
)
from cvassure.provenance.verify import (
    CHAIN_BROKEN,
    NONCE_REUSE,
    SEQUENCE_GAP,
    SEQUENCE_REPLAY,
    SIGNATURE_INVALID,
    TIMESTAMP_REGRESSION,
    findings_from_result,
    verify_log,
)


@pytest.fixture
def keys(tmp_path):
    priv, pub = init_keys(tmp_path / "keys")
    return load_private_key(priv), load_public_key(pub)


def build_chain(private_key, n=100):
    chain = ReceiptChain(private_key)
    for i in range(n):
        chain.append(
            input_sha256=f"{i:064x}",
            model_weight_digest="a" * 64,
            preproc_config={"resize": 224, "normalise": "imagenet"},
            inference_config={"topk": 1},
            output={"label": "truck", "confidence": 0.91},
            timestamp_utc=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}.000000+00:00",
        )
    return [r.to_dict() for r in chain.receipts]


# -- happy path -------------------------------------------------------------


def test_clean_chain_of_100_verifies(keys):
    priv, pub = keys
    result = verify_log(build_chain(priv, 100), pub)
    assert result.ok
    assert result.render() == "PASS  Receipts 1-100 verified."


def test_sequence_starts_at_one_and_never_repeats(keys):
    priv, _ = keys
    seqs = [r["seq"] for r in build_chain(priv, 20)]
    assert seqs == list(range(1, 21))


def test_nonces_are_unique(keys):
    priv, _ = keys
    nonces = [r["nonce"] for r in build_chain(priv, 100)]
    assert len(set(nonces)) == 100
    assert all(len(n) == 32 for n in nonces)


def test_wrong_public_key_fails_everything(keys, tmp_path):
    priv, _ = keys
    _, other_pub = init_keys(tmp_path / "other")
    result = verify_log(build_chain(priv, 5), load_public_key(other_pub))
    assert result.has(SIGNATURE_INVALID)


# -- the six named failure modes -------------------------------------------


def test_signature_invalid_when_a_field_is_edited(keys):
    priv, pub = keys
    log = build_chain(priv, 100)
    log[41]["output"] = {"label": "ambulance", "confidence": 0.91}
    result = verify_log(log, pub)
    assert result.has(SIGNATURE_INVALID)
    assert any(f.seq == 42 for f in result.failures if f.code == SIGNATURE_INVALID)
    assert "Receipt #42" in result.render()


def test_chain_broken_when_a_link_is_rewritten(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log[5]["prev_receipt_hash"] = "f" * 64
    result = verify_log(log, pub)
    assert result.has(CHAIN_BROKEN)


def test_editing_an_old_record_breaks_the_link_that_follows_it(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log[2]["output"] = {"label": "tank", "confidence": 0.5}
    result = verify_log(log, pub)
    # the edited record fails its own signature, and record #4 no longer
    # points at anything that exists
    assert result.has(SIGNATURE_INVALID)
    assert result.has(CHAIN_BROKEN)
    assert {f.seq for f in result.failures} == {3, 4}
    assert result.n_verified == 2


def test_sequence_gap_when_a_record_is_deleted(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    del log[4]
    result = verify_log(log, pub)
    assert result.has(SEQUENCE_GAP)
    assert "missing" in result.render()


def test_sequence_replay_when_an_old_record_is_resubmitted(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log.append(copy.deepcopy(log[3]))
    result = verify_log(log, pub)
    assert result.has(SEQUENCE_REPLAY)


def test_nonce_reuse_is_caught_even_with_a_fresh_sequence_number(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log[7]["nonce"] = log[2]["nonce"]
    result = verify_log(log, pub)
    assert result.has(NONCE_REUSE)


def test_timestamp_regression(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log[6]["timestamp_utc"] = "2020-01-01T00:00:00.000000+00:00"
    result = verify_log(log, pub)
    assert result.has(TIMESTAMP_REGRESSION)


def test_reorder_is_caught(keys):
    priv, pub = keys
    log = build_chain(priv, 10)
    log[3], log[6] = log[6], log[3]
    result = verify_log(log, pub)
    assert result.has(SEQUENCE_REPLAY) or result.has(SEQUENCE_GAP)
    assert result.has(CHAIN_BROKEN)


def test_malformed_record_does_not_crash(keys):
    priv, pub = keys
    log = build_chain(priv, 5)
    log[2] = {"seq": 3}
    result = verify_log(log, pub)
    assert not result.ok


# -- output quality ---------------------------------------------------------


def test_failure_output_names_the_receipt_and_explains_itself(keys):
    priv, pub = keys
    log = build_chain(priv, 500)
    log[346]["output"] = {"label": "bus"}
    text = verify_log(log, pub).render()
    assert "Receipt #347" in text
    assert "SIGNATURE_INVALID" in text
    assert "changed after it was signed" in text


def test_findings_are_produced_for_the_report(keys):
    priv, pub = keys
    log = build_chain(priv, 20)
    log[9]["output"] = {"label": "bus"}
    findings = findings_from_result(verify_log(log, pub))
    assert findings
    assert all(f.asset_type == "receipt" for f in findings)
    assert any(f.disposition == "quarantine" for f in findings)


def test_clean_log_produces_one_reassuring_finding(keys):
    priv, pub = keys
    findings = findings_from_result(verify_log(build_chain(priv, 20), pub))
    assert len(findings) == 1
    assert findings[0].disposition == "accept"
    assert findings[0].attack_class == "clean"


# -- file round trip and CLI ------------------------------------------------


def test_edit_one_digit_in_the_file_and_verification_fails(keys, tmp_path):
    """The live demo, as a test."""
    priv, pub = keys
    path = tmp_path / "inference.jsonl"
    write_receipts(build_chain(priv, 100), path)
    assert verify_log(read_receipts(path), pub).ok

    lines = path.read_text().splitlines()
    rec = json.loads(lines[49])
    rec["output"]["confidence"] = 0.92  # one digit
    lines[49] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    result = verify_log(read_receipts(path), pub)
    assert not result.ok
    assert any(f.seq == 50 for f in result.failures)


def test_cli_round_trip(tmp_path, capsys):
    keydir = tmp_path / "keys"
    assert main(["provenance", "init-keys", "--out", str(keydir)]) == 0

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(
                {
                    "input_sha256": f"{i:064x}",
                    "model_weight_digest": "b" * 64,
                    "preproc_config": {"resize": 224},
                    "inference_config": {"topk": 1},
                    "output": {"label": "truck"},
                }
            )
            for i in range(25)
        )
        + "\n"
    )
    signed = tmp_path / "signed.jsonl"
    assert main(["provenance", "sign", "--receipts", str(raw), "--out", str(signed),
                 "--key", str(keydir / "priv.pem")]) == 0
    assert main(["provenance", "verify", "--receipts", str(signed),
                 "--pubkey", str(keydir / "pub.pem")]) == 0
    assert "PASS" in capsys.readouterr().out

    text = signed.read_text().replace('"truck"', '"bus"', 1)
    signed.write_text(text)
    assert main(["provenance", "verify", "--receipts", str(signed),
                 "--pubkey", str(keydir / "pub.pem")]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_private_key_is_not_world_readable(tmp_path):
    priv, _ = init_keys(tmp_path / "k")
    assert oct(priv.stat().st_mode)[-3:] == "600"
