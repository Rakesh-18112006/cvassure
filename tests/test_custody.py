import json

import pytest

from cvassure.cli import main
from cvassure.ingest.dataset import load_dataset
from cvassure.provenance.custody import (
    CHAIN_BROKEN,
    HANDOFF_MISMATCH,
    HOP_GAP,
    HOP_REPLAY,
    SIGNATURE_INVALID,
    TIMESTAMP_REGRESSION,
    UNKNOWN_ACTOR,
    CustodyChain,
    dataset_content_digest,
    findings_from_custody_result,
    init_actor_keys,
    load_public_key,
    read_custody_chain,
    verify_custody_chain,
    write_custody_chain,
)

ACTORS = ("COLLECTOR", "LABELLER", "TRAINER", "DEPLOYER")


@pytest.fixture
def keyring_and_keys(tmp_path):
    """Four organisations, each with its own keypair — the normal case."""
    privs, pubs = {}, {}
    for actor in ACTORS:
        priv, pub = init_actor_keys(tmp_path / actor)
        from cvassure.provenance.custody import load_private_key

        privs[actor] = load_private_key(priv)
        pubs[actor] = load_public_key(pub)
    return privs, pubs


def build_chain(privs, *, n_hops=4):
    """A clean four-hop chain: collect -> label -> train -> deploy, each hop
    signed by a different organisation."""
    chain = CustodyChain()
    stages = ("data_collection", "labelling", "training", "deployment")
    for i in range(n_hops):
        actor = ACTORS[i % len(ACTORS)]
        chain.add_hop(
            stage=stages[i % len(stages)],
            actor_id=actor,
            output_digest=f"{i:064x}",
            description=f"hop {i} by {actor}",
            private_key=privs[actor],
        )
    return [r.to_dict() for r in chain.records]


# -- happy path ---------------------------------------------------------------


def test_clean_chain_of_four_hops_verifies(keyring_and_keys):
    privs, pubs = keyring_and_keys
    result = verify_custody_chain(build_chain(privs), pubs)
    assert result.ok
    assert "Nothing was substituted" in result.render()


def test_hops_start_at_one_and_increase(keyring_and_keys):
    privs, _ = keyring_and_keys
    hops = [r["hop"] for r in build_chain(privs, n_hops=6)]
    assert hops == list(range(1, 7))


def test_each_hop_declares_the_previous_hops_output_as_its_input(keyring_and_keys):
    privs, _ = keyring_and_keys
    log = build_chain(privs, n_hops=3)
    assert log[0]["input_digest"] == "0" * 64  # GENESIS
    assert log[1]["input_digest"] == log[0]["output_digest"]
    assert log[2]["input_digest"] == log[1]["output_digest"]


def test_wrong_public_key_for_an_actor_fails_that_actors_hops(keyring_and_keys, tmp_path):
    privs, pubs = keyring_and_keys
    _, other_pub = init_actor_keys(tmp_path / "impostor")
    bad_keyring = dict(pubs)
    bad_keyring["COLLECTOR"] = load_public_key(other_pub)
    result = verify_custody_chain(build_chain(privs), bad_keyring)
    assert result.has(SIGNATURE_INVALID)


def test_unregistered_actor_is_reported_by_name(keyring_and_keys):
    privs, pubs = keyring_and_keys
    keyring_missing_one = {k: v for k, v in pubs.items() if k != "TRAINER"}
    result = verify_custody_chain(build_chain(privs), keyring_missing_one)
    assert result.has(UNKNOWN_ACTOR)
    assert any(f.actor_id == "TRAINER" for f in result.failures)


# -- the failure modes that matter for a supply chain --------------------------


def test_signature_invalid_when_a_field_is_edited(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=4)
    log[1]["description"] = "quietly changed after signing"
    result = verify_custody_chain(log, pubs)
    assert result.has(SIGNATURE_INVALID)


def test_chain_broken_when_a_link_is_rewritten(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=4)
    log[2]["prev_hash"] = "f" * 64
    result = verify_custody_chain(log, pubs)
    assert result.has(CHAIN_BROKEN)


def test_handoff_mismatch_when_a_hop_is_signed_over_a_fabricated_input(keyring_and_keys):
    """The headline case: an organisation signs off on having received
    something that does not match what the previous organisation actually
    produced — a substitution between two honestly-signing parties, which no
    signature check alone can see."""
    privs, pubs = keyring_and_keys
    chain = CustodyChain()
    chain.add_hop(
        stage="data_collection", actor_id="COLLECTOR", output_digest="a" * 64,
        description="collected 500 images", private_key=privs["COLLECTOR"],
    )
    # LABELLER signs, correctly and honestly from its own point of view, but
    # what it claims to have received ("b"*64) is not what COLLECTOR actually
    # produced ("a"*64) — exactly what a swapped-in-transit dataset looks like.
    chain.add_hop(
        stage="labelling", actor_id="LABELLER", output_digest="c" * 64,
        description="labelled the images I received", private_key=privs["LABELLER"],
        input_digest="b" * 64,
    )
    log = [r.to_dict() for r in chain.records]
    result = verify_custody_chain(log, pubs)
    assert result.has(HANDOFF_MISMATCH)
    failure = next(f for f in result.failures if f.code == HANDOFF_MISMATCH)
    assert failure.hop == 2
    assert failure.actor_id == "LABELLER"
    # and nothing else fires — both signatures are individually valid
    assert not result.has(SIGNATURE_INVALID)
    assert not result.has(CHAIN_BROKEN)


def test_first_hop_must_declare_the_origin_marker(keyring_and_keys):
    privs, pubs = keyring_and_keys
    chain = CustodyChain()
    chain.add_hop(
        stage="data_collection", actor_id="COLLECTOR", output_digest="a" * 64,
        description="claims something came before it", private_key=privs["COLLECTOR"],
        input_digest="d" * 64,
    )
    result = verify_custody_chain([r.to_dict() for r in chain.records], pubs)
    assert result.has(HANDOFF_MISMATCH)


def test_hop_gap_when_a_hop_is_deleted(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=6)
    del log[2]
    result = verify_custody_chain(log, pubs)
    assert result.has(HOP_GAP)


def test_hop_replay_when_an_old_hop_is_resubmitted(keyring_and_keys):
    import copy

    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=6)
    log.append(copy.deepcopy(log[1]))
    result = verify_custody_chain(log, pubs)
    assert result.has(HOP_REPLAY)


def test_timestamp_regression(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=4)
    log[2]["timestamp_utc"] = "2000-01-01T00:00:00.000000+00:00"
    result = verify_custody_chain(log, pubs)
    assert result.has(TIMESTAMP_REGRESSION)


def test_malformed_hop_does_not_crash(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=4)
    log[1] = {"hop": 2}
    result = verify_custody_chain(log, pubs)
    assert not result.ok


# -- findings bridge ------------------------------------------------------------


def test_findings_are_produced_for_the_report(keyring_and_keys):
    privs, pubs = keyring_and_keys
    log = build_chain(privs, n_hops=4)
    log[3]["description"] = "tampered"
    findings = findings_from_custody_result(verify_custody_chain(log, pubs))
    assert findings
    assert all(f.asset_type == "custody" for f in findings)
    assert any(f.disposition == "quarantine" for f in findings)


def test_clean_chain_produces_one_reassuring_finding(keyring_and_keys):
    privs, pubs = keyring_and_keys
    findings = findings_from_custody_result(verify_custody_chain(build_chain(privs), pubs))
    assert len(findings) == 1
    assert findings[0].disposition == "accept"
    assert findings[0].attack_class == "clean"


def test_findings_bridge_handles_a_handoff_mismatch(keyring_and_keys):
    """The Finding schema requires ``reason`` to read as one sentence under
    400 characters (see ``check_plain_english``) — this failed in practice
    once, because HANDOFF_MISMATCH's detail embeds real sha256 digests and an
    actor id, and nothing upstream of ``findings_from_custody_result`` had
    ever been exercised on this specific code path."""
    privs, pubs = keyring_and_keys
    chain = CustodyChain()
    chain.add_hop(
        stage="data_collection", actor_id="COLLECTOR", output_digest="a" * 64,
        description="collected 500 images", private_key=privs["COLLECTOR"],
    )
    chain.add_hop(
        stage="labelling", actor_id="LABELLER", output_digest="c" * 64,
        description="labelled the images I received", private_key=privs["LABELLER"],
        input_digest="b" * 64,
    )
    findings = findings_from_custody_result(
        verify_custody_chain([r.to_dict() for r in chain.records], pubs)
    )
    assert findings
    assert findings[0].disposition == "quarantine"
    assert "LABELLER" in findings[0].reason


def test_findings_bridge_bounds_reason_length_for_a_long_actor_id(tmp_path):
    """``actor_id`` is free text typed into the web form, with no length
    limit of its own — a long one must not push ``reason`` over the schema's
    per-Finding budget."""
    long_name = "A" * 300
    # the directory name is unrelated to the actor id under test — keep it
    # short so this exercises the reason-length fix, not filesystem limits.
    priv, pub = init_actor_keys(tmp_path / "keys")
    from cvassure.provenance.custody import load_private_key

    key = load_private_key(priv)
    chain = CustodyChain()
    chain.add_hop(
        stage="data_collection", actor_id=long_name, output_digest="a" * 64,
        description="collected the imagery", private_key=key,
    )
    chain.add_hop(
        stage="labelling", actor_id=long_name, output_digest="c" * 64,
        description="labelled the images I received", private_key=key,
        input_digest="b" * 64,
    )
    findings = findings_from_custody_result(
        verify_custody_chain([r.to_dict() for r in chain.records], {long_name: load_public_key(pub)})
    )
    assert findings  # would have raised PlainEnglishError before the fix
    assert findings[0].disposition == "quarantine"


# -- content digests --------------------------------------------------------------


def test_dataset_digest_is_the_same_regardless_of_where_the_files_live(synth_root, tmp_path):
    import shutil

    ds = load_dataset(synth_root)
    copy_root = tmp_path / "copy"
    shutil.copytree(synth_root, copy_root)
    ds_copy = load_dataset(copy_root)
    assert dataset_content_digest(ds.samples) == dataset_content_digest(ds_copy.samples)


def test_dataset_digest_changes_if_a_label_changes(synth_root):
    from dataclasses import replace

    ds = load_dataset(synth_root)
    original = dataset_content_digest(ds.samples)
    relabelled = list(ds.samples)
    relabelled[0] = replace(relabelled[0], label="not_" + str(relabelled[0].label))
    assert dataset_content_digest(relabelled) != original


def test_images_only_digest_ignores_labels(synth_root):
    from dataclasses import replace

    ds = load_dataset(synth_root)
    relabelled = [replace(s, label="anything") for s in ds.samples]
    assert dataset_content_digest(ds.samples, include_labels=False) == dataset_content_digest(
        relabelled, include_labels=False
    )


# -- file round trip and CLI ---------------------------------------------------


def test_file_round_trip_and_edit_breaks_verification(keyring_and_keys, tmp_path):
    privs, pubs = keyring_and_keys
    path = tmp_path / "custody.jsonl"
    write_custody_chain(build_chain(privs, n_hops=4), path)
    assert verify_custody_chain(read_custody_chain(path), pubs).ok

    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["description"] = "edited after the fact"
    lines[1] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")

    result = verify_custody_chain(read_custody_chain(path), pubs)
    assert not result.ok
    assert any(f.hop == 2 for f in result.failures)


def test_cli_round_trip(tmp_path, capsys, synth_root):
    keys_root = tmp_path / "keys"
    assert main(["custody", "init-actor", "--out", str(keys_root / "COLLECTOR")]) == 0
    assert main(["custody", "init-actor", "--out", str(keys_root / "LABELLER")]) == 0

    chain_path = tmp_path / "chain.jsonl"
    assert main([
        "custody", "add-hop", "--chain", str(chain_path), "--stage", "data_collection",
        "--actor-id", "COLLECTOR", "--key", str(keys_root / "COLLECTOR" / "priv.pem"),
        "--description", "collected the field imagery", "--dataset", str(synth_root),
        "--images-only",
    ]) == 0

    assert main([
        "custody", "add-hop", "--chain", str(chain_path), "--stage", "labelling",
        "--actor-id", "LABELLER", "--key", str(keys_root / "LABELLER" / "priv.pem"),
        "--description", "labelled the same imagery", "--dataset", str(synth_root),
    ]) == 0

    assert main([
        "custody", "verify", "--chain", str(chain_path), "--keys-dir", str(keys_root),
    ]) == 0
    assert "PASS" in capsys.readouterr().out

    text = chain_path.read_text().replace("labelled the same imagery", "swapped in transit", 1)
    chain_path.write_text(text)
    assert main([
        "custody", "verify", "--chain", str(chain_path), "--keys-dir", str(keys_root),
    ]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_private_key_is_not_world_readable(tmp_path):
    priv, _ = init_actor_keys(tmp_path / "k")
    assert oct(priv.stat().st_mode)[-3:] == "600"
