"""Synthetic terminal/index/history-proof extraction; no replay payloads."""

from __future__ import annotations

import copy

import pytest

from scripts import sf_dlite_extract128 as extract
from scripts import sf_dlite_value_sidecar as core

OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5",
           "c2c3", "g8f6", "d2d3", "d7d6", "e1g1", "e8g8",
           "f1e1", "f8e8", "b1d2", "c6b8"]


def packed(value: dict) -> bytes:
    return core.canonical(value) + b"\n"


def fixture() -> tuple[bytes, bytes, bytes, dict[str, bytes]]:
    stack_sha = core.digest(core.canonical(OPENING))
    notes = []
    proofs = {}
    winners = []
    child = []
    for ordinal, source in enumerate(extract.SOURCE_PROOFS):
        uid = ["a" * 64, source, "root", ordinal,
               16 if source == "BT4-v9" else 0]
        digest = str(ordinal + 1) * 64
        note = {"ordinal": ordinal, "uid": uid, "source": source,
                "input_digest": digest,
                "context": [stack_sha, "0:0:0", 6, "legal", "query"]}
        notes.append(packed(note))
        winners.append({"winner_uid": uid, "winner_ordinal": ordinal,
                        "input_digest": digest})
        report = {"source": source, "game_id": ordinal, "root_id": "root",
                  "history_chain": {
                      "schema": "tri_source_full_game_history_chain_v1",
                      "root_start_fen":
                      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                      "opening_uci": OPENING, "played_uci": ["a2a4"],
                      "row_index": [{"uid": uid, "input_digest": digest,
                                     "history_stack_sha256": stack_sha,
                                     "pov_white": True, "rule50": 6}]}}
        proofs[source] = packed(report)
        child.append({"wave": 1, "source": source,
                      "proof_sha256": core.digest(proofs[source])})
    index_raw = b"".join(notes)
    receipt = {"schema": "tri_source_83991_cpu_diagnostic_v7",
               "status": "COMPLETE_DIAGNOSTIC_ZERO_CREDIT",
               "credit": extract.ZERO, "source_child_results": child,
               "resolution": {
                   "two_pass_readback": {"rows": 3,
                                         "index_sha256": core.digest(index_raw)},
                   "provisional_dedup": {"gross_rows": 3,
                                         "unique_rows": 3,
                                         "winners": winners}}}
    receipt_raw = packed(receipt)
    terminal = {"schema": "tri_source_83991_cpu_terminal_v7",
                "status": receipt["status"], "credit": extract.ZERO,
                "receipt_sha256": core.digest(receipt_raw)}
    return receipt_raw, packed(terminal), index_raw, proofs


def run(payload: tuple[bytes, bytes, bytes, dict[str, bytes]]):
    receipt, terminal, index, proofs = payload
    return extract.extract_metadata(receipt_raw=receipt, terminal_raw=terminal,
                                    index_raw=index, proof_raw=proofs,
                                    sample_size=2, expected_gross=3)


def test_exact_separate_domain_sample_and_history_join() -> None:
    winners, proofs, manifest = run(fixture())
    rows = [extract.json_object(line, "row")
            for line in extract.lines(winners, 2)]
    expected = sorted([extract.json_object(line, "row") for line in
                       extract.lines(fixture()[2], 3)],
                      key=lambda row: extract.sample_rank(row["uid"]))[:2]
    assert [row["uid"] for row in rows] == [row["uid"] for row in expected]
    assert manifest["sample_size"] == 2
    assert manifest["winners_sha256"] == core.digest(winners)
    assert manifest["proofs_sha256"] == core.digest(proofs)
    assert manifest["status"] == "METADATA_ONLY_ZERO_LABEL_AND_CORPUS_CREDIT"


def test_tampered_proof_and_index_and_terminal_refuse() -> None:
    receipt, terminal, index, proofs = fixture()
    bad_proofs = copy.deepcopy(proofs)
    bad_proofs["Ceres-v8"] += b"\n"
    with pytest.raises(core.Hold, match="proof SHA"):
        run((receipt, terminal, index, bad_proofs))
    with pytest.raises(core.Hold, match="wave index"):
        run((receipt, terminal, index + b"\n", proofs))
    broken_terminal = extract.json_object(terminal, "terminal")
    broken_terminal["receipt_sha256"] = "0" * 64
    with pytest.raises(core.Hold, match="terminal complete"):
        run((receipt, packed(broken_terminal), index, proofs))


def test_winner_index_mismatch_refuses() -> None:
    receipt, terminal, index, proofs = fixture()
    modified = extract.json_object(receipt, "receipt")
    modified["resolution"]["provisional_dedup"]["winners"] = [
        {**winner, "winner_uid": [*winner["winner_uid"][:3], 100,
                                  winner["winner_uid"][4]]}
        if ordinal == 0 else winner
        for ordinal, winner in enumerate(
            modified["resolution"]["provisional_dedup"]["winners"])]
    new_receipt = packed(modified)
    new_terminal = extract.json_object(terminal, "terminal")
    new_terminal["receipt_sha256"] = core.digest(new_receipt)
    with pytest.raises(core.Hold, match="winner/index identity"):
        run((new_receipt, packed(new_terminal), index, proofs))


def test_missing_selected_history_refuses_even_with_rehashed_receipt() -> None:
    receipt, terminal, index, proofs = fixture()
    _winners, _selected_proofs, manifest = run((receipt, terminal, index, proofs))
    selected = [extract.json_object(line, "winner") for line in
                extract.lines(_winners, 2)]
    source = selected[0]["source"]
    game = extract.json_object(proofs[source], "game")
    game["history_chain"]["row_index"] = []
    changed_proofs = dict(proofs)
    changed_proofs[source] = packed(game)
    changed_receipt = extract.json_object(receipt, "receipt")
    for child in changed_receipt["source_child_results"]:
        if child["source"] == source:
            child["proof_sha256"] = core.digest(changed_proofs[source])
    changed_receipt_raw = packed(changed_receipt)
    changed_terminal = extract.json_object(terminal, "terminal")
    changed_terminal["receipt_sha256"] = core.digest(changed_receipt_raw)
    assert manifest["sample_size"] == 2
    with pytest.raises(core.Hold, match="all selected complete history proofs"):
        run((changed_receipt_raw, packed(changed_terminal), index,
             changed_proofs))
