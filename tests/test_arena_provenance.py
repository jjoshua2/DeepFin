"""Arena provenance: sha256 of what each seat loads and of the openings, banked and checked.

The arena names checkpoints by PATH in its fingerprint and result rows. These
tests pin that every result row and the game-log header also carry the sha256
of the bytes played, that a resume onto rewritten files is refused, and that a
log written before provenance existed still resumes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.eval.model_provenance import ProvenanceError, seat_files, seat_provenance
from chess_anti_engine.utils.game_log import read_game_log
from tests.test_match_resume import _openings_file, _run_arena, _SimulatedCrash


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cand(tmp_path: Path, data: bytes = b"candidate weights v1") -> str:
    path = tmp_path / "cand.pt"
    path.write_bytes(data)
    return str(path)


# ---------------------------------------------------------------------------
# model_provenance
# ---------------------------------------------------------------------------

def test_a_checkpoint_file_and_directory_hash_what_the_loader_reads(tmp_path: Path) -> None:
    f = tmp_path / "net.pt"
    f.write_bytes(b"weights")
    assert seat_provenance(str(f))["sha256"] == _sha(b"weights")
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "trainer.pt").write_bytes(b"trainer weights")
    prov = seat_provenance(str(d))
    assert prov["sha256"] == _sha(b"trainer weights")
    assert prov["files"][0]["path"] == str(d / "trainer.pt")


def test_a_spec_records_every_net_it_names_and_verifies_its_shas(tmp_path: Path) -> None:
    policy, value = tmp_path / "p.pt", tmp_path / "v.pt"
    policy.write_bytes(b"policy net")
    value.write_bytes(b"value net")
    spec = tmp_path / "x.hybrid.json"
    spec.write_text(json.dumps({"policy": str(policy), "policy_sha256": _sha(b"policy net"),
                                "value": str(value), "mutate": None}))
    files = seat_files(str(spec))
    assert [(f["role"], f["sha256"]) for f in files] == [
        ("spec", _sha(spec.read_bytes())), ("policy", _sha(b"policy net")), ("value", _sha(b"value net"))]
    spec.write_text(json.dumps({"policy": str(policy), "policy_sha256": "0" * 64}))
    with pytest.raises(ProvenanceError, match="policy_sha256"):
        seat_files(str(spec))


# ---------------------------------------------------------------------------
# arena_standard: rows, header, resume
# ---------------------------------------------------------------------------

def test_every_result_row_and_the_header_carry_the_sha256s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cand = _cand(tmp_path)
    log_path = tmp_path / "prov.games.jsonl"
    record = _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand)
    openings = _openings_file(tmp_path)
    assert record["candidate_sha256"] == _sha(b"candidate weights v1")
    assert record["candidate_files"] == [{"role": "checkpoint", "path": cand, "sha256": _sha(b"candidate weights v1")}]
    assert record["reference_sha256"] is None          # "ref.pt" does not exist: recorded as unknown
    assert record["openings_sha256"] == _sha(openings.read_bytes())
    assert record["game_log_provenance"] == "this_run"
    log = read_game_log(log_path)
    assert log.info["provenance"]["candidate"]["sha256"] == record["candidate_sha256"]
    assert "provenance" not in log.settings, "settings are fingerprinted; provenance must stay outside them"


def test_a_resume_with_the_same_files_is_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cand = _cand(tmp_path)
    log_path = tmp_path / "same.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, crash_after=3)
    record = _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, resume=True)
    assert record["game_log_provenance"] == "recorded"
    assert record["resumed_pairs"] == 1


def test_a_resume_onto_a_rewritten_net_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Same path (so the settings fingerprint matches), different bytes."""
    cand = _cand(tmp_path)
    log_path = tmp_path / "rewritten.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, crash_after=3)
    _cand(tmp_path, b"candidate weights v2")
    with pytest.raises(SystemExit, match="candidate sha256"):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, resume=True)


def test_a_log_written_before_provenance_still_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cand = _cand(tmp_path)
    log_path = tmp_path / "legacy.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, crash_after=3)
    lines = log_path.read_text().splitlines()
    header: dict[str, Any] = json.loads(lines[0])
    del header["info"]                                  # the pre-change header shape
    log_path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    record = _run_arena(monkeypatch, tmp_path, log_path=log_path, candidate=cand, resume=True)
    assert record["game_log_provenance"] == "absent"
    assert record["candidate_sha256"] == _sha(b"candidate weights v1")
    assert record["pairs"] == 4


def test_a_spec_whose_recorded_sha_is_wrong_is_refused_before_play(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    net = tmp_path / "p.pt"
    net.write_bytes(b"policy net")
    spec = tmp_path / "bad.hybrid.json"
    spec.write_text(json.dumps({"policy": str(net), "policy_sha256": "f" * 64}))
    with pytest.raises(SystemExit, match="policy_sha256"):
        _run_arena(monkeypatch, tmp_path, log_path=tmp_path / "bad.games.jsonl", candidate=str(spec))
    assert not (tmp_path / "bad.games.jsonl").exists(), "refused before the log is created"
