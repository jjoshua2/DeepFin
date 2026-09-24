"""Fixture-only receipt contract tests; no ordinary GPU bank is claimed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import bt4_ordinary_bank_audit as audit
from scripts import bt4_own_game_teacher_source as source
from tests.test_bt4_own_game_teacher_source import _bank


def test_reviewed_ordinary_retry_verifier_is_allowlisted() -> None:
    assert "aa2e16ab1a78973f46839bc6228c4d783ceb6097db55f9f5a5d576fec3103d86" in (
        audit.REVIEWED_FULL_STRICT_VERIFIERS
    )


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
            *, fails: bool = False) -> dict:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    stage_dir = tmp_path / "ordinary"
    stage_dir.mkdir()
    bank, _, _, _ = _bank(stage_dir / "bank", model="a" * 64, value=1.0)
    proof = bank / "provider_proof.json"
    proof.write_text('{"fixture_only":true}\n')
    summary = json.loads((bank / "summary.json").read_text())
    summary["provider_proof_sha256"] = audit.sha_file(proof)
    (bank / "summary.json").write_text(json.dumps(summary))
    summary_sha = audit.sha_file(bank / "summary.json")
    facts = {"accepted_rows": 1, "attempted_plies": 1, "completed_games": 1,
             "discarded_games": 0, "discarded_by_termination": {},
             "sixman_games": 1, "natural_games": 0,
             "summary_sha256": summary_sha,
             "provider_proof_sha256": audit.sha_file(proof),
             "pinned_source_origins": {"synthetic_test_only": "fixture"}}
    verifier = tmp_path / "full_strict_fixture.py"
    if fails:
        verifier.write_text("def verify_bank(plan, stage, bank):\n"
                            "    raise ValueError('fixture verifier refused')\n")
    else:
        verifier.write_text("def verify_bank(plan, stage, bank):\n"
                            f"    return {facts!r}\n")
    verifier_sha = audit.sha_file(verifier)
    reviewed = frozenset((*audit.REVIEWED_FULL_STRICT_VERIFIERS, verifier_sha))
    monkeypatch.setattr(audit, "REVIEWED_FULL_STRICT_VERIFIERS", reviewed)
    monkeypatch.setattr(source, "REVIEWED_FULL_STRICT_VERIFIERS", reviewed)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"supervisor_sha256": verifier_sha,
                                "output_root": str(tmp_path),
                                "stages": [{"name": "ordinary"}]}))
    plan_sha = audit.sha_file(plan)
    terminal = stage_dir / "terminal.json"
    terminal.write_text(json.dumps({"plan_sha256": plan_sha, "stage": "ordinary",
                                    "status": "PASS_ORDINARY", "returncode": 0,
                                    "summary_sha256": summary_sha,
                                    "provider_proof_sha256": audit.sha_file(proof),
                                    **facts}))
    return {"bank": bank, "summary_sha": summary_sha, "plan": plan,
            "plan_sha": plan_sha, "verifier": verifier, "verifier_sha": verifier_sha,
            "terminal": terminal, "terminal_sha": audit.sha_file(terminal),
            "out": tmp_path / "independent-audit.json"}


def run_fixture(values: dict) -> dict:
    return audit.audit_bank(auditor_sha256=audit.sha_file(Path(audit.__file__)),
                            plan_path=values["plan"], plan_sha256=values["plan_sha"],
                            verifier_path=values["verifier"],
                            verifier_sha256=values["verifier_sha"], stage_name="ordinary",
                            bank=values["bank"], terminal_path=values["terminal"],
                            terminal_sha256=values["terminal_sha"], output=values["out"])


def test_new_profile_binds_bank_plan_verifier_and_adapter(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch)
    receipt = run_fixture(values)
    assert receipt["status"] == audit.STATUS
    assert receipt["summary_sha256"] == values["summary_sha"]
    assert receipt["provider_proof_sha256"] == audit.sha_file(values["bank"] / "provider_proof.json")
    assert receipt["verifier"]["sha256"] == values["verifier_sha"]
    assert source.inspect_bank(values["bank"], values["summary_sha"], values["out"],
                               audit.sha_file(values["out"]))["completed_rows"] == 1


def test_changed_bank_is_refused(tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch)
    run_fixture(values)
    receipt_sha = audit.sha_file(values["out"])
    (values["bank"] / "unlisted.txt").write_text("changed")
    with pytest.raises(ValueError, match="current complete bank"):
        source.inspect_bank(values["bank"], values["summary_sha"], values["out"], receipt_sha)


def test_wrong_auditor_pin_is_refused(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch)
    run_fixture(values)
    receipt = json.loads(values["out"].read_text())
    receipt["verifier"]["sha256"] = "0" * 64
    values["out"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="function differs"):
        source.inspect_bank(values["bank"], values["summary_sha"], values["out"],
                            audit.sha_file(values["out"]))


def test_rewritten_strict_fact_is_refused_even_with_new_receipt_pin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch)
    run_fixture(values)
    receipt = json.loads(values["out"].read_text())
    receipt["facts"]["discarded_by_termination"] = {"max_plies_unresolved": 1}
    values["out"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="terminal does not bind"):
        source.inspect_bank(values["bank"], values["summary_sha"], values["out"],
                            audit.sha_file(values["out"]))


def test_verifier_failure_leaves_no_pass_receipt(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch, fails=True)
    with pytest.raises(ValueError, match="fixture verifier refused"):
        run_fixture(values)
    assert not values["out"].exists()
    assert json.loads((tmp_path / "independent-audit.json.writing").read_text())["status"] == "INCOMPLETE"


def test_unreviewed_verifier_and_wrong_stage_refused_before_audit(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    values = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "REVIEWED_FULL_STRICT_VERIFIERS", frozenset())
    with pytest.raises(ValueError, match="lacks a reviewed"):
        run_fixture(values)
    monkeypatch.setattr(audit, "REVIEWED_FULL_STRICT_VERIFIERS",
                        frozenset({values["verifier_sha"]}))
    with pytest.raises(ValueError, match="stage absent"):
        audit.audit_bank(auditor_sha256=audit.sha_file(Path(audit.__file__)),
                         plan_path=values["plan"], plan_sha256=values["plan_sha"],
                         verifier_path=values["verifier"],
                         verifier_sha256=values["verifier_sha"], stage_name="wrong",
                         bank=values["bank"], terminal_path=values["terminal"],
                         terminal_sha256=values["terminal_sha"], output=values["out"])
    assert not values["out"].exists()
