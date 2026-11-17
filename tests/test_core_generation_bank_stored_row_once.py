"""Actual tiny accepted-bank -> Trainer -> checkpoint -> resume tests.

Authorities and payloads are synthetic. These tests supply no production family,
scientific, historical exposure, GPU or training qualification credit.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import stat
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import core_finite_learner as core
from scripts import core_generation_bank_stored_row_once as once
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import frozen
from tests.test_core_generation_bank_learner import bank, contract as bank_contract
from tests.test_core_generation_bank_policy_anchor import anchored_contract


def contract(tmp_path: Path, *, split: bool = True, anchor: bool = False, batch_size: int = 2) -> tuple[dict[str, Any], dict[str, str]]:
    c, _ = anchored_contract(tmp_path) if anchor else bank_contract(tmp_path)
    c["schema"] = once.ANCHOR_CONTRACT if anchor else once.CONTRACT
    c["batch_size"] = batch_size
    total = (4 + batch_size - 1) // batch_size
    c["updates_per_segment"] = [1, total - 1] if split and total > 1 else [total]
    if total == 1:
        c["lr"]["warmup_steps"] = 0
        c["lr"]["warmup_start"] = c["lr"]["peak"]
    c["source_files"] = {name: core.file_sha(core.REPO / name) for name in once.SOURCE_FILES}
    c["sampling"] = {"schema": once.SAMPLING, "order_seed": 1673, "expected_rows": 4,
                     "presentations_per_row": 1, "mirror_shares_quota": True,
                     "partial_batch": "EMIT_FINAL_REMAINDER_ONCE_NO_PADDING", "lineage_uuid": c["run_uuid"],
                     "order": once.ORDER, "permutation_sha256": core.digest(once.permutation_for(4, 1673)),
                     "quota_ledger": str(tmp_path / "quota.jsonl")}
    return c, frozen(tmp_path / "once-contract.json", c)


def stage(tmp_path: Path, cref: dict[str, str], binding: dict[str, Any], segment: int,
          prior: dict[str, Any] | None, name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + ".stage.json"), {"schema": once.STAGE, "contract_sha256": cref["sha256"],
        "segment": segment, "banks": [binding], "parent": prior, "output": str(tmp_path / name)})


def closed_parent(tmp_path: Path, cref: dict[str, str], boundary: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", "scripts.core_generation_bank_stored_row_once", "--contract", cref["path"],
               "--contract-sha256", cref["sha256"], "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    terminal = frozen(tmp_path / f"once-owner-{boundary['segment']}-terminal.json", {
        "status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "f" * 64,
        "command": command, "source_revision": "synthetic-owner"})
    output = Path(boundary["checkpoint"]["path"]).parent
    return {"boundary": {"path": str(output / "boundary.json"), "sha256": core.file_sha(output / "boundary.json")},
            "owner_terminal": terminal, "owner_sha256": "f" * 64, "command": command}


@pytest.mark.parametrize(("anchor", "batch_size"), [(False, 2), (True, 2), (False, 3), (True, 3)])
def test_real_cli_checkpoint_and_closed_segment_restore_complete_once(tmp_path: Path, anchor: bool, batch_size: int) -> None:
    c, cref = contract(tmp_path, anchor=anchor, batch_size=batch_size)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first_ref = stage(tmp_path, cref, binding, 1, None, "first")
    assert once.main(["--contract", cref["path"], "--contract-sha256", cref["sha256"],
                      "--stage", first_ref["path"], "--stage-sha256", first_ref["sha256"]]) == 0
    first = json.loads((tmp_path / "first/boundary.json").read_text())
    prior = closed_parent(tmp_path, cref, first)
    second = once.execute(cref, stage(tmp_path, cref, binding, 2, prior, "second"))
    saved = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert first["pass_complete"] is False
    assert second["pass_complete"] is True
    assert second["quota_state"]["cursor"] == 4
    assert second["quota_state"]["last_batch_rows"] == 4 - batch_size
    assert second["quota_state"]["batches"] == second["step"] == 2
    assert second["drawn_rows"] == second["applied_views"] == second["metrics"]["train_samples_seen"] == 4 - batch_size
    assert saved[once.CHECKPOINT_KEY]["quota_state"] == second["quota_state"]
    assert saved[once.CHECKPOINT_KEY]["draw_journals"] == second["draw_journals"]
    records = [json.loads(line) for ref in second["draw_journals"] for line in Path(ref["path"]).read_text().splitlines()]
    ids = [(r["generation"], r["source_sha256"], r["row_index"]) for rec in records for r in rec["rows"]]
    assert len(ids) == len(set(ids)) == 4
    assert [rec["draw_batch"] for rec in records] == [0, 1]
    assert [len(rec["rows"]) for rec in records] == [batch_size, 4 - batch_size]
    assert second["metrics"]["transient_cuda_retry_batches"] == 0
    delivery = once.validate_closed_segment(cref, closed_parent(tmp_path, cref, second))
    assert delivery["pass_complete"] is True
    assert delivery["cumulative_presentations"] == 4
    assert delivery["optimizer_updates"] == 2
    if anchor:
        assert second["policy_anchor"] == c["policy_anchor"]
        assert second["policy_anchor_diagnostics"]["reference_forwards"] == 1


@pytest.mark.parametrize(("anchor", "batch_size"), [(False, 2), (True, 2), (False, 3), (True, 3)])
def test_uninterrupted_and_real_checkpoint_resume_match_updates_rng_and_draw_order(tmp_path: Path, anchor: bool, batch_size: int) -> None:
    c, cref = contract(tmp_path, anchor=anchor, batch_size=batch_size)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = once.execute(cref, stage(tmp_path, cref, binding, 1, None, "first"))
    second = once.execute(cref, stage(tmp_path, cref, binding, 2, closed_parent(tmp_path, cref, first), "second"))
    direct = copy.deepcopy(c)
    direct["updates_per_segment"] = [2]
    direct["sampling"]["quota_ledger"] = str(tmp_path / "direct-quota.jsonl")
    direct_ref = frozen(tmp_path / "direct.contract.json", direct)
    full = once.execute(direct_ref, stage(tmp_path, direct_ref, binding, 1, None, "full"))
    a = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step"):
        require_equal(a[key], b[key], "stored-row split vs uninterrupted " + key)
    assert second["rng"] == full["rng"]
    assert second["quota_state"] == full["quota_state"]
    assert second["quota_ledger_checkpoint"]["sha256"] == full["quota_ledger_checkpoint"]["sha256"]
    split_bytes = b"".join(Path(ref["path"]).read_bytes() for ref in second["draw_journals"])
    assert hashlib.sha256(split_bytes).hexdigest() == full["draws_sha256"]


@pytest.mark.parametrize("fault", ["wrong_population", "dose", "twopasses", "mirror_quota", "legacy_schema", "ledger_output", "order", "permutation_sha"])
def test_preflight_refusals_precede_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, _ = contract(tmp_path)
    if fault == "wrong_population":
        c["sampling"]["expected_rows"] = 5
        c["sampling"]["permutation_sha256"] = core.digest(once.permutation_for(5, 1673))
        c["updates_per_segment"] = [1, 1, 1]
    elif fault == "dose":
        c["updates_per_segment"] = [1]
        c["lr"]["warmup_steps"] = 0
        c["lr"]["warmup_start"] = c["lr"]["peak"]
    elif fault == "twopasses":
        c["sampling"]["presentations_per_row"] = 2
    elif fault == "mirror_quota":
        c["sampling"]["mirror_shares_quota"] = False
    elif fault == "ledger_output":
        c["sampling"]["quota_ledger"] = str(tmp_path / "refused/quota.jsonl")
    elif fault == "order":
        c["sampling"]["order"] = "sha256_seed_index_sort_v1"
    elif fault == "permutation_sha":
        c["sampling"]["permutation_sha256"] = "0" * 64
    else:
        c["schema"] = "core_generation_bank_learning_v1"
    cref = frozen(tmp_path / "bad.contract.json", c)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000) if fault == "wrong_population" else {}
    calls: list[Any] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    with pytest.raises(ValueError, match=r"stored-row|partial batch|dose"):
        once.execute(cref, stage(tmp_path, cref, binding, 1, None, "refused"))
    assert not calls
    assert not (tmp_path / "refused/learner.pt").exists()
    assert not (tmp_path / "refused").exists()


def test_actual_resume_refuses_newer_journal_than_checkpoint_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = once.execute(cref, stage(tmp_path, cref, binding, 1, None, "first"))
    prior = closed_parent(tmp_path, cref, first)
    invalid = copy.deepcopy(first)
    invalid["quota_state"]["cursor"] = invalid["quota_state"]["batches"] = 0
    invalid["quota_state"]["last_batch_rows"] = 0
    invalid["quota_state"]["row_prefix_sha256"] = hashlib.sha256(b"").hexdigest()
    prior["boundary"] = frozen(tmp_path / "older-state.boundary.json", invalid)
    # The accepted output path must stay genuine; mutate only the synthetic
    # closed boundary under the fixture's output and rebind its content hash.
    target = tmp_path / "first/boundary.json"
    original_dir_mode = stat.S_IMODE(target.parent.stat().st_mode)
    original_file_mode = stat.S_IMODE(target.stat().st_mode)
    assert original_dir_mode == 0o555
    assert original_file_mode == 0o444
    target.parent.chmod(original_dir_mode | 0o200)
    try:
        target.chmod(0o644)
        target.unlink()
        prior["boundary"] = frozen(target, invalid)
    finally:
        if target.exists():
            target.chmod(original_file_mode)
        target.parent.chmod(original_dir_mode)
    assert stat.S_IMODE(target.stat().st_mode) == original_file_mode
    assert stat.S_IMODE(target.parent.stat().st_mode) == original_dir_mode
    calls: list[Any] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    with pytest.raises(ValueError, match="checkpoint/journal cursor mismatch"):
        once.execute(cref, stage(tmp_path, cref, binding, 2, prior, "refused"))
    assert not calls


@pytest.mark.parametrize("fault", ["forward", "optimizer", "nonfinite"])
def test_real_training_fault_spends_one_batch_no_retry_or_clean_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, cref = contract(tmp_path, split=False)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    original = core.make_trainer
    calls: list[str] = []
    def make(*args: Any, **kwargs: Any) -> Any:
        trainer = original(*args, **kwargs)
        trainer.prefetch_batches = True
        if fault == "forward":
            def broken_forward(*_args: Any, **_kwargs: Any) -> Any:
                calls.append("forward")
                raise RuntimeError("CUDA synthetic forward failure")
            monkeypatch.setattr(trainer.model, "forward", broken_forward)
        elif fault == "optimizer":
            def broken_optimizer(*_args: Any, **_kwargs: Any) -> Any:
                calls.append("optimizer")
                raise RuntimeError("CUDA synthetic optimizer failure")
            monkeypatch.setattr(trainer.opt, "step", broken_optimizer)
        else:
            def nonfinite(grad: Any) -> Any:
                calls.append("gradient")
                return grad * float("inf")
            next(trainer.model.parameters()).register_hook(nonfinite)
        return trainer
    monkeypatch.setattr(core, "make_trainer", make)
    with pytest.raises(ValueError, match="finite core update failed; retries forbidden") as caught:
        once.execute(cref, stage(tmp_path, cref, binding, 1, None, "failed"))
    cause = caught.value.__cause__
    assert isinstance(cause, RuntimeError)
    if fault == "forward":
        assert "CUDA synthetic forward failure" in str(cause)
        assert calls == ["forward"]
    elif fault == "optimizer":
        from chess_anti_engine.train.aurora import OptimizerStepFailed
        assert isinstance(cause, OptimizerStepFailed)
        assert isinstance(cause.__cause__, RuntimeError)
        assert "CUDA synthetic optimizer failure" in str(cause.__cause__)
        assert calls == ["optimizer"]
    else:
        assert "cannot discard a non-finite-gradient update" in str(cause)
        assert calls == ["gradient"]
    records = (tmp_path / "failed/draws.jsonl").read_text().splitlines()
    assert len(records) == 1
    assert calls
    assert not (tmp_path / "failed/boundary.json").exists()
    assert core.failure_marker(tmp_path / "failed").exists()


def test_failed_sibling_tail_cannot_resume_into_fresh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref = contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = once.execute(cref, stage(tmp_path, cref, binding, 1, None, "first"))
    prior = closed_parent(tmp_path, cref, first)
    # Simulate a durable post-checkpoint reservation, without replaying a model.
    with Path(c["sampling"]["quota_ledger"]).open("ab") as ledger:
        ledger.write((tmp_path / "first/draws.jsonl").read_bytes())
        ledger.flush()
        os.fsync(ledger.fileno())
    calls: list[Any] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    with pytest.raises(ValueError, match="quota ledger ahead/behind"):
        once.execute(cref, stage(tmp_path, cref, binding, 2, prior, "fresh-sibling"))
    assert not calls


def test_single_actual_partial_batch_completes_without_padding(tmp_path: Path) -> None:
    c, cref = contract(tmp_path, split=False, batch_size=6)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    result = once.execute(cref, stage(tmp_path, cref, binding, 1, None, "tail-only"))
    assert result["step"] == result["applied_updates"] == 1
    assert result["quota_state"]["cursor"] == result["quota_state"]["last_batch_rows"] == 4
    assert result["drawn_rows"] == result["applied_views"] == result["metrics"]["train_samples_seen"] == 4
    records = [json.loads(line) for line in (tmp_path / "tail-only/draws.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert len(records[0]["rows"]) == 4
    assert result["pass_complete"] is True
