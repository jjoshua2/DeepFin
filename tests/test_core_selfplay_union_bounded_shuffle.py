"""Actual opt-in entry/Trainer/full-state integration on tiny synthetic fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
import sys
import threading
from typing import Any

import pytest
import torch

from scripts import core_finite_learner as core
from scripts import core_selfplay_union_bounded_shuffle as entry
from scripts import core_selfplay_union_bounded_shuffle_proof as proof
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import frozen
from tests.test_core_selfplay_union_stored_row_once import contract as original_contract, method as original_method


def selected_method(tmp_path: Path, c: dict[str, Any], name: str) -> dict[str, str]:
    base = original_method(tmp_path, c, name + "-base")
    value = json.loads(Path(base["path"]).read_bytes())
    value["training"].update(sampling_order=entry.ORDER, replay_limits=c["sampling"]["limits"])
    value["scope"] = "synthetic caller-selected bounded software fixture, not compute or scientific authority"
    return frozen(tmp_path / (name + ".method.json"), value)


def contract(tmp_path: Path, *, split: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    old, _ = original_contract(tmp_path, split=split)
    c = copy.deepcopy(old)
    c["schema"] = entry.CONTRACT
    c["study_uuid"] = "55555555-5555-4555-8555-555555555555"
    c["source_files"] = {p: core.file_sha(core.REPO / p) for p in entry.SOURCE_FILES}
    c["sampling"].pop("permutation_sha256")
    c["sampling"].update(schema=entry.SAMPLING, order=entry.ORDER,
        limits={"total_bytes": 32 << 20, "source_bytes": 8 << 20, "buffer_bytes": 2 << 20,
                "ready_bytes": 1 << 20, "consumer_bytes": 20 << 20, "scratch_bytes": 1 << 20,
                "buffer_rows": 3, "ready_batches": 2})
    c["method"] = selected_method(tmp_path, c, "bounded")
    return c, frozen(tmp_path / "bounded.contract.json", c)


def stage(tmp_path: Path, c: dict[str, Any], cref: dict[str, str], segment: int,
          parent: dict[str, Any] | None, name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + ".stage.json"), {"schema": entry.STAGE, "contract_sha256": cref["sha256"],
        "segment": segment, "population": c["population"], "parent": parent, "output": str(tmp_path / name)})


def binding(tmp_path: Path, cref: dict[str, str], boundary: dict[str, Any], *, observed: bool = True) -> dict[str, Any]:
    module = "scripts.core_selfplay_union_bounded_shuffle_proof" if observed else "scripts.core_selfplay_union_bounded_shuffle"
    command = [sys.executable, "-B", "-m", module, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
               "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    name = Path(boundary["checkpoint"]["path"]).parent.name
    terminal = frozen(tmp_path / (name + ".terminal.json"), {"status": "PASS", "returncode": 0,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "f" * 64,
        "command": command, "scope": "synthetic trusted software closure; not an actual R7 claim"})
    path = Path(boundary["checkpoint"]["path"]).parent / "boundary.json"
    return {"boundary": {"path": str(path), "sha256": core.file_sha(path)}, "owner_terminal": terminal,
            "owner_sha256": "f" * 64, "command": command}


def test_real_bounded_entry_split_resume_matches_continuous_model_rng_and_tail(tmp_path: Path) -> None:
    c, cref = contract(tmp_path)
    first = proof.execute(cref, stage(tmp_path, c, cref, 1, None, "first"))
    assert first["applied_updates"] == 1
    assert first["applied_views"] == first["drawn_rows"] == 2
    assert first["quota_state"]["ready"]  # prefetched physical IDs are NOT reserved presentations
    assert first["quota_state"]["emitted"] == 2
    first_rows = [json.loads(line) for line in Path(first["draw_journals"][0]["path"]).read_text().splitlines()]
    assert sum(len(record["rows"]) for record in first_rows) == 2
    second = proof.execute(cref, stage(tmp_path, c, cref, 2, binding(tmp_path, cref, first), "second"))
    assert second["step"] == second["quota_state"]["batches"] == 3
    assert second["applied_updates"] == 2
    assert second["applied_views"] == second["metrics"]["train_samples_seen"] == 3
    assert second["quota_state"]["emitted"] == 5
    assert second["pass_complete"] is True
    assert second["quota_state"]["ready"] == []
    records = [json.loads(line) for ref in second["draw_journals"] for line in Path(ref["path"]).read_text().splitlines()]
    ids = [(row["generation"], row["source_sha256"], row["row_index"]) for record in records for row in record["rows"]]
    assert len(ids) == len(set(ids)) == 5
    assert [len(record["rows"]) for record in records] == [2, 2, 1]
    direct = copy.deepcopy(c)
    direct["updates_per_segment"] = [3]
    direct["sampling"]["quota_ledger"] = str(tmp_path / "direct-quota.jsonl")
    direct["method"] = selected_method(tmp_path, direct, "direct")
    dr = frozen(tmp_path / "direct.contract.json", direct)
    full = proof.execute(dr, stage(tmp_path, direct, dr, 1, None, "full"))
    a = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "bounded continuous vs split " + key)
    assert second["rng"] == full["rng"]
    assert second["quota_state"] == full["quota_state"]
    assert second["quota_ledger_checkpoint"]["sha256"] == full["quota_ledger_checkpoint"]["sha256"]
    cert = second["own_live_certificate"]
    assert cert["actual_calls"] == {"forward": 2, "optimizer_pre": 2, "optimizer_post": 2, "save": 1}
    assert cert["bounded_state_sha256"] == core.digest(second["quota_state"])
    assert cert["extra_forwards"] == 0
    assert a[entry.CHECKPOINT_KEY]["quota_state"] == second["quota_state"]
    assert entry.validate_closed_model_delivery(cref, binding(tmp_path, cref, second))["pass_complete"] is True
    assert not any(t.name == "bounded-replay-prefetch" and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("fault", ["tensor", "forward", "backward", "optimizer"])
def test_actual_trainer_faults_spend_reserved_rows_without_boundary_or_redraw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, cref = contract(tmp_path, split=False)
    real = core.make_trainer

    def broken(*args: Any, **kwargs: Any):
        trainer = real(*args, **kwargs)
        if fault == "tensor":
            def tensor(_batch: Any):
                raise RuntimeError("synthetic tensor failure")
            monkeypatch.setattr(trainer, "_host_batch_to_tensors", tensor)
        elif fault == "forward":
            def forward(_model: Any, _args: Any):
                raise RuntimeError("synthetic forward failure")
            trainer.model.register_forward_pre_hook(forward)
        elif fault == "optimizer":
            def opt(*_args: Any, **_kwargs: Any):
                raise RuntimeError("synthetic optimizer failure")
            monkeypatch.setattr(trainer.opt, "step", opt)
        return trainer

    monkeypatch.setattr(core, "make_trainer", broken)
    if fault == "backward":
        def backward(*_args: Any, **_kwargs: Any):
            raise RuntimeError("synthetic backward failure")
        monkeypatch.setattr(torch.Tensor, "backward", backward)
    with pytest.raises(ValueError, match="finite core update failed; retries forbidden"):
        entry.execute(cref, stage(tmp_path, c, cref, 1, None, "failed"))
    assert not (tmp_path / "failed/learner.pt").exists()
    assert not (tmp_path / "failed/boundary.json").exists()
    failure = json.loads(core.failure_marker(tmp_path / "failed").read_bytes())
    assert failure["drawn_rows"] == 2
    assert failure["completed_optimizer_calls"] == 0
    assert Path(c["sampling"]["quota_ledger"]).stat().st_size > 0
    assert not any(t.name == "bounded-replay-prefetch" and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize("fault", ["ledger_tail", "checkpoint_state"])
def test_actual_resume_refuses_tainted_prefix_before_source_or_model_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, cref = contract(tmp_path)
    first = entry.execute(cref, stage(tmp_path, c, cref, 1, None, "first"))
    parent = binding(tmp_path, cref, first, observed=False)
    if fault == "ledger_tail":
        with Path(c["sampling"]["quota_ledger"]).open("ab") as stream:
            stream.write(b"tainted-tail\n")
    else:
        changed = copy.deepcopy(first)
        changed["quota_state"]["ready"][0].reverse()
        changed["quota_state"]["sha256"] = core.digest({k: v for k, v in changed["quota_state"].items() if k != "sha256"})
        path = Path(parent["boundary"]["path"])
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o555
        path.parent.chmod(0o755)
        try:
            path.chmod(0o644)
            path.unlink()
            parent["boundary"] = frozen(path, changed)
        finally:
            path.chmod(0o444)
            path.parent.chmod(0o555)
    calls: list[str] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args, **_kwargs: calls.append("model"))
    monkeypatch.setattr(entry, "from_fixed_manifest", lambda *_args, **_kwargs: calls.append("source"))
    with pytest.raises((ValueError, TypeError), match=r"ahead/behind|ready order|frontier|differs"):
        entry.execute(cref, stage(tmp_path, c, cref, 2, parent, "refused"))
    assert calls == []
    assert not (tmp_path / "refused/learner.pt").exists()


def test_bounded_model_metadata_consumer_reconstructs_order_without_raw_payload_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref = contract(tmp_path, split=False)
    full = proof.execute(cref, stage(tmp_path, c, cref, 1, None, "full"))
    parent = binding(tmp_path, cref, full)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: pytest.fail("metadata consumer decoded model"))
    monkeypatch.setattr(entry, "from_fixed_manifest", lambda *_args, **_kwargs: pytest.fail("metadata consumer read source"))
    real_hash = core.file_sha

    def metadata_hash(path: Path) -> str:
        assert path.suffix == ".json"
        return real_hash(path)

    monkeypatch.setattr(core, "file_sha", metadata_hash)
    result = entry.validate_closed_model_delivery(cref, parent)
    assert result["actual_views"] == 5
    assert result["selection_audit"]["payload_reads"] == 0
    assert result["own_live_certificate"]["bounded_state_sha256"] == core.digest(full["quota_state"])


def test_passive_bounded_own_live_observer_does_not_change_model_or_rng(tmp_path: Path) -> None:
    c, cref = contract(tmp_path, split=False)
    observed = proof.execute(cref, stage(tmp_path, c, cref, 1, None, "observed"))
    direct = copy.deepcopy(c)
    direct["sampling"]["quota_ledger"] = str(tmp_path / "bare-quota.jsonl")
    dr = frozen(tmp_path / "bare.contract.json", direct)
    bare = entry.execute(dr, stage(tmp_path, direct, dr, 1, None, "bare"))
    a = torch.load(observed["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(bare["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "bounded passive witness " + key)
    assert observed["rng"] == bare["rng"]
    assert observed["quota_state"] == bare["quota_state"]
    assert observed["own_live_certificate"]["actual_calls"]["forward"] == 3
    assert "own_live_certificate" not in bare


@pytest.mark.parametrize("fault", ["model", "optimizer", "private_rng", "old_family"])
def test_own_live_saved_and_private_state_tamper_refuses_bounded_delivery(tmp_path: Path, fault: str) -> None:
    c, cref = contract(tmp_path, split=False)

    class Tampered(proof.OwnLiveProof):
        def after_save(self, context: dict[str, Any]) -> None:
            super().after_save(context)
            if fault == "private_rng":
                context["replay"]._shuffle.random()
                return
            path = context["checkpoint"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if fault == "model":
                next(iter(payload["model"].values())).add_(1)
            elif fault == "optimizer":
                next(iter(payload["opt"]["state"].values()))["step"] += 1
            else:
                payload[entry.global_union.CHECKPOINT_KEY] = {"pretend_old_family": True}
            torch.save(payload, path)

    with pytest.raises(ValueError, match=r"saved payload differs|live state changed"):
        entry.execute(cref, stage(tmp_path, c, cref, 1, None, "tampered"), observer=Tampered())
    assert not (tmp_path / "tampered/boundary.json").exists()
    assert not (tmp_path / "tampered/boundary.pending.json").exists()


def test_cpu_trainer_releases_both_caller_and_generator_numpy_aliases_before_next_allocation(tmp_path: Path) -> None:
    import weakref

    c, cref = contract(tmp_path, split=False)
    released: list[list[Any]] = []

    class Lifetime(proof.OwnLiveProof):
        def before_train(self, context: dict[str, Any]) -> None:
            super().before_train(context)
            replay = context["replay"]
            sample = replay.sample_batch_arrays
            complete = replay.complete_batch
            current: list[Any] = []

            def sampled(batch_size: int):
                assert all(reference() is None for group in released for reference in group)
                arrays = sample(batch_size)
                current[:] = [weakref.ref(value) for value in arrays.values()]
                return arrays

            def completed() -> None:
                complete()
                released.append(list(current))
                assert all(reference() is None for reference in current)
                current.clear()

            replay.sample_batch_arrays = sampled
            replay.complete_batch = completed

    boundary = entry.execute(cref, stage(tmp_path, c, cref, 1, None, "cpu-lifetime"), observer=Lifetime())
    assert boundary["step"] == 3
    assert len(released) == 3
    assert all(reference() is None for group in released for reference in group)
