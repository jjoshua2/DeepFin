"""Synthetic owned-wrapper tests; no experiment or serving qualification.

The inert 494-buffer module cannot forward. Only the capped metadata fixture
uses the existing Tiny training fixture to obtain its real public API shapes;
its subsequently replaced buffers/receipts are explicitly fictional test data.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from chess_anti_engine.uci import __main__ as uci
from chess_anti_engine.uci import model_loader
from scripts import closed_actor_uci as owned
from scripts import core_closed_once_actor as bridge
from scripts import core_finite_learner as core
from scripts.core_selfplay_union_stored_row_once_proof import fingerprint
from tests.test_core_closed_once_actor import actor_fixture
from tests.test_core_finite_learner import frozen


class Buffers(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        for i in range(494):
            self.register_buffer(f"tensor{i}", torch.zeros(1))

    def forward(self, *_args: Any, **_kwargs: Any) -> Any:
        self.forward_calls += 1
        pytest.fail("the admission wrapper must not forward")


def fixture(tmp_path: Path, family: str = "global_union", *, fault: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    tmp_path.mkdir(exist_ok=True)
    actor: dict[str, Any]
    payload: dict[str, Any]
    if family == "capped_cycle":
        from scripts import core_capped_on_policy_cycle as capped
        from scripts import core_capped_on_policy_cycle_proof as proof
        from tests.test_core_capped_on_policy_cycle import cycle_contract, delivery, origin

        original = tmp_path / "origin"
        original.mkdir()
        oc, initial = origin(original)
        c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
        boundary = proof.execute(cref, sref)
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        lineage = delivery(receipts, c, cref, boundary, cycle=True)
        model = bridge.closed_delivery(lineage)["model"]
        actor = {"kind": bridge.KIND, **model["checkpoint"], "architecture": model["architecture"],
                 "expected_checkpoint_step": model["step"], "predecessor": bridge.closed_delivery(lineage)["root"]["closed_binding"],
                 "lineage": lineage}
        payload = {"model": {}, "arch": model["architecture"], "step": model["step"],
                   capped.CHECKPOINT_KEY: model["tag"], "frozen_reference_policy_anchor_v1": None}
    else:
        arch = json.loads(core.canonical({"_schema_version": ARCH_SCHEMA_VERSION,
                                         **asdict(ModelConfig(kind="tiny"))}))
        actor, payload = actor_fixture(tmp_path, family=family, architecture=arch, full_state=True)
    accepted = copy.deepcopy(bridge.validate_actor(actor))
    payload["model"] = {f"tensor{i}": torch.tensor([float(i)]) for i in range(494)}
    expected_hash = fingerprint(payload["model"], torch)
    # Build self-consistent accepted metadata around the expected synthetic state;
    # faults alter only the actual payload, not the accepted model identity.
    actual: dict[str, Any] = copy.deepcopy(payload)
    if fault == "arch":
        actual["arch"]["num_layers"] += 1
    elif fault == "step":
        actual["step"] += 1
    elif fault == "partial":
        actual["model"].pop("tensor493")
    elif fault == "state":
        actual["model"]["tensor0"] += 1
    elif fault == "tag":
        actual[accepted["model"]["tag_key"]]["identity"]["run_uuid"] = "bad"
    elif fault == "numpy":
        actual["unsupported_rng"] = np.random.RandomState(1).get_state()
    elif fault == "reference":
        actual["frozen_reference_policy_anchor_v1"] = None
    checkpoint = Path(actor["path"])
    checkpoint.chmod(0o644)
    torch.save(actual, checkpoint)
    checkpoint.chmod(0o444)
    checkpoint_ref = {"path": str(checkpoint), "sha256": core.file_sha(checkpoint)}
    boundary = accepted["boundary"]
    boundary["checkpoint"] = checkpoint_ref
    cert = boundary["own_live_certificate"]
    cert["checkpoint"] = checkpoint_ref
    cert["saved_model_state_sha256"] = expected_hash
    cert["saved_model"]["model_state_sha256"] = expected_hash
    boundary_path = checkpoint.parent / "boundary.json"
    boundary_path.chmod(0o644)
    boundary_path.write_text(core.canonical(boundary) + "\n")
    boundary_path.chmod(0o444)
    bref = {"path": str(boundary_path), "sha256": core.file_sha(boundary_path)}
    root = accepted["root"]
    root["boundary"] = bref
    root["closed_binding"]["boundary"] = bref
    root["model"]["checkpoint"] = checkpoint_ref
    root["model"]["model_state_sha256"] = expected_hash
    rref = frozen(tmp_path / "uci-root.json", root)
    prior_independent = bridge.read(actor["lineage"]["delivery"]["independent"])
    iref = frozen(tmp_path / "uci-independent.json", dict(root, status=prior_independent["status"], root_acceptance=rref, findings=[]))
    actor.update(**checkpoint_ref, predecessor=root["closed_binding"])
    actor["lineage"]["delivery"] = {"root": rref, "independent": iref}
    # Positive metadata baseline is real, serialized and checked before faults.
    assert bridge.validate_actor(actor)["model"]["model_state_sha256"] == expected_hash
    return actor, actual


def build_inert(monkeypatch: pytest.MonkeyPatch) -> list[Buffers]:
    built: list[Buffers] = []

    def build(_config: Any) -> Buffers:
        model = Buffers()
        built.append(model)
        return model

    monkeypatch.setattr(model_loader, "build_model", build)
    return built


@pytest.mark.parametrize("family", ["global_union", "policy_anchor", "capped_cycle"])
def test_real_public_loader_validates_same_object_and_restores_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str) -> None:
    actor, _ = fixture(tmp_path, family)
    built = build_inert(monkeypatch)
    originals = (torch.load, model_loader.build_model, model_loader.load_model_from_checkpoint, uci.load_model_from_checkpoint)
    with owned.guarded_loader(actor) as observed:
        actual = uci._load_models(actor["path"], ["cpu"])[0]
        assert actual is built[0]
        assert observed["validated_before_evaluator"] is True
        assert observed["safe_decodes"] == observed["model_builds"] == observed["public_loader_calls"] == 1
        assert observed["state_entries"] == 494
        assert observed["returned_model_id"] == id(actual)
        assert not actual.training
        assert actual.forward_calls == 0
    assert observed["hooks_restored"] is True
    assert (torch.load, model_loader.build_model, model_loader.load_model_from_checkpoint, uci.load_model_from_checkpoint) == originals


@pytest.mark.parametrize("fault", ["arch", "step", "partial", "state", "tag", "reference"])
def test_actual_payload_refusals_happen_before_constructor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    actor, _ = fixture(tmp_path, "policy_anchor", fault=fault)
    built = build_inert(monkeypatch)
    original = torch.load
    with pytest.raises(ValueError, match=r"closed UCI|closed once actor"), owned.guarded_loader(actor):
        uci._load_models(actor["path"], ["cpu"])
    assert built == []
    assert torch.load is original


@pytest.mark.parametrize("fault", ["missing_pair", "independent", "path", "hash", "arch", "step"])
def test_metadata_refusals_precede_any_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    actor, _ = fixture(tmp_path)
    if fault == "missing_pair":
        actor["lineage"]["delivery"].pop("independent")
    elif fault == "independent":
        pair = actor["lineage"]["delivery"]
        value = bridge.read(pair["independent"])
        value["root_acceptance"]["sha256"] = "0" * 64
        pair["independent"] = frozen(tmp_path / "wrong-ind.json", value)
    elif fault == "path":
        actor["path"] = str(tmp_path / "different.pt")
    elif fault == "hash":
        actor["sha256"] = "0" * 64
    elif fault == "arch":
        actor["architecture"]["num_layers"] += 1
    else:
        actor["expected_checkpoint_step"] += 1
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("invalid caller metadata reached decoder")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="closed once actor"), owned.guarded_loader(actor):
        pytest.fail("invalid caller metadata was accepted")


@pytest.mark.parametrize("fault", ["missing", "shape", "changed_after_load"])
def test_actual_returned_state_requires_complete_load_and_typed_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    actor, _ = fixture(tmp_path)
    built = build_inert(monkeypatch)
    original = model_loader.build_model
    def corrupt(config: Any) -> Buffers:
        model = cast(Buffers, original(config))
        if fault == "missing":
            model.register_buffer("extra", torch.zeros(1))
        elif fault == "shape":
            model.tensor0 = torch.zeros(2)
        else:
            original_to = model.to
            def changed(*args: Any, **kwargs: Any) -> Buffers:
                original_to(*args, **kwargs)
                getattr(model, "tensor0").add_(1)
                return model
            monkeypatch.setattr(model, "to", changed)
        return model
    monkeypatch.setattr(model_loader, "build_model", corrupt)
    with pytest.raises((ValueError, RuntimeError), match=r"Incomplete state-dict|closed UCI"), owned.guarded_loader(actor):
        uci._load_models(actor["path"], ["cpu"])
    assert len(built) == 1
    assert built[0].forward_calls == 0


def test_unsafe_numpy_payload_is_refused_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path, fault="numpy")
    built = build_inert(monkeypatch)
    actual_load = torch.load
    calls: list[bool] = []
    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["weights_only"])
        return actual_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", counted)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"), owned.guarded_loader(actor):
        uci._load_models(actor["path"], ["cpu"])
    assert calls == [True]
    assert built == []


def test_fresh_from_import_alias_is_restored_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path)
    original = model_loader.load_model_from_checkpoint
    # A real fresh canonical import captures the guard by value, just as runpy's
    # import-light package entry does. Restore the prior test module afterwards.
    with monkeypatch.context() as context:
        context.delitem(sys.modules, "chess_anti_engine.uci.__main__")
        imported: list[Any] = []
        def fail_after_import() -> None:
            import importlib
            fresh = importlib.import_module("chess_anti_engine.uci.__main__")
            imported.append(fresh)
            assert fresh.load_model_from_checkpoint is model_loader.load_model_from_checkpoint
            assert fresh.load_model_from_checkpoint is not original
            raise RuntimeError("synthetic")
        with pytest.raises(RuntimeError, match="synthetic"), owned.guarded_loader(actor) as observed:
            fail_after_import()
        assert observed["hooks_restored"] is True
        assert imported[0].load_model_from_checkpoint is original


def test_real_public_cli_reaches_evaluator_with_same_admitted_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path)
    built = build_inert(monkeypatch)
    seen: list[Any] = []
    class Engine:
        quit_requested = False
        exit_code = 0
        def warmup_search(self) -> None:
            pass
        def dispatch(self, _cmd: Any) -> None:
            pass
        def close(self) -> None:
            pass
    def evaluator(models: list[Any], *_args: Any, **_kwargs: Any) -> Any:
        assert models == [built[0]]
        assert models[0].forward_calls == 0
        seen.extend(models)
        return lambda *_args, **_kwargs: object()
    monkeypatch.setattr(uci, "enforce_startup_gate", lambda: None)
    monkeypatch.setattr(uci, "_make_evaluator_factory", evaluator)
    monkeypatch.setattr(uci, "_build_engine", lambda **_kwargs: Engine())
    monkeypatch.setattr(sys, "stdin", iter(["isready\n", "quit\n"]))
    aref = frozen(tmp_path / "actor.json", actor)
    witness = tmp_path / "witness.json"
    assert owned.main(["--actor", aref["path"], "--actor-sha256", aref["sha256"], "--witness", str(witness),
                       "--", "--checkpoint", actor["path"], "--device", "cpu", "--no-compile"]) == 0
    observed = json.loads(witness.read_text())["load_observation"]
    assert seen == built
    assert len(seen) == 1
    assert seen[0].forward_calls == 0
    assert observed["validated_before_evaluator"]
    assert observed["hooks_restored"]
    assert witness.stat().st_mode & 0o222 == 0


def test_ordinary_public_loader_remains_unwrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_inert(monkeypatch)
    path = tmp_path / "ordinary.pt"
    torch.save({"model": Buffers().state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION,
                **asdict(ModelConfig(kind="tiny"))}, "step": 680276}, path)
    actual = model_loader.load_model_from_checkpoint(path, device="cpu")
    assert actual is built[0]
    assert actual.forward_calls == 0
    assert not actual.training


def test_changed_checkpoint_bytes_refuse_before_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path)
    checkpoint = Path(actor["path"])
    checkpoint.chmod(0o644)
    with checkpoint.open("ab") as stream:
        stream.write(b"changed-after-acceptance")
    checkpoint.chmod(0o444)
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("changed checkpoint reached decoder")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="checkpoint SHA differs"), owned.guarded_loader(actor):
        pytest.fail("changed checkpoint was accepted")


@pytest.mark.parametrize("fault", ["different_cli_path", "duplicate_loader", "early_success"])
def test_public_path_and_single_load_refusals_restore_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    actor, _ = fixture(tmp_path)
    build_inert(monkeypatch)
    original = model_loader.load_model_from_checkpoint
    if fault == "different_cli_path":
        aref = frozen(tmp_path / "actor.json", actor)
        with pytest.raises(ValueError, match="exact explicit --checkpoint"):
            owned.run_closed_actor_uci(aref, ["--checkpoint", str(tmp_path / "other.pt")], witness_path=tmp_path / "failed-witness.json")
        witness = json.loads((tmp_path / "failed-witness.json").read_text())
        assert witness["status"] == "CLOSED_ACTOR_PUBLIC_UCI_UNSUCCESSFUL"
        assert witness["load_observation"]["safe_decodes"] == 0
    else:
        def exercise() -> None:
            with owned.guarded_loader(actor):
                if fault == "duplicate_loader":
                    uci._load_models(actor["path"], ["cpu"])
                    uci._load_models(actor["path"], ["cpu"])
                # A help/selfcheck/early exit cannot turn zero loads into success.
        with pytest.raises(ValueError, match="closed UCI"):
            exercise()
    assert model_loader.load_model_from_checkpoint is original
    assert uci.load_model_from_checkpoint is original


def test_selfcheck_refuses_before_decode_dispatch_or_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path)
    aref = frozen(tmp_path / "actor.json", actor)
    witness_path = tmp_path / "refused-selfcheck.json"
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("diagnostic mode reached decoder, dispatcher or child")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(owned.runpy, "run_module", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="diagnostic --selfcheck is unsupported"):
        owned.main(["--actor", aref["path"], "--actor-sha256", aref["sha256"], "--witness", str(witness_path),
                    "--", "--checkpoint", actor["path"], "--selfcheck"])
    witness = json.loads(witness_path.read_text())
    assert witness["status"] == "CLOSED_ACTOR_PUBLIC_UCI_UNSUCCESSFUL"
    assert witness["error_type"] == "ValueError"
    assert witness["public_uci_exit_code"] is None
    assert "load_observation" not in witness


def test_later_unrelated_tensor_load_and_constructor_are_not_intercepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actor, _ = fixture(tmp_path)
    built = build_inert(monkeypatch)
    cache = tmp_path / "unrelated-cache.pt"
    torch.save({"cache": torch.tensor([42.0])}, cache)
    original_load = torch.load
    original_build = model_loader.build_model
    original_public_loader = model_loader.load_model_from_checkpoint
    with owned.guarded_loader(actor) as observed:
        admitted = uci._load_models(actor["path"], ["cpu"])[0]
        assert torch.load is original_load
        assert model_loader.build_model is original_build
        cached = torch.load(cache, map_location="cpu", weights_only=True)
        later = model_loader.build_model(model_loader.model_config_from_arch(actor["architecture"]))
        assert cached["cache"].tolist() == [42.0]
        assert later is built[1]
        assert admitted is built[0]
        assert later is not admitted
        assert all(model.forward_calls == 0 for model in built)
        assert observed["safe_decodes"] == observed["model_builds"] == observed["public_loader_calls"] == 1
        assert uci.load_model_from_checkpoint is model_loader.load_model_from_checkpoint
        assert uci.load_model_from_checkpoint is not original_public_loader
    assert observed["hooks_restored"] is True
