"""Owned opt-in provenance guard around the unchanged public UCI entry point.

This supplies no admission, resource budget, playing profile or qualification.
The caller must select one coherent source and an owned operation before use.
Ordinary ``deepfin --checkpoint`` and the public loader remain unchanged.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import threading
from typing import Any
from unittest.mock import patch

SCHEMA = "CLOSED_ONCE_ACTOR_PUBLIC_UCI_OBSERVATION_V1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def guarded_loader(actor: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Observe the one original load; validate it before UCI can get the model.

    No journal/data consumer, forward, optimizer restore or alternate decoder is
    called. These process-local hooks are scoped to a dedicated owned UCI process.
    Do not use the guard alongside another model loader in the same process.
    """
    from scripts import core_closed_once_actor as bridge
    from scripts.core_selfplay_union_stored_row_once_proof import file_hash, fingerprint

    # Freeze the caller's metadata before handing work to the UCI builder thread.
    selected = json.loads(_canonical(actor))
    delivery = bridge.validate_actor(selected)
    expected = delivery["model"]
    checkpoint = Path(selected["path"])
    before = file_hash(checkpoint)
    _require(before[0] == selected["sha256"], "closed UCI checkpoint SHA differs")

    import torch
    from chess_anti_engine.uci import model_loader

    source_root = Path(bridge.__file__).resolve().parents[1]
    for helper, relative in ((model_loader, "chess_anti_engine/uci/model_loader.py"),
                             (file_hash, "scripts/core_selfplay_union_stored_row_once_proof.py"),
                             (fingerprint, "scripts/core_selfplay_union_stored_row_once_proof.py")):
        _require(Path(inspect.getfile(helper)).resolve() == source_root / relative,
                 "closed UCI mixed source namespace")
    original_load = torch.load
    original_build = model_loader.build_model
    original_loader = model_loader.load_model_from_checkpoint
    normalized_arch = asdict(model_loader.model_config_from_arch(selected["architecture"]))
    lock = threading.Lock()
    builds: list[Any] = []
    observed: dict[str, Any] = {
        "schema": SCHEMA, "coherent_source_root": str(source_root),
        "checkpoint": expected["checkpoint"], "delivery": selected["lineage"]["delivery"],
        "family": selected["lineage"]["family"], "expected_step": expected["step"],
        "public_loader_calls": 0, "safe_decodes": 0, "model_builds": 0,
        "validated_before_evaluator": False, "hooks_restored": False,
        "scope": "state/metadata observation only; guard calls no model forward or optimizer",
    }

    def capture_load(*args: Any, **kwargs: Any) -> Any:
        with lock:
            _require(observed["public_loader_calls"] == 1 and observed["safe_decodes"] == 0
                     and len(args) == 1 and isinstance(args[0], (str, Path))
                     and Path(args[0]) == checkpoint and kwargs.get("weights_only") is True,
                     "closed UCI unexpected path, unsafe or duplicate decode")
            observed["safe_decodes"] += 1
        payload = original_load(*args, **kwargs)
        _require(isinstance(payload, dict)
                 and _canonical(payload.get("arch")) == _canonical(selected["architecture"])
                 and type(payload.get("step")) is int and payload["step"] == selected["expected_checkpoint_step"],
                 "closed UCI embedded architecture or step differs")
        state = payload.get("model")
        _require(isinstance(state, dict) and len(state) == 494
                 and all(isinstance(value, torch.Tensor) for value in state.values()),
                 "closed UCI incomplete494 payload")
        observed["payload_model_state_sha256"] = bridge.validate_payload(selected, payload, torch)
        observed["checkpoint_step"] = payload["step"]
        return payload

    def capture_build(config: Any) -> Any:
        with lock:
            _require(observed["safe_decodes"] == 1 and observed["model_builds"] == 0
                     and "payload_model_state_sha256" in observed
                     and _canonical(asdict(config)) == _canonical(normalized_arch),
                     "closed UCI unexpected build or architecture")
            observed["model_builds"] += 1
        model = original_build(config)
        builds.append(model)
        return model

    def load(path: str | Path, *, device: str = "cpu", model_config: Any = None,
             require_complete: bool | None = None) -> Any:
        with lock:
            _require(observed["public_loader_calls"] == 0 and Path(path) == checkpoint
                     and model_config is None and require_complete is not False,
                     "closed UCI unexpected loader path, override or duplicate load")
            observed["public_loader_calls"] += 1
        # This is the original public loader with its normal safe decoder and
        # constructor. The opt-in boundary explicitly requires complete loading.
        model = original_loader(path, device=device, require_complete=True)
        _require(len(builds) == 1 and builds[0] is model, "closed UCI returned a different model")
        state = model.state_dict()
        _require(len(state) == 494 and all(isinstance(value, torch.Tensor) for value in state.values()),
                 "closed UCI incomplete494 returned model")
        actual_hash = fingerprint(state, torch)
        report = getattr(model, "tolerant_load_report", None)
        _require(report is not None and not report.shape_skipped and not report.missing and not report.unexpected,
                 "closed UCI incomplete public load report")
        _require(actual_hash == observed["payload_model_state_sha256"] == expected["model_state_sha256"],
                 "closed UCI actual served model state differs")
        _require(file_hash(checkpoint) == before, "closed UCI checkpoint changed during load")
        observed.update(model_state_sha256=actual_hash, state_entries=len(state),
                        returned_model_id=id(model), checkpoint_stamp=before[1],
                        validated_before_evaluator=True, safe_weights_only=True, complete_load=True)
        return model

    with ExitStack() as hooks:
        main = sys.modules.get("chess_anti_engine.uci.__main__")
        try:
            hooks.enter_context(patch.object(torch, "load", capture_load))
            hooks.enter_context(patch.object(model_loader, "build_model", capture_build))
            hooks.enter_context(patch.object(model_loader, "load_model_from_checkpoint", load))
            # The canonical entry imports its loader by value. Cover an already
            # imported alias too; a fresh import captures the patched function.
            if main is not None:
                filename = getattr(main, "__file__", None)
                _require(isinstance(filename, str)
                         and Path(filename).resolve() == source_root / "chess_anti_engine/uci/__main__.py"
                         and getattr(main, "load_model_from_checkpoint", None) is original_loader,
                         "closed UCI unexpected captured loader alias")
                hooks.enter_context(patch.object(main, "load_model_from_checkpoint", load))
            yield observed
            _require(observed["validated_before_evaluator"] is True
                     and observed["public_loader_calls"] == observed["safe_decodes"] == observed["model_builds"] == 1,
                     "closed UCI exited without exactly one admitted model")
            _require(file_hash(checkpoint) == before, "closed UCI checkpoint changed during serving")
        finally:
            current = sys.modules.get("chess_anti_engine.uci.__main__")
            if main is None and current is not None and getattr(current, "load_model_from_checkpoint", None) is load:
                # A new module's from-import alias is not one of patch.object's
                # original targets. Restore that newly captured alias explicitly.
                current.load_model_from_checkpoint = original_loader
            hooks.close()
            observed["hooks_restored"] = (torch.load is original_load
                                          and model_loader.build_model is original_build
                                          and model_loader.load_model_from_checkpoint is original_loader
                                          and (current is None or current.load_model_from_checkpoint is original_loader))


def _write_witness(path: Path, value: dict[str, Any]) -> None:
    _require(path.is_absolute() and path.resolve() == path and path.parent.is_dir()
             and not path.exists(), "closed UCI witness must be a fresh canonical external file")
    payload = (_canonical(value) + "\n").encode()
    _require(len(payload) <= 4 << 20, "closed UCI witness exceeds compact bound")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_closed_actor_uci(actor_ref: dict[str, str], public_argv: Sequence[str], *, witness_path: Path) -> int:
    """Run the real public CLI once; a witness is not Root/owner acceptance."""
    from scripts import core_finite_learner as core

    output = core.external(str(witness_path))
    _require(output.parent.is_dir() and not output.exists(), "closed UCI witness already exists or lacks parent")
    _, actor = core.read_ref(actor_ref)
    argv = list(public_argv)
    _require(bool(argv) and all(isinstance(arg, str) for arg in argv), "closed UCI requires public arguments")
    witness: dict[str, Any] = {"schema": SCHEMA, "actor": actor_ref, "public_uci_exit_code": None,
                               "error_type": None, "execution_authority_issued": False}
    exit_code: int | None = None
    try:
        with guarded_loader(actor) as observation:
            witness["load_observation"] = observation
            # Execute the public package entry, rather than a parallel parser or
            # engine. The loader guard also checks the path actually parsed by it.
            _require(argv.count("--checkpoint") == 1
                     and argv.index("--checkpoint") + 1 < len(argv)
                     and argv[argv.index("--checkpoint") + 1] == actor["path"],
                     "closed UCI requires the exact explicit --checkpoint file")
            spec = importlib.util.find_spec("chess_anti_engine.uci.__main__")
            _require(spec is not None and spec.origin is not None
                     and Path(spec.origin).resolve() == Path(observation["coherent_source_root"]) / "chess_anti_engine/uci/__main__.py",
                     "closed UCI foreign public entry point")
            old_argv = sys.argv
            try:
                sys.argv = ["deepfin", *argv]
                try:
                    runpy.run_module("chess_anti_engine.uci", run_name="__main__", alter_sys=True)
                    exit_code = 0
                except SystemExit as error:
                    if error.code is None or type(error.code) is int:
                        exit_code = 0 if error.code is None else error.code
                    if exit_code != 0:
                        raise
            finally:
                sys.argv = old_argv
        _require(exit_code == 0, "closed UCI did not finish successfully")
        return 0
    except BaseException as error:
        witness["error_type"] = type(error).__name__
        raise
    finally:
        witness["public_uci_exit_code"] = exit_code
        witness["status"] = ("CLOSED_ACTOR_PUBLIC_UCI_OBSERVED_PENDING_OWNER_CLOSURE"
                             if exit_code == 0 and witness["error_type"] is None
                             else "CLOSED_ACTOR_PUBLIC_UCI_UNSUCCESSFUL")
        _write_witness(output, witness)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--actor-sha256", required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("public_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    _require(args.public_argv[:1] == ["--"], "closed UCI requires literal -- before public arguments")
    return run_closed_actor_uci({"path": args.actor, "sha256": args.actor_sha256},
                                args.public_argv[1:], witness_path=args.witness)


if __name__ == "__main__":
    raise SystemExit(main())
