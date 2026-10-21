"""Checkpoint architecture wiring with fake models, loads, devices and match."""

from __future__ import annotations

import ast
import dataclasses
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from chess_anti_engine.uci.model_loader import model_config_from_arch


class MatchBoundary(Exception):
    pass


class MemoryPath(PurePosixPath):
    def exists(self) -> bool:
        return True


class FakeModel:
    def __init__(self, cfg: ModelConfig, events: list[Any]) -> None:
        self.cfg = cfg
        self.history_rep_fix = cfg.history_rep_fix
        self.input_history_encoding = cfg.input_history_encoding
        self.input_extra_features = cfg.input_extra_features
        self._inference_only = False
        self.events = events

    def to(self, device: Any) -> FakeModel:
        self.events.append(("to", self, device))
        return self

    def eval(self) -> FakeModel:
        self.events.append(("eval", self))
        return self


@pytest.fixture
def harness() -> Any:
    h = SimpleNamespace(
        checkpoints={}, built=[], weights=[], events=[], compiled=[],
        fail_weights=False, fail_planes=False, matches=[],
    )

    def checkpoint(path: str, **kwargs: Any) -> Any:
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        h.events.append(("checkpoint", path))
        return h.checkpoints[Path(path).name]

    def build(cfg: ModelConfig) -> FakeModel:
        h.events.append(("build", cfg))
        model = FakeModel(cfg, h.events)
        h.built.append(model)
        return model

    def weights(model: FakeModel, state: Any, **kwargs: Any) -> None:
        h.weights.append((model, state, kwargs))
        h.events.append(("weights", model))
        if h.fail_weights:
            raise RuntimeError("synthetic incomplete load")

    def planes(model: FakeModel, count: int, **_kwargs: Any) -> None:
        h.events.append(("planes", model, count))
        if h.fail_planes:
            raise ValueError("synthetic slot-plane mismatch")

    def compile_model(model: FakeModel, **kwargs: Any) -> Any:
        h.compiled.append((model, kwargs))
        h.events.append(("compile", model))
        return SimpleNamespace(wrapped=model)

    def match(a: FakeModel, b: FakeModel, **kwargs: Any) -> None:
        h.matches.append((a, b, kwargs))
        raise MatchBoundary("stop before match execution and pending-output writes")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("unexpected download, headers or sleep")

    env: dict[str, Any] = {
        "dataclasses": dataclasses, "ModelConfig": ModelConfig,
        "model_config_from_arch": model_config_from_arch,
        "torch": SimpleNamespace(load=checkpoint), "build_model": build,
        "load_state_dict_tolerant": weights, "require_model_planes": planes,
        "_maybe_compile_inference_model": compile_model,
        "time": SimpleNamespace(time=lambda: 1.0, sleep=forbidden),
        "_sha256_file": lambda _path: "best",
        "_download_and_verify_shared": forbidden, "_worker_headers": forbidden,
        "OpeningConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "play_match_batch": match,
    }
    source = (Path(__file__).resolve().parents[1] / "chess_anti_engine/worker.py").read_text()
    tree = ast.parse(source)
    session = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorkerSession")
    methods = {}
    for name in ("_load_and_compile_model", "_run_arena"):
        node = next(node for node in session.body if isinstance(node, ast.FunctionDef) and node.name == name)
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<worker-arena-methods>", "exec"), env)
        methods[name] = env[name]
    cls = type("ArenaLoadSession", (), methods)
    h.session = cls()
    h.session.args = SimpleNamespace(
        compile_inference=False, threaded_selfplay=False, compile_mode="default",
    )
    h.session.inference_client = None
    h.session.device = "cpu"
    h.session._boot_hang_watchdog = SimpleNamespace(stage=lambda _name: nullcontext())
    h.session.log = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    return h


def _arch(cfg: ModelConfig) -> dict[str, Any]:
    return {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}


def _load(h: Any, cfg: ModelConfig, *, embedded: bool = True) -> Any:
    return h.session._load_and_compile_model(
        Path("/synthetic/best.pt"), cfg, label="worker-best", sha_short="best",
        checkpoint_arch=embedded,
    )


@pytest.mark.parametrize("current_fix,best_fix", [(False, True), (True, False)])
def test_best_uses_own_topology_and_repetition_identity(
    harness: Any, current_fix: bool, best_fix: bool,
) -> None:
    active = ModelConfig(embed_dim=64, num_layers=2, history_rep_fix=current_fix)
    saved = ModelConfig(
        embed_dim=96, num_layers=3, embed_dim_by_layer=(96, 96, 96),
        ffn_mult_by_layer=(2.0, 3.0, 4.0), history_rep_fix=best_fix,
        input_history_encoding="lc0_root", use_gradient_checkpointing=True,
    )
    before = dataclasses.asdict(active)
    state = {"synthetic-weight": object()}
    harness.checkpoints["best.pt"] = {"model": state, "arch": _arch(saved)}
    loaded = _load(harness, active)
    expected = dataclasses.replace(saved, use_gradient_checkpointing=False)
    assert loaded.cfg == expected
    assert loaded.history_rep_fix is best_fix
    assert loaded.input_history_encoding == "lc0_root"
    assert harness.weights == [(loaded, state, {"label": "worker-best", "require_complete": True})]
    assert dataclasses.asdict(active) == before
    assert loaded._inference_only
    assert [event[0] for event in harness.events] == ["checkpoint", "build", "weights", "to", "eval"]


def test_default_current_model_load_keeps_manifest_authority(harness: Any) -> None:
    explicit = ModelConfig(embed_dim=64, history_rep_fix=True, use_gradient_checkpointing=False)
    harness.checkpoints["best.pt"] = {
        "model": {}, "arch": _arch(ModelConfig(embed_dim=96, history_rep_fix=False)),
    }
    loaded = harness.session._load_and_compile_model(
        Path("/synthetic/best.pt"), explicit, label="worker-model", sha_short="best",
    )
    assert loaded.cfg is explicit
    assert harness.weights[0][2]["require_complete"] is False


@pytest.mark.parametrize("legacy", ["raw", "container", "none", "non-dict"])
def test_legacy_best_retains_explicit_config_fallback(harness: Any, legacy: str) -> None:
    cfg = ModelConfig(history_rep_fix=True, use_gradient_checkpointing=False)
    state = {"synthetic-weight": object()}
    payload: Any = state if legacy == "raw" else {"model": state}
    if legacy == "none":
        payload["arch"] = None
    elif legacy == "non-dict":
        payload["arch"] = "legacy metadata"
    harness.checkpoints["best.pt"] = payload
    loaded = _load(harness, cfg)
    assert loaded.cfg is cfg
    assert harness.weights[0] == (loaded, state, {"label": "worker-best", "require_complete": False})


@pytest.mark.parametrize("bad", ["missing-schema", "noninteger-schema", "future-schema", "unknown-key"])
def test_invalid_embedded_arch_refuses_before_construction(harness: Any, bad: str) -> None:
    arch = _arch(ModelConfig())
    if bad == "missing-schema":
        del arch["_schema_version"]
    elif bad == "noninteger-schema":
        arch["_schema_version"] = "20"
    elif bad == "future-schema":
        arch["_schema_version"] = ARCH_SCHEMA_VERSION + 1
    else:
        arch["unknown-topology"] = 1
    harness.checkpoints["best.pt"] = {"model": {}, "arch": arch}
    with pytest.raises(ValueError):
        _load(harness, ModelConfig())
    assert not harness.built
    assert not harness.weights
    assert [event[0] for event in harness.events] == ["checkpoint"]


def test_minimal_legacy_arch_keeps_serving_decoder_defaults(harness: Any) -> None:
    harness.checkpoints["best.pt"] = {
        "model": {}, "arch": {"_schema_version": ARCH_SCHEMA_VERSION, "kind": "tiny"},
    }
    loaded = _load(harness, ModelConfig(history_rep_fix=True))
    assert loaded.cfg.kind == "tiny"
    assert loaded.history_rep_fix is False
    assert harness.weights[0][2]["require_complete"] is True


def test_complete_load_failure_prevents_device_transfer(harness: Any) -> None:
    harness.checkpoints["best.pt"] = {"model": {}, "arch": _arch(ModelConfig())}
    harness.fail_weights = True
    with pytest.raises(RuntimeError, match="synthetic incomplete"):
        _load(harness, ModelConfig())
    assert harness.weights[0][2]["require_complete"] is True
    assert [event[0] for event in harness.events] == ["checkpoint", "build", "weights"]


def test_slot_guard_receives_checkpoint_model_before_weights_or_device(harness: Any) -> None:
    saved = ModelConfig(input_extra_features="v2_threats")
    harness.checkpoints["best.pt"] = {"model": {}, "arch": _arch(saved)}
    harness.session.inference_client = SimpleNamespace(input_planes=146)
    harness.fail_planes = True
    with pytest.raises(ValueError, match="slot-plane"):
        _load(harness, ModelConfig(input_extra_features="v1"))
    guard = next(event for event in harness.events if event[0] == "planes")
    assert guard[1].cfg.input_extra_features == saved.input_extra_features
    assert guard[2] == 146
    assert not harness.weights
    assert not any(event[0] == "to" for event in harness.events)


def _prepare_arena(h: Any, active_fix: bool, best_fix: bool) -> tuple[FakeModel, ModelConfig]:
    cfg = ModelConfig(history_rep_fix=active_fix, use_gradient_checkpointing=False)
    current = FakeModel(cfg, h.events)
    session = h.session
    session.model, session.model_cfg_active = current, cfg
    session.model_sha, session.last_best_sha, session.best_model = "current", "old", None
    session.cache_dir = MemoryPath("/synthetic")
    session.rng, session.opening_book_path = object(), None
    h.checkpoints["best_best.pt"] = {"model": {}, "arch": _arch(ModelConfig(history_rep_fix=best_fix))}
    return current, cfg


@pytest.mark.parametrize("active_fix,best_fix", [(False, True), (True, False)])
def test_arena_caller_constructs_independent_best_identity(
    harness: Any, active_fix: bool, best_fix: bool,
) -> None:
    current, cfg = _prepare_arena(harness, active_fix, best_fix)
    with pytest.raises(MatchBoundary):
        harness.session._run_arena({"best_model": {"sha256": "best"}}, {"arena": {}})
    a, b, _kwargs = harness.matches[0]
    assert a is current
    assert a.history_rep_fix is active_fix
    assert b is harness.session.best_model
    assert b.history_rep_fix is best_fix
    assert harness.session.model_cfg_active is cfg
    assert harness.session.last_best_sha == "best"
    assert harness.weights[0][2]["require_complete"] is True


@pytest.mark.parametrize("failure", ["arch", "weights"])
def test_failed_best_load_keeps_prior_arena_identity(harness: Any, failure: str) -> None:
    current, cfg = _prepare_arena(harness, False, True)
    old_best = object()
    harness.session.best_model = old_best
    if failure == "arch":
        harness.checkpoints["best_best.pt"]["arch"]["unsupported-topology"] = 1
    else:
        harness.fail_weights = True
    with pytest.raises((ValueError, RuntimeError)):
        harness.session._run_arena({"best_model": {"sha256": "best"}}, {"arena": {}})
    assert harness.session.best_model is old_best
    assert harness.session.last_best_sha == "old"
    assert harness.session.model is current
    assert harness.session.model_cfg_active is cfg
    assert not harness.matches


def test_compile_receives_checkpoint_model_after_load(harness: Any) -> None:
    harness.checkpoints["best.pt"] = {
        "model": {}, "arch": _arch(ModelConfig(history_rep_fix=True)),
    }
    harness.session.args.compile_inference = True
    loaded = _load(harness, ModelConfig(history_rep_fix=False))
    compiled_model, kwargs = harness.compiled[0]
    assert compiled_model.history_rep_fix is True
    assert kwargs == {"device": "cpu", "mode": "default", "use_fp8": False}
    assert loaded.wrapped is compiled_model
    assert [event[0] for event in harness.events] == [
        "checkpoint", "build", "weights", "to", "eval", "compile",
    ]
