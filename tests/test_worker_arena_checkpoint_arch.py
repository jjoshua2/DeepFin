"""Worker arena: the best checkpoint is built from its OWN architecture.

On the pre-fix code ``_run_arena`` built the best model from the ACTIVE
manifest config and loaded it with the tolerant loader, so a best checkpoint
of a different width/depth silently kept fresh-initialised tensors and the
arena scored the current net against a half-random opponent.

The methods under test are extracted from ``worker.py``'s AST and executed
against injected collaborators (the same seam as
``test_worker_model_load_identity.py``): importing ``chess_anti_engine.worker``
pulls in compiled extensions, which would make this a test of whether those are
built. The ``real_*`` tests inject the REAL ``torch.load`` / ``build_model`` /
``load_state_dict_tolerant`` / ``model_config_from_arch`` with tiny models, so
the refusal is proven on real state dicts rather than on a fake.

Originally from PR #1110 (commits 2f2911111, 416a4189a); adapted to the
contained-refusal design (``_ArenaBestModelRefused``).
"""

from __future__ import annotations

import ast
import dataclasses
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from chess_anti_engine.model import (
    ARCH_SCHEMA_VERSION,
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
)
from chess_anti_engine.uci.model_loader import model_config_from_arch

WORKER_PY = Path(__file__).resolve().parents[1] / "chess_anti_engine/worker.py"


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


def _worker_class_node() -> ast.ClassDef:
    tree = ast.parse(WORKER_PY.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerSession"
    )


def _exec_nodes(nodes: list[ast.stmt], env: dict[str, Any]) -> None:
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<worker-arena-methods>", "exec"), env)


def _build_session(env: dict[str, Any]) -> Any:
    tree = ast.parse(WORKER_PY.read_text())
    refused = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ArenaBestModelRefused"
    )
    _exec_nodes([refused], env)
    session = _worker_class_node()
    methods = {}
    for name in ("_load_and_compile_model", "_run_arena"):
        node = next(
            n for n in session.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        _exec_nodes([node], env)
        methods[name] = env[name]
    cls = type("ArenaLoadSession", (), methods)
    # Class body comes from the real source AST at runtime; this is the
    # deliberate dynamic boundary for the fake collaborators, not WorkerSession.
    s: Any = cls()
    s.args = SimpleNamespace(
        compile_inference=False, threaded_selfplay=False, compile_mode="default",
        poll_seconds=0.0,
    )
    s.inference_client = None
    s.device = "cpu"
    s._boot_hang_watchdog = SimpleNamespace(stage=lambda _name: nullcontext())
    s.log = SimpleNamespace(info=lambda *_a, **_k: None)
    s._refused_best_sha = None
    s.refused_cls = env["_ArenaBestModelRefused"]
    return s


@pytest.fixture
def harness() -> Any:
    h = SimpleNamespace(
        checkpoints={}, built=[], weights=[], events=[], compiled=[],
        fail_weights=False, fail_planes=False, matches=[], sleeps=[],
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
        pytest.fail("unexpected download or headers")

    env: dict[str, Any] = {
        "dataclasses": dataclasses, "ModelConfig": ModelConfig,
        "model_config_from_arch": model_config_from_arch,
        "torch": SimpleNamespace(load=checkpoint), "build_model": build,
        "load_state_dict_tolerant": weights, "require_model_planes": planes,
        "_maybe_compile_inference_model": compile_model,
        "time": SimpleNamespace(time=lambda: 1.0, sleep=h.sleeps.append),
        "_sha256_file": lambda _path: "best",
        "_download_and_verify_shared": forbidden, "_worker_headers": forbidden,
        "OpeningConfig": SimpleNamespace,
        "play_match_batch": match,
    }
    h.session = _build_session(env)
    h.refused_cls = h.session.refused_cls
    return h


def _arch(cfg: ModelConfig) -> dict[str, Any]:
    return {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}


def _load(h: Any, cfg: ModelConfig, *, embedded: bool = True) -> Any:
    return h.session._load_and_compile_model(
        Path("/synthetic/best.pt"), cfg, label="worker-best", sha_short="best",
        checkpoint_arch=embedded,
    )


@pytest.mark.parametrize(("current_fix", "best_fix"), [(False, True), (True, False)])
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
    """The selfplay model (checkpoint_arch=False) is unchanged: manifest config, tolerant."""
    explicit = ModelConfig(embed_dim=64, history_rep_fix=True, use_gradient_checkpointing=False)
    harness.checkpoints["best.pt"] = {
        "model": {}, "arch": _arch(ModelConfig(embed_dim=96, history_rep_fix=False)),
    }
    loaded = harness.session._load_and_compile_model(
        Path("/synthetic/best.pt"), explicit, label="worker-model", sha_short="best",
    )
    assert loaded.cfg is explicit
    assert harness.weights[0][2]["require_complete"] is False


def test_default_current_model_load_failure_is_not_converted(harness: Any) -> None:
    """A selfplay-model load failure keeps its original exception type."""
    harness.checkpoints["best.pt"] = {"model": {}}
    harness.fail_weights = True
    with pytest.raises(RuntimeError, match="synthetic incomplete load") as info:
        harness.session._load_and_compile_model(
            Path("/synthetic/best.pt"), ModelConfig(), label="worker-model", sha_short="best",
        )
    assert not isinstance(info.value, harness.refused_cls)


@pytest.mark.parametrize("legacy", ["raw", "container", "none", "non-dict"])
def test_legacy_best_falls_back_to_active_config_but_must_load_completely(
    harness: Any, legacy: str,
) -> None:
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
    assert harness.weights[0] == (loaded, state, {"label": "worker-best", "require_complete": True})


@pytest.mark.parametrize("bad", ["missing-schema", "noninteger-schema", "future-schema", "unknown-key"])
def test_invalid_embedded_arch_refuses_before_construction(harness: Any, bad: str) -> None:
    arch = _arch(ModelConfig())
    if bad == "missing-schema":
        del arch["_schema_version"]
        expected = "missing an integer _schema_version"
    elif bad == "noninteger-schema":
        arch["_schema_version"] = "20"
        expected = "missing an integer _schema_version"
    elif bad == "future-schema":
        arch["_schema_version"] = ARCH_SCHEMA_VERSION + 1
        expected = "is newer than this loader"
    else:
        arch["unknown-topology"] = 1
        expected = "embedded arch has unknown keys"
    harness.checkpoints["best.pt"] = {"model": {}, "arch": arch}
    with pytest.raises(harness.refused_cls, match=expected):
        _load(harness, ModelConfig())
    assert not harness.built
    assert not harness.weights
    assert [event[0] for event in harness.events] == ["checkpoint"]


def test_complete_load_failure_prevents_device_transfer(harness: Any) -> None:
    harness.checkpoints["best.pt"] = {"model": {}, "arch": _arch(ModelConfig())}
    harness.fail_weights = True
    with pytest.raises(harness.refused_cls, match="synthetic incomplete"):
        _load(harness, ModelConfig())
    assert harness.weights[0][2]["require_complete"] is True
    assert [event[0] for event in harness.events] == ["checkpoint", "build", "weights"]


def test_slot_guard_receives_checkpoint_model_before_weights_or_device(harness: Any) -> None:
    saved = ModelConfig(input_extra_features="v2_threats")
    harness.checkpoints["best.pt"] = {"model": {}, "arch": _arch(saved)}
    harness.session.inference_client = SimpleNamespace(input_planes=146)
    harness.fail_planes = True
    with pytest.raises(harness.refused_cls, match="slot-plane"):
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


@pytest.mark.parametrize(("active_fix", "best_fix"), [(False, True), (True, False)])
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
def test_failed_best_load_keeps_prior_identity_and_is_not_retried(
    harness: Any, failure: str,
) -> None:
    current, cfg = _prepare_arena(harness, False, True)
    old_best = object()
    harness.session.best_model = old_best
    if failure == "arch":
        harness.checkpoints["best_best.pt"]["arch"]["unsupported-topology"] = 1
        expected = "embedded arch has unknown keys"
    else:
        harness.fail_weights = True
        expected = "synthetic incomplete load"
    with pytest.raises(harness.refused_cls, match=expected):
        harness.session._run_arena({"best_model": {"sha256": "best"}}, {"arena": {}})
    assert harness.session.best_model is old_best
    assert harness.session.last_best_sha == "old"
    assert harness.session.model is current
    assert harness.session.model_cfg_active is cfg
    assert harness.session._refused_best_sha == "best"
    assert not harness.matches
    # Second poll with the same sha: no reload, no match, no raise; just waits.
    n_events = len(harness.events)
    harness.session._run_arena({"best_model": {"sha256": "best"}}, {"arena": {}})
    assert len(harness.events) == n_events
    assert not harness.matches
    assert harness.sleeps == [0.0]


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


# ---------------------------------------------------------------------------
# Real models: real torch.load / build_model / load_state_dict_tolerant.
# ---------------------------------------------------------------------------

_ACTIVE = ModelConfig(embed_dim=64, num_layers=2, num_heads=4, use_gradient_checkpointing=False)
_DEEPER = ModelConfig(embed_dim=64, num_layers=3, num_heads=4, ffn_mult=3.0)


@pytest.fixture
def real_session() -> Any:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("unexpected call")

    env: dict[str, Any] = {
        "dataclasses": dataclasses, "ModelConfig": ModelConfig,
        "model_config_from_arch": model_config_from_arch,
        "torch": torch, "build_model": build_model,
        "load_state_dict_tolerant": load_state_dict_tolerant,
        "require_model_planes": forbidden,
        "_maybe_compile_inference_model": forbidden,
    }
    return _build_session(env)


def _save(path: Path, cfg: ModelConfig, *, embed_arch: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(1234)
    sd = build_model(cfg).state_dict()
    payload: dict[str, Any] = {"model": sd}
    if embed_arch:
        payload["arch"] = _arch(cfg)
    torch.save(payload, str(path))
    return sd


def test_real_best_of_different_depth_loads_every_weight_exactly(
    real_session: Any, tmp_path: Path,
) -> None:
    path = tmp_path / "best.pt"
    saved = _save(path, _DEEPER, embed_arch=True)
    model = real_session._load_and_compile_model(
        path, _ACTIVE, label="worker-best", sha_short="best", checkpoint_arch=True,
    )
    got = model.state_dict()
    assert set(got) == set(saved)
    for key, tensor in saved.items():
        assert torch.equal(got[key], tensor), key


def test_real_arch_less_best_of_different_depth_is_refused(
    real_session: Any, tmp_path: Path,
) -> None:
    """Pre-fix behaviour on a mismatched best: tolerant partial load, no error."""
    path = tmp_path / "best.pt"
    _save(path, _DEEPER, embed_arch=False)
    with pytest.raises(real_session.refused_cls, match="Incomplete state-dict load"):
        real_session._load_and_compile_model(
            path, _ACTIVE, label="worker-best", sha_short="best", checkpoint_arch=True,
        )


def test_real_selfplay_model_load_stays_tolerant(real_session: Any, tmp_path: Path) -> None:
    """checkpoint_arch=False keeps the manifest-authoritative tolerant load."""
    path = tmp_path / "model.pt"
    _save(path, _DEEPER, embed_arch=True)
    model = real_session._load_and_compile_model(
        path, _ACTIVE, label="worker-model", sha_short="m",
    )
    assert len(model.state_dict()) == len(build_model(_ACTIVE).state_dict())


def test_run_contains_arena_refusal() -> None:
    """Source-shape pin (not behavioural): run()'s arena branch catches the refusal.

    ``run()`` cannot be exercised without a server; the behavioural half (the
    refusal is raised, prior identity kept, sha not retried) is covered above.
    An uncaught raise there reaches ``finally: self._cleanup()`` and exits the
    whole worker over one unloadable checkpoint.
    """
    run = next(
        n for n in _worker_class_node().body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    arena_calls = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Try)
        and any(
            isinstance(c, ast.Expr) and isinstance(c.value, ast.Call)
            and isinstance(c.value.func, ast.Attribute) and c.value.func.attr == "_run_arena"
            for c in node.body
        )
    ]
    assert len(arena_calls) == 1
    handled = {
        h.type.id for h in arena_calls[0].handlers if isinstance(h.type, ast.Name)
    }
    assert "_ArenaBestModelRefused" in handled
