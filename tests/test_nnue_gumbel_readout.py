from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.nnue import _nnue_ext
from scripts import gen_random_selfplay_shards as gen
from scripts import nnue_gumbel_readout as readout
from tests.test_nnue_native_eval import write_synthetic_pack


# ── the exact key sets the compiled extension publishes ──────────────────────
# ⚑⚑ A FAKE THAT PUBLISHES A SCHEMA THE C LAYER DOES NOT IS A TEST SUITE THAT
# CANNOT SEE THE BUG. It happened, and it cost the headline cell: `fastq_stats`
# shipped a Py_BuildValue format string with 25 specifiers for 26 pairs, so the
# LAST pair -- `see_recapture_exempt` -- was silently dropped from the dict.
# `ReadoutArmSource` subscripts every consumed knob, so every FastQ worker raised
# KeyError before playing a ply; the fake published the key, so every test here
# passed. `test_the_fake_publishes_exactly_the_c_layers_schema` reads all three
# key sets off LIVE handles and refuses to let the fake drift from them again.
_ARM_COUNTER_KEYS = (
    "calls", "calls_in_check", "nodes", "resolved_leaves", "terminal_mate",
    "terminal_draw", "depth_cutoffs", "qnodes", "qterminal_draw", "qply_cutoffs",
    "nnue_evals", "dag_nodes_interned", "dag_hits_within_call",
    "dag_hits_cross_call", "dag_budget_trips",
)
_ARM_PEAK_KEYS = ("max_depth_seen", "qmax_ply_seen")
_ARM_STORE_KEYS = ("dag_node_count", "dag_edge_count", "dag_memory_bytes")
_FASTQ_COUNTER_KEYS = (
    "calls", "nodes", "evasion_nodes", "nodes_created", "nodes_created_in_check",
    "nnue_evals", "hits_within_call", "hits_cross_call", "quiet_certificates",
    "quiet_certificate_hits", "quiet_returns", "see_prunes", "delta_prunes",
    "recapture_exemptions", "stand_pat_cutoffs", "move_cutoffs", "budget_trips",
    "path_ceilings", "cycle_draws", "terminal_mate", "terminal_draw",
)

#: One ``arm_dag_stats`` snapshot, satisfying the identity ``arm_dag_stats``'s own
#: docstring names: ``state_inits + state_makes == node_count``.
_DAG_SNAPSHOT: dict[str, int] = {
    "root_id": 0,
    "node_count": 110,
    "edge_count": 170,
    "payload_capacity": 256,
    "probes": 200,
    "hits": 50,
    "inserts": 110,
    "collision_steps": 7,
    "edge_reuses": 3,
    "state_inits": 40,
    "state_makes": 70,
    "nnue_evals": 108,
    "node_reuses": 12,
    "dag_memory_bytes": 12345,
    "nnue_payload_bytes": 6789,
    "memory_bytes": 19134,
}


class _FakeHandle:
    """A context that SNAPSHOTS the globals at open, exactly as the C one does."""

    def __init__(self, arm: str, arm_cfg: dict[str, int], fastq_cfg: dict[str, int]):
        self.arm = arm
        self.arm_cfg = dict(arm_cfg)
        self.fastq_cfg = dict(fastq_cfg)


class FakeExt:
    RESOLVER_MAX_DEPTH = 12
    QSEARCH_MAX_PLY = 4
    QSEARCH_CHECK_PLIES = 1
    QSEARCH_DAG_NODE_CAP = 0
    FASTQ_MAX_QPLY = 4
    FASTQ_NODE_CAP = 32
    FASTQ_DELTA_MARGIN = 200
    FASTQ_RECAPTURE_EXEMPT = 1
    RESOLVER_MATE_BASE = 100000
    RESOLVER_MAX_PLIES = 128
    RESOLVER_MATE_PLY_STEP = 1

    def __init__(
        self,
        *,
        clamp_node_cap: int | None = None,
        arm_counters: dict[str, int] | None = None,
        fastq_counters: dict[str, int] | None = None,
        dag_snapshot: dict[str, int] | None = None,
    ) -> None:
        self.events: list[tuple[Any, ...]] = []
        #: The GLOBALS a future arm_open() will snapshot -- not a context.
        self.arm_cfg = {
            "resolver_max_depth": self.RESOLVER_MAX_DEPTH,
            "qsearch_max_ply": self.QSEARCH_MAX_PLY,
            "qsearch_check_plies": self.QSEARCH_CHECK_PLIES,
            "dag_node_cap": self.QSEARCH_DAG_NODE_CAP,
        }
        self.fastq_cfg = {
            "max_qply": self.FASTQ_MAX_QPLY,
            "node_cap": self.FASTQ_NODE_CAP,
            "delta_margin": self.FASTQ_DELTA_MARGIN,
            "see_recapture_exempt": self.FASTQ_RECAPTURE_EXEMPT,
        }
        #: When set, ``fastq_set_config`` CLAMPS node_cap to this value and
        #: returns the clamped configuration -- the shape a real C setter has.
        #: ⚑ THE ECHO AND THE CONTEXT THEN AGREE WITH EACH OTHER AND NOT WITH
        #: THE CALLER, which is precisely what a requested-vs-realized check
        #: built on the setter's return value cannot see: it would compare the
        #: clamped value against itself and pass.
        self.clamp_node_cap = clamp_node_cap
        self.arm_counters = dict(arm_counters or {})
        self.fastq_counters = dict(fastq_counters or {})
        self.dag_snapshot = dict(dag_snapshot or _DAG_SNAPSHOT)
        self.resets = 0

    def set_arm_config(self, resolver: int, qply: int, checks: int, cap: int = 0):
        self.events.append(("set_arm_config", resolver, qply, checks, cap))
        self.arm_cfg = {
            "resolver_max_depth": resolver,
            "qsearch_max_ply": qply,
            "qsearch_check_plies": checks,
            "dag_node_cap": cap,
        }
        return dict(self.arm_cfg)

    def fastq_set_config(self, max_qply: int, node_cap: int, margin: int, exempt: int):
        self.events.append(("fastq_set_config", max_qply, node_cap, margin, exempt))
        stored_cap = node_cap if self.clamp_node_cap is None else self.clamp_node_cap
        self.fastq_cfg = {
            "max_qply": max_qply,
            "node_cap": stored_cap,
            "delta_margin": margin,
            "see_recapture_exempt": exempt,
        }
        # ⚑ The echo is what was STORED, clamp included -- as a C setter's is.
        return dict(self.fastq_cfg)

    def arm_open(self, arm: str, path: str):
        self.events.append(("arm_open", arm, path))
        return _FakeHandle(arm, self.arm_cfg, self.fastq_cfg)

    def arm_stats(self, handle: _FakeHandle) -> dict[str, int]:
        if handle.arm == readout.ARM_FASTQ:
            raise AssertionError("FastQ must never use arm_stats")
        out: dict[str, int] = dict.fromkeys(
            _ARM_COUNTER_KEYS + _ARM_PEAK_KEYS + _ARM_STORE_KEYS, 0,
        )
        out.update(handle.arm_cfg)
        out["dag_enabled"] = int(handle.arm == readout.ARM_QSEARCH_DAG)
        out.update(self.arm_counters)
        return out

    def fastq_stats(self, handle: _FakeHandle) -> dict[str, int]:
        if handle.arm != readout.ARM_FASTQ:
            raise AssertionError("non-FastQ arm must never use fastq_stats")
        out: dict[str, int] = dict.fromkeys(_FASTQ_COUNTER_KEYS, 0)
        out["max_ply_seen"] = 0
        out.update(handle.fastq_cfg)
        out.update(self.fastq_counters)
        return out

    def arm_dag_stats(self, handle: _FakeHandle) -> dict[str, int]:
        if handle.arm == readout.ARM_QSEARCH:
            raise AssertionError("plain qsearch has no DAG")
        return dict(self.dag_snapshot)

    def arm_dag_reset(self, handle: _FakeHandle):
        if handle.arm == readout.ARM_QSEARCH:
            raise AssertionError("plain qsearch has no DAG")
        self.resets += 1

    def load(self, path: str):
        return ("weights", path)

    def source_sha256(self, _handle: object):
        return "a" * 64

    def simd_active(self):
        return False


@pytest.fixture(scope="module")
def synthetic_pack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("readout") / "tiny.pack"
    write_synthetic_pack(path)
    return path


@pytest.fixture(scope="module", params=["synthetic", "real"])
def live_pack(request: pytest.FixtureRequest, synthetic_pack: Path) -> Path:
    """Both nets from one body; the real one skips when the env var is unset."""
    if request.param == "synthetic":
        return synthetic_pack
    env = os.environ.get("CAE_NNUE_TEST_PACK")
    if not env:
        pytest.skip("needs the real NNUE pack (set CAE_NNUE_TEST_PACK)")
    path = Path(env)
    if not path.is_file():
        pytest.skip(f"CAE_NNUE_TEST_PACK does not exist: {path}")
    return path


@pytest.fixture(autouse=True)
def _extension_defaults():
    """Leave the compiled globals as they were found.

    The live-handle tests set knobs, and `set_arm_config` / `fastq_set_config`
    are PROCESS-wide: a test that inherited another's extreme would assert the
    wrong configuration by accident.
    """
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH,
        _nnue_ext.QSEARCH_MAX_PLY,
        _nnue_ext.QSEARCH_CHECK_PLIES,
        _nnue_ext.QSEARCH_DAG_NODE_CAP,
    )
    _nnue_ext.fastq_set_config(
        _nnue_ext.FASTQ_MAX_QPLY,
        _nnue_ext.FASTQ_NODE_CAP,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )
    yield
    _nnue_ext.set_arm_config(
        _nnue_ext.RESOLVER_MAX_DEPTH,
        _nnue_ext.QSEARCH_MAX_PLY,
        _nnue_ext.QSEARCH_CHECK_PLIES,
        _nnue_ext.QSEARCH_DAG_NODE_CAP,
    )
    _nnue_ext.fastq_set_config(
        _nnue_ext.FASTQ_MAX_QPLY,
        _nnue_ext.FASTQ_NODE_CAP,
        _nnue_ext.FASTQ_DELTA_MARGIN,
        _nnue_ext.FASTQ_RECAPTURE_EXEMPT,
    )


def _args(arm: str, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "arm": arm,
        "nnue_resolver_max_depth": None,
        "nnue_qsearch_max_ply": None,
        "nnue_qsearch_check_plies": None,
        "dag_node_cap": None,
        "allow_binding_dag_node_cap": False,
        "fastq_max_qply": None,
        "fastq_node_cap": None,
        "fastq_delta_margin": None,
        "fastq_recapture_exempt": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _FenOnly:
    """Minimal bank-writer board, including schema-3 history metadata."""

    def __init__(self, fen: str) -> None:
        self._fen = fen
        fields = fen.split()
        self.halfmove_clock = int(fields[4]) if len(fields) > 4 else 0
        self.hash_stack_len = 0
        self.hist_len = 0

    def fen(self) -> str:
        return self._fen


def _pack(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pack"
    path.write_bytes(b"not-a-real-net; fake extension owns this test")
    return path


def _source(
    ext: FakeExt, tmp_path: Path, arm: str = readout.ARM_QSEARCH_DAG, **overrides: Any,
) -> readout.ReadoutArmSource:
    cfg = readout.resolve_arm_config(_args(arm, **overrides), ext)
    return readout.ReadoutArmSource(
        config=cfg,
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )


def _run_config(
    arm_config: readout.ResolvedArmConfig, **overrides: Any,
) -> readout.RunConfig:
    values: dict[str, Any] = {
        "arm_config": arm_config,
        "pack": Path("x"),
        "pack_file_sha256": "b" * 64,
        "games": 4,
        "workers": 2,
        "seed": 123,
        "sims": 32,
        "topk": gen.MAX_LEGAL_MOVES,
        "max_plies": 100,
        "all_root_moves": True,
        "cp_per_internal_unit": 0.28,
        "cp_slope": 0.006,
        "cp_draw_width": 120.0,
        "bank_path": None,
        "run_id": "test",
        "nice": 0,
    }
    values.update(overrides)
    return readout.RunConfig(**values)


def _worker_result(
    provider: dict[str, int], **overrides: Any,
) -> readout.WorkerResult:
    values: dict[str, Any] = {
        "worker_id": 0,
        "games": 1,
        "plies": 10,
        "setup_s": 0.5,
        "elapsed_s": 2.0,
        "terminations": {},
        "policy_shape": {},
        "root_budget": {},
        "provider_stats": provider,
        "dag": readout.DagGameStats(),
        "game_records": [],
        "eval_batches": 1,
        "eval_rows": 1,
        "arm_batches": 1,
        "arm_leaves": 10,
        "arm_roots": 1,
        "mate_band_leaves": 0,
        "mate_band_roots": 0,
        "bank_rows": 0,
        "bank_file": None,
        "kernel": "scalar",
        "pack_file_sha256": "b" * 64,
        "pack_source_sha256": "a" * 64,
        "nice_realized": 0,
    }
    values.update(overrides)
    return readout.WorkerResult(**values)


# ── the fake's schema, against the compiled extension's ──────────────────────


def test_the_fake_publishes_exactly_the_c_layers_schema(live_pack: Path) -> None:
    """⚑⚑ THE FAKE MAY NOT PUBLISH A KEY THE EXTENSION DOES NOT, OR VICE VERSA.

    Both directions are real failures that have happened here. A key the fake
    invents makes every consumer of it untested against production -- the
    `nodes` / `node_count` split below. A key the fake publishes and the C layer
    drops makes a KeyError in production invisible to the suite -- the
    `see_recapture_exempt` format-string bug, which killed the FastQ cell before
    it played a ply while this file stayed green.

    Read off LIVE handles, never off a transcription: a transcription is the
    same drift one level down.
    """
    fake = FakeExt()
    fake_q = fake.arm_open(readout.ARM_QSEARCH_DAG, str(live_pack))
    fake_f = fake.arm_open(readout.ARM_FASTQ, str(live_pack))

    real_q = _nnue_ext.arm_open(readout.ARM_QSEARCH_DAG, str(live_pack))
    real_f = _nnue_ext.arm_open(readout.ARM_FASTQ, str(live_pack))

    assert set(fake.arm_stats(fake_q)) == set(_nnue_ext.arm_stats(real_q))
    assert set(fake.fastq_stats(fake_f)) == set(_nnue_ext.fastq_stats(real_f))
    assert set(fake.arm_dag_stats(fake_q)) == set(_nnue_ext.arm_dag_stats(real_q))
    assert set(fake.arm_dag_stats(fake_f)) == set(_nnue_ext.arm_dag_stats(real_f))


def test_every_published_key_is_classified_for_merging(live_pack: Path) -> None:
    """A new C counter must be classified BEFORE it can be merged.

    Read off a live handle rather than the fake, so a counter added to the C
    layer fails here instead of being summed by whichever rule is the fallback.
    """
    handle = _nnue_ext.arm_open(readout.ARM_QSEARCH_DAG, str(live_pack))
    for key in _nnue_ext.arm_stats(handle):
        readout.ARM_KEY_CLASSES.classify(key)
    fastq = _nnue_ext.arm_open(readout.ARM_FASTQ, str(live_pack))
    for key in _nnue_ext.fastq_stats(fastq):
        readout.FASTQ_KEY_CLASSES.classify(key)


def test_the_realized_snapshot_carries_every_fastq_knob(live_pack: Path) -> None:
    """The dependency this harness cannot run without, asserted from its side.

    ``ReadoutArmSource`` builds ``realized`` by SUBSCRIPTING every consumed key,
    so a knob missing from ``fastq_stats()`` is a KeyError in every worker. This
    is that subscript, on a live context.
    """
    _nnue_ext.fastq_set_config(3, 16, 150, 0)
    handle = _nnue_ext.arm_open(readout.ARM_FASTQ, str(live_pack))
    snapshot = _nnue_ext.fastq_stats(handle)
    assert {k: snapshot[k] for k in readout.ResolvedArmConfig(
        arm=readout.ARM_FASTQ, fastq_max_qply=3, fastq_node_cap=16,
        fastq_delta_margin=150, fastq_recapture_exempt=0,
    ).consumed()} == {
        "max_qply": 3, "node_cap": 16, "delta_margin": 150,
        "see_recapture_exempt": 0,
    }


# ── arm specs and knob ownership ─────────────────────────────────────────────


def test_arm_specs_are_the_three_intended_experimental_cells() -> None:
    expected = {
        "nnue-qsearch": readout.ArmSpec("nnue-qsearch", False, "arm", True, False),
        "nnue-qsearch-dag": readout.ArmSpec(
            "nnue-qsearch-dag", True, "arm", True, False,
        ),
        "nnue-fastq": readout.ArmSpec("nnue-fastq", True, "fastq", False, True),
    }
    assert expected == readout.ARM_SPECS


def test_defaults_come_from_the_extension_for_the_selected_arm() -> None:
    ext = FakeExt()
    q = readout.resolve_arm_config(_args(readout.ARM_QSEARCH_DAG), ext)
    assert q.consumed() == {
        "resolver_max_depth": 12,
        "qsearch_max_ply": 4,
        "qsearch_check_plies": 1,
        "dag_node_cap": 0,
    }
    f = readout.resolve_arm_config(_args(readout.ARM_FASTQ), ext)
    assert f.consumed() == {
        "max_qply": 4,
        "node_cap": 32,
        "delta_margin": 200,
        "see_recapture_exempt": 1,
    }


def test_knobs_for_another_provider_are_refused_not_ignored() -> None:
    ext = FakeExt()
    with pytest.raises(ValueError, match="does not consume --fastq"):
        readout.resolve_arm_config(
            _args(readout.ARM_QSEARCH_DAG, fastq_node_cap=64), ext,
        )
    with pytest.raises(ValueError, match="does not consume qsearch"):
        readout.resolve_arm_config(
            _args(readout.ARM_FASTQ, nnue_qsearch_max_ply=8), ext,
        )
    with pytest.raises(ValueError, match="has no DAG"):
        readout.resolve_arm_config(
            _args(readout.ARM_QSEARCH, dag_node_cap=10), ext,
        )


def test_a_binding_dag_node_cap_is_refused_because_it_voids_the_control() -> None:
    """``set_arm_config``'s own docstring: a binding cap stops matching the oracle."""
    ext = FakeExt()
    with pytest.raises(ValueError, match="no longer be a control"):
        readout.resolve_arm_config(
            _args(readout.ARM_QSEARCH_DAG, dag_node_cap=64), ext,
        )
    # 0 is OFF, and is what the DAG cell resolves to by default.
    assert readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH_DAG, dag_node_cap=0), ext,
    ).dag_node_cap == 0
    # The explicit opt-out exists, and it is explicit.
    assert readout.resolve_arm_config(
        _args(
            readout.ARM_QSEARCH_DAG, dag_node_cap=64,
            allow_binding_dag_node_cap=True,
        ),
        ext,
    ).dag_node_cap == 64


# ── construction: configure before open, and the caller's own dict ───────────


def test_fastq_config_is_snapshotted_before_open_and_read_back_from_fastq_stats(
    tmp_path: Path,
) -> None:
    ext = FakeExt()
    source = _source(
        ext, tmp_path, readout.ARM_FASTQ,
        fastq_max_qply=6, fastq_node_cap=48, fastq_delta_margin=333,
        fastq_recapture_exempt=0,
    )
    assert ext.events[:2] == [
        ("fastq_set_config", 6, 48, 333, 0),
        ("arm_open", readout.ARM_FASTQ, str(source.pack)),
    ]
    assert source.realized == {
        "max_qply": 6,
        "node_cap": 48,
        "delta_margin": 333,
        "see_recapture_exempt": 0,
    }
    # FakeExt raises if ReadoutArmSource chose the resolver counter surface.
    assert source.provider_stats()["node_cap"] == 48


def test_a_context_opened_before_it_was_configured_is_caught() -> None:
    """⚑ THE FAKE CAN NOW FAIL THIS, WHICH IS THE WHOLE POINT.

    The previous fake's ``arm_open`` touched no configuration and its
    ``arm_stats`` echoed the LIVE globals, so a source that configured after
    opening would still have read back its own request: the test could not fail.
    ``_FakeHandle`` snapshots at open, exactly as a C context does, so opening
    first now realizes the DEFAULTS and the constructor's own check fires.
    """
    ext = FakeExt()
    cfg = readout.resolve_arm_config(
        _args(readout.ARM_FASTQ, fastq_node_cap=48), ext,
    )
    opened_first = ext.arm_open(readout.ARM_FASTQ, "irrelevant")
    assert opened_first.fastq_cfg["node_cap"] == FakeExt.FASTQ_NODE_CAP
    ext.fastq_set_config(*readout.readout_arm_config_plan(cfg).setter_args)
    assert ext.fastq_stats(opened_first)["node_cap"] == FakeExt.FASTQ_NODE_CAP
    assert ext.fastq_stats(ext.arm_open(readout.ARM_FASTQ, "x"))["node_cap"] == 48


def test_requested_is_the_callers_dict_not_the_setters_echo(tmp_path: Path) -> None:
    """⚑ A CLAMP INSIDE THE SETTER MUST NOT PASS THE REALIZED CHECK.

    ``self.requested`` used to be built from ``set_arm_config``'s return value.
    That value is the PRODUCER's copy AFTER the setter clamped it -- and the
    context snapshots the same clamped number, so the check compared the clamp
    against itself and passed while the caller's 48 was never honoured anywhere.
    Comparing against the caller's own dict is the only reading that can catch
    it. MUTANT: rebuild ``requested`` from the setter's return value and this
    raises nothing.
    """
    ext = FakeExt(clamp_node_cap=32)
    with pytest.raises(RuntimeError, match="did not realize the requested knobs"):
        _source(ext, tmp_path, readout.ARM_FASTQ, fastq_node_cap=48)
    # ⚑ And the clamp really is invisible from the setter's side: both the echo
    # and the context agree on 32, so only the caller's dict knows about 48.
    echo = FakeExt(clamp_node_cap=32).fastq_set_config(4, 48, 200, 1)
    assert echo["node_cap"] == 32


def test_qsearch_dag_uses_arm_stats_and_resets_only_its_graph(tmp_path: Path) -> None:
    ext = FakeExt()
    source = _source(
        ext, tmp_path, readout.ARM_QSEARCH_DAG,
        nnue_resolver_max_depth=10, nnue_qsearch_max_ply=3,
        nnue_qsearch_check_plies=0,
    )
    assert ext.events[:2] == [
        ("set_arm_config", 10, 3, 0, 0),
        ("arm_open", readout.ARM_QSEARCH_DAG, str(source.pack)),
    ]
    assert source.provider_stats()["dag_enabled"] == 1
    assert source.dag_stats() == _DAG_SNAPSHOT
    source.reset_game()
    source.reset_game()
    assert ext.resets == 2


def test_plain_qsearch_never_touches_the_dag_surface(tmp_path: Path) -> None:
    ext = FakeExt()
    source = _source(ext, tmp_path, readout.ARM_QSEARCH)
    assert source.dag_stats() is None
    source.reset_game()
    assert ext.resets == 0


# ── the two subclasses add no state of their own ─────────────────────────────


def test_the_readout_arm_source_adds_no_state_of_its_own(tmp_path: Path) -> None:
    """⚑⚑ THE DRIFT TEST. It exists because the drift already happened.

    ``ReadoutArmSource`` used to skip ``super().__init__()`` and reproduce the
    parent's body to get past its arm whitelist. A reproduced constructor
    inherits none of the original's later edits, and this one had already lost
    ``consumes_qsearch``. The parent now takes a widenable ``_ALLOWED_ARMS`` and
    an ``ArmConfigPlan``, so the subclass calls ``super().__init__()`` normally
    -- and this assertion is what keeps it that way.
    """
    ext = FakeExt()
    pack = _pack(tmp_path)
    parent = gen.NnueArmValueSource(
        arm=gen.VALUE_SOURCE_NNUE_QSEARCH,
        pack=pack,
        resolver_max_depth=12,
        qsearch_max_ply=4,
        qsearch_check_plies=1,
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        ext=ext,
    )
    child = _source(FakeExt(), tmp_path, readout.ARM_QSEARCH)
    assert set(vars(child)) == set(vars(parent))
    # And the attribute the reproduction had lost is present and correct.
    assert parent.consumes_qsearch is True
    assert child.consumes_qsearch is True
    assert _source(FakeExt(), tmp_path, readout.ARM_FASTQ).consumes_qsearch is False


def test_the_readout_evaluator_adds_no_state_of_its_own(tmp_path: Path) -> None:
    """The same drift test for the evaluator, which NO test instantiated at all."""
    parent = gen.UniformPriorEvaluator(
        value_source=gen.VALUE_SOURCE_ZERO, expected_planes=175,
    )
    child = readout.ReadoutEvaluator(
        source=_source(FakeExt(), tmp_path, readout.ARM_QSEARCH_DAG),
        expected_planes=175,
        input_history_encoding=gen.LC0_HISTORY_ROOT_LEGACY_META,
        input_extra_features=gen.EXTRA_FEATURES_V2_THREATS,
    )
    assert set(vars(child)) == set(vars(parent))
    assert child.value_source == readout.ARM_QSEARCH_DAG
    assert child.nnue_source is not None


def test_widening_the_whitelist_did_not_widen_the_guards_it_protects(
    tmp_path: Path,
) -> None:
    """⚑ BOTH ClassVars are widened, and the parent's are untouched.

    ``_ALLOWED_VALUE_SOURCES`` alone would let a readout arm through the XOR
    cross-check with ``nnue_source`` quietly optional -- the source present and
    unused, which is the defect that check exists for.
    """
    assert readout.ReadoutEvaluator._ALLOWED_VALUE_SOURCES == readout.READOUT_ARMS
    assert readout.ReadoutEvaluator._NATIVE_VALUE_SOURCES == readout.READOUT_ARMS
    assert gen.UniformPriorEvaluator._ALLOWED_VALUE_SOURCES == gen.VALUE_SOURCES
    assert gen.UniformPriorEvaluator._NATIVE_VALUE_SOURCES == gen.NNUE_ARM_SOURCES
    assert readout.ReadoutArmSource._ALLOWED_ARMS == readout.READOUT_ARMS
    assert gen.NnueArmValueSource._ALLOWED_ARMS == gen.NNUE_ARM_SOURCES

    # The generator still refuses the DAG arms it does not run.
    with pytest.raises(ValueError, match="arm must be one of"):
        gen.NnueArmValueSource(
            arm=readout.ARM_QSEARCH_DAG,
            pack=_pack(tmp_path),
            resolver_max_depth=12,
            qsearch_max_ply=4,
            qsearch_check_plies=1,
            cp_per_internal_unit=0.28,
            cp_slope=0.006,
            cp_draw_width=120.0,
            ext=FakeExt(),
        )
    # And the parent still refuses a native source it was not told about.
    with pytest.raises(ValueError, match="disagree"):
        gen.UniformPriorEvaluator(
            value_source=gen.VALUE_SOURCE_ZERO,
            expected_planes=175,
            nnue_source=_source(FakeExt(), tmp_path, readout.ARM_QSEARCH),
        )


# ── DAG per-game statistics ──────────────────────────────────────────────────


def test_the_dag_summary_reads_the_c_layers_own_key_names() -> None:
    """⚑ THE FALLBACK MUTANT.

    ``add`` used to read ``stats.get("nodes", stats.get("node_count", 0))``. The
    C layer publishes ``node_count`` and has never published ``nodes``, so the
    live branch was the FALLBACK -- and the fake returned the dead names, so the
    tests only ever exercised the branch production cannot reach. Deleting the
    fallback in that shape left every test green while every real run reported
    ``nodes_per_game: 0.0``.
    """
    stats = readout.DagGameStats()
    stats.add({**_DAG_SNAPSHOT, "node_count": 100, "edge_count": 150,
               "hits": 20, "probes": 120, "inserts": 100,
               "memory_bytes": 1_000_000, "state_inits": 40, "state_makes": 60})
    stats.add({**_DAG_SNAPSHOT, "node_count": 50, "edge_count": 70,
               "hits": 10, "probes": 60, "inserts": 50,
               "memory_bytes": 1_000_000, "state_inits": 20, "state_makes": 30})
    summary = stats.summary()
    assert summary["nodes_per_game"] == 75
    assert summary["nodes_peak_per_game"] == 100
    assert summary["canonical_hit_rate"] == pytest.approx(30 / 180)
    assert summary["state_inits"] == 60
    assert summary["state_makes"] == 90
    # reset retains capacity, so this is intentionally a worker resource peak,
    # not a claim that the second game itself needed a fresh MB.
    assert summary["memory_peak_per_worker_bytes"] == 1_000_000
    assert stats.state_identity_violations == 0
    # ⚑ And a snapshot missing the real key is a hard failure, not a 0.
    with pytest.raises(KeyError):
        readout.DagGameStats().add({"nodes": 11, "edges": 17})


def test_a_broken_state_identity_is_counted_and_makes_the_cell_inadmissible() -> None:
    """``state_inits + state_makes == node_count`` (arm_dag_stats' own docstring)."""
    stats = readout.DagGameStats()
    stats.add({**_DAG_SNAPSHOT, "state_makes": _DAG_SNAPSHOT["state_makes"] + 1})
    assert stats.state_identity_violations == 1

    cfg = _run_config(readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH_DAG), FakeExt(),
    ))
    ext = FakeExt()
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH_DAG, "x"))
    report = readout._aggregate(
        [_worker_result(provider, dag=stats)], cfg, wall_s=1.0,
    )
    assert report["identities"]["dag_state_identity_ok"] is False
    assert report["admissible"] is False
    assert any("state_inits" in r for r in report["inadmissible_reasons"])


# ── merging provider counters ────────────────────────────────────────────────


def test_an_unclassified_provider_key_is_refused_not_summed() -> None:
    """The parent's rule, reused: a new C counter is classified before it merges."""
    with pytest.raises(ValueError, match="not classified"):
        readout.merge_provider_stats(
            [{"a_brand_new_c_counter": 3}], readout.ARM_KEY_CLASSES,
        )


def test_store_sizes_peaks_counters_and_config_each_merge_by_their_own_rule() -> None:
    left = {
        "calls": 10, "max_depth_seen": 4, "dag_node_count": 100,
        "resolver_max_depth": 12,
    }
    right = {
        "calls": 5, "max_depth_seen": 7, "dag_node_count": 40,
        "resolver_max_depth": 12,
    }
    merged, conflicts = readout.merge_provider_stats(
        [left, right], readout.ARM_KEY_CLASSES,
    )
    assert merged["calls"] == 15          # counter: sum
    assert merged["max_depth_seen"] == 7  # peak: max
    assert merged["dag_node_count"] == 140  # store size: a RESOURCE total
    assert merged["resolver_max_depth"] == 12
    assert conflicts == {}


def test_a_config_disagreement_is_recorded_not_raised() -> None:
    """⚑ Raising here discards a finished multi-hour run at the aggregation step.

    The parent settled this: ``NnueArmStats.merge`` records ``context_conflicts``
    and publishes. A run whose workers disagreed is not one cell, which is a
    finding to REPORT, not a reason to have no report.
    """
    merged, conflicts = readout.merge_provider_stats(
        [{"resolver_max_depth": 12}, {"resolver_max_depth": 8}],
        readout.ARM_KEY_CLASSES,
    )
    assert conflicts == {"resolver_max_depth": [8, 12]}
    assert merged["resolver_max_depth"] == 8  # the shallowest worker's ceiling


# ── FastQ rates ──────────────────────────────────────────────────────────────


def test_fastq_rates_are_null_when_nothing_ran() -> None:
    """⚑ ``calls == 0`` must not read as a healthy zero.

    ``x / max(1, calls)`` made "nothing tripped over four million calls" and
    "nothing ran at all" the same 0.0 -- and the doc's read order names 0.0 as
    the healthy value for ``budget_trip_rate``.
    """
    empty: dict[str, int] = dict.fromkeys(_FASTQ_COUNTER_KEYS, 0)
    rates = readout._fastq_rates(empty)
    assert rates["calls"] == 0.0
    assert rates["budget_trip_rate"] is None
    assert rates["nnue_evals_per_call"] is None


def test_a_missing_fastq_counter_is_a_hard_failure_not_a_zero() -> None:
    """The other half: a counter that vanished from the C layer read 0.0."""
    provider: dict[str, int] = dict.fromkeys(_FASTQ_COUNTER_KEYS, 4)
    del provider["budget_trips"]
    with pytest.raises(KeyError, match="budget_trips"):
        readout._fastq_rates(provider)


def test_the_evaluate_once_identity_is_checked_and_can_fail() -> None:
    """docs/fastq_design.md §7: an ASSERTABLE counter identity, now asserted."""
    ext = FakeExt()
    cfg = _run_config(readout.resolve_arm_config(_args(readout.ARM_FASTQ), ext))
    handle = ext.arm_open(readout.ARM_FASTQ, "x")

    good = ext.fastq_stats(handle)
    good.update({"calls": 10, "nnue_evals": 90, "nodes_created_in_check": 10,
                 "nodes_created": 100})
    ok = readout._aggregate([_worker_result(good)], cfg, wall_s=1.0)
    assert ok["identities"]["evaluate_once_identity_ok"] is True
    assert ok["admissible"] is True

    bad = dict(good, nodes_created=101)
    broken = readout._aggregate([_worker_result(bad)], cfg, wall_s=1.0)
    assert broken["identities"]["evaluate_once_identity_ok"] is False
    assert broken["admissible"] is False
    with pytest.raises(RuntimeError, match="inadmissible"):
        readout.assert_admissible(
            {"admissible": False, "inadmissible_reasons": ["x"]},
        )


def test_a_binding_dag_cap_that_actually_tripped_makes_the_report_inadmissible() -> None:
    ext = FakeExt()
    cfg = _run_config(readout.resolve_arm_config(
        _args(
            readout.ARM_QSEARCH_DAG, dag_node_cap=8,
            allow_binding_dag_node_cap=True,
        ),
        ext,
    ))
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH_DAG, "x"))
    provider["dag_budget_trips"] = 3
    report = readout._aggregate([_worker_result(provider)], cfg, wall_s=1.0)
    assert report["admissible"] is False
    assert any("dag_budget_trips" in r for r in report["inadmissible_reasons"])


# ── the report's own shape ───────────────────────────────────────────────────


def test_the_report_names_the_window_each_throughput_number_measures() -> None:
    """⚑ Two windows, two names.

    ``plies_per_s`` divides by the parent's end-to-end clock (pool startup and
    every worker's ``ext.load()`` + mmap included); ``search_plies_per_s``
    divides by the widest worker's SEARCH window. One number that silently mixed
    them is how the banking cost landed on the headline cell.
    """
    ext = FakeExt()
    cfg = _run_config(readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext))
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH, "x"))
    report = readout._aggregate(
        [
            _worker_result(provider, worker_id=0, plies=10, setup_s=1.0, elapsed_s=2.0),
            _worker_result(provider, worker_id=1, plies=10, setup_s=1.5, elapsed_s=4.0),
        ],
        cfg,
        wall_s=10.0,
    )
    assert report["plies"] == 20
    assert report["wall_s"] == 10.0
    assert report["search_wall_s"] == 4.0
    assert report["plies_per_s"] == pytest.approx(2.0)
    assert report["search_plies_per_s"] == pytest.approx(5.0)
    assert report["worker_wall_s"] == pytest.approx(6.0)
    assert report["worker_setup_wall_s"] == pytest.approx(2.5)
    # ⚑ The old name claimed CPU time it never measured.
    assert "worker_cpu_s" not in report


def test_the_report_publishes_realized_knobs_beside_requested_ones() -> None:
    ext = FakeExt()
    arm_config = readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH, nnue_resolver_max_depth=9), ext,
    )
    cfg = _run_config(arm_config)
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH, "x"))
    report = readout._aggregate(
        [_worker_result(
            provider,
            arm_config_requested={"resolver_max_depth": 9},
            arm_config_realized={"resolver_max_depth": 9},
        )],
        cfg,
        wall_s=1.0,
    )
    assert report["arm_config"] == arm_config.consumed()
    assert report["arm_config_realized"] == {"resolver_max_depth": 9}


def test_the_classification_block_names_the_store_sizes_for_what_they_are() -> None:
    ext = FakeExt()
    cfg = _run_config(readout.resolve_arm_config(
        _args(readout.ARM_QSEARCH_DAG), ext,
    ))
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH_DAG, "x"))
    report = readout._aggregate([_worker_result(provider)], cfg, wall_s=1.0)
    block = report["provider_stats_classification"]
    assert set(block["store_endpoint_sizes"]) == {
        "dag_node_count", "dag_edge_count", "dag_memory_bytes",
    }
    assert "dag_enabled" in block["config"]
    assert "max_depth_seen" in block["peaks"]
    assert "nnue_evals" in block["counters"]


# ── digests ──────────────────────────────────────────────────────────────────


def test_the_game_digest_moves_with_every_field_it_names() -> None:
    """A digest that ignored a field would silently pass a diverged cell."""
    base: dict[str, Any] = {
        "game_index": 3, "start_fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "move_trace": "e2e4 e7e5", "result": "1-0", "termination": "checkmate",
    }
    reference = readout.game_digest(
        game_index=base["game_index"], start_fen=base["start_fen"],
        move_trace=base["move_trace"], result=base["result"],
        termination=base["termination"],
    )
    for field_name, other in (
        ("game_index", 4),
        ("start_fen", "8/8/8/8/8/8/8/K6k b - - 0 1"),
        ("move_trace", "e2e4 e7e6"),
        ("result", "0-1"),
        ("termination", "fifty_moves"),
    ):
        changed: dict[str, Any] = {**base, field_name: other}
        assert readout.game_digest(
            game_index=changed["game_index"], start_fen=changed["start_fen"],
            move_trace=changed["move_trace"], result=changed["result"],
            termination=changed["termination"],
        ) != reference


def test_games_digest_is_order_independent_but_content_sensitive() -> None:
    a = readout.GameRecord(game=0, plies=1, result="1-0", termination="t", digest="aa")
    b = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="bb")
    assert readout.games_digest([a, b]) == readout.games_digest([b, a])
    c = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="cc")
    assert readout.games_digest([a, b]) != readout.games_digest([a, c])


def test_searches_digest_refuses_an_unpopulated_component() -> None:
    missing = readout.GameRecord(
        game=0, plies=1, result="1-0", termination="t", digest="aa",
    )
    with pytest.raises(ValueError, match="no search_digest"):
        readout.searches_digest([missing])


def test_the_oracle_voids_the_decomposition_when_the_digests_differ() -> None:
    same = {
        readout.ARM_QSEARCH: [{
            "repeat": 0, "games_digest": "same", "searches_digest": "search-same",
        }],
        readout.ARM_QSEARCH_DAG: [{
            "repeat": 0, "games_digest": "same", "searches_digest": "search-same",
        }],
    }
    assert readout._oracle(same)["digests_agree"] is True
    differ = {
        readout.ARM_QSEARCH: [{
            "repeat": 0, "games_digest": "left", "searches_digest": "search-same",
        }],
        readout.ARM_QSEARCH_DAG: [{
            "repeat": 0, "games_digest": "right", "searches_digest": "search-same",
        }],
    }
    assert readout._oracle(differ)["digests_agree"] is False
    # One cell alone cannot claim the comparison it did not make.
    alone = {readout.ARM_FASTQ: [{
        "repeat": 0, "games_digest": "x", "searches_digest": "sx",
    }]}
    assert readout._oracle(alone) == {
        "arms": [readout.ARM_QSEARCH, readout.ARM_QSEARCH_DAG],
        "available": False,
        "digests_agree": None,
        "game_digests_agree": None,
        "search_digests_agree": None,
    }


# ── scheduling, cadence and the leaf bank ────────────────────────────────────


def test_worker_assignment_is_schedule_independent_for_game_seed() -> None:
    cfg = _run_config(
        readout.ResolvedArmConfig(
            arm=readout.ARM_FASTQ,
            fastq_max_qply=4,
            fastq_node_cap=32,
            fastq_delta_margin=200,
            fastq_recapture_exempt=1,
        ),
        games=7,
        workers=3,
    )
    specs = readout._build_worker_specs(cfg)
    assert [s.game_indices for s in specs] == [(0, 3, 6), (1, 4), (2, 5)]
    assert sorted(g for s in specs for g in s.game_indices) == list(range(7))
    # _run_worker seeds each game as cfg.seed + game_index, not by local index;
    # moving a game between workers therefore cannot change its trajectory.
    assert [cfg.seed + g for g in range(cfg.games)] == list(range(123, 130))


def test_the_dag_reset_cadence_is_a_choice_and_is_reported() -> None:
    """fastq_design.md 4.4: the cadence is CHOSEN FROM MEASUREMENT."""
    assert readout.parse_dag_reset("game") == 1
    assert readout.parse_dag_reset("never") == 0
    assert readout.parse_dag_reset("every-4-games") == 4
    with pytest.raises(ValueError, match="must be 'game'"):
        readout.parse_dag_reset("sometimes")
    with pytest.raises(ValueError, match="N >= 1"):
        readout.parse_dag_reset("every-0-games")
    assert readout._dag_reset_label(0) == "never"
    assert readout._dag_reset_label(1) == "game"
    assert readout._dag_reset_label(4) == "every-4-games"


def test_the_bank_path_names_the_arm_the_repeat_and_the_worker() -> None:
    """Three cells aimed at one path must not merge into one file."""
    base = Path("data/leaves.jsonl")
    paths = {
        readout._worker_bank_path(base, arm=arm, repeat=r, worker_id=w)
        for arm in readout.READOUT_ARMS for r in (0, 1) for w in (0, 1)
    }
    assert len(paths) == 12
    assert readout._worker_bank_path(
        base, arm=readout.ARM_FASTQ, repeat=2, worker_id=3,
    ) == Path("data/leaves.nnue-fastq.r02.w03.jsonl")
    assert readout._worker_bank_path(None, arm="x", repeat=0, worker_id=0) is None


def test_a_rerun_refuses_to_append_into_the_previous_runs_bank(
    tmp_path: Path,
) -> None:
    """⚑ ``"x"``, not ``"a"``. An appended file is two runs read as one."""
    ext = FakeExt()
    bank = tmp_path / "leaves.jsonl"
    first = readout.ReadoutArmSource(
        config=readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        leaf_bank=bank,
        ext=ext,
    )
    assert first.leaf_bank_path == bank
    first.close()
    with pytest.raises(FileExistsError):
        readout.ReadoutArmSource(
            config=readout.resolve_arm_config(_args(readout.ARM_QSEARCH), FakeExt()),
            pack=_pack(tmp_path),
            cp_per_internal_unit=0.28,
            cp_slope=0.006,
            cp_draw_width=120.0,
            leaf_bank=bank,
            ext=FakeExt(),
        )


def test_banked_rows_carry_their_identity_and_the_mate_constants(
    tmp_path: Path,
) -> None:
    """⚑ A cp reconstruction must not need constants absent from the artifact.

    Above the mate band floor the banked value is a mate distance in PLIES.
    Turning it back into a score needs RESOLVER_MATE_BASE /
    RESOLVER_MATE_PLY_STEP / RESOLVER_MAX_PLIES -- three BUILD-VINTAGE constants
    that schema 1 did not bank. A reader who guessed would run the mate rows
    through the centipawn slope.
    """
    ext = FakeExt()
    bank = tmp_path / "leaves.jsonl"
    source = readout.ReadoutArmSource(
        config=readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),
        pack=_pack(tmp_path),
        cp_per_internal_unit=0.28,
        cp_slope=0.006,
        cp_draw_width=120.0,
        leaf_bank=bank,
        ext=ext,
        bank_identity={"run_id": "r", "seed": 7, "repeat": 1, "worker_id": 2},
    )
    boards: list[Any] = [
        _FenOnly("8/8/8/8/8/8/8/K6k w - - 0 1"),
        _FenOnly("8/8/8/8/8/8/8/K6k b - - 0 1"),
    ]
    raw = np.array([250.0, 99_950.0], dtype=np.float64)
    _q, is_mate = source.q_from_values(raw)
    source._bank_batch(boards, raw, is_mate, role="leaf", cluster=(3, 4))
    source.close()

    rows = [json.loads(line) for line in bank.read_text().splitlines()]
    assert len(rows) == 2
    assert [r["is_mate"] for r in rows] == [False, True]
    for row in rows:
        assert row["schema"] == gen.LEAF_BANK_SCHEMA
        assert row["run_id"] == "r"
        assert row["seed"] == 7
        assert row["repeat"] == 1
        assert row["worker_id"] == 2
        assert row["resolver_mate_base"] == FakeExt.RESOLVER_MATE_BASE
        assert row["resolver_mate_ply_step"] == FakeExt.RESOLVER_MATE_PLY_STEP
        assert row["resolver_max_plies"] == FakeExt.RESOLVER_MAX_PLIES
        assert row["game"] == 3
        assert row["ply"] == 4
    # ⚑ And the band verdict is the ARM's, reproducible from the banked
    # constants alone -- no build-vintage import needed.
    first_row = rows[0]
    floor = (
        first_row["resolver_mate_base"]
        - first_row["resolver_max_plies"] * first_row["resolver_mate_ply_step"]
    )
    assert [abs(r["value"]) >= floor for r in rows] == [r["is_mate"] for r in rows]


# ── the plan: validation before spawn, and interleaved repeats ───────────────


def test_a_bad_topk_is_refused_before_any_worker_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ It used to surface as N identical pickled exceptions after pool startup."""
    spawned: list[object] = []

    def record(spec: readout.WorkerSpec) -> None:
        spawned.append(spec)

    monkeypatch.setattr(readout, "_run_worker", record)
    pack = _pack(tmp_path)
    cfg = _run_config(
        readout.resolve_arm_config(_args(readout.ARM_QSEARCH), FakeExt()),
        pack=pack,
        topk=8,          # < MAX_LEGAL_MOVES, with --all-root-moves on
        workers=1,
        games=1,
    )
    with pytest.raises(ValueError, match="--all-root-moves needs --topk"):
        readout.run_cell(cfg)
    assert spawned == []


def test_repeats_interleave_the_cells_rather_than_running_them_in_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑ (repeat, then cell), so machine drift lands on every cell equally."""
    ext = FakeExt()
    pack = _pack(tmp_path)
    seen: list[tuple[str, int]] = []

    def fake_cell(cfg: readout.RunConfig) -> dict[str, Any]:
        seen.append((cfg.arm_config.arm, cfg.repeat))
        return {
            "arm": cfg.arm_config.arm,
            "repeat": cfg.repeat,
            "games_digest": f"digest-{cfg.repeat}",
            "searches_digest": f"search-{cfg.repeat}",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "lc0_ext_path": "/tmp/_lc0_ext.so", "lc0_ext_sha256": "d" * 64,
                "lc0_ext_loaded_build_id": "3" * 40,
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "nnue_ext_loaded_build_id": "1" * 40,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
                "mcts_ext_loaded_build_id": "2" * 40,
            }],
        }

    monkeypatch.setattr(readout, "run_cell", fake_cell)
    plan = readout.ReadoutPlan(
        arm_configs=(
            readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),
            readout.resolve_arm_config(_args(readout.ARM_QSEARCH_DAG), ext),
        ),
        pack=pack,
        games=2, workers=1, seed=1, sims=8, topk=gen.MAX_LEGAL_MOVES,
        max_plies=10, all_root_moves=True,
        cp_per_internal_unit=0.28, cp_slope=0.006, cp_draw_width=120.0,
        bank_path=None, run_id="t", nice=0,
        dag_reset_every=readout.DAG_RESET_EVERY_GAME, repeats=3,
    )
    report = readout.run(plan)
    assert seen == [
        (readout.ARM_QSEARCH, 0), (readout.ARM_QSEARCH_DAG, 0),
        (readout.ARM_QSEARCH, 1), (readout.ARM_QSEARCH_DAG, 1),
        (readout.ARM_QSEARCH, 2), (readout.ARM_QSEARCH_DAG, 2),
    ]
    assert report["order"] == [
        {"arm": a, "repeat": r} for (a, r) in seen
    ]
    assert report["provenance"]["repeats"] == 3
    assert report["provenance"]["banking"] is False
    assert report["provenance"]["seed"] == 1
    assert report["provenance"]["max_plies"] == 10
    assert report["provenance"]["kernel"] == "scalar"
    assert report["provenance"]["pack_file_sha256"] == readout._sha256_file(pack)
    assert report["oracle"]["digests_agree"] is True
    assert report["admissible"] is True


def test_the_provenance_block_carries_what_makes_three_cells_one_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell that ran on a different kernel is not the same experiment."""
    ext = FakeExt()
    pack = _pack(tmp_path)
    kernels = iter(("avx2", "scalar"))

    def fake_cell(cfg: readout.RunConfig) -> dict[str, Any]:
        return {
            "arm": cfg.arm_config.arm,
            "repeat": cfg.repeat,
            "games_digest": "same",
            "searches_digest": "search-same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": next(kernels), "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "lc0_ext_path": "/tmp/_lc0_ext.so", "lc0_ext_sha256": "d" * 64,
                "lc0_ext_loaded_build_id": "3" * 40,
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "nnue_ext_loaded_build_id": "1" * 40,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
                "mcts_ext_loaded_build_id": "2" * 40,
            }],
        }

    monkeypatch.setattr(readout, "run_cell", fake_cell)
    plan = readout.ReadoutPlan(
        arm_configs=(
            readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),
            readout.resolve_arm_config(_args(readout.ARM_QSEARCH_DAG), ext),
        ),
        pack=pack,
        games=2, workers=1, seed=1, sims=8, topk=gen.MAX_LEGAL_MOVES,
        max_plies=10, all_root_moves=True,
        cp_per_internal_unit=0.28, cp_slope=0.006, cp_draw_width=120.0,
        bank_path=None, run_id="t", nice=0,
        dag_reset_every=readout.DAG_RESET_EVERY_GAME, repeats=1,
    )
    report = readout.run(plan)
    assert report["admissible"] is False
    assert any("different NNUE kernels" in r for r in report["inadmissible_reasons"])
    assert report["provenance"]["kernel"] == ["avx2", "scalar"]


def test_the_cli_accepts_the_whole_matrix_in_one_invocation(tmp_path: Path) -> None:
    pack = _pack(tmp_path)
    args = readout.build_parser().parse_args([
        "--arm", "nnue-qsearch", "--arm", "nnue-qsearch-dag",
        "--nnue-pack", str(pack), "--repeats", "2",
        "--dag-reset", "every-3-games",
    ])
    assert args.arm == ["nnue-qsearch", "nnue-qsearch-dag"]
    assert args.repeats == 2
    plan = readout.plan_from_args(args)
    assert plan.dag_reset_every == 3
    assert [c.arm for c in plan.arm_configs] == [
        readout.ARM_QSEARCH, readout.ARM_QSEARCH_DAG,
    ]


def test_the_report_divides_throughput_by_ACTIVE_workers_not_requested() -> None:
    """⚑ THE DENOMINATOR IS THE ONE THAT RAN, and it is pinned here.

    `_build_worker_specs` drops empty buckets, so asking for 8 workers on 1
    game runs 1. A report that published the REQUESTED count as the throughput
    divisor would understate per-worker throughput by 8x while looking
    perfectly well-formed — this repo's signature defect (a number that does
    not mean what its name says) landing on the readout's headline metric.

    The follow-up commit that fixed this covered `_build_worker_specs` only,
    so the reported field itself was unpinned: reverting `_aggregate` to
    `cfg.workers` failed no test. This closes that.

    ⚑ It is `workers_active`, not `workers`. Schema 2 shipped `workers` meaning
    REQUESTED, so redefining it in place would hand a stale consumer a
    plausible wrong number; the rename plus the REPORT_SCHEMA 3 bump makes that
    reader fail loudly instead. Both are asserted below.

    Mutants that must kill it: `"workers_active": cfg.workers`; restoring the
    key name to `"workers"`; leaving REPORT_SCHEMA at 2.
    """
    ext = FakeExt()
    cfg = _run_config(
        readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),
        workers=8, games=1,
    )
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH, "x"))
    report = readout._aggregate([_worker_result(provider)], cfg, wall_s=1.0)

    assert report["workers_active"] == 1, (
        "one result was aggregated, so one worker ran"
    )
    assert report["workers_requested"] == 8, (
        "the request is provenance and must survive alongside the realized count"
    )
    assert report["workers_active"] != report["workers_requested"], (
        "if these agree the test cannot see the bug it exists for"
    )
    assert "workers" not in report, (
        "schema 2's `workers` meant requested; a stale reader must KeyError "
        "rather than silently read a redefined key"
    )
    assert readout.REPORT_SCHEMA >= 3, (
        "the key rename is a breaking report change and must carry a bump"
    )



# --- #474 current-head measurement-integrity regression tests ---

def test_plan_refuses_nonpositive_max_plies(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--nnue-pack", str(tmp_path / "pack"),
        "--max-plies", "0",
    ])
    with pytest.raises(ValueError, match="max-plies must be positive"):
        readout.plan_from_args(args)


def test_multi_arm_matrix_scopes_explicit_knobs_to_the_consuming_cells(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--arm", readout.ARM_QSEARCH_DAG,
        "--arm", readout.ARM_FASTQ,
        "--nnue-pack", str(tmp_path / "pack"),
        "--nnue-qsearch-max-ply", "3",
        "--dag-node-cap", "0",
        "--fastq-max-qply", "6",
    ])
    plan = readout.plan_from_args(args)
    by_arm = {cfg.arm: cfg for cfg in plan.arm_configs}
    assert by_arm[readout.ARM_QSEARCH].qsearch_max_ply == 3
    assert by_arm[readout.ARM_QSEARCH].fastq_max_qply is None
    assert by_arm[readout.ARM_QSEARCH_DAG].dag_node_cap == 0
    assert by_arm[readout.ARM_FASTQ].fastq_max_qply == 6
    assert by_arm[readout.ARM_FASTQ].qsearch_max_ply is None


def test_matrix_still_refuses_a_knob_no_selected_arm_consumes(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--nnue-pack", str(tmp_path / "pack"),
        "--fastq-max-qply", "6",
    ])
    with pytest.raises(ValueError, match=r"no.*nnue-fastq|nnue-fastq is not selected"):
        readout.plan_from_args(args)


def test_persistent_dag_snapshots_are_differenced_not_resummed() -> None:
    first = dict(_DAG_SNAPSHOT)
    second = dict(first)
    second.update({
        "node_count": 115,
        "edge_count": 177,
        "probes": 220,
        "hits": 54,
        "inserts": 115,
        "state_makes": 75,
        "memory_bytes": first["memory_bytes"] + 1024,
    })
    assert second["state_inits"] + second["state_makes"] == second["node_count"]
    stats = readout.DagGameStats()
    stats.add(first)
    stats.add(second, previous=first)
    summary = stats.summary()
    assert summary["nodes_per_game"] == pytest.approx((110 + 5) / 2)
    assert summary["edges_per_game"] == pytest.approx((170 + 7) / 2)
    assert summary["hits"] == 54
    assert summary["probes"] == 220
    assert summary["state_makes"] == 75


def test_dag_delta_refuses_a_counter_that_goes_backwards_without_reset() -> None:
    first = dict(_DAG_SNAPSHOT)
    second = dict(first)
    second["hits"] -= 1
    stats = readout.DagGameStats()
    stats.add(first)
    with pytest.raises(ValueError, match="went backwards"):
        stats.add(second, previous=first)


def test_search_output_digest_refuses_an_empty_trace() -> None:
    with pytest.raises(ValueError, match="empty search-output trace"):
        readout.search_output_digest([])


def test_file_stamp_detects_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "stable.bin"
    path.write_bytes(b"before")
    stamp = readout._file_stamp(path)
    path.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed while this worker was running"):
        readout._assert_file_unchanged("test file", path, stamp)


def test_search_output_digest_catches_a_target_change_that_game_digest_cannot() -> None:
    row_a = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([True, True]),
        search_value=0.125,
    )
    row_b = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.6, 0.4], dtype=np.float32),
        legal_mask=np.array([True, True]),
        search_value=0.125,
    )
    same_game = readout.game_digest(
        game_index=0, start_fen="start", move_trace="e2e4", result="*", termination="max",
    )
    left = readout.GameRecord(0, 1, "*", "max", same_game, readout.search_output_digest([row_a]))
    right = readout.GameRecord(0, 1, "*", "max", same_game, readout.search_output_digest([row_b]))
    cells = {
        readout.ARM_QSEARCH: [{"repeat": 0, "games_digest": readout.games_digest([left]),
                               "searches_digest": readout.searches_digest([left])}],
        readout.ARM_QSEARCH_DAG: [{"repeat": 0, "games_digest": readout.games_digest([right]),
                                   "searches_digest": readout.searches_digest([right])}],
    }
    oracle = readout._oracle(cells)
    assert oracle["game_digests_agree"] is True
    assert oracle["search_digests_agree"] is False
    assert oracle["digests_agree"] is False


def test_search_output_digest_catches_a_root_value_change_with_same_policy() -> None:
    row_a = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([True, True]),
        search_value=0.125,
    )
    row_b = argparse.Namespace(
        ply_index=0,
        policy_probs=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([True, True]),
        search_value=0.25,
    )
    assert readout.search_output_digest([row_a]) != readout.search_output_digest([row_b])


def test_source_provenance_change_during_matrix_voids_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext = FakeExt()
    pack = _pack(tmp_path)
    snapshots = iter((
        {
            "git_head": "same", "git_tracked_dirty": True,
            "git_tracked_diff_sha256": "a" * 64,
        },
        {
            "git_head": "same", "git_tracked_dirty": True,
            "git_tracked_diff_sha256": "b" * 64,
        },
    ))
    monkeypatch.setattr(readout, "_git_provenance", lambda: next(snapshots))

    def fake_cell(cfg: readout.RunConfig) -> dict[str, Any]:
        return {
            "arm": cfg.arm_config.arm,
            "repeat": cfg.repeat,
            "games_digest": "same",
            "searches_digest": "search-same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "lc0_ext_path": "/tmp/_lc0_ext.so", "lc0_ext_sha256": "d" * 64,
                "lc0_ext_loaded_build_id": "3" * 40,
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "nnue_ext_loaded_build_id": "1" * 40,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
                "mcts_ext_loaded_build_id": "2" * 40,
            }],
        }

    monkeypatch.setattr(readout, "run_cell", fake_cell)
    plan = readout.ReadoutPlan(
        arm_configs=(readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),),
        pack=pack, games=1, workers=1, seed=1, sims=8,
        topk=gen.MAX_LEGAL_MOVES, max_plies=10, all_root_moves=True,
        cp_per_internal_unit=0.28, cp_slope=0.006, cp_draw_width=120.0,
        bank_path=None, run_id="t", nice=0,
        dag_reset_every=readout.DAG_RESET_EVERY_GAME, repeats=1,
    )
    report = readout.run(plan)
    assert report["admissible"] is False
    assert any("source provenance changed" in r for r in report["inadmissible_reasons"])
    assert report["provenance"]["git_head"] == "same"
    assert report["provenance"]["git_end_head"] == "same"
    assert report["provenance"]["git_tracked_dirty"] is True
    assert report["provenance"]["git_end_tracked_dirty"] is True
    assert report["provenance"]["git_tracked_diff_sha256"] == "a" * 64
    assert report["provenance"]["git_end_tracked_diff_sha256"] == "b" * 64
    assert report["provenance"]["git_changed_during_run"] is True


def test_unavailable_git_provenance_voids_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext = FakeExt()
    pack = _pack(tmp_path)
    unavailable = {
        "git_head": None,
        "git_tracked_dirty": None,
        "git_tracked_diff_sha256": None,
    }
    monkeypatch.setattr(readout, "_git_provenance", lambda: dict(unavailable))

    def fake_cell(cfg: readout.RunConfig) -> dict[str, Any]:
        return {
            "arm": cfg.arm_config.arm,
            "repeat": cfg.repeat,
            "games_digest": "same",
            "searches_digest": "search-same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "lc0_ext_path": "/tmp/_lc0_ext.so", "lc0_ext_sha256": "d" * 64,
                "lc0_ext_loaded_build_id": "3" * 40,
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "nnue_ext_loaded_build_id": "1" * 40,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
                "mcts_ext_loaded_build_id": "2" * 40,
            }],
        }

    monkeypatch.setattr(readout, "run_cell", fake_cell)
    plan = readout.ReadoutPlan(
        arm_configs=(readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),),
        pack=pack, games=1, workers=1, seed=1, sims=8,
        topk=gen.MAX_LEGAL_MOVES, max_plies=10, all_root_moves=True,
        cp_per_internal_unit=0.28, cp_slope=0.006, cp_draw_width=120.0,
        bank_path=None, run_id="t", nice=0,
        dag_reset_every=readout.DAG_RESET_EVERY_GAME, repeats=1,
    )
    report = readout.run(plan)
    assert report["admissible"] is False
    assert report["provenance"]["git_provenance_available"] is False
    assert any("source provenance is unavailable" in r for r in report["inadmissible_reasons"])


def test_quality_scope_explicitly_forbids_paired_attribution_from_end_to_end_cells() -> None:
    assert readout.QUALITY_SCOPE["population"] == "end_to_end_arm_selected"
    assert readout.QUALITY_SCOPE["paired_evaluator_quality"] is False
    assert readout.QUALITY_SCOPE["deep_sf_paired_input_admissible"] is False


def test_native_module_identity_reads_the_loaded_elf_build_id() -> None:
    from chess_anti_engine.encoding import _lc0_ext
    from chess_anti_engine.mcts import _mcts_tree

    for module in (_lc0_ext, _nnue_ext, _mcts_tree):
        path, digest, build_id, stamp = readout._module_identity(module)
        assert digest == readout._sha256_file(Path(path))
        assert stamp[0] == digest
        assert len(build_id) >= 16
        int(build_id, 16)


def test_late_integrity_failure_is_preserved_as_inadmissible_cell() -> None:
    ext = FakeExt()
    cfg = _run_config(readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext))
    provider = ext.arm_stats(ext.arm_open(readout.ARM_QSEARCH, "x"))
    worker = _worker_result(provider)
    worker.integrity_reasons.append("NNUE pack changed while this worker was running")
    report = readout._aggregate([worker], cfg, wall_s=1.0)
    assert report["admissible"] is False
    assert any(
        "late integrity check failed" in reason
        for reason in report["inadmissible_reasons"]
    )


def test_leaf_bank_marks_when_fen_does_not_reconstruct_repetition_history() -> None:
    import io

    sink = io.StringIO()
    source: Any = object.__new__(gen.NnueArmValueSource)
    source._bank = sink
    source.arm = readout.ARM_QSEARCH
    source.pack_file_sha256 = "f" * 64
    source.cp_per_internal_unit = 0.28
    source.cp_slope = 0.006
    source.cp_draw_width = 120.0
    source.mate_base = 100000.0
    source.mate_ply_step = 1.0
    source.mate_max_plies = 128.0
    source.bank_identity = {}
    source.realized = {}
    source.bank_rows = 0
    board: Any = argparse.Namespace(
        fen=lambda: "8/8/8/8/8/8/8/K6k w - - 7 9",
        halfmove_clock=7,
        hash_stack_len=3,
        hist_len=7,
    )
    gen.NnueArmValueSource._bank_batch(
        source, [board], np.array([10.0]), np.array([False]),
        role="leaf", cluster=(2, 3),
    )
    row = json.loads(sink.getvalue())
    assert row["schema"] >= 3
    assert row["halfmove_clock"] == 7
    assert row["hash_stack_len"] == 3
    assert row["fen_reconstructs_full_search_state"] is False
