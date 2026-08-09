"""Every UCI search option must reach the search, or say that it cannot.

The acceptance criterion for an option here is NOT "a config field got
assigned". It is: what observation proves this reached the search? So every
test below drives the option through the real ``parse_command`` ->
``Engine.dispatch`` -> ``setoption`` path and then observes either

  * a different search — the root visit distribution and/or bestmove of a real
    ``go``, run against a deterministic evaluator so a difference cannot be
    thread noise (``test_gumbel_shape_option_changes_the_search``); or
  * an explicit refusal / NO-EFFECT report when the option cannot act on the
    live search path (``test_..._reported_inert_...``).

The failure mode this file exists for is real and recent in this repo:
``c_puct`` and the ``fpu_*`` family were accepted, stored, and printed as
realized Gumbel search configuration while ``full_tree=True`` made the PUCT
descent they drive unreachable. An operator could tune them for a whole
tournament against a number that never moved.

The NULL CONTROL is ``test_a_cosmetic_option_changes_nothing``: an option that
genuinely should not touch the search must leave the signature byte-identical.
Without it, a "the search changed" assertion could be passing because the
harness is noisy rather than because the knob works.
"""
from __future__ import annotations

import dataclasses
import re

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.mcts.gumbel import (
    PLAY_SEARCH_DEFAULTS,
    PY_ONLY_GUMBEL_KNOBS,
    GumbelConfig,
)
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.mcts.search_options import (
    OPTIONS_BY_NAME,
    SEARCH_OPTIONS,
    SEARCH_PATHS,
    branch_note,
    inert_reason,
    realized_rows,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker
from chess_anti_engine.uci.walker_pool import WalkerPoolConfig

# A quiet middlegame-ish position with plenty of legal moves, so the root has
# more candidates than `topk` and the halving schedule really runs.
FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

# Enough sims that the DESCENT-site transforms (QVisitExp / QVisitFloor /
# QGlobalScale) have a tree deep enough to act on. Measured: at 64 nodes three
# of them are indistinguishable from baseline, which would have produced a
# false "inert" reading for a knob that works. Resolution before threshold.
NODES = 2048


class _DetEval:
    """Deterministic logits from the encoded planes — a fixed random projection.

    Deterministic is the whole point: it makes the classic Gumbel path's root
    visit distribution reproducible, so a difference between two runs is the
    knob and nothing else. ``test_the_harness_is_deterministic`` asserts it.
    """

    def __init__(self, planes: int) -> None:
        rs = np.random.default_rng(1234)
  # 1/sqrt(fan_in): without it the pre-tanh sum overflows float32 on a
  # 175-plane board and the logits go NaN, which the search then treats as a
  # position rather than as a broken evaluator.
        self._w = (
            rs.normal(size=(planes * 64, 16)) / np.sqrt(planes * 64.0)
        ).astype(np.float32)
        self._p = rs.normal(size=(16, POLICY_SIZE)).astype(np.float32)
        self._v = rs.normal(size=(16, 3)).astype(np.float32)

    def evaluate_encoded(self, x, relations=None):
        del relations
  # The leaf buffer is `np.empty`-allocated and its PAD rows carry stale
  # content the search discards; sanitising here keeps a NaN from that
  # padding out of the logits instead of letting it look like a position.
        a = np.nan_to_num(
            np.asarray(x, dtype=np.float32).reshape(np.shape(x)[0], -1),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        h = np.tanh(a @ self._w)
        return (h @ self._p).astype(np.float32), (h @ self._v).astype(np.float32)


def _make_engine(*, threads: int = 1) -> Engine:
    planes = input_plane_count("v2_threats")
    worker = SearchWorker(
        _DetEval(planes),
        device="cpu",
        gumbel_cfg=dataclasses.replace(
            GumbelConfig(
                simulations=256, add_noise=False, temperature=0.0,
                input_extra_features="v2_threats",
            ),
            **PLAY_SEARCH_DEFAULTS,
        ),
        chunk_sims=256,
        n_walkers=threads,
        vloss_weight=3,
    )
    return Engine(worker=worker, options=EngineOptions(threads=threads))


def _setoption(engine: Engine, line: str) -> None:
    """Drive one option through the REAL command path, not the handler."""
    engine.dispatch(parse_command(line))


def _signature(engine: Engine) -> tuple[int, int, tuple[int, ...]]:
    """Run a real ``go``; return (best root action, tree size, root visits).

    Goes through ``position`` / ``go`` dispatch so the observation is of the
    search the engine would run in a game, not of a helper called by hand.

    The tree size is in here because the ROOT visit distribution alone cannot
    see the DESCENT-site transforms. Under the shipped play shape the root uses
    the log transform (``QVisitExpRoot=-1``), which dominates the root's
    sequential halving, so ``QVisitExp`` / ``QGlobalScale`` / ``QVisitFloor``
    change which leaves get expanded deeper in the tree while leaving the root
    counts identical at 2048 sims. Judged on root visits alone all three would
    have read INERT — a wrong verdict produced by the instrument, not the code.
    Node count sees them (8045 -> 8018 / 8042 / 8068) and the null control
    still does not move it.
    """
    engine.dispatch(parse_command(f"position fen {FEN}"))
    engine.dispatch(parse_command(f"go nodes {NODES}"))
    engine._wait_for_search()
    worker = engine._worker
    if worker._tree is None or worker._root_id is None:
        return -1, 0, ()
    actions, counts = worker._tree.get_children_visits(worker._root_id)
    actions = np.asarray(actions)
    counts = np.asarray(counts)
    best = int(actions[int(np.argmax(counts))]) if counts.size else -1
    return best, int(worker._tree.node_count()), tuple(
        int(v) for v in counts[np.argsort(actions)]
    )


def _run(
    lines: tuple[str, ...] = (), *, threads: int = 1,
) -> tuple[int, int, tuple[int, ...]]:
    engine = _make_engine(threads=threads)
    try:
        for line in lines:
            _setoption(engine, line)
        return _signature(engine)
    finally:
        engine._worker.close()


# --- harness controls --------------------------------------------------------


def test_the_harness_is_deterministic() -> None:
    """Precondition for every 'the search changed' assertion below.

    If this fails, a knob test could pass on run-to-run noise. The walker pool
    genuinely IS nondeterministic, which is why the walker-mode assertions in
    this file are structural rather than behavioural.
    """
    assert _run() == _run()


def test_a_cosmetic_option_changes_nothing() -> None:
    """NULL CONTROL. MultiPV changes how many lines are REPORTED, not searched.

    An option that must not move the search, driven through the same path, at
    the same budget. It has to leave the signature byte-identical or every
    positive result in this file is suspect.
    """
    base = _run()
    assert _run(("setoption name MultiPV value 3",)) == base
    assert _run(("setoption name UCI_ShowWDL value true",)) == base


# --- the search-shape options really shape the search ------------------------

# (option, probe value, extra setoptions applied to BOTH arms).
#
# The context column is not a convenience. `QVisitExpRoot` chooses between a
# LOG and a LINEAR root value-transform, and at the shipped `CScaleRoot=7` both
# land far enough up the sigma(q) curve that the completed-Q term already
# dominates the prior — the root ranking is then identical either way and the
# knob is unobservable at every value tried (0, 0.5, 1, 2, 98, -10). Lowering
# the root scale into the regime where the transform still competes with the
# prior is what makes the observation possible. Recording that here rather than
# quietly dropping the option is the point: "no probe value moved it" would
# otherwise have read as "the plumbing is broken".
#
# `searchconfig` now REPORTS that inertness rather than printing [LIVE] over it
# — see `inert_reason`'s q_visit_exp_root arms and the calibration test
# `test_an_inert_root_exponent_verdict_implies_an_unobservable_search` below.
#
# `GumbelScale` is absent: it is nondeterministic by construction and has its
# own test.
_MOVES_THE_SEARCH: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("PolicyTemperature", "3.0", ()),
    ("CScale", "0.9", ()),
    ("CVisit", "5.0", ()),
    ("CScaleRoot", "0.5", ()),
    ("CVisitRoot", "10.0", ()),
    ("QVisitExp", "0.4", ()),
    ("QVisitExpRoot", "1.0", ("setoption name CScaleRoot value 0.05",)),
    ("QVisitFloor", "20.0", ()),
    ("QGlobalScale", "true", ()),
    ("HalvingDiv", "4", ()),
    ("Topk", "4", ()),
    ("ChunkSims", "64", ()),
    ("VLossWeight", "0", ()),
    ("MinibatchSize", "1", ()),
)


@pytest.mark.parametrize(
    ("name", "value", "context"), _MOVES_THE_SEARCH,
    ids=[row[0] for row in _MOVES_THE_SEARCH],
)
def test_gumbel_shape_option_changes_the_search(
    name: str, value: str, context: tuple[str, ...],
) -> None:
    """The deciding test: setoption -> a demonstrably different search.

    Fails if the option is dropped anywhere between `parse_command` and the C
    search — a missing dispatch entry, a value written to a copy of the config,
    or a knob the path does not read.
    """
    base = _run(context)
    got = _run((*context, f"setoption name {name} value {value}"))
    assert got != base, (
        f"{name}={value} left the search byte-identical: same best root move, "
        "same tree size, same root visit distribution. It did not reach the "
        "search."
    )


def test_root_noise_scale_reaches_the_search_and_zero_stays_deterministic() -> None:
    """`GumbelScale` had no reachable mechanism at all before this option.

    `SearchWorker._run_gumbel_chunk` hard-coded `add_noise=False`, so
    `GumbelConfig.gumbel_scale` could not be observed from the engine at any
    value. The zero half of this test is what proves the default is still the
    deterministic search every prior build ran.
    """
    base = _run()
    assert _run(("setoption name GumbelScale value 0.0",)) == base

    noisy = [_run(("setoption name GumbelScale value 1.5",)) for _ in range(4)]
    assert any(sig != base for sig in noisy), (
        "GumbelScale 1.5 never perturbed the root candidate set"
    )


def test_root_noise_is_mirrored_into_the_shared_config_for_rpg() -> None:
    """The second half of GumbelScale, which the classic-path test cannot see.

    Root-parallel Gumbel needs >= 2 devices, so its behaviour is not testable
    here. What IS testable is the mechanism: RPG reads `add_noise` /
    `gumbel_scale` off the SHARED GumbelConfig by reference at candidate-
    selection time, so `set_root_noise_scale` has to write them there too or
    the option is a silent null on that path — with no warning, because the
    registry declares it live on `rpg`.

    A mutation that drops the mirror survived the behavioural suite; this is
    the test that kills it. Both halves are asserted: that the worker writes
    the fields, and that RPG is what reads them (source-level, since a
    two-device search is unavailable).
    """
    import inspect

    from chess_anti_engine.uci import root_parallel_gumbel as rpg

    engine = _make_engine()
    try:
        _setoption(engine, "setoption name GumbelScale value 1.5")
        cfg = engine._worker._cfg
        assert cfg.add_noise is True
        assert cfg.gumbel_scale == pytest.approx(1.5)

        _setoption(engine, "setoption name GumbelScale value 0")
        assert engine._worker._cfg.add_noise is False
    finally:
        engine._worker.close()

    src = inspect.getsource(rpg)
    assert "self._gcfg.add_noise" in src, (
        "RPG no longer reads add_noise off the shared config — re-decide "
        "whether GumbelScale is still live on the rpg path in SEARCH_OPTIONS"
    )
    assert "self._gcfg.gumbel_scale" in src


# --- inert options are never silently accepted -------------------------------


@pytest.mark.parametrize("name", ["CPuct", "CPuctFactor", "CPuctBase", "FpuReduction"])
def test_puct_option_is_reported_inert_under_classic_gumbel(
    name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact defect this surface could have mass-produced.

    Two assertions, and both matter: the engine SAYS the value cannot act
    (an operator reading a match log finds out), and the search is in fact
    unchanged (so the message is true rather than defensive boilerplate).
    """
    base = _run()
    engine = _make_engine()
    try:
        _setoption(engine, f"setoption name {name} value 9.5")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out, f"{name} was accepted silently under Gumbel"
        assert "gumbel" in out
        assert _signature(engine) == base, (
            f"{name} is advertised as inert under Gumbel but moved the search"
        )
    finally:
        engine._worker.close()


@pytest.mark.parametrize(
    "name", ["CScale", "CVisit", "Topk", "PolicyTemperature", "HalvingDiv"],
)
def test_gumbel_option_is_reported_inert_on_the_walker_pool(
    name: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Threads>1 is plain PUCT: every Gumbel shape knob stops applying.

    The report is asserted behaviourally-by-proxy rather than by a visit-count
    diff, because the walker pool races N threads onto a shared tree and its
    visit distribution is NOT reproducible — a diff there would be
    uninterpretable. `test_the_walker_pool_config_carries_no_gumbel_knob`
    supplies the structural half.
    """
    engine = _make_engine(threads=2)
    try:
        capsys.readouterr()
        _setoption(engine, f"setoption name {name} value 3")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out
        assert "walker" in out
    finally:
        engine._worker.close()


def test_the_walker_pool_config_carries_no_gumbel_knob() -> None:
    """Structural proof behind the walker-mode inertness claim.

    The pool cannot read what it was never given. Adding a Gumbel field to
    `WalkerPoolConfig` should fail this test and force a re-decision of the
    `live_in` sets in the registry.
    """
    fields = {f.name for f in dataclasses.fields(WalkerPoolConfig)}
    gumbel_only = {
        o.field for o in SEARCH_OPTIONS if "walker" not in o.live_in
    }
    assert not (fields & gumbel_only), sorted(fields & gumbel_only)


def test_minibatch_size_is_inert_off_the_classic_gumbel_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MinibatchSize shipped before this PR with no mode gate at all.

    It is `run_gumbel_root_many_c(target_batch=...)` — a C-path function
    argument — so on the walker pool it was accepted, stored, echoed back as
    "MinibatchSize set to N", and read by nothing.
    """
    engine = _make_engine(threads=2)
    try:
        capsys.readouterr()
        _setoption(engine, "setoption name MinibatchSize value 512")
        out = capsys.readouterr().out
        assert "NO EFFECT" in out
    finally:
        engine._worker.close()


def test_searchconfig_reports_the_path_that_actually_ran(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UseVL=true on an evaluator without the slot API is accepted and ignored.

    The readback must report the path that WILL run (classic gumbel), not the
    one that was requested, or it becomes another way to be confidently wrong
    about the engine's configuration.
    """
    engine = _make_engine()
    try:
        _setoption(engine, "setoption name UseVL value true")
        assert engine._worker.realized_search_path() == "gumbel"
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        out = capsys.readouterr().out
        assert "searchconfig path=gumbel" in out
        assert re.search(r"searchconfig CScale = 0\.025 \[LIVE\]", out)
        assert re.search(r"searchconfig CPuct = [0-9.]+ \[INERT\]", out)
        # Every registered option gets a row: an omitted row is
        # indistinguishable from an unsupported build.
        for opt in SEARCH_OPTIONS:
            assert f"searchconfig {opt.name} = " in out
    finally:
        engine._worker.close()


def test_searchconfig_reports_a_value_the_setoption_actually_applied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Readback and search must agree, or the readback is decoration."""
    engine = _make_engine()
    try:
        _setoption(engine, "setoption name PolicyTemperature value 2.5")
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        assert "searchconfig PolicyTemperature = 2.5 [LIVE]" in capsys.readouterr().out
        assert engine._worker._cfg.policy_temp == pytest.approx(2.5)
    finally:
        engine._worker.close()


# --- parsing and ranges ------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expect"),
    [
        ("setoption name PolicyTemperature value 0.4", "out of range"),
        ("setoption name PolicyTemperature value 5.5", "out of range"),
        ("setoption name PolicyTemperature value abc", "not a number"),
        ("setoption name Topk value 1", "out of range"),
        ("setoption name HalvingDiv value 1.5", "not a integer"),
        ("setoption name QGlobalScale value yes", "expected true/false"),
    ],
)
def test_a_rejected_value_says_so_and_keeps_the_old_one(
    line: str, expect: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """UCI gives a GUI no way to see a rejected option, so silence here is
    indistinguishable from success. A bad `PolicyTemperature 0` would also
    divide the policy logits by zero."""
    engine = _make_engine()
    try:
        before = dict(engine._worker.realized_search_values())
        capsys.readouterr()
        _setoption(engine, line)
        assert expect in capsys.readouterr().out
        assert engine._worker.realized_search_values() == before
    finally:
        engine._worker.close()


def test_policy_temperature_accepts_the_documented_range_ends() -> None:
    engine = _make_engine()
    try:
        for value in ("0.5", "5.0"):
            _setoption(engine, f"setoption name PolicyTemperature value {value}")
            assert engine._worker._cfg.policy_temp == pytest.approx(float(value))
    finally:
        engine._worker.close()


# --- the surface cannot drift from the code ----------------------------------


def test_advertised_defaults_match_the_live_worker() -> None:
    """A handshake default that is not what the engine runs is the same lie.

    `_build_engine` copies the constructed worker's realized values back into
    EngineOptions for exactly this reason; this test fails if someone
    reintroduces a hand-typed default.
    """
    from chess_anti_engine.uci.__main__ import _build_engine

    planes = input_plane_count("v2_threats")
    options = EngineOptions()
    engine = _build_engine(
        evaluator=_DetEval(planes),
        primary_device="cpu",
        chunk_sims=777,
        topk=9,
        c_scale=0.077,
        policy_temp=1.5,
        n_walkers=1,
        vloss_weight=2,
        walker_gather=1,
        pucv_vloss_mode=0,
        max_batch=64,
        vl_gather=64,
        eval_cache_entries=0,
        use_multi_gpu_pucv=False,
        input_extra_features="v2_threats",
        options=options,
    )
    try:
        realized = engine._worker.realized_search_values()
        for opt in SEARCH_OPTIONS:
            assert options.search_value(opt.field) == realized[opt.field], (
                f"handshake would advertise {opt.name}="
                f"{options.search_value(opt.field)} while the worker runs "
                f"{realized[opt.field]}"
            )
        # ...and the CLI values really landed, not just agreed with each other.
        assert realized["chunk_sims"] == 777
        assert realized["topk"] == 9
        assert realized["policy_temp"] == pytest.approx(1.5)
    finally:
        engine._worker.close()


def test_every_registry_field_resolves_on_engine_options() -> None:
    options = EngineOptions()
    for opt in SEARCH_OPTIONS:
        options.search_value(opt.field)  # raises AttributeError on a typo


def test_every_registry_option_is_reachable_by_its_uci_name() -> None:
    """A registry entry with no dispatch route is an option that cannot be set.

    `_handle_setoption` lowercases the name and looks it up in OPTIONS_BY_NAME,
    so this is the property that makes every declared option settable.
    """
    for opt in SEARCH_OPTIONS:
        assert OPTIONS_BY_NAME[opt.name.lower()] is opt


def test_float_options_declare_a_range_and_int_options_declare_bounds() -> None:
    for opt in SEARCH_OPTIONS:
        if opt.kind in ("string", "spin"):
            assert opt.lo is not None, opt.name
            assert opt.hi is not None, opt.name
            assert opt.lo < opt.hi, opt.name
            assert opt.lo <= float(opt.default) <= opt.hi, opt.name


def test_policy_temp_is_reachable_from_the_play_search_surface() -> None:
    """It was a live positive control in the inert-knob audit and the ONE shape
    knob no play-path config could set. Absence is not "at its default"."""
    assert "policy_temp" in PLAY_SEARCH_DEFAULTS
    assert PLAY_SEARCH_DEFAULTS["policy_temp"] == 1.0
    assert "policy_temp" in {o.field for o in SEARCH_OPTIONS}


def test_the_uci_surface_never_exposes_the_training_time_target_temperature() -> None:
    """`policy_target_temp` is a TRAINING knob on the policy TARGET. It never
    runs during play, and an operator who found it in the UCI option list would
    reasonably conclude otherwise."""
    names = {o.name.lower() for o in SEARCH_OPTIONS}
    assert "policytargettemperature" not in names
    assert not any("target" in n for n in names)


# --- the C path cannot silently drop a Python-only knob ----------------------


def test_the_c_path_refuses_a_python_only_knob() -> None:
    """The OTHER way a knob returns a flawless null.

    `volatility_q_scale` is a real, implemented GumbelConfig field — on the
    PYTHON search. `run_gumbel_root_many_c` has no code for it, so a caller
    that set it and landed on the C path would get a complete, reproducible,
    wrong measurement. The guard is on the dispatch boundary, so it covers
    every caller at once rather than one CLI.
    """
    cfg = GumbelConfig(
        simulations=8, add_noise=False, temperature=0.0,
        input_extra_features="v2_threats", volatility_q_scale=0.5,
    )
    planes = input_plane_count("v2_threats")
    with pytest.raises(ValueError, match="volatility_q_scale"):
        run_gumbel_root_many_c(
            model=None, boards=[chess.Board(FEN)], device="cpu",
            rng=np.random.default_rng(0), cfg=cfg, evaluator=_DetEval(planes),
        )


def test_the_python_only_denylist_names_fields_that_exist() -> None:
    fields = set(GumbelConfig.__dataclass_fields__)
    assert fields >= PY_ONLY_GUMBEL_KNOBS
    # ...and none of them is exposed over UCI, since the UCI engine is
    # C-path-only and would have to refuse them at every `go`.
    assert not (PY_ONLY_GUMBEL_KNOBS & {o.field for o in SEARCH_OPTIONS})


def test_a_default_config_still_enters_the_c_path() -> None:
    """Negative control for the guard: it must not refuse ordinary configs."""
    cfg = GumbelConfig(
        simulations=8, add_noise=False, temperature=0.0,
        input_extra_features="v2_threats",
    )
    planes = input_plane_count("v2_threats")
    out = run_gumbel_root_many_c(
        model=None, boards=[chess.Board(FEN)], device="cpu",
        rng=np.random.default_rng(0), cfg=cfg, evaluator=_DetEval(planes),
    )
    assert out[1]  # an action was chosen


# --- the inertness model itself ---------------------------------------------


def test_rpg_reports_the_descent_knobs_inert_while_the_root_knobs_are_set() -> None:
    """Root-parallel Gumbel runs Gumbel only at the ROOT; the intra-candidate
    descent is a PUCT chunker. So `CScale` is read there only as the root
    fallback — and at the shipped defaults (CScaleRoot=7) it is not read at
    all. A static per-path table would have called it live."""
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    c_scale = OPTIONS_BY_NAME["cscale"]
    assert inert_reason(c_scale, "rpg", values) is not None
    assert inert_reason(c_scale, "gumbel", values) is None

    values["c_scale_root"] = -1.0
    assert inert_reason(c_scale, "rpg", values) is None


def test_topk_is_live_on_both_gumbel_paths_and_dead_on_the_puct_ones() -> None:
    topk = OPTIONS_BY_NAME["topk"]
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    assert inert_reason(topk, "gumbel", values) is None
    assert inert_reason(topk, "rpg", values) is None
    for path in ("walker", "pucv", "pucv_pool"):
        assert inert_reason(topk, path, values) is not None


def test_the_puct_family_is_live_exactly_where_a_puct_descent_runs() -> None:
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    for name in ("cpuct", "cpuctfactor", "cpuctbase", "fpureduction"):
        opt = OPTIONS_BY_NAME[name]
        assert inert_reason(opt, "gumbel", values) is not None
        for path in ("walker", "pucv", "pucv_pool", "rpg"):
            assert inert_reason(opt, path, values) is None


# --- a NO EFFECT claim must be true of the TRANSITION, not of an arm --------

# (label, context applied to both arms, option, value A, value B).
#
# The bug this table exists for: `inert_reason` answered "are all values inside
# this arm equivalent?" while `Engine._set_search_option` used the answer to
# assert "this setoption had no effect". They diverge on any setoption that
# CROSSES an arm boundary, and the engine then printed `NO EFFECT on the live
# search` for a command that changed its own move. The first two rows are that
# case, in both directions -- the reverse one printed "Only crossing to >= 0 can
# change anything" having just crossed from >= 0.
#
# The old cell list was all within-arm, which is exactly why it passed.
_SETOPTION_EFFECT_CELLS: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    ("QVisitExpRoot CROSS-ARM at CScaleRoot=0.05, LOG -> POWER",
     ("setoption name CScaleRoot value 0.05",), "QVisitExpRoot", "-1.0", "1.0"),
    ("QVisitExpRoot CROSS-ARM at CScaleRoot=0.05, POWER -> LOG",
     ("setoption name CScaleRoot value 0.05",), "QVisitExpRoot", "1.0", "-1.0"),
    ("QVisitExpRoot CROSS-ARM at the shipped CScaleRoot=7",
     (), "QVisitExpRoot", "-1.0", "1.0"),
    ("QVisitExpRoot within the LOG arm", (), "QVisitExpRoot", "-1.0", "-10"),
    ("QVisitExpRoot within the POWER arm", (), "QVisitExpRoot", "0", "2"),
    ("QVisitExpRoot into the >=90 sentinel", (), "QVisitExpRoot", "98", "95"),
    # A genuinely unreachable knob: NO EFFECT here must stay true.
    ("CPuct under classic Gumbel", (), "CPuct", "1.75", "9.5"),
)


def _apply_and_observe(
    context: tuple[str, ...], option: str, value: str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, tuple[int, int, tuple[int, ...]]]:
    """Set one option through the real dispatch; return (its info string, search)."""
    engine = _make_engine()
    try:
        for line in context:
            _setoption(engine, line)
        capsys.readouterr()
        _setoption(engine, f"setoption name {option} value {value}")
        message = "".join(
            line for line in capsys.readouterr().out.splitlines(keepends=True)
            if option in line
        )
        return message, _signature(engine)
    finally:
        engine._worker.close()


@pytest.mark.parametrize(
    ("label", "context", "option", "value_a", "value_b"), _SETOPTION_EFFECT_CELLS,
    ids=[c[0] for c in _SETOPTION_EFFECT_CELLS],
)
def test_a_no_effect_report_implies_the_search_did_not_move(
    label: str, context: tuple[str, ...], option: str,
    value_a: str, value_b: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """THE deciding test, and it is on the CLAIM the engine actually makes.

    The previous version asserted `inert_reason(...) is not None => identical
    search`, which is a statement about a predicate. The operator never sees the
    predicate; they see `NO EFFECT on the live search`. So the implication that
    has to hold is:

        the engine printed NO EFFECT  =>  the search is byte-identical.

    Cross-arm cells are included precisely because the old list had none.
    """
    _, sig_a = _apply_and_observe(context, option, value_a, capsys)
    message, sig_b = _apply_and_observe(context, option, value_b, capsys)

    if "NO EFFECT" in message:
        assert sig_a == sig_b, (
            f"{label}: the engine reported NO EFFECT for "
            f"{option}={value_b} but the search MOVED: {sig_a[:2]} vs "
            f"{sig_b[:2]}. A knob that took effect and was reported as ignored "
            "is the same defect as one silently dropped, sign-flipped.\n"
            f"message: {message.strip()}"
        )


def test_the_reviewer_cross_arm_case_reports_honestly_and_moves_the_search(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Named regression for the exact reproduction that blocked this PR.

    `CScaleRoot 0.05`, `QVisitExpRoot -1.0 -> 1.0`: best root action 306 -> 553,
    tree 8037 -> 7764, and the engine used to call it NO EFFECT.
    """
    context = ("setoption name CScaleRoot value 0.05",)
    _, sig_log = _apply_and_observe(context, "QVisitExpRoot", "-1.0", capsys)
    message, sig_pow = _apply_and_observe(context, "QVisitExpRoot", "1.0", capsys)

    assert sig_log != sig_pow, (
        "harness precondition: this transition must move the search, otherwise "
        "the test cannot detect a false NO EFFECT"
    )
    assert "NO EFFECT" not in message
    assert "reaches the live search" in message


def test_the_shipped_root_exponent_is_branch_pinned_not_inert() -> None:
    """`QVisitExpRoot = -1.0` selects the shipped LOG root transform.

    Reporting it INERT, in the same column as `CPuct` (which genuinely cannot
    reach this path), told an operator proving their config that a load-bearing
    parameter was doing nothing.
    """
    values = {o.field: o.default for o in SEARCH_OPTIONS}
    opt = OPTIONS_BY_NAME["qvisitexproot"]
    assert values["q_visit_exp_root"] == float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"])

    assert inert_reason(opt, "gumbel", values) is None
    note = branch_note(opt, "gumbel", values)
    assert note is not None
    assert "every value < 0 is the same search" in note

    rows = {name: (status, why) for name, _, status, why in realized_rows("gumbel", values)}
    assert rows["QVisitExpRoot"][0] == "BRANCH"
    assert rows["CPuct"][0] == "INERT"


def test_the_branch_note_is_exact_and_tracks_the_current_value() -> None:
    """Each arm is read off the C, so the note must follow the value it is about.

    Also a gate-can-fail check: an option with no branch structure must get no
    note on any path.
    """
    opt = OPTIONS_BY_NAME["qvisitexproot"]
    for value, expected in (
        (-1.0, "every value < 0"), (-10.0, "every value < 0"),
        (0.0, "the power branch"), (2.0, "the power branch"),
        (98.0, "every value in [90, 99]"),
    ):
        values = {o.field: o.default for o in SEARCH_OPTIONS}
        values["q_visit_exp_root"] = value
        for path in ("gumbel", "rpg"):
            note = branch_note(opt, path, values)
            assert note is not None, (path, value)
            assert expected in note, (path, value, note)

    values = {o.field: o.default for o in SEARCH_OPTIONS}
    for name in ("topk", "cscale", "policytemperature", "halvingdiv"):
        for path in SEARCH_PATHS:
            assert branch_note(OPTIONS_BY_NAME[name], path, values) is None


def test_searchconfig_counts_branch_pinned_apart_from_inert(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The readback's summary line must not fold the two together."""
    engine = _make_engine()
    try:
        capsys.readouterr()
        engine.dispatch(parse_command("searchconfig"))
        out = capsys.readouterr().out
    finally:
        engine._worker.close()

    assert "QVisitExpRoot = -1.0 [BRANCH]" in out
    # Value not pinned: the harness's realized CPuct is GumbelConfig's, not the
    # registry default. The STATUS is what this test is about.
    assert re.search(r"searchconfig CPuct = \S+ \[INERT\]", out)
    assert "1 branch-pinned" in out


# --- audit_targets --gumbel passthrough --------------------------------------


def test_audit_targets_gumbel_overrides_reach_the_play_profile() -> None:
    from scripts.audit_targets import build_search_profiles, parse_gumbel_overrides

    overrides = parse_gumbel_overrides(["policy_temp=2.2", "topk=8,halving_div=4"])
    profiles = build_search_profiles(
        {}, play_sims=64, play_topk=None, gumbel_overrides=overrides,
    )
    assert profiles["search"].overrides == overrides
    # ...and NOT the training rows, which describe the stored target.
    assert profiles["train"].overrides == ()
    assert profiles["train_fast"].overrides == ()

    both = build_search_profiles(
        {}, play_sims=64, play_topk=None, gumbel_overrides=overrides,
        override_training_rows=True,
    )
    assert both["train"].overrides == overrides


@pytest.mark.parametrize(
    ("spec", "expect"),
    [
        ("c_puct=3", "cannot affect a Gumbel search"),
        ("fpu_reduction=0.1", "cannot affect a Gumbel search"),
        ("not_a_field=1", "is not a GumbelConfig field"),
        ("topk", "expected k=v"),
        ("topk=abc", "is not a number"),
    ],
)
def test_audit_targets_refuses_an_override_that_would_measure_a_null(
    spec: str, expect: str,
) -> None:
    from scripts.audit_targets import parse_gumbel_overrides

    with pytest.raises(SystemExit, match=re.escape(expect)):
        parse_gumbel_overrides([spec])


def test_audit_targets_aborts_when_an_override_fails_to_reach_the_config() -> None:
    """THE dispatch guard, tested by making the plumbing fail on purpose.

    `_SearchProfile` carries a FIXED field list, so before this guard an
    override for a field outside it parsed, printed in the run header, and was
    dropped — a complete, reproducible, wrong audit. The guard compares against
    the config the runner is about to be handed.
    """
    from scripts.audit_targets import _assert_overrides_realized

    cfg = GumbelConfig(policy_temp=1.0)
    with pytest.raises(SystemExit, match="did not reach the search config"):
        _assert_overrides_realized(
            cfg, (("policy_temp", 2.2),), where="test",
        )
    # Negative control: a config that DID take the override passes.
    _assert_overrides_realized(
        GumbelConfig(policy_temp=2.2), (("policy_temp", 2.2),), where="test",
    )


def test_audit_targets_override_coercion_keeps_int_fields_int() -> None:
    """`--gumbel topk=8` must not land a float on an int field: the C
    `start_gumbel_sims` signature rejects it mid-search, which fails long after
    the checkpoint and Stockfish have been paid for."""
    from scripts.audit_targets import _coerce_override

    assert isinstance(_coerce_override(GumbelConfig().topk, 8.0), int)
    assert isinstance(_coerce_override(GumbelConfig().c_scale, 8), float)
    assert _coerce_override(GumbelConfig().q_global_scale, 1.0) is True


def test_realized_search_path_mirrors_the_dispatch_branch_order() -> None:
    """`realized_search_path` is only trustworthy if it reads the same state
    `_run_one_chunk` branches on, in the same order."""
    import inspect

    src = inspect.getsource(SearchWorker._run_one_chunk)
    order: list[str] = []
    for m in re.findall(r"self\.(_rpg_pool|_pucv_pool|_walker_pool|_pucv) is not None", src):
        # The `searchmoves` guard tests `_rpg_pool` a second time before the
        # branch chain; dedupe so this compares dispatch ORDER, not mentions.
        if m not in order:
            order.append(m)
    reported = re.findall(
        r"self\.(_rpg_pool|_pucv_pool|_walker_pool|_pucv) is not None",
        inspect.getsource(SearchWorker.realized_search_path),
    )
    assert reported == order[: len(reported)]
