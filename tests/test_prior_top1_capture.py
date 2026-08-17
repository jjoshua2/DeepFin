"""Issue #425 step: persist the generating net's raw root prior top-1.

The ΔQ experiment needs, for ONE model θ, ``a_P = argmax π_θ(s)`` paired against
``a_M = MCTS_θ(s)``. Offline that pair is unobtainable: the replay schema never
persisted the generating prior (``_NetRecord.policy_probs`` is the SEARCH-improved
target and is the only policy that reaches a shard), so a current checkpoint's
prior can only ever be paired against a *historical* net's played move. Capturing
the prior at selfplay time makes both halves come from the same θ by construction.

The question these tests are written against is not "is the arithmetic right" but
**"does the value take effect on the production path, and what observation proves
it did"**. So:

  * capture runs through the REAL record-append functions (both the C fast path
    and the Python fallback), never a re-derivation the loop does not execute;
  * correctness is checked against an INDEPENDENT numpy argmax computed in the
    test, not against the other path -- two paths sharing one wrong rule agree
    perfectly;
  * the payoff assertion reads a shard **written to disk and read back** through
    the production zarr writer, not an in-memory object;
  * the gating flag is proven by the DIFFERENCE it makes to that written shard.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.moves.encode import compact_policy_index, full_policy_index
from chess_anti_engine.replay.shard import load_shard_arrays, samples_to_arrays, save_local_shard_arrays
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.finalize import _build_replay_samples
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.network_turn import (
    _append_records_via_c,
    _append_records_via_python,
)
from chess_anti_engine.selfplay.state import SelfplayState

# Presence check without importing the extension for its side effects.
_HAS_C = importlib.util.find_spec("chess_anti_engine.mcts._mcts_tree") is not None

requires_c = pytest.mark.skipif(not _HAS_C, reason="C ply path unavailable")

_BATCH = 24


def _session_state(game: GameConfig) -> SelfplayState:
    """A real ``SelfplayState``, built the way ``play_batch`` builds one."""
    evaluator = Mock(spec=["evaluate_encoded"])
    stockfish = Mock(spec=["search", "nodes"])
    stockfish.nodes = 0
    return SelfplayState.create(
        model=None,
        device="cpu",
        rng=np.random.default_rng(0),
        stockfish=stockfish,
        evaluator=evaluator,
        batch_size=_BATCH,
        continuous=False,
        target=_BATCH,
        opponent=OpponentConfig(),
        temp=TemperatureConfig(),
        search=SearchConfig(),
        opening=OpeningConfig(),
        diff_focus=DiffFocusConfig(),
        game=game,
    )


def _ply_inputs(state: SelfplayState) -> dict[str, Any]:
    """One deterministic ply batch, shared by every path under test.

    ⚑ The improved policy (``probs``) is deliberately built so its argmax is a
    DIFFERENT move from the prior's argmax wherever the position allows it. That
    is the whole semantic content of this field: if the capture silently read the
    search-improved policy (or the played action) instead of the raw prior, every
    other assertion here would still pass. ``_prior_disagrees_with_search``
    below asserts the disagreement is actually present, so the discrimination
    cannot quietly evaporate.
    """
    n = _BATCH
    rng = np.random.default_rng(7)
    pol = (rng.standard_normal((n, POLICY_SIZE)) * 2.0).astype(np.float32)
    probs = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    actions: list[int | None] = []
    values: list[float | None] = []
    for i in range(n):
        mask = legal_move_mask(state.boards[i])
        legal = np.flatnonzero(mask)
        prior_top = int(legal[int(np.argmax(pol[i][legal]))])
        # Put the search's mass on legal moves OTHER than the prior's argmax.
        others = legal[legal != prior_top]
        k = max(2, len(others) // 4)
        chosen = rng.choice(others, size=k, replace=False)
        visits = rng.integers(1, 40, size=k).astype(np.float32)
        probs[i, chosen] = visits / visits.sum()
        actions.append(int(chosen[int(np.argmax(probs[i, chosen]))]))
        values.append(float(np.clip(rng.uniform(-1.0, 1.0), -1.0, 1.0)))
    return {
        "pol_logits": pol,
        "wdl_logits": rng.standard_normal((n, 3)).astype(np.float32) * 1.5,
        "probs_list": [probs[i] for i in range(n)],
        "actions": actions,
        "values_list": values,
    }


def _expected_prior(pol_logits: np.ndarray, board) -> tuple[int, float]:
    """Masked-softmax argmax computed independently of the production helper.

    Written with plain numpy on purpose: sharing ``_prior_top1`` would make this
    a tautology, and a rule that is wrong in the C path and the Python path alike
    is exactly what a parity check cannot see.
    """
    mask = np.asarray(legal_move_mask(board), dtype=bool)
    lg = np.asarray(pol_logits, dtype=np.float64).copy()
    lg[~mask] = -np.inf
    top = int(np.argmax(lg))
    e = np.exp(lg - lg[top])
    e[~mask] = 0.0
    return top, float(e[top] / e.sum())


def _run_c(game: GameConfig) -> tuple[SelfplayState, dict[str, Any], list]:
    state = _session_state(game)
    boards_before = [state.boards[i].copy() for i in range(_BATCH)]
    inp = _ply_inputs(state)
    _append_records_via_c(
        state, list(range(_BATCH)),
        cb_encode_list=list(state.cboards[:_BATCH]),
        pol_logits=inp["pol_logits"], wdl_logits_raw=inp["wdl_logits"],
        actions=inp["actions"], values_list=inp["values_list"],
        probs_list=inp["probs_list"],
        gumbel_diags=[None] * _BATCH, is_full_py=[True] * _BATCH,
        sample_weights=[1.0] * _BATCH, diff_focus=DiffFocusConfig(),
    )
    return state, inp, boards_before


def _run_python(game: GameConfig) -> tuple[SelfplayState, dict[str, Any], list]:
    state = _session_state(game)
    boards_before = [state.boards[i].copy() for i in range(_BATCH)]
    inp = _ply_inputs(state)
    logits = inp["wdl_logits"].astype(np.float64)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    wdl_est = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    planes = input_plane_count(state.game.input_extra_features)
    _append_records_via_python(
        state, list(range(_BATCH)),
        xs_batch=np.zeros((_BATCH, planes, 8, 8), dtype=np.float32),
        pol_logits=inp["pol_logits"], wdl_est=wdl_est,
        probs_list=inp["probs_list"], actions=inp["actions"],
        values_list=inp["values_list"],
        gumbel_diags=[None] * _BATCH, masks_list=[None] * _BATCH,
        is_full=np.ones(_BATCH, dtype=bool), sample_weights=[1.0] * _BATCH,
        diff_focus=DiffFocusConfig(),
    )
    return state, inp, boards_before


def _records(state: SelfplayState) -> list:
    return [state.samples_per_game[i][0] for i in range(_BATCH)]


# ---------------------------------------------------------------------------
# Capture: the value, on both real ply paths
# ---------------------------------------------------------------------------


@requires_c
def test_prior_disagrees_with_search_in_the_fixture() -> None:
    """Guard against the discrimination silently evaporating.

    Every assertion below distinguishes "read the prior" from "read the search"
    only while the two actually differ. If a future edit to ``_ply_inputs`` let
    them coincide, the suite would keep passing while testing nothing.
    """
    state, inp, boards = _run_c(GameConfig())
    agreements = 0
    for i, rec in enumerate(_records(state)):
        prior_idx, _ = _expected_prior(inp["pol_logits"][i], boards[i])
        search_idx = int(np.argmax(inp["probs_list"][i]))
        assert rec.prior_top1_index == prior_idx
        if prior_idx == search_idx or prior_idx == int(inp["actions"][i]):
            agreements += 1
    assert agreements == 0, "fixture no longer separates prior from search"


@requires_c
def test_c_path_captures_the_independently_computed_masked_argmax() -> None:
    state, inp, boards = _run_c(GameConfig())
    for i, rec in enumerate(_records(state)):
        idx, prob = _expected_prior(inp["pol_logits"][i], boards[i])
        assert rec.prior_top1_index == idx
        assert rec.prior_top1_prob == pytest.approx(prob, abs=1e-9)


def test_python_path_captures_the_independently_computed_masked_argmax() -> None:
    state, inp, boards = _run_python(GameConfig())
    for i, rec in enumerate(_records(state)):
        idx, prob = _expected_prior(inp["pol_logits"][i], boards[i])
        assert rec.prior_top1_index == idx
        assert rec.prior_top1_prob == pytest.approx(prob, abs=1e-9)


@requires_c
def test_both_ply_paths_capture_identical_values() -> None:
    """``_append_records_via_python`` only runs when the C extension is missing;
    a fallback that stores a different value is the defect, not the safety net.
    """
    c_state, _, _ = _run_c(GameConfig())
    py_state, _, _ = _run_python(GameConfig())
    for c_rec, py_rec in zip(_records(c_state), _records(py_state), strict=True):
        assert c_rec.prior_top1_index == py_rec.prior_top1_index
        assert c_rec.prior_top1_prob == pytest.approx(py_rec.prior_top1_prob, abs=1e-12)


@requires_c
@pytest.mark.parametrize("runner", [_run_c, _run_python])
def test_flag_off_captures_nothing_on_either_path(runner) -> None:
    state, _, _ = runner(GameConfig(record_prior_top1=False))
    for rec in _records(state):
        assert rec.prior_top1_index is None
        assert rec.prior_top1_prob is None


@requires_c
def test_prior_top1_prob_is_a_probability() -> None:
    state, _, _ = _run_c(GameConfig())
    for rec in _records(state):
        assert 0.0 < float(rec.prior_top1_prob) <= 1.0


# ---------------------------------------------------------------------------
# The payoff: a shard WRITTEN to disk and READ BACK
# ---------------------------------------------------------------------------


def _write_and_reload(state: SelfplayState, tmp_path: Path, name: str) -> dict[str, Any]:
    """records -> real finalize -> real production zarr writer -> read back."""
    samples: list = []
    for i in range(_BATCH):
        recs = list(state.samples_per_game[i])
        samples.extend(
            _build_replay_samples(
                state, i, recs,
                result="1/2-1/2",
                tb_policy_overrides={},
                vol_targets=[None] * len(recs),
                sf_vol_targets=[None] * len(recs),
                total_plies_played=len(recs),
                ply_to_index={int(r.ply_index): j for j, r in enumerate(recs)},
            ),
        )
    assert samples, "finalize produced no rows"
    path = save_local_shard_arrays(tmp_path / f"{name}.zarr", arrs=samples_to_arrays(samples))
    # load_shard_arrays runs validate_arrays, which range-checks every field in
    # POLICY_INDEX_FIELDS against the shard's policy width. A -1 sentinel or an
    # un-remapped full-4672 id in a compact shard fails HERE.
    arrs, _meta = load_shard_arrays(path)
    return arrs


@requires_c
def test_prior_top1_reaches_a_written_shard_and_decodes_to_a_legal_move(tmp_path) -> None:
    """The deciding observation for this change.

    Asserts on what the shard on disk CONTAINS after a real write/read round
    trip -- certifying a value in memory and dispatching it to storage are
    different events, and only the second one is what the ΔQ dataset reads.
    """
    state, inp, boards = _run_c(GameConfig(record_prior_top1=True))
    arrs = _write_and_reload(state, tmp_path, "on")

    assert "prior_top1_index" in arrs, "field never reached the written shard"
    assert "prior_top1_prob" in arrs
    has = np.asarray(arrs["has_prior_top1"]).astype(bool)
    idx = np.asarray(arrs["prior_top1_index"]).astype(np.int64)
    prob = np.asarray(arrs["prior_top1_prob"]).astype(np.float64)
    legal = np.asarray(arrs["legal_mask"])
    assert has.all(), "every net-turn row should carry the prior"

    # (a) the stored index names a LEGAL move of that row's own position
    covered = 0
    for row in np.flatnonzero(has):
        assert int(legal[row, idx[row]]) == 1, f"row {row}: prior top-1 is illegal"
        assert 0.0 < prob[row] <= 1.0
        covered += 1
    assert covered == int(has.sum())

    # (b) and it is the move we actually meant -- translated out of the shard's
    # compact encoding back to the full action id the test computed.
    # Subset, not equality: finalize's difficulty-focus subsampling legitimately
    # drops whole rows before the shard, so the population here is "rows that
    # reached storage". `has.all()` above is the strong half -- of the rows that
    # DID land, every one carries a prior.
    expected = {_expected_prior(inp["pol_logits"][i], boards[i])[0] for i in range(_BATCH)}
    stored_full = {full_policy_index(int(v)) for v in idx[has]}
    assert stored_full <= expected
    assert len(stored_full) >= 3, "too few rows survived to make this meaningful"


@requires_c
def test_the_flag_changes_what_lands_in_a_written_shard(tmp_path) -> None:
    """Proof by execution that the knob is not decorative.

    A ``record_*`` flag wired at five of its six sites reads as working; the only
    thing that distinguishes wired from dead is a DIFFERENCE in the bytes on
    disk. Same inputs, same code, flag flipped.
    """
    on = _write_and_reload(_run_c(GameConfig(record_prior_top1=True))[0], tmp_path, "on")
    off = _write_and_reload(_run_c(GameConfig(record_prior_top1=False))[0], tmp_path, "off")

    assert np.asarray(on["has_prior_top1"]).astype(bool).all()
    # Off: either the column is pruned out of the shard entirely, or it is
    # present with every presence flag clear. Both are "no rows carry it"; what
    # must NOT happen is a covered row.
    assert int(np.asarray(off.get("has_prior_top1", np.zeros(1))).astype(bool).sum()) == 0
    assert int(np.asarray(off.get("has_prior_top1_prob", np.zeros(1))).astype(bool).sum()) == 0


@requires_c
def test_the_kill_switch_also_kills_priors_carried_in_from_a_resumed_game(
    tmp_path,
) -> None:
    """⚑ The flag gates the WRITE, not only the capture.

    ``record_prior_top1`` is `_RESUME_COMPAT_EXEMPT`, so a game suspended by a
    session that had it ON resumes into a session that has it OFF still carrying
    priors on its restored records. If finalize only asked "does this record hold
    a prior?", the kill switch would leave those rows covered — a switch that
    turns the field off for new plies but not for the shard is not a kill switch,
    and it breaks the pre-committed "0% coverage when disabled" arm of this
    change's yardstick.

    Reproduced without the resume machinery, which is not what is under test:
    records built with the flag ON (so they genuinely carry values), finalized
    under a game config with the flag OFF — exactly the state a resumed game is
    in. The in-memory non-vacuity assertion is the load-bearing one: without it
    this would pass just as well against records that never had a prior.
    """
    state, _inp, _boards = _run_c(GameConfig(record_prior_top1=True))
    carried = sum(
        1 for i in range(_BATCH)
        for r in state.samples_per_game[i] if r.prior_top1_index is not None
    )
    assert carried >= 3, "fixture carries no priors — the assertion below is vacuous"

    # The session's frozen GameConfig now says OFF, as it would after a restart
    # that flipped the kill switch while these games were in flight.
    object.__setattr__(state, "game", dataclasses.replace(
        state.game, record_prior_top1=False,
    ))
    assert state.game.record_prior_top1 is False
    off = _write_and_reload(state, tmp_path, "carried_off")

    assert int(np.asarray(off.get("has_prior_top1", np.zeros(1))).astype(bool).sum()) == 0
    assert int(
        np.asarray(off.get("has_prior_top1_prob", np.zeros(1))).astype(bool).sum()
    ) == 0
    # ...and the records themselves were not mutated -- finalize declines to
    # WRITE the value, it does not erase selfplay's in-memory state.
    still = sum(
        1 for i in range(_BATCH)
        for r in state.samples_per_game[i] if r.prior_top1_index is not None
    )
    assert still == carried


@requires_c
def test_written_index_is_compact_and_not_a_raw_4672_id(tmp_path) -> None:
    """The encoding translation at finalize is real.

    ``_NetRecord`` holds the internal full-4672 action id; a shard in the
    compact ``lc0_1858`` encoding must hold the COMPACT id. Skipping the
    translation leaves an id that is in-range-but-wrong for small values and
    out-of-range for large ones -- the silent half is what this catches.
    """
    state, inp, boards = _run_c(GameConfig(record_prior_top1=True))
    arrs = _write_and_reload(state, tmp_path, "enc")
    width = int(np.asarray(arrs["policy_target"]).shape[1])
    idx = np.asarray(arrs["prior_top1_index"]).astype(np.int64)
    has = np.asarray(arrs["has_prior_top1"]).astype(bool)
    assert ((idx[has] >= 0) & (idx[has] < width)).all()

    full_ids = [_expected_prior(inp["pol_logits"][i], boards[i])[0] for i in range(_BATCH)]
    assert {int(v) for v in idx[has]} <= {compact_policy_index(v) for v in full_ids}
    assert has.sum() >= 3
    # ...and every one of those full ids genuinely differs from its compact id,
    # so the assertion above cannot be satisfied by an identity mapping (i.e. by
    # storing the raw 4672 id and never translating).
    assert all(compact_policy_index(v) != v for v in full_ids)


# ---------------------------------------------------------------------------
# The config route: yaml schema -> TrialConfig -> reco -> GameConfig
# ---------------------------------------------------------------------------


def _reco_for(config: dict[str, Any]) -> dict[str, Any]:
    """The reco the server would publish for a flattened ``config``."""
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker

    return build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=1000, mcts_simulations=8,
    )


def _game_config_the_worker_builds(reco: dict[str, Any]) -> GameConfig:
    """The ``GameConfig`` the REAL worker hop builds out of ``reco``.

    ``WorkerSession._build_selfplay_configs`` is the kill switch's ONLY actuator:
    nothing downstream re-reads the reco, and since this delta that hop gates the
    shard bytes rather than only the capture. A test that certifies the reco and
    then constructs a ``GameConfig`` itself observes neither -- replacing the hop
    with a literal ``True`` left the whole suite green. So assert on what the
    CONSUMER received.

    The GameConfig is matched by config identity (``cfgs`` is keyed by the
    ``play_batch`` kwarg name), and the uniqueness assert keeps that key from
    silently becoming the wrong row if a second GameConfig is ever added.
    """
    import logging
    import threading
    from types import SimpleNamespace

    from chess_anti_engine.worker import WorkerSession

    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.prior_top1_capture")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, reco)
    game_keys = [k for k, v in cfgs.items() if isinstance(v, GameConfig)]
    assert game_keys == ["game"], f"GameConfig is no longer uniquely keyed: {game_keys}"
    game = cfgs["game"]
    assert isinstance(game, GameConfig)
    return game


def test_the_reco_to_gameconfig_hop_actually_reads_the_flag() -> None:
    """The kill switch's sole actuator, isolated from every hop above it.

    Driven at the reco level only (the yaml/TrialConfig/publisher stages are held
    at their defaults), and -- decisively -- at the NON-default value: an
    assertion pinned to ``True`` is invariant to the mutation that matters, which
    is replacing ``bool(reco.get("record_prior_top1", True))`` with a literal.
    """
    base = _reco_for({})

    for want in (True, False):
        reco = dict(base)
        reco["record_prior_top1"] = want
        game = _game_config_the_worker_builds(reco)
        assert game.record_prior_top1 is want, (
            f"the reco->GameConfig hop ignored record_prior_top1={want}; "
            f"the consumer received {game.record_prior_top1!r}"
        )

    # The OFF case is the one that binds: it must disagree with the dataclass
    # default, so no hop that ignores the reco entirely can satisfy it.
    off = dict(base)
    off["record_prior_top1"] = False
    assert (
        _game_config_the_worker_builds(off).record_prior_top1
        != GameConfig().record_prior_top1
    )

    # An old server's manifest omits the key; the worker must fall back to the
    # dataclass default rather than inventing one of its own.
    absent = {k: v for k, v in base.items() if k != "record_prior_top1"}
    assert (
        _game_config_the_worker_builds(absent).record_prior_top1
        == GameConfig().record_prior_top1
    )


def test_flag_survives_the_whole_config_route() -> None:
    """A knob is dead unless every hop carries it. Walks the real hops rather
    than asserting on any single one of them -- including the last one, the
    worker's reco -> GameConfig hop, which is where the value is consumed."""
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    for want in (True, False):
        flat = flatten_run_config_defaults(
            {"selfplay": {"record_prior_top1": want}},
        )
        assert flat["record_prior_top1"] is want, "yaml schema dropped the key"
        tc = TrialConfig.from_dict(flat)
        assert tc.record_prior_top1 is want, "TrialConfig.from_dict dropped it"
        reco = _reco_for(flat)
        assert bool(reco["record_prior_top1"]) is want, "reco dropped it"
        assert _game_config_the_worker_builds(reco).record_prior_top1 is want, (
            "the worker's reco -> GameConfig hop dropped it"
        )


def test_flag_defaults_on_everywhere_it_is_declared() -> None:
    """Default-ON is deliberate: the field only pays off if it accumulates
    passively from the deploy, with no yaml key (an unknown key is fatal at
    launch). A default that disagreed between hops would make the effective
    value depend on which hop happened to supply it."""
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    assert GameConfig().record_prior_top1 is True
    assert TrialConfig.from_dict(flatten_run_config_defaults({})).record_prior_top1 is True


def test_flag_is_restart_keyed_and_resume_exempt() -> None:
    """It is baked into the frozen GameConfig at session start, so it must be a
    restart key or a mid-run change would never reach a worker."""
    from chess_anti_engine.worker import WorkerSession

    assert "record_prior_top1" in WorkerSession._RECO_RESTART_KEYS
    assert "record_prior_top1" in WorkerSession._RESUME_COMPAT_EXEMPT_KEYS


def test_dataclasses_replace_keeps_the_flag() -> None:
    """GameConfig is rebuilt with dataclasses.replace in several worker paths."""
    g = dataclasses.replace(GameConfig(), record_prior_top1=False)
    assert g.record_prior_top1 is False
