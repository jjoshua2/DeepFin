"""``gumbel_policy_temp`` must reach the SEARCH, not just a config object.

lc0's PolicyTemperature softens the policy prior before it seeds the tree. The
knob existed on ``GumbelConfig`` but was unreachable from configuration: absent
from ``PLAY_SEARCH_DEFAULTS`` and from every selfplay config dataclass, so
production ran the hard-coded 1.0 and no yaml edit could change it.

These tests are written against this project's signature defect --- a value that
is accepted and then silently ignored. So they OBSERVE the search output moving,
rather than asserting that a field was assigned:

* ``test_search_output_softens_...`` runs a real Gumbel search twice and reads
  ``gumbel_policy_diagnostics(...)["entropy"]`` --- literally the per-ply term
  that is summed into the published ``gumbel_policy_entropy_mean``.
* ``test_tempering_disables_the_compact_bf16_leaf_transport`` watches which
  evaluator method the search calls, and with what input dtype, which pins the
  cost mechanism on both of the transports the gate switches.
* ``test_the_dense_leaf_prior_is_tempered_exactly_once`` asserts on the array
  ``continue_gumbel_sims`` is HANDED. ⚑ The published in-effect proof
  (``gumbel_policy_entropy_mean``) is computed from the correctly-tempered ROOT,
  so it fires at full pre-registered strength even if leaf priors are left
  untempered --- it cannot tell "working" from "half-working". The dense leaf is
  new production code (before this PR ``policy_temp`` was unreachable, so
  ``_use_legal_bf16`` was always True and that branch never ran in selfplay) and
  the gate this PR adds forces it whenever T != 1. A leaf ``T^2`` and a wholly
  untempered leaf both survived this file plus five other suites with ZERO kills
  until this guard existed.
* ``test_null_control_...`` is the null control: it must pass under every
  mutation of the plumbing, so a mutation run that kills everything is
  recognisable as a broken harness rather than a good test suite.

⚑ Every search below runs at the PRODUCTION CALL SHAPE. ``_search_entropies``
passes ``pre_pol_logits``/``pre_wdl_logits``/``tree`` and the ``per_game_*``
lists exactly as ``network_turn._run_mcts_group`` does, because production
passes all of them on every ply and the cached-root branch
(``gumbel_c.py:610-614``) is reached ONLY when ``pre_pol_logits`` is not None.
An earlier revision of this file called the search without them: an outright
``NameError`` on that branch left all 11 tests green, and a silent ``T^2``
double-division of the root prior --- the exact hazard the comment there exists
to prevent --- survived this file plus the arena, gumbel-edge-case and
uci-parity suites with zero kills. A comment is not a test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.moves import (
    MODEL_POLICY_SIZE,
    POLICY_SIZE,
    policy_batch_to_full_if_needed,
)
from chess_anti_engine.mcts.gumbel import (
    POLICY_TEMP_MAX,
    POLICY_TEMP_MIN,
    GumbelConfig,
    apply_policy_temp,
    gumbel_policy_diagnostics,
    policy_temp_active,
)
from chess_anti_engine.selfplay.config import GameConfig, SearchConfig
from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
from chess_anti_engine.worker import WorkerSession
from tests.test_reco_coverage import _bare_session, _reco_from

# The broker/evaluator transport is over the FULL action space: the model's
# compact lc0_1858 head is expanded before the legal gather, and
# `evaluate_legal_bf16` indexes with full 4672 action ids.
_EVAL_POLICY = POLICY_SIZE


class _StubEvaluator:
    """Deterministic per-position logits over both transports the search can pick.

    Deterministic in the ROW CONTENT, not in call order, so the two arms see the
    same network for the same position even though they visit leaves in a
    different order once the priors change.
    """

    pads_batches_internally = True

    def __init__(
        self,
        *,
        legal_bf16: bool = True,
        input_bf16: bool = False,
        compact_root: bool = False,
        policy_width: int = _EVAL_POLICY,
    ) -> None:
        self._legal_bf16 = bool(legal_bf16)
        self.supports_input_bf16_bits = bool(input_bf16)
        self.supports_compact_root_policy = bool(compact_root)
        self._policy_width = int(policy_width)
        self.dense_calls = 0
        self.legal_bf16_calls = 0
        # Which evaluator method received each batch, in call order. The ROOT is
        # always evaluated first, so ``calls[0]`` names the ROOT transport --- the
        # only outside-visible signal of which root branch gumbel_c chose.
        self.calls: list[str] = []
        # ⚑ The dtype of ``x`` IS the input transport: gumbel_c allocates the
        # leaf encode buffer ``np.uint16 if _use_input_bf16 else np.float32``
        # (gumbel_c.py:1213). Recording it is the only way to observe the bf16
        # INPUT half of the gate from outside the search.
        self.leaf_input_dtypes: list[np.dtype[Any]] = []
        # Raw, PRE-temperature leaf logits, most recent batch first. The dense
        # leaf arm evaluates and then immediately hands the tempered result to
        # ``continue_gumbel_sims``, so "last returned" is the batch the tree is
        # about to receive.
        self.last_leaf_policy: np.ndarray | None = None

    @property
    def supports_legal_bf16(self) -> bool:
        return self._legal_bf16

    def _dense_logits(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ⚑ The dense leaf arm is handed `_enc_buf[:padded]`, and every row above
        # `n_leaves` is STALE, UNINITIALISED buffer content (`np.empty`, see
        # `_record_batch_dup`'s docstring). Those rows can hold NaN/inf, and the
        # search discards their outputs -- but a stub that RAISES on them turns
        # whatever was in that memory into a flaky test. A real evaluator just
        # forwards the garbage, so sanitise rather than crash. Finite rows keep
        # their exact previous key, so the content-determinism the two arms of
        # the parity test rely on is unchanged.
        arr = np.nan_to_num(
            np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0,
        )
        n = int(arr.shape[0])
        flat = arr.reshape(n, -1)
        keys = (flat * np.arange(1, flat.shape[1] + 1, dtype=np.float32)).sum(axis=1)
        width = self._policy_width
        pol = np.empty((n, width), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i, key in enumerate(keys):
            seed = int(abs(float(key)) * 97.0) & 0x7FFFFFFF if np.isfinite(key) else 0
            rng = np.random.default_rng(seed)
            pol[i] = (rng.standard_normal(width) * 2.5).astype(np.float32)
            wdl[i] = (rng.standard_normal(3) * 0.5).astype(np.float32)
        return pol, wdl

    def evaluate_encoded(self, x: np.ndarray, relations=None):
        assert relations is None
        self.dense_calls += 1
        self.calls.append("evaluate_encoded")
        self.leaf_input_dtypes.append(np.asarray(x).dtype)
        pol, wdl = self._dense_logits(x)
        self.last_leaf_policy = pol.copy()
        return pol, wdl

    def evaluate_legal_bf16(self, x, legal_flat, legal_counts):
        self.legal_bf16_calls += 1
        self.calls.append("evaluate_legal_bf16")
        self.leaf_input_dtypes.append(np.asarray(x).dtype)
        pol, wdl = self._dense_logits(x)
        counts = np.asarray(legal_counts, dtype=np.int64)
        flat = np.asarray(legal_flat, dtype=np.int64)
        rows = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        compact = pol[rows, flat].astype(np.float32, copy=False)
        import torch

        bits = torch.from_numpy(compact).to(torch.bfloat16).view(torch.uint16)
        return bits.numpy(), wdl


def _boards():
    import chess

    return [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board("r2q1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 b - - 0 11"),
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ]


def _root_trunk_logits(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic RAW root logits, shaped like the trunk's own heads.

    ``network_turn`` hands the search ``pol_logits[group, :]`` --- the compact
    ``policy_own`` head, ``MODEL_POLICY_SIZE`` wide and NOT yet expanded to the
    4672 action space, and NOT yet tempered. The expansion and the single
    division by T both happen inside ``_policy_logits_to_full``. Feeding compact
    logits here is what makes that branch (and its ``T^2`` hazard) reachable.
    """
    rng = np.random.default_rng(4242)
    pol = (rng.standard_normal((n, MODEL_POLICY_SIZE)) * 2.5).astype(np.float32)
    wdl = (rng.standard_normal((n, 3)) * 0.5).astype(np.float32)
    return pol, wdl


def _run_production_shape(
    policy_temp: float,
    *,
    evaluator: Any = None,
    sims: int = 64,
    pre_pol: np.ndarray | None = None,
    pre_wdl: np.ndarray | None = None,
    cached_root: bool = True,
    tree: Any = None,
    add_noise: bool = True,
):
    """Run the C Gumbel search with the kwarg set production actually uses.

    Mirrors ``selfplay/network_turn.py::_run_mcts_group``: ``pre_pol_logits``,
    ``pre_wdl_logits``, a carried ``tree``, the three ``per_game_*`` lists,
    ``target_batch`` and ``vloss_weight``. ``cached_root=False`` drops only the
    two ``pre_*`` arrays, which is the UCI-style path where the search evaluates
    its own roots --- used by the exactly-once parity test below.

    ``per_game_add_noise`` is ``True``, not ``False``: production computes
    ``noise_py = [scale > 0.0 for scale in scale_py]``
    (``network_turn.py``, ``_run_network_turn_group``) and every full-sim ply
    carries a positive Gumbel scale, so root noise is ON for every ply this
    experiment is measured over. Passing ``[False]`` also silently made the
    accompanying ``per_game_gumbel_scale=[1.0]`` inert, i.e. two of the three
    ``per_game_*`` lists the docstring claims to mirror were not being
    exercised at all.

    ``tree`` lets a caller substitute a recorder for the C ``MCTSTree`` and
    observe what the LEAF transport is handed; ``None`` means a fresh tree.

    NOT mirrored, so the docstring does not imply otherwise: ``root_node_ids``
    is ``None`` here where production passes ``[state.root_ids[...]]``, so the
    tree-REUSE branch (``gumbel_c.py:797-812``) is never taken. Every ply here
    is a cold root. That matters for reading the leaf test above --- in
    production a reused root is not re-expanded, so from ply 2 onward the root
    prior IS the leaf transport's output and a leaf break compounds.

    Uses the PRODUCTION mapping (`build_selfplay_gumbel_config`) so a break in
    the SearchConfig -> GumbelConfig wiring shows up here, not only in a
    dedicated wiring test.
    """
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts._mcts_tree import MCTSTree
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    boards = _boards()
    n = len(boards)
    cboards = [CBoard.from_board(b) for b in boards]
    search = SearchConfig(
        gumbel_topk=8, gumbel_policy_temp=float(policy_temp), gumbel_c_scale=0.025,
    )
    cfg = build_selfplay_gumbel_config(
        search=search, game=GameConfig(), simulations=sims,
    )
    ev = evaluator if evaluator is not None else _StubEvaluator()
    if cached_root and (pre_pol is None or pre_wdl is None):
        pre_pol, pre_wdl = _root_trunk_logits(n)
    probs, actions, _values, masks, _tree, _rids, diags = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(20260809),
        cfg=cfg, evaluator=ev, cboards=cboards,
        pre_pol_logits=pre_pol if cached_root else None,
        pre_wdl_logits=pre_wdl if cached_root else None,
        tree=MCTSTree() if tree is None else tree, root_node_ids=None,
        per_game_simulations=[sims] * n,
        per_game_add_noise=[bool(add_noise)] * n,
        per_game_gumbel_scale=[1.0] * n,
        allow_terminal_root_shortcuts=True, return_diagnostics=True,
        vloss_weight=1, target_batch=0,
    )
    return boards, probs, actions, masks, diags, ev


def _search_entropies(policy_temp: float, *, evaluator=None, sims: int = 64):
    """Mean improved-policy entropy over a real Gumbel search at ``policy_temp``.

    Production call shape --- see ``_run_production_shape``.
    """
    boards, probs, actions, masks, diags, ev = _run_production_shape(
        policy_temp, evaluator=evaluator, sims=sims,
    )
    ents = []
    for i, board in enumerate(boards):
        legal = np.flatnonzero(masks[i])
        diag = diags[i] or gumbel_policy_diagnostics(
            probs=probs[i], action=int(actions[i]), legal=legal, candidates=None,
        )
        ents.append(float(diag["entropy"]))
        assert len(list(board.legal_moves)) > 0
    return np.array(ents, dtype=np.float64), probs, masks, ev


# ---------------------------------------------------------------------------
# Default is a no-op
# ---------------------------------------------------------------------------


def test_default_is_an_exact_no_op() -> None:
    """Merging this must not move production until someone opts in."""
    assert SearchConfig().gumbel_policy_temp == 1.0
    assert TrialConfig().gumbel_policy_temp == 1.0
    cfg = build_selfplay_gumbel_config(
        search=SearchConfig(), game=GameConfig(), simulations=256,
    )
    assert cfg.policy_temp == 1.0
    assert not policy_temp_active(cfg.policy_temp)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(apply_policy_temp(logits, cfg=cfg), logits)


# ---------------------------------------------------------------------------
# The observation: a real search softens
# ---------------------------------------------------------------------------


def test_search_output_softens_when_the_configured_temperature_rises() -> None:
    """The deciding test. Not "was the field set" --- "did the search change".

    ``entropy`` here is the same per-ply number ``selfplay/finalize.py`` sums
    into ``gumbel_policy_entropy_sum``, which the trainer publishes as
    ``gumbel_policy_entropy_mean``. So this asserts movement in the exact
    quantity the live in-effect proof reads.
    """
    ent_flat, _p1, _m1, _e1 = _search_entropies(1.0)
    ent_soft, _p2, _m2, _e2 = _search_entropies(2.5)

    assert float(ent_soft.mean()) > float(ent_flat.mean()) + 0.05, (
        f"policy_temp did not reach the search: entropy {ent_flat.mean():.4f} "
        f"-> {ent_soft.mean():.4f} nats"
    )


def test_sharpening_moves_the_other_way() -> None:
    """T<1 must SHARPEN. A gate stuck 'on' would soften in both directions.

    The 0.05 margin matters as much as the direction: with a bare ``<`` this
    test PASSES against an ``apply_policy_temp`` neutered to a no-op, because
    float noise alone satisfies a strict inequality between two means that are
    supposed to be identical. Same margin as the softening twin above.
    """
    ent_flat, _p1, _m1, _e1 = _search_entropies(1.0)
    ent_sharp, _p2, _m2, _e2 = _search_entropies(0.4)

    assert float(ent_sharp.mean()) < float(ent_flat.mean()) - 0.05, (
        f"policy_temp=0.4 did not sharpen: {ent_flat.mean():.4f} -> "
        f"{ent_sharp.mean():.4f} nats"
    )


# ---------------------------------------------------------------------------
# The cost mechanism, pinned as an invariant rather than a comment
# ---------------------------------------------------------------------------


def test_tempering_disables_the_compact_bf16_leaf_transport() -> None:
    """T != 1 must fall back to the dense leaf transport.

    Two things at once. (1) It is the reason the knob is not free: the compact
    legal-bf16 leaf path is a C softmax with no temperature hook, so keeping it
    would leave LEAF priors untempered while root priors were tempered --- a
    half-applied intervention that would still look 'in effect' at the root.
    (2) It is the whole measured cost: dense float32 POLICY_SIZE leaf transport
    instead of compact bf16, plus the loss of the bf16 INPUT transport that is
    gated on the same boolean (``_use_input_bf16 = _has_input_bf16 and
    _use_legal_bf16``, gumbel_c.py:1198).

    BOTH halves are observed, not just the first. The input half is read off the
    DTYPE the evaluator is handed: gumbel_c allocates the leaf encode buffer as
    ``np.uint16 if _use_input_bf16 else np.float32`` (gumbel_c.py:1213), so
    uint16 in means the bf16 input transport is live and float32 means it was
    lost. An earlier revision of this docstring claimed to pin the input half
    while the stub reported ``supports_input_bf16_bits = False``, which made
    ``_use_input_bf16`` False in BOTH arms --- the claim was unobservable, so
    the stub now opts in.

    Asserted by watching which evaluator method the search calls and with what,
    so it cannot be satisfied by a comment or by reading the source.
    """
    ev_flat = _StubEvaluator(legal_bf16=True, input_bf16=True)
    _ents, _p, _m, _e = _search_entropies(1.0, evaluator=ev_flat)
    assert ev_flat.legal_bf16_calls > 0, (
        "T=1.0 must keep the compact bf16 leaf transport --- if this fails the "
        "cost table in the ledger was measured against the wrong baseline"
    )
    assert set(ev_flat.leaf_input_dtypes) == {np.dtype(np.uint16)}, (
        "T=1.0 must keep the bf16 INPUT transport too; the leaf encode buffer "
        f"came through as {sorted({str(d) for d in ev_flat.leaf_input_dtypes})}"
    )

    ev_temp = _StubEvaluator(legal_bf16=True, input_bf16=True)
    _ents2, _p2, _m2, _e2 = _search_entropies(1.5, evaluator=ev_temp)
    assert ev_temp.legal_bf16_calls == 0, (
        "T=1.5 used the compact bf16 leaf transport, whose C softmax has no "
        "temperature hook: leaf priors would be UNTEMPERED while root priors "
        "were tempered"
    )
    assert ev_temp.dense_calls > 0
    assert set(ev_temp.leaf_input_dtypes) == {np.dtype(np.float32)}, (
        "T=1.5 kept the bf16 INPUT transport: it is gated on the same boolean "
        "as the leaf transport, so the ledger's cost line is measuring one "
        f"fallback where production takes two. Saw "
        f"{sorted({str(d) for d in ev_temp.leaf_input_dtypes})}"
    )


def test_the_root_noise_the_harness_mirrors_is_not_an_inert_flag() -> None:
    """``per_game_add_noise=True`` is production's value AND a live one.

    ``_run_mcts_group`` computes ``noise_py = [scale > 0.0 for scale in
    scale_py]``, which is True on every full-sim ply, so root Gumbel noise is on
    for every ply this experiment is measured over. An earlier revision of this
    harness passed ``[False]``, which also made the ``per_game_gumbel_scale``
    list it passes alongside inert --- two of the three ``per_game_*`` lists the
    docstring claims to mirror were doing nothing.

    Setting it right is only half the fix: a flag that production sets and the
    search ignores is the same defect one layer down. So compare the two arms at
    the SAME seed and require them to differ. If they do not, the harness is not
    running the noisy regime it says it runs, whatever the argument says.
    """
    _b_on, probs_on, acts_on, _m1, _d1, _e1 = _run_production_shape(1.5, add_noise=True)
    _b_off, probs_off, acts_off, _m2, _d2, _e2 = _run_production_shape(
        1.5, add_noise=False,
    )
    differs = list(acts_on) != list(acts_off) or any(
        float(np.max(np.abs(np.asarray(probs_on[i], dtype=np.float64)
                            - np.asarray(probs_off[i], dtype=np.float64)))) > 0.0
        for i in range(len(probs_on))
    )
    assert differs, (
        "per_game_add_noise made no difference to the search: the harness is "
        "claiming production's noisy root regime while running the deterministic "
        "one, and every entropy number above is measured off the wrong regime"
    )


# ---------------------------------------------------------------------------
# The DENSE LEAF prior --- the transport this experiment switches on
# ---------------------------------------------------------------------------


class _LeafTransportRecorder:
    """Proxy for the C ``MCTSTree`` that records what the LEAF transport is HANDED.

    ⚑ Why the observation has to be here and nowhere else. The pre-registered
    in-effect proof for this experiment is ``gumbel_policy_entropy_mean``, which
    is built from the ROOT improved policy. Leave the leaf priors untempered and
    that metric still moves its full pre-registered amount --- the proof cannot
    tell "working" from "half-working", which is exactly the half-applied
    intervention ``test_tempering_disables_the_compact_bf16_leaf_transport``
    names as its own motivation. So the leaf needs its own observation.

    Intercepting ``_policy_logits_to_full`` does NOT substitute: a regression
    that tempers again *outside* that function is invisible to a spy inside it.
    What the tree receives is the last value before the priors enter the search,
    so that is what gets asserted on.
    """

    def __init__(self, tree: Any, evaluator: _StubEvaluator, temp: float) -> None:
        self._tree = tree
        self._evaluator = evaluator
        self._temp = float(temp)
        self.dense_leaf_batches = 0
        self.bf16_leaf_batches = 0
        self.residuals: list[float] = []

    def __getattr__(self, name: str) -> Any:
  # Reached only for names the proxy does not define, so the hasattr()
  # probes gumbel_c uses to choose a leaf transport still see the real
  # tree's API and the transport gate is not perturbed by the recorder.
        return getattr(object.__getattribute__(self, "_tree"), name)

    def continue_gumbel_sims(self, pol: np.ndarray, wdl: np.ndarray) -> Any:
        raw = self._evaluator.last_leaf_policy
        assert raw is not None, "the dense leaf arm ran without an evaluator call"
        got = np.asarray(pol, dtype=np.float32)
        rows = int(got.shape[0])
        want = policy_batch_to_full_if_needed(
            np.asarray(raw[:rows], dtype=np.float32) / np.float32(self._temp),
            fill_value=-1e9,
        )
        self.dense_leaf_batches += 1
        self.residuals.append(float(np.max(np.abs(got.astype(np.float64) - want))))
        return self._tree.continue_gumbel_sims(pol, wdl)

    def continue_gumbel_sims_legal_bf16(self, pol: np.ndarray, wdl: np.ndarray) -> Any:
        self.bf16_leaf_batches += 1
        return self._tree.continue_gumbel_sims_legal_bf16(pol, wdl)


@pytest.mark.parametrize(
    "width", [POLICY_SIZE, MODEL_POLICY_SIZE], ids=["full_leaf", "compact_leaf"],
)
@pytest.mark.parametrize("temp", [1.0, 1.5, 0.4])
def test_the_dense_leaf_prior_is_tempered_exactly_once(temp: float, width: int) -> None:
    """``continue_gumbel_sims`` must receive ``to_full(raw_leaf_logits / T)``.

    The deciding leaf test. ``gumbel_c.py:1302`` is new production code: before
    this PR ``policy_temp`` was unreachable, so ``_use_legal_bf16`` was always
    True and the dense leaf branch never ran in selfplay. The gate this PR adds
    FORCES that branch whenever T != 1, and it compounds --- a reused root
    (``gumbel_c.py:797-812``) is not re-expanded, so from ply 2 onward the root
    prior production searches from IS the leaf transport's output.

    T=1.0 is the control, and it is a real one rather than a skipped case: the
    ``legal_bf16=False`` evaluator forces the dense branch at every temperature,
    so this same assertion runs at T=1.0, where both a leaf ``T^2`` and a wholly
    untempered leaf are arithmetic no-ops and must stay green.

    Both leaf widths are covered because ``_policy_logits_to_full`` divides
    BEFORE it scatters: production's compact ``lc0_1858`` head exercises the
    composition, a full-4672 evaluator exercises the division alone.
    """
    ev = _StubEvaluator(legal_bf16=False, policy_width=width)
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    rec = _LeafTransportRecorder(MCTSTree(), ev, temp)
    _run_production_shape(temp, evaluator=ev, tree=rec)

    assert rec.dense_leaf_batches > 0, (
        "no dense leaf batch reached the tree, so this test asserted nothing --- "
        "the leaf transport moved and the guard needs re-pointing"
    )
    assert rec.bf16_leaf_batches == 0
    assert max(rec.residuals) == 0.0, (
        f"T={temp}, leaf width {width}: the prior the LEAF transport handed the "
        f"tree is not to_full(raw/T) --- max|leaf_prior - to_full(raw/T)| = "
        f"{max(rec.residuals)} over {rec.dense_leaf_batches} batches. Either the "
        "leaf is tempered a second time (T^2) or not at all; both leave the root "
        "correctly tempered, so gumbel_policy_entropy_mean would still fire its "
        "full pre-registered effect on a half-applied intervention."
    )


@pytest.mark.parametrize("temp", [1.0, 1.5])
def test_the_production_gate_sends_tempered_priors_to_the_dense_leaf(temp: float) -> None:
    """The same invariant on the branch the PRODUCTION gate actually selects.

    The test above forces the dense arm by taking ``evaluate_legal_bf16`` away.
    This one leaves the evaluator fully capable, exactly as the broker is, and
    observes the routing the ``policy_temp_active`` gate performs: at T=1.0 the
    tree must see ONLY compact bf16 leaf batches, at T=1.5 ONLY dense ones ---
    and those dense ones must carry the tempered prior.
    """
    ev = _StubEvaluator(legal_bf16=True)
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    rec = _LeafTransportRecorder(MCTSTree(), ev, temp)
    _run_production_shape(temp, evaluator=ev, tree=rec)

    if temp == 1.0:
        assert rec.bf16_leaf_batches > 0, (
            "T=1.0 must keep the compact bf16 leaf transport; the ledger's cost "
            "table was measured against that baseline"
        )
        assert rec.dense_leaf_batches == 0, (
            "T=1.0 fell back to the dense leaf transport, so the ledger's cost "
            "baseline is the expensive path, not the cheap one"
        )
        return

    assert rec.dense_leaf_batches > 0, (
        "T != 1 must route every leaf batch through the tempering-aware dense "
        "transport --- the C bf16 softmax has no temperature hook"
    )
    assert rec.bf16_leaf_batches == 0, (
        "T != 1 still used the compact bf16 leaf transport for some batches: "
        "those leaf priors are UNTEMPERED while the root's are tempered"
    )
    assert max(rec.residuals) == 0.0, (
        f"T={temp}: dense leaf prior != to_full(raw/T), max residual "
        f"{max(rec.residuals)} over {rec.dense_leaf_batches} batches"
    )


@pytest.mark.parametrize("temp", [1.0, 1.5])
def test_tempering_disables_the_compact_root_transport(temp: float) -> None:
    """The OTHER gate this PR adds, and the only one that skips tempering wholesale.

    ``_use_compact_root`` (gumbel_c.py:593-602) gained a
    ``not policy_temp_active(...)`` clause in this PR. It has to be there: the
    compact-root arm assigns ``_root_compact_logits`` and the tempering line
    below is guarded by ``if _root_compact_logits is None`` (gumbel_c.py:689),
    so with the clause removed the ROOT prior is not tempered at all --- not
    doubled, not halved, simply skipped, while every leaf is tempered normally.

    Observable from outside because the two root branches call DIFFERENT
    evaluator methods: compact-root evaluates the root through
    ``evaluate_legal_bf16``, the dense root through ``evaluate_encoded``. The
    root is always the first batch, so ``calls[0]`` names the branch taken.

    ⚑ Reachability, so a later reader does not over- or under-read this: the
    gate needs ``pre_pol_logits is None`` (the search evaluates its own roots)
    AND a broker-class evaluator advertising ``supports_compact_root_policy``,
    and no production caller combines them today --- selfplay always passes
    cached root logits. It is guarded anyway because "no caller does this today"
    is the same class of argument as ``gate_games: 0``: a config fact, not an
    invariant, and this line is one ``pre_pol_logits`` refactor away from live.
    """
    ev = _StubEvaluator(legal_bf16=True, input_bf16=True, compact_root=True)
    _run_production_shape(temp, evaluator=ev, cached_root=False)

    assert ev.calls, "the search never called the evaluator"
    if temp == 1.0:
        assert ev.calls[0] == "evaluate_legal_bf16", (
            "T=1.0 must keep the compact root transport; if this fails the gate "
            "is off for a reason unrelated to policy_temp and this test is "
            "no longer observing what it claims"
        )
        return
    assert ev.calls[0] == "evaluate_encoded", (
        "T != 1 used the compact ROOT transport, which bypasses "
        "_policy_logits_to_full entirely (gumbel_c.py:689): the root prior "
        "would be UNTEMPERED while every leaf prior was tempered"
    )


# ---------------------------------------------------------------------------
# The T^2 hazard on the cached root prior --- the branch production always takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temp", [1.0, 1.5])
def test_caching_the_root_logits_does_not_change_the_search(temp: float) -> None:
    """Black-box exactly-once: the cached-root path must equal the eval path.

    ``gumbel_c.py:610-614`` assigns ``pre_pol_logits`` RAW because the
    unconditional ``_policy_logits_to_full`` below it applies the temperature
    for every path; tempering here as well would divide the root prior by T
    TWICE. Today that hazard is documented only in a comment, and a comment is
    not a check: the double-division is silent, plausible, and doubles the
    intervention while still reading as "in effect".

    So run the same search twice at the same T --- once letting it evaluate its
    own roots, once handing it the very logits that evaluation produced --- and
    require the improved policy to be IDENTICAL. Under a ``T^2`` regression the
    cached arm divides twice and the eval arm once, so the two disagree. This
    asserts no implementation detail: it is the invariant that makes cached root
    logits legitimate at all (UCI passes them, selfplay passes them).

    T=1.5 is the value going live, so the parametrisation is not decorative.
    """
    class _Recording(_StubEvaluator):
        first_root: tuple[np.ndarray, np.ndarray] | None = None

        def evaluate_encoded(self, x: np.ndarray, relations=None):
            pol, wdl = super().evaluate_encoded(x, relations=relations)
            if self.first_root is None:
                self.first_root = (pol.copy(), wdl.copy())
            return pol, wdl

    ev_uncached = _Recording()
    _b, probs_uncached, acts_uncached, _m, _d, _e = _run_production_shape(
        temp, evaluator=ev_uncached, cached_root=False,
    )
    root = ev_uncached.first_root
    assert root is not None, "the uncached arm never evaluated a root batch"
    assert root[0].shape[0] == len(_b), "captured a leaf batch, not the root batch"

    _b2, probs_cached, acts_cached, _m2, _d2, _e2 = _run_production_shape(
        temp, evaluator=_Recording(), pre_pol=root[0], pre_wdl=root[1],
    )

    assert list(acts_cached) == list(acts_uncached), (
        f"T={temp}: handing the search its own root logits changed the move it "
        "picked -- the cached-root branch does not apply policy_temp the same "
        "number of times as the evaluated-root branch"
    )
    for i in range(len(_b)):
        np.testing.assert_allclose(
            np.asarray(probs_cached[i], dtype=np.float64),
            np.asarray(probs_uncached[i], dtype=np.float64),
            atol=0.0, rtol=0.0,
            err_msg=(
                f"T={temp}, board {i}: cached-root and evaluated-root searches "
                "disagree; a T^2 double-division of the root prior looks exactly "
                "like this"
            ),
        )


def test_the_cached_root_prior_is_divided_by_t_exactly_once(monkeypatch: Any) -> None:
    """The same claim as a numeric identity, localized to the one line.

    The parity test above says the two paths agree; this says WHICH value they
    agree on, so a regression that tempered both paths twice would still be
    caught. Intercepts ``_policy_logits_to_full`` --- the single place the
    division lives --- and checks (a) the root call receives the caller's RAW
    logits, (b) its output is ``to_full(raw)/T``, (c) exactly one call carries
    the root batch shape.
    """
    from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
    from chess_anti_engine.moves import policy_batch_to_full_if_needed

    temp = 1.5
    pre_pol, pre_wdl = _root_trunk_logits(len(_boards()))
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    original = gumbel_c_mod._policy_logits_to_full

    def _spy(pol_logits, *, cfg):
        captured = np.array(pol_logits, dtype=np.float32, copy=True)
        out = original(pol_logits, cfg=cfg)
        seen.append((captured, np.array(out, copy=True)))
        return out

    monkeypatch.setattr(gumbel_c_mod, "_policy_logits_to_full", _spy)
    _run_production_shape(temp, pre_pol=pre_pol, pre_wdl=pre_wdl)

    assert seen, "_policy_logits_to_full was never called -- the branch moved"
    root_shape = (pre_pol.shape[0], MODEL_POLICY_SIZE)
    root_calls = [(i, o) for i, o in seen if i.shape == root_shape]
    assert len(root_calls) == 1, (
        f"expected exactly one root-shaped {root_shape} call, saw {len(root_calls)}"
    )
    root_in, root_out = root_calls[0]
    assert seen[0][0].shape == root_shape, "the root call is no longer the first"

    assert float(np.max(np.abs(root_in - pre_pol))) == 0.0, (
        "the root prior was already tempered before _policy_logits_to_full: "
        f"max|in - raw| = {float(np.max(np.abs(root_in - pre_pol)))}, "
        f"max|in - raw/T| = {float(np.max(np.abs(root_in - pre_pol / temp)))}. "
        "That is the T^2 double-division gumbel_c.py:611-613 warns about."
    )
    want = policy_batch_to_full_if_needed(pre_pol / temp, fill_value=-1e9)
    finite = want > -1e8
    np.testing.assert_allclose(root_out[finite], want[finite], rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# yaml -> trainer -> manifest -> worker -> SearchConfig
# ---------------------------------------------------------------------------


def test_the_key_survives_every_hop_to_the_worker() -> None:
    """The hop that historically drops values is trainer -> manifest -> worker."""
    assert "gumbel_policy_temp" in SELFPLAY_CONFIG_KEYS, (
        "the live-yaml validator is all-or-nothing: an unlisted key rejects the "
        "WHOLE reload"
    )

    tc = TrialConfig.from_dict({"gumbel_policy_temp": 1.5})
    assert tc.gumbel_policy_temp == 1.5

    reco = _reco_from({"gumbel_policy_temp": 1.5, "mcts_simulations": 256})
    assert reco["gumbel_policy_temp"] == 1.5, (
        "not published in the manifest: the worker would run the 1.0 default "
        "while the yaml said 1.5, and the ledger would carry a verdict for an "
        "experiment that never ran"
    )

    session = _bare_session()
    cfgs, _sf = WorkerSession._build_selfplay_configs(session, reco)
    assert cfgs["search"].gumbel_policy_temp == 1.5

    cfg = build_selfplay_gumbel_config(
        search=cfgs["search"], game=GameConfig(), simulations=256,
    )
    assert cfg.policy_temp == 1.5


def test_a_change_restarts_the_worker_session() -> None:
    """``_build_selfplay_configs`` runs once per session and freezes SearchConfig.

    A live-applied change would be accepted by the manifest and never reach a
    search --- silently ignored, with a plausible-looking config to prove it was
    'set'.
    """
    assert "gumbel_policy_temp" in WorkerSession._RECO_RESTART_KEYS
    assert "gumbel_policy_temp" not in WorkerSession._RECO_LIVE_KEYS
    assert "gumbel_policy_temp" in WorkerSession._RESUME_COMPAT_EXEMPT_KEYS


_REJECTED_TEMPS = [
    0.0,
    -1.0,
    float("nan"),
    float("inf"),
  # ⚑ Finite, positive, and still degenerate. 1e300 divides every float32
  # logit to exactly 0.0 -- a uniform prior, i.e. search with the policy head
  # switched off -- and 1e-300 divides them to +/-inf. `math.isfinite` closes
  # the `inf` LITERAL, not the pathology, so the band closes the class.
    1e300,
    1e-300,
  # ⚑ LITERALS, deliberately not written relative to POLICY_TEMP_MIN/MAX.
  # Everything below that was relative -- these two entries, and the accepted
  # set in the twin test -- simply FOLLOWS the band, so widening the band to
  # [1e-30, 1e30] re-opened the exact pathology the band was added to close
  # with nothing failing. A test that derives its expectation from the thing
  # under test cannot fail. 100.0 and 0.001 pin the band's MAGNITUDE: moving a
  # bound past either is now a deliberate act that has to edit this list.
    100.0,
    0.001,
    POLICY_TEMP_MAX * 1.0001,
    POLICY_TEMP_MIN * 0.9999,
]


@pytest.mark.parametrize("bad", _REJECTED_TEMPS)
def test_a_degenerate_temperature_is_rejected_not_swallowed(bad: float) -> None:
    """``apply_policy_temp`` treats out-of-band values as no-ops --- reject at load.

    Otherwise ``gumbel_policy_temp: 0`` is accepted by the validator, published
    to every worker, and does nothing: a knob that cannot fail. And
    ``gumbel_policy_temp: 1e300`` is worse than nothing -- accepted, published,
    ACTIVE, and it deletes the policy prior.
    """
    with pytest.raises(ValueError, match="gumbel_policy_temp"):
        TrialConfig.from_dict({"gumbel_policy_temp": bad})


@pytest.mark.parametrize("good", [POLICY_TEMP_MIN, 0.4, 1.0, 1.5, 2.5, POLICY_TEMP_MAX])
def test_the_band_admits_every_temperature_anyone_actually_uses(good: float) -> None:
    """The band must not be so tight it rejects the experiment it is guarding.

    A bound is only defensible if its accepted set is checked too; otherwise
    "reject absurd values" quietly becomes "reject the value going live".
    Endpoints are inclusive, and 1.5 (production) / 1.2 (the decided floor) /
    0.4 and 2.5 (the values these tests search at) all sit inside.
    """
    assert TrialConfig.from_dict({"gumbel_policy_temp": good}).gumbel_policy_temp == good
    assert policy_temp_active(good) == (good != 1.0)


def test_the_band_edges_still_leave_a_prior_worth_having() -> None:
    """The band's PURPOSE, asserted on behaviour instead of on its own numbers.

    Both parametrised tests above take their edge cases from
    ``POLICY_TEMP_MIN``/``POLICY_TEMP_MAX``, so widening the band moves them
    too and neither can ever fail --- a test that derives its expectations from
    the thing under test is decoration. The literals now in ``_REJECTED_TEMPS``
    fix that at two fixed points; this fixes it at the concept.

    The band exists so that a value the loader ACCEPTS still leaves a usable
    prior. That is checkable directly: temper a realistic 16-nat logit spread at
    each edge and require the prior to survive it.

    * At the soft edge the ordering must still carry real mass --- top:bottom at
      least 2:1. A 16-nat spread gives 2.23 at T=20 and 1.17 at T=100, so this
      binds at T ~= 23 and no widening past that can pass.
    * At the sharp edge the tempered logits must stay finite. This is the
      ``1e-300 -> +/-inf`` failure, and float32 overflows around T ~= 1e-38.
    """
    logits = np.array([[8.0, 4.0, 0.0, -4.0, -8.0]], dtype=np.float32)

    soft = apply_policy_temp(
        logits, cfg=dataclasses.replace(GumbelConfig(), policy_temp=POLICY_TEMP_MAX),
    )
    e = np.exp(soft.astype(np.float64) - soft.max())
    p = e / e.sum()
    ratio = float(p.max() / p.min())
    assert ratio >= 2.0, (
        f"at POLICY_TEMP_MAX={POLICY_TEMP_MAX} the prior is effectively uniform "
        f"(top:bottom {ratio:.3f} on a 16-nat spread): the loader would accept a "
        f"value that searches with the policy head switched off, which is the "
        f"pathology the band was added to close"
    )

    sharp = apply_policy_temp(
        logits, cfg=dataclasses.replace(GumbelConfig(), policy_temp=POLICY_TEMP_MIN),
    )
    assert np.all(np.isfinite(sharp)), (
        f"at POLICY_TEMP_MIN={POLICY_TEMP_MIN} tempering overflows float32 to "
        f"+/-inf: {sharp}"
    )
    assert float(sharp.max() - sharp.min()) > 0.0


@pytest.mark.parametrize("bad", _REJECTED_TEMPS)
def test_every_temperature_the_loader_rejects_is_inert_in_the_search(bad: float) -> None:
    """The predicate must agree with the loader on EVERY entry point.

    The yaml is validated, but ``scripts/arena_standard.py --cand-gumbel
    policy_temp=1e300`` reaches ``dataclasses.replace`` without that validator.
    ``inf`` used to satisfy ``policy_temp_active`` and turn the prior uniform
    (``logits/inf`` is all zeros) --- an arena searching with its policy head
    effectively off, while the realized-shape log line reported tempering as
    working. ``policy_temp_active`` is THE definition of "tempering is on", so
    every value the loader refuses must be a no-op here: same instrument, same
    rejection set, no entry point left holed.
    """
    assert not policy_temp_active(bad)
    cfg = dataclasses.replace(GumbelConfig(), policy_temp=bad)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(apply_policy_temp(logits, cfg=cfg), logits)


# ---------------------------------------------------------------------------
# NULL CONTROL --- must fail under NO mutation of this plumbing
# ---------------------------------------------------------------------------


def test_null_control_search_returns_a_normalised_legal_policy() -> None:
    """Deliberately insensitive to policy_temp.

    If a mutation run kills this too, the mutation broke the search rather than
    the temperature wiring, and the other kills prove nothing.
    """
    for temp in (1.0, 1.5):
        _ents, probs, masks, _ev = _search_entropies(temp)
        for i in range(len(probs)):
            p = np.asarray(probs[i], dtype=np.float64)
            assert p.shape == masks[i].shape
            assert float(p.sum()) == pytest.approx(1.0, abs=1e-5)
            assert float(p[~masks[i]].sum()) == 0.0
