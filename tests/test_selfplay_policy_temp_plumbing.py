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
  evaluator method the search calls, which pins BOTH the cost mechanism and the
  invariant that leaf priors are never left untempered.
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

from chess_anti_engine.moves import MODEL_POLICY_SIZE, POLICY_SIZE
from chess_anti_engine.mcts.gumbel import (
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

    supports_input_bf16_bits = False
    pads_batches_internally = True

    def __init__(self, *, legal_bf16: bool = True) -> None:
        self._legal_bf16 = bool(legal_bf16)
        self.dense_calls = 0
        self.legal_bf16_calls = 0

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
        pol = np.empty((n, _EVAL_POLICY), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i, key in enumerate(keys):
            seed = int(abs(float(key)) * 97.0) & 0x7FFFFFFF if np.isfinite(key) else 0
            rng = np.random.default_rng(seed)
            pol[i] = (rng.standard_normal(_EVAL_POLICY) * 2.5).astype(np.float32)
            wdl[i] = (rng.standard_normal(3) * 0.5).astype(np.float32)
        return pol, wdl

    def evaluate_encoded(self, x: np.ndarray, relations=None):
        assert relations is None
        self.dense_calls += 1
        return self._dense_logits(x)

    def evaluate_legal_bf16(self, x, legal_flat, legal_counts):
        self.legal_bf16_calls += 1
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
):
    """Run the C Gumbel search with the kwarg set production actually uses.

    Mirrors ``selfplay/network_turn.py::_run_mcts_group``: ``pre_pol_logits``,
    ``pre_wdl_logits``, a carried ``tree``, the three ``per_game_*`` lists,
    ``target_batch`` and ``vloss_weight``. ``cached_root=False`` drops only the
    two ``pre_*`` arrays, which is the UCI-style path where the search evaluates
    its own roots --- used by the exactly-once parity test below.

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
        tree=MCTSTree(), root_node_ids=None,
        per_game_simulations=[sims] * n,
        per_game_add_noise=[False] * n,
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
    gated on the same boolean.

    Asserted by watching which evaluator method the search calls, so it cannot
    be satisfied by a comment or by reading the source.
    """
    ev_flat = _StubEvaluator(legal_bf16=True)
    _ents, _p, _m, _e = _search_entropies(1.0, evaluator=ev_flat)
    assert ev_flat.legal_bf16_calls > 0, (
        "T=1.0 must keep the compact bf16 leaf transport --- if this fails the "
        "cost table in the ledger was measured against the wrong baseline"
    )

    ev_temp = _StubEvaluator(legal_bf16=True)
    _ents2, _p2, _m2, _e2 = _search_entropies(1.5, evaluator=ev_temp)
    assert ev_temp.legal_bf16_calls == 0, (
        "T=1.5 used the compact bf16 leaf transport, whose C softmax has no "
        "temperature hook: leaf priors would be UNTEMPERED while root priors "
        "were tempered"
    )
    assert ev_temp.dense_calls > 0


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


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_temperature_is_rejected_not_swallowed(bad: float) -> None:
    """``apply_policy_temp`` treats <= 0 as a no-op --- reject it at load.

    Otherwise ``gumbel_policy_temp: 0`` is accepted by the validator, published
    to every worker, and does nothing: a knob that cannot fail.
    """
    with pytest.raises(ValueError, match="gumbel_policy_temp"):
        TrialConfig.from_dict({"gumbel_policy_temp": bad})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_every_temperature_the_loader_rejects_is_inert_in_the_search(bad: float) -> None:
    """The predicate must agree with the loader on EVERY entry point.

    The yaml is validated, but ``scripts/arena_standard.py --cand-gumbel
    policy_temp=inf`` reaches ``dataclasses.replace`` without that validator.
    ``inf`` used to satisfy ``policy_temp_active`` and turn the prior uniform
    (``logits/inf`` is all zeros) --- an arena searching with its policy head
    effectively off, while the realized-shape log line reported tempering as
    working. ``policy_temp_active`` is THE definition of "tempering is on", so
    the same four values the loader refuses must be no-ops here.
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
