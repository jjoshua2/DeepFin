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
"""

from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.mcts.gumbel import (
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
        arr = np.asarray(x, dtype=np.float32)
        n = int(arr.shape[0])
        flat = arr.reshape(n, -1)
        keys = (flat * np.arange(1, flat.shape[1] + 1, dtype=np.float32)).sum(axis=1)
        pol = np.empty((n, _EVAL_POLICY), dtype=np.float32)
        wdl = np.empty((n, 3), dtype=np.float32)
        for i, key in enumerate(keys):
            rng = np.random.default_rng(int(abs(key) * 97.0) & 0x7FFFFFFF)
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


def _search_entropies(policy_temp: float, *, evaluator=None, sims: int = 64):
    """Mean improved-policy entropy over a real Gumbel search at ``policy_temp``.

    Uses the PRODUCTION mapping (`build_selfplay_gumbel_config`) so a break in
    the SearchConfig -> GumbelConfig wiring shows up here, not only in a
    dedicated wiring test.
    """
    import chess

    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

    boards = [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board("r2q1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP2BPPP/R2Q1RK1 b - - 0 11"),
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ]
    cboards = [CBoard.from_board(b) for b in boards]
    search = SearchConfig(
        gumbel_topk=8, gumbel_policy_temp=float(policy_temp), gumbel_c_scale=0.025,
    )
    cfg = build_selfplay_gumbel_config(
        search=search, game=GameConfig(), simulations=sims,
    )
    ev = evaluator if evaluator is not None else _StubEvaluator()
    probs, actions, _values, _masks, _tree, _rids, diags = run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(20260809),
        cfg=cfg, evaluator=ev, cboards=cboards,
        allow_terminal_root_shortcuts=True, return_diagnostics=True,
        vloss_weight=1, target_batch=0,
    )
    ents = []
    for i, board in enumerate(boards):
        legal = np.flatnonzero(_masks[i])
        diag = diags[i] or gumbel_policy_diagnostics(
            probs=probs[i], action=int(actions[i]), legal=legal, candidates=None,
        )
        ents.append(float(diag["entropy"]))
        assert len(list(board.legal_moves)) > 0
    return np.array(ents, dtype=np.float64), probs, _masks, ev


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
