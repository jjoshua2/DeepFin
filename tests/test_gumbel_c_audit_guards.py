"""Guards and surface fixes from the 2026-08-03 play-path code audit.

Three findings, all of the house pattern — a value accepted and then silently
ignored, or a cost paid with nothing said about it:

* **F4** — ``tree_gumbel_select_child`` (``_mcts_tree.c:2941-2944``) does not
  implement the ``VLOSS_MODE_VIRTUAL_MEAN`` PARENT accounting the comment at
  2968-2983 says it mirrors from ``tree_select_child:660-673``. It mirrors the
  CHILD term and not the parent term, so under VIRTUAL_MEAN ``parent_Q`` — the
  FPU for every unvisited child and the ``weighted_q`` fallback — still carries
  exactly the parallel-PUCT pessimism VIRTUAL_MEAN exists to remove. No caller
  passes ``vloss_mode=1`` to the Gumbel path today, so it is a trap armed for
  whoever wires the knob through. The Python entry point now refuses it.
* **F5** — ``_use_pipeline`` built fresh sub-trees, ignored the caller's
  ``tree``/``root_node_ids`` and returned root ids ``[-1]*n``, so a caller that
  asked for a persistent tree lost every root — as a function of BATCH SIZE.
* **F10** — the inlined temperature sampler is now the shared
  ``mcts/sampling.py`` primitive, as ``gumbel.py`` already used.

The F5 and F10 tests drive the real C search on CPU with a synthetic evaluator.
"""
from __future__ import annotations

import dataclasses
import logging
import os

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.mcts import gumbel_c as gumbel_c_mod
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import (
    VLOSS_MODE_LEGACY,
    VLOSS_MODE_VIRTUAL_MEAN,
    run_gumbel_root_many_c,
)
from chess_anti_engine.moves import POLICY_SIZE

_FENS = (
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4rrk1/pp1n1ppp/2pb1q2/3p4/3P4/2NBPN2/PPQ2PPP/2R2RK1 w - - 0 14",
    "2kr3r/ppp2ppp/2n1b3/3q4/3P4/2P1BN2/PP3PPP/R2Q1RK1 b - - 0 12",
)


class _HashEvaluator:
    """Deterministic in the encoded position; no torch model, no GPU."""

    supports_legal_bf16 = False

    def __init__(self, seed: int = 20260803) -> None:
        rng = np.random.default_rng(seed)
        self._pol = (rng.standard_normal((997, POLICY_SIZE)) * 2.0).astype(np.float32)
        self._wdl = rng.standard_normal((997, 3)).astype(np.float32)

    def _lookup(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = int(x.shape[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        idx = np.array(
            [int.from_bytes(flat[i].tobytes()[:6], "little") % 997 for i in range(n)]
        )
        return self._pol[idx], self._wdl[idx]

    def evaluate_encoded(self, x, relations=None):
        del relations
        return self._lookup(x)


class _AsyncHashEvaluator(_HashEvaluator):
    """Same values, but exposing ``evaluate_encoded_async`` — which is the ONLY
    thing that arms ``_use_pipeline`` (``_has_async``)."""

    def evaluate_encoded_async(self, x, relations=None):
        del relations
        pol, wdl = self._lookup(x)
        return torch.from_numpy(pol.copy()), torch.from_numpy(wdl.copy()), None


def _cfg(**kw: float) -> GumbelConfig:
    return dataclasses.replace(
        GumbelConfig(simulations=8, topk=4, temperature=0.0, add_noise=False), **kw,
    )


def _run(boards, *, cfg=None, evaluator=None, seed=5, **kw):
    gumbel_c_mod._COMPILED_BATCH_BUCKETS = ()
    return run_gumbel_root_many_c(
        None, boards, device="cpu", rng=np.random.default_rng(seed),
        cfg=cfg if cfg is not None else _cfg(),
        evaluator=evaluator if evaluator is not None else _HashEvaluator(),
        target_batch=0, vloss_weight=1, **kw,
    )


# --- F4: the VIRTUAL_MEAN trap -------------------------------------------


def test_the_vloss_mode_constants_mirror_the_c_defines() -> None:
    assert (VLOSS_MODE_LEGACY, VLOSS_MODE_VIRTUAL_MEAN) == (0, 1)


def test_gumbel_refuses_virtual_mean_until_the_c_parent_branch_is_mirrored() -> None:
    """Fails on pre-fix `main`: the call ran a half-mirrored descent instead."""
    with pytest.raises(ValueError, match="F4") as exc:
        _run([chess.Board(_FENS[0])], vloss_mode=VLOSS_MODE_VIRTUAL_MEAN)
    message = str(exc.value)
    assert "vloss_mode" in message
    assert "F4" in message
    assert "parent" in message.lower()


def test_legacy_vloss_mode_is_untouched() -> None:
    """Negative control: the guard must not fire on what production runs."""
    probs, actions, *_ = _run(
        [chess.Board(f) for f in _FENS], vloss_mode=VLOSS_MODE_LEGACY,
    )
    assert len(probs) == len(_FENS)
    assert all(0 <= int(a) < POLICY_SIZE for a in actions)


def test_audit_targets_rejects_vloss_mode_1_at_parse_time() -> None:
    """The reviewer's best catch, and the PR's own thesis turned on itself.

    ``_net_candidates`` only forwards ``vloss_mode`` when ``vloss_weight > 0``,
    and the script never prints or records it -- so ``--vloss-mode 1`` at the
    DEFAULT ``--vloss-weight 0`` was accepted, dropped, and left no trace. That
    is the "value accepted and then silently ignored" pattern this PR exists to
    remove, sitting in the flag the PR just re-documented as raising. With a
    weight it did raise, but only after the audit set, the checkpoint and the
    evaluator had loaded.

    Asserted as a subprocess against a checkpoint path that does not exist: if
    the guard ever moves after the loads, the run dies on the missing file
    instead and this fails.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/audit_targets.py",
         "--checkpoint", "/nonexistent/ckpt.pt", "--vloss-mode", "1"],
        cwd=repo, capture_output=True, text=True, timeout=600, check=False,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "F4" in output, output[-2000:]
    assert "VIRTUAL_MEAN" in output
    # It must have died on OUR guard, not on the missing checkpoint/audit set.
    assert "FileNotFoundError" not in output, output[-2000:]


def test_the_guard_fires_before_any_search_work() -> None:
    """A guard that only fires after the tree is built is a slow crash, not a
    guard: assert it rejects with no evaluator calls."""

    class _Exploding(_HashEvaluator):
        def evaluate_encoded(self, x, relations=None):  # pragma: no cover
            del x, relations
            raise AssertionError("search ran despite the vloss_mode guard")

    with pytest.raises(ValueError, match="F4"):
        _run(
            [chess.Board(_FENS[0])],
            evaluator=_Exploding(),
            vloss_mode=VLOSS_MODE_VIRTUAL_MEAN,
        )


# --- F5: the pipeline must not eat the caller's tree ----------------------


def _boards_for_pipeline() -> list[chess.Board]:
    # >= 64 boards is the pipeline threshold (gumbel_c.py `_use_pipeline`).
    return [chess.Board(_FENS[i % len(_FENS)]) for i in range(64)]


def test_a_supplied_tree_keeps_its_roots_at_the_pipeline_batch_size() -> None:
    """The deciding assertion. Pre-fix this returned [-1]*64 and left the tree
    empty: tree reuse silently switched off at 64 boards."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    boards = _boards_for_pipeline()
    result = _run(boards, evaluator=_AsyncHashEvaluator(), tree=tree)
    returned_tree, root_ids = result[4], result[5]

    assert returned_tree is tree, "the caller's tree must come back, not a sub-tree"
    assert len(root_ids) == len(boards)
    assert all(int(r) >= 0 for r in root_ids), (
        f"pipeline discarded the caller's roots: {sorted({int(r) for r in root_ids})}"
    )


def test_the_roots_the_call_returns_are_actually_reusable_next_ply() -> None:
    """Root ids are only worth returning if the caller can feed them back."""
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    tree = MCTSTree()
    boards = _boards_for_pipeline()
    first = _run(boards, evaluator=_AsyncHashEvaluator(), tree=tree)
    root_ids = [int(r) for r in first[5]]
    assert all(tree.is_expanded(r) for r in root_ids)

    second = _run(
        boards, evaluator=_AsyncHashEvaluator(), tree=tree, root_node_ids=root_ids,
    )
    assert [int(r) for r in second[5]] == root_ids


def test_the_fallback_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    gumbel_c_mod._PIPELINE_TREE_WARNED = False
    from chess_anti_engine.mcts._mcts_tree import MCTSTree

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.mcts.gumbel_c"):
        _run(_boards_for_pipeline(), evaluator=_AsyncHashEvaluator(), tree=MCTSTree())
        _run(_boards_for_pipeline(), evaluator=_AsyncHashEvaluator(), tree=MCTSTree())

    hits = [r for r in caplog.records if "F5" in r.getMessage()]
    assert len(hits) == 1, "the F5 warning must be one-shot, not once per ply"


def test_the_pipeline_still_runs_when_no_tree_was_supplied() -> None:
    """Negative control: callers that never asked for tree carry keep the
    throughput optimisation, and still get the -1 sentinel."""
    boards = _boards_for_pipeline()
    result = _run(boards, evaluator=_AsyncHashEvaluator(), tree=None)
    assert all(int(r) == -1 for r in result[5])


# --- F7: the compact-legal transport gate must price itself ---------------


def test_a_non_unit_policy_temp_announces_the_transport_it_disables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _LegalBf16Evaluator(_HashEvaluator):
        supports_legal_bf16 = True

        def evaluate_legal_bf16(self, x, legal_flat, legal_counts):  # pragma: no cover
            del x, legal_flat, legal_counts
            raise AssertionError("compact-legal transport used despite policy_temp")

    gumbel_c_mod._LEGAL_BF16_TEMP_WARNED = False
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.mcts.gumbel_c"):
        _run(
            [chess.Board(_FENS[0])],
            cfg=_cfg(policy_temp=0.5),
            evaluator=_LegalBf16Evaluator(),
        )
    hits = [r for r in caplog.records if "F7" in r.getMessage()]
    assert len(hits) == 1
    assert "policy_temp" in hits[0].getMessage()


def test_policy_temp_one_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Negative control: production runs policy_temp=1.0 and must stay silent."""
    gumbel_c_mod._LEGAL_BF16_TEMP_WARNED = False
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.mcts.gumbel_c"):
        _run([chess.Board(_FENS[0])], cfg=_cfg(policy_temp=1.0))
    assert not [r for r in caplog.records if "F7" in r.getMessage()]


# --- F10: one temperature sampler, not two -------------------------------


def test_temperature_zero_plays_the_survivor_not_the_first_legal_action() -> None:
    """One candidate makes the survivor known independently of sampling code."""
    board = chess.Board()
    legal = _legal_indices(board)
    expected = int(legal.max())

    class _PreferredActionEvaluator:
        supports_legal_bf16 = False

        def evaluate_encoded(self, x, relations=None):
            del relations
            policy = np.zeros((len(x), POLICY_SIZE), dtype=np.float32)
            policy[:, expected] = 8.0
            return policy, np.zeros((len(x), 3), dtype=np.float32)

    probs, actions, *_ = _run(
        [board], cfg=_cfg(topk=1, temperature=0.0), evaluator=_PreferredActionEvaluator(),
    )
    support = np.flatnonzero(probs[0])
    assert len(support) > 1
    assert int(support[0]) != expected
    assert int(actions[0]) == expected


def test_a_positive_temperature_still_samples_from_the_returned_policy() -> None:
    boards = [chess.Board(_FENS[0])]
    seen = set()
    for seed in range(24):
        _, actions, *_ = _run(boards, cfg=_cfg(temperature=1.5), seed=seed)
        seen.add(int(actions[0]))
    assert len(seen) > 1, "temperature>0 collapsed to a single action"


def _legal_indices(board: chess.Board) -> np.ndarray:
    from chess_anti_engine.encoding._lc0_ext import CBoard

    return CBoard.from_board(board).legal_move_indices()
