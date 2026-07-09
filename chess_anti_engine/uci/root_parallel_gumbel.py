"""Root-parallel Gumbel search over evaluator groups (multi-GPU search v2, v1 scope).

Parallelizes the *validated* Gumbel path (docs/design_multi_gpu_search.md §2):
within a sequential-halving phase, each surviving root candidate's simulation
budget is independent work, farmed out to per-device evaluator groups. Root
decisions (candidate sampling, the halving schedule, completed-Q scoring, and
final selection) are computed single-threaded at phase barriers from
per-candidate subtree stats, so they are identical to a serial run given the
same per-candidate simulation outcomes.

Scouted classic-Gumbel decisions this orchestrator reproduces exactly
(sources: mcts/gumbel_c.py ``run_gumbel_root_many_c``, mcts/_mcts_tree.c
``gss_begin_round`` / ``gss_score_and_halve``, mcts/gumbel.py):

- **Candidate sampling**: Gumbel noise + log prior + top-m over the searched
  root legal set, with ``m = min(topk, n_legal, max(2, (budget+1)//2))`` and
  ``m = 1`` when ``budget <= 1``. Reused verbatim via
  ``gumbel._select_top_m_with_gumbel`` (UCI runs ``add_noise=False`` so the
  noise term is zero and no RNG stream is consumed — same as classic).
  Root priors are the legal-masked softmax of ``_policy_logits_to_full``
  logits (policy_temp + compact-to-full), computed from the SAME cached root
  eval the classic chunk path passes as ``pre_pol_logits`` — the root position
  eval runs on the primary device exactly like the classic path.
  Immediate-terminal root handling matches ``gumbel_c``: a mate-in-1 becomes
  the finished action; terminal-draw children are pruned from a winning root
  (``immediate_terminal_cboard_policy_or_draws``).

- **Sequential-halving schedule** (``gss_begin_round``): per round,
  ``rounds_left`` counts ceil-divisions of the surviving count by
  ``halving_div`` down to 1; ``vpa = max(1, budget // (n_cands *
  rounds_left))`` (all budget to a lone survivor); after the round the budget
  is debited ``vpa * n_cands`` (floored at 0) and the top
  ``clamp(ceil(n/div), 1, n-1)`` candidates survive. Exposed as pure helpers
  (``halving_visits_per_action`` / ``halving_keep_count`` /
  ``halving_schedule``) so tests can pin the arithmetic against the C
  implementation's.

- **Completed-Q halving score** (``gss_score_and_halve``): per candidate,
  ``gumbel + log(max(prior, 1e-12)) + q_scale * (q_hat - min_q) / max(max_q -
  min_q, 1e-8)`` where ``q_hat`` is the child's mean value from the root POV
  when visited, else the mctx mixed value; min/max run over ALL root children
  (unvisited ones contribute the mixed value); the running root value is
  ``(root_q_init + sum(q_c * N_c)) / (1 + sum(N_c))`` — algebraically the
  ``W[root]/N[root]`` the classic single-tree backprop maintains (here
  descents are rooted at candidate nodes, so the root aggregate is
  reconstructed from child stats). ``q_scale`` reuses
  ``gumbel._root_sigma_scale`` — the ROOT-site transform (c_scale_root /
  c_visit_root / q_visit_exp_root; the production root-LOG transform), which
  is deliberately different from the descent transform (see
  ``--c-scale-root`` / ``--q-visit-exp-root`` in uci/__main__.py).

- **Final selection**: the best-scored survivor of the last halving round
  (``remaining[0]``), i.e. the classic ``temperature=0`` play (the UCI chunk
  path forces ``temperature=0.0``). The returned root value is the survivor
  child's mean Q from the root POV when visited, else the initial root eval —
  bit-matching classic step 4's ``values_out``.

Deliberate v1 deviations (all pre-recorded in the design):

- **Intra-candidate descents use the batched-VL PUCT primitive**
  (``PucvChunker`` / ``batch_descend_puct`` rooted at the candidate node),
  NOT the classic non-root deterministic completed-Q argmax descent. This is
  the single-GPU-validated parallel gather regime; per-candidate outcomes
  therefore differ from classic per-sim, while every root decision is exact
  for the outcomes produced. Pending accounting defaults to virtual-mean.
- ``s = 1`` evaluator groups only; no late-phase budget splitting (when
  survivors < groups the extra groups idle); no cross-move tree reuse
  (fresh per-candidate arenas each move); no TB probing inside candidate
  subtrees.

All candidate subtrees live in ONE caller-provided ``MCTSTree`` (each rooted
at the corresponding root-child node) so the ``SearchWorker`` PV / visit-lead
/ hashfull plumbing works unchanged; the batched primitives never consult the
transposition table, so subtrees stay disjoint and the ownership rule (one
group per candidate at a time — asserted at claim) is structural.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _policy_logits_to_full,
    _root_sigma_scale,
    _select_top_m_with_gumbel,
)
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits as _wdl_to_q
from chess_anti_engine.mcts.puct_vl import _PLANES, PucvChunker
from chess_anti_engine.mcts.root_tactics import (
    immediate_terminal_cboard_policy_or_draws,
)
from chess_anti_engine.moves import POLICY_SIZE

_log = logging.getLogger(__name__)


# --- pure halving-schedule arithmetic (mirrors _mcts_tree.c) ------------------


def halving_rounds_left(n_candidates: int, halving_div: int = 2) -> int:
    """Ceil-divisions of ``n_candidates`` by ``halving_div`` down to 1.

    Mirrors the ``while (tmp > 1) { rounds_left++; tmp = (tmp+div-1)/div; }``
    loop in ``gss_begin_round``.
    """
    div = max(2, int(halving_div))
    rounds = 0
    tmp = int(n_candidates)
    while tmp > 1:
        rounds += 1
        tmp = (tmp + div - 1) // div
    return rounds


def halving_visits_per_action(
    n_candidates: int, budget_remaining: int, halving_div: int = 2,
) -> int:
    """Per-candidate sims for the current round (``gss_begin_round``)."""
    if n_candidates <= 1:
        return int(budget_remaining)
    rounds_left = halving_rounds_left(n_candidates, halving_div)
    return max(1, int(budget_remaining) // (int(n_candidates) * rounds_left))


def halving_keep_count(n_candidates: int, halving_div: int = 2) -> int:
    """Survivors after one halving round (``gss_score_and_halve``)."""
    div = max(2, int(halving_div))
    keep = (int(n_candidates) + div - 1) // div
    return max(1, min(keep, int(n_candidates) - 1))


def halving_schedule(
    n_candidates: int, budget: int, halving_div: int = 2,
) -> list[tuple[int, int]]:
    """Full ``(n_cands, visits_per_action)`` phase list for a sim budget.

    Pure companion of the run loop, for tests pinning the schedule against
    the classic C implementation's arithmetic.
    """
    phases: list[tuple[int, int]] = []
    rem = int(n_candidates)
    b = int(budget)
    while b > 0 and rem > 0:
        vpa = halving_visits_per_action(rem, b, halving_div)
        phases.append((rem, vpa))
        b = max(0, b - vpa * rem)
        if rem > 1:
            rem = halving_keep_count(rem, halving_div)
    return phases


# --- config / stats -----------------------------------------------------------


@dataclass
class RootParallelGumbelConfig:
    n_groups: int
    # Per-candidate leaf gather for the batched-VL descents inside one
    # candidate's subtree (the PucvChunker submit size).
    gather: int = 512
    # In-tree descent selection below the candidate root (PUCT+VL regime).
    c_puct: float = 2.5
    fpu_at_root: float = 0.0
    fpu_reduction: float = 1.2
    vloss_weight: int = 3
    # Pending accounting for the intra-candidate batched descents. Defaults to
    # virtual-mean (design §2: the mild single-GPU-validated regime). v1 pins
    # this independently of the UCI PUCVPendingMode default (which is legacy
    # for the pucv pool).
    vloss_mode: int = 1
    # Input channel count (146 v1 / 175 v2_threats); sized from the model.
    input_planes: int = _PLANES
    compute_relations: bool = False


@dataclass(frozen=True)
class RootParallelPhase:
    index: int
    candidates: int
    visits_per_action: int
    sims_completed: int
    survivor_actions: tuple[int, ...]


@dataclass(frozen=True)
class RootParallelGumbelStats:
    target_sims: int = 0
    elapsed_seconds: float = 0.0
    phases: tuple[RootParallelPhase, ...] = ()
    group_sims: tuple[int, ...] = ()


@dataclass
class _CandidateArena:
    """One root candidate's subtree, rooted at its root-child node ``cid``."""
    action: int
    cid: int
    cboard: Any = None  # CBoard after the candidate move; set on first touch
    terminal_value: float | None = None  # candidate move ends the game
    expanded: bool = False


@dataclass
class _WorkItem:
    arena: _CandidateArena
    budget: int
    phase_index: int


@dataclass
class _SearchState:
    tree: MCTSTree
    root_id: int
    root_cb: Any  # CBoard
    fen: str
    pri: np.ndarray  # full POLICY_SIZE float64 priors (post draw-prune softmax)
    search_legal: np.ndarray  # candidate-eligible actions (int64)
    root_q_init: float
    finished_action: int | None = None
    finished_value: float | None = None
    arenas: dict[int, _CandidateArena] = field(default_factory=dict)


class RootParallelGumbelPool:
    """Run root-parallel Gumbel over N per-device evaluator groups.

    Same two construction modes as ``MultiGpuPucvPool``: pre-built evaluators
    (CPU / eager / test paths) or per-group factories invoked on the group's
    own worker thread (REQUIRED for torch.compile + cudagraphs — the compiled
    model's cudagraph state is thread-local). When ``devices`` is provided,
    each worker calls ``torch.cuda.set_device`` before invoking its factory
    (the cudagraph-TLS trap; see design §1).
    """

    def __init__(
        self,
        cfg: RootParallelGumbelConfig,
        gumbel_cfg: GumbelConfig,
        evaluator_factories: Sequence[Callable[[], Any]] | None = None,
        *,
        evaluators: Sequence[Any] | None = None,
        devices: Sequence[str] | None = None,
        rng: np.random.Generator | None = None,
        info_string_cb: Callable[[str], None] | None = None,
    ) -> None:
        if cfg.n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {cfg.n_groups}")
        if (evaluator_factories is None) == (evaluators is None):
            raise ValueError(
                "specify exactly one of evaluator_factories / evaluators",
            )
        if devices is not None and len(devices) != cfg.n_groups:
            raise ValueError(
                f"need {cfg.n_groups} devices, got {len(devices)}",
            )
        if evaluator_factories is not None:
            if len(evaluator_factories) != cfg.n_groups:
                raise ValueError(
                    f"need {cfg.n_groups} factories, got {len(evaluator_factories)}",
                )
            self._factories: list[Callable[[], Any]] | None = list(evaluator_factories)
            self._evals: list[Any] = [None] * cfg.n_groups
        else:
            assert evaluators is not None
            if len(evaluators) != cfg.n_groups:
                raise ValueError(
                    f"need {cfg.n_groups} evaluators, got {len(evaluators)}",
                )
            for k, ev in enumerate(evaluators):
                if not hasattr(ev, "evaluate_inplace_async"):
                    raise TypeError(
                        f"evaluator[{k}] missing evaluate_inplace_async",
                    )
                if getattr(ev, "n_slots", 1) < 2:
                    raise ValueError(
                        f"evaluator[{k}] needs n_slots >= 2 for pipelining",
                    )
            self._factories = None
            self._evals = list(evaluators)
        self._cfg = cfg
        self._gcfg = gumbel_cfg
        self._devices = list(devices) if devices is not None else None
        self._rng = rng if rng is not None else np.random.default_rng()
        self._info_cb = info_string_cb

        self._state: _SearchState | None = None

        # Phase dispatch: a work queue drained by the persistent group
        # threads, plus a pending counter + event forming the phase barrier.
        self._work_q: queue.SimpleQueue[_WorkItem | None] = queue.SimpleQueue()
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._phase_done = threading.Event()
        self._job_stop: threading.Event | None = None

        # Ownership invariant (design §2 "the heart"): one group per candidate
        # at a time. `_claim` asserts it; `touch_hook` (tests) sees
        # claim/touch/release events.
        self._owner_lock = threading.Lock()
        self._owned: dict[int, int] = {}
        self.owner_history: list[tuple[int, int, int]] = []  # (phase, action, group)
        self.touch_hook: Callable[[str, int, int], None] | None = None

        self._stats_lock = threading.Lock()
        self._group_sims = [0] * cfg.n_groups
        self._phase_sims = 0
        self._last_stats = RootParallelGumbelStats(
            group_sims=tuple(0 for _ in range(cfg.n_groups)),
        )

        self._errors: list[BaseException] = []
        self._init_errors: list[BaseException] = []
        self._init_done = threading.Barrier(cfg.n_groups + 1)
        self._shutdown = threading.Event()
        self._chunkers: list[PucvChunker | None] = [None] * cfg.n_groups
        self._threads = [
            threading.Thread(
                target=self._worker_loop, args=(k,),
                name=f"rpg-group-{k}", daemon=True,
            )
            for k in range(cfg.n_groups)
        ]
        for th in self._threads:
            th.start()
        self._init_done.wait()
        if self._init_errors:
            self.close()
            raise self._init_errors[0]

    def __enter__(self) -> RootParallelGumbelPool:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def n_groups(self) -> int:
        return self._cfg.n_groups

    def close(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for _ in self._threads:
            self._work_q.put(None)
        for th in self._threads:
            th.join(timeout=2.0)
        # Unblock any orchestrator stuck at a phase barrier mid-close.
        self._phase_done.set()

    def last_stats(self) -> RootParallelGumbelStats:
        with self._stats_lock:
            return self._last_stats

    # --- root preparation (single-threaded, before any chunk) ----------------

    def prepare_root(
        self,
        *,
        tree: MCTSTree,
        board: chess.Board,
        pol_logits: np.ndarray,
        wdl_logits: np.ndarray,
    ) -> int:
        """Install the search root in ``tree`` and return its node id.

        Idempotent for the same (tree, position): repeated ``go`` on one
        position keeps the arenas. Mirrors the classic root init in
        ``run_gumbel_root_many_c`` — legal-masked softmax priors from the
        (policy_temp'd, full-space) cached root logits, ``add_root(1,
        root_q)``, mate shortcut, and terminal-draw pruning at a winning root.
        """
        fen = board.fen()
        st = self._state
        if st is not None and st.tree is tree and st.fen == fen:
            return st.root_id

        root_cb = CBoard.from_board(board)
        pol_full = _policy_logits_to_full(
            np.asarray(pol_logits, dtype=np.float32)[None, :], cfg=self._gcfg,
        )[0]
        root_q = float(_wdl_to_q(np.asarray(wdl_logits, dtype=np.float32).reshape(-1)))

        finished_action: int | None = None
        finished_value: float | None = None
        pri = np.zeros(POLICY_SIZE, dtype=np.float64)
        search_legal = np.zeros(0, dtype=np.int64)

        legal_idx = root_cb.legal_move_indices()
        if root_cb.is_game_over():
            finished_action = -1
            finished_value = float(root_cb.terminal_value())
            root_q = finished_value
            rid = int(tree.add_root(1, root_q))
        elif legal_idx.size == 0:
            finished_action = -1
            finished_value = root_q
            rid = int(tree.add_root(1, root_q))
        else:
            want_draws = root_q > 0.0 and legal_idx.size > 1
            terminal_mate, terminal_draws = immediate_terminal_cboard_policy_or_draws(
                root_cb, legal_idx, detect_draws=want_draws,
            )
            if root_q > 0.0 and legal_idx.size > 1 and terminal_draws:
                draw_arr = np.fromiter(terminal_draws, dtype=np.int32)
                keep = ~np.isin(legal_idx, draw_arr)
                if keep.any():
                    legal_idx = legal_idx[keep]
            search_legal = legal_idx.astype(np.int64)

            ll = pol_full[search_legal].astype(np.float64)
            ll -= ll.max()
            e = np.exp(ll)
            s = float(e.sum())
            priors = (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)
            pri[search_legal] = priors

            rid = int(tree.add_root(1, root_q))
            tree.expand(rid, search_legal.astype(np.int32), priors)

            if terminal_mate is not None:
                _probs, mate_action, mate_value = terminal_mate
                finished_action = int(mate_action)
                finished_value = float(mate_value)
            elif search_legal.size == 1:
                finished_action = int(search_legal[0])
                finished_value = root_q

        self._state = _SearchState(
            tree=tree,
            root_id=rid,
            root_cb=root_cb,
            fen=fen,
            pri=pri,
            search_legal=search_legal,
            root_q_init=root_q,
            finished_action=finished_action,
            finished_value=finished_value,
        )
        return rid

    # --- chunk entry point ----------------------------------------------------

    def run(
        self,
        *,
        target_sims: int,
        stop_event: threading.Event,
    ) -> tuple[float, int]:
        """One sim-budget chunk of root-parallel Gumbel; returns (root value,
        best action).

        Same chunk contract as the classic path: each call re-samples
        candidates (a no-op re-derivation under ``add_noise=False``) and runs
        a full halving schedule over ``target_sims``, accumulating stats in
        the persistent tree. ``stop_event`` is polled by group workers between
        per-candidate gather-batches; a stop finalizes from whatever budgets
        completed (decisions from partial budgets — classic-equivalent).
        """
        st = self._state
        if st is None:
            raise RuntimeError("prepare_root() must be called before run()")
        if self._shutdown.is_set():
            raise RuntimeError("RootParallelGumbelPool is closed")
        if st.finished_action is not None:
            assert st.finished_value is not None
            return float(st.finished_value), int(st.finished_action)

        started = time.perf_counter()
        budget = max(1, int(target_sims))
        cands, gumbels = _select_top_m_with_gumbel(
            legal=st.search_legal,
            pri=st.pri,
            sim_budget=budget,
            topk=int(self._gcfg.topk),
            add_noise=bool(self._gcfg.add_noise),
            gumbel_scale=float(self._gcfg.gumbel_scale),
            rng=self._rng,
        )
        remaining = [int(a) for a in cands]
        first_candidate = remaining[0]

        with self._stats_lock:
            self._group_sims = [0] * self._cfg.n_groups
        self._errors = []
        self._job_stop = stop_event

        phases: list[RootParallelPhase] = []
        phase_index = 0
        while budget > 0 and remaining:
            n_cands = len(remaining)
            vpa = halving_visits_per_action(n_cands, budget, self._gcfg.halving_div)
            items = [
                _WorkItem(self._arena_for(st, a), vpa, phase_index)
                for a in remaining
            ]
            sims_done = self._dispatch_phase(items)
            if self._errors:
                raise self._errors[0]
            stopped = stop_event.is_set()
            remaining = self._score_and_halve(st, remaining, gumbels)
            budget = max(0, budget - vpa * n_cands)
            phases.append(RootParallelPhase(
                index=phase_index,
                candidates=n_cands,
                visits_per_action=vpa,
                sims_completed=sims_done,
                survivor_actions=tuple(remaining),
            ))
            if self._info_cb is not None:
                self._info_cb(
                    f"rpg phase={phase_index} cands={n_cands} vpa={vpa} "
                    f"sims={sims_done} survivors={len(remaining)} "
                    f"budget_left={budget}{' STOPPED' if stopped else ''}"
                )
            phase_index += 1
            if stopped:
                break

        best = remaining[0] if remaining else first_candidate
        value = self._final_value(st, best)
        with self._stats_lock:
            self._last_stats = RootParallelGumbelStats(
                target_sims=int(target_sims),
                elapsed_seconds=time.perf_counter() - started,
                phases=tuple(phases),
                group_sims=tuple(self._group_sims),
            )
        return value, int(best)

    # --- orchestrator internals -------------------------------------------------

    def _arena_for(self, st: _SearchState, action: int) -> _CandidateArena:
        arena = st.arenas.get(action)
        if arena is None:
            cid = int(st.tree.find_child(st.root_id, int(action)))
            assert cid >= 0, f"candidate action {action} missing at the root"
            arena = _CandidateArena(action=int(action), cid=cid)
            st.arenas[action] = arena
        return arena

    def _dispatch_phase(self, items: list[_WorkItem]) -> int:
        with self._stats_lock:
            self._phase_sims = 0
        with self._pending_lock:
            self._pending = len(items)
        self._phase_done.clear()
        for it in items:
            self._work_q.put(it)
        self._phase_done.wait()
        with self._stats_lock:
            return self._phase_sims

    def _score_and_halve(
        self,
        st: _SearchState,
        remaining: list[int],
        gumbels: dict[int, float],
    ) -> list[int]:
        """Exact mirror of ``gss_score_and_halve`` (see module docstring)."""
        if len(remaining) <= 1:
            return remaining
        actions, visits, qs = st.tree.get_children_q(st.root_id, 0.0)
        visits_f = visits.astype(np.float64)
        visited = visits_f > 0.0
        n_total = float(visits_f.sum())
        # Running root value: the W[root]/N[root] a classic single-tree
        # backprop would hold (child q is already root-POV in get_children_q).
        root_q = (
            st.root_q_init + float((qs[visited] * visits_f[visited]).sum())
        ) / (n_total + 1.0)
        pri_children = np.maximum(st.pri[actions], np.finfo(np.float64).tiny)
        sum_probs = float(pri_children[visited].sum())
        if sum_probs > 0.0 and np.isfinite(sum_probs):
            weighted_q = float(
                (pri_children[visited] * qs[visited]).sum(),
            ) / sum_probs
        else:
            weighted_q = root_q
        mixed_value = (root_q + n_total * weighted_q) / (n_total + 1.0)
        q_hat = np.where(visited, qs, mixed_value)
        min_q = float(q_hat.min())
        max_q = float(q_hat.max())
        q_denom = max(max_q - min_q, 1e-8)
        max_visit = int(visits.max(initial=0))
        q_scale = _root_sigma_scale(max_visit=max_visit, cfg=self._gcfg)

        slot = {int(a): j for j, a in enumerate(actions)}
        scores: list[float] = []
        for a in remaining:
            j = slot.get(int(a))
            qh = float(q_hat[j]) if j is not None else root_q
            log_prior = float(np.log(max(float(st.pri[int(a)]), 1e-12)))
            scores.append(
                float(gumbels.get(int(a), 0.0))
                + log_prior
                + q_scale * ((qh - min_q) / q_denom)
            )
        # Stable descending sort matches the C selection sort's strict-greater
        # tie behavior (earlier candidate wins ties).
        order = sorted(range(len(remaining)), key=lambda i: -scores[i])
        keep = halving_keep_count(len(remaining), self._gcfg.halving_div)
        return [remaining[i] for i in order[:keep]]

    def _final_value(self, st: _SearchState, best: int) -> float:
        """Classic step-4 value: survivor child's root-POV Q if visited,
        else the initial root eval."""
        actions, visits, qs = st.tree.get_children_q(st.root_id, st.root_q_init)
        mask = actions == int(best)
        if bool(mask.any()) and int(visits[mask][0]) > 0:
            return float(qs[mask][0])
        return float(st.root_q_init)

    # --- ownership ---------------------------------------------------------------

    def _claim(self, action: int, group: int, phase: int) -> None:
        with self._owner_lock:
            owner = self._owned.get(action)
            assert owner is None, (
                f"ownership violation: candidate {action} already owned by "
                f"group {owner}, claimed by group {group}"
            )
            self._owned[action] = group
            if self.touch_hook is not None:
  # History is test instrumentation only — unbounded across a long
  # game, so record it only when a hook shows a test is watching.
                self.owner_history.append((phase, action, group))
        if self.touch_hook is not None:
            self.touch_hook("claim", action, group)

    def _release(self, action: int, group: int) -> None:
        with self._owner_lock:
            self._owned.pop(action, None)
        if self.touch_hook is not None:
            self.touch_hook("release", action, group)

    # --- group worker ---------------------------------------------------------------

    def _worker_loop(self, idx: int) -> None:
        try:
            if self._factories is not None:
                if self._devices is not None:
                    device = self._devices[idx]
                    if device.startswith("cuda"):
                        # cudagraph-TLS trap: bind the device on THIS thread
                        # before the factory compiles/captures anything.
                        import torch
                        torch.cuda.set_device(device)
                evaluator = self._factories[idx]()
                if not hasattr(evaluator, "evaluate_inplace_async"):
                    raise TypeError(
                        f"factory[{idx}] returned an evaluator missing "
                        "evaluate_inplace_async",
                    )
                if getattr(evaluator, "n_slots", 1) < 2:
                    raise ValueError(
                        f"factory[{idx}] returned an evaluator with n_slots < 2",
                    )
                self._evals[idx] = evaluator
            else:
                evaluator = self._evals[idx]
            cfg = self._cfg
            self._chunkers[idx] = PucvChunker(
                evaluator,
                gather=cfg.gather,
                c_puct=cfg.c_puct,
                fpu_at_root=cfg.fpu_at_root,
                fpu_reduction=cfg.fpu_reduction,
                vloss_weight=cfg.vloss_weight,
                vloss_mode=cfg.vloss_mode,
                input_planes=cfg.input_planes,
                compute_relations=cfg.compute_relations,
            )
        except BaseException as exc:
            self._init_errors.append(exc)
            self._init_done.wait()
            return
        self._init_done.wait()

        chunker = self._chunkers[idx]
        assert chunker is not None
        while not self._shutdown.is_set():
            item = self._work_q.get()
            if item is None or self._shutdown.is_set():
                return
            stop_event = self._job_stop
            try:
                sims = self._run_item(idx, evaluator, chunker, item, stop_event)
                with self._stats_lock:
                    self._group_sims[idx] += sims
                    self._phase_sims += sims
            except BaseException as exc:  # surfaced by run() after the barrier
                _log.exception("rpg-group-%d raised; requesting stop", idx)
                self._errors.append(exc)
                if stop_event is not None:
                    stop_event.set()
            finally:
                with self._pending_lock:
                    self._pending -= 1
                    if self._pending <= 0:
                        self._phase_done.set()

    def _run_item(
        self,
        group_idx: int,
        evaluator: Any,
        chunker: PucvChunker,
        item: _WorkItem,
        stop_event: threading.Event | None,
    ) -> int:
        st = self._state
        assert st is not None
        arena = item.arena
        self._claim(arena.action, group_idx, item.phase_index)
        try:
            done = 0
            if not arena.expanded:
                done += self._expand_candidate(evaluator, arena, st)
            if arena.terminal_value is not None:
                # Classic semantics: every forced sim into a terminal child
                # backprops the terminal value (visit counts keep accruing so
                # schedule/scoring arithmetic matches the serial run).
                n = item.budget - done
                if n > 0:
                    path = np.array([arena.cid], dtype=np.int32)
                    st.tree.backprop_many([path] * n, [arena.terminal_value] * n)
                    done += n
                return done
            gather = max(1, int(self._cfg.gather))
            while done < item.budget:
                if stop_event is not None and stop_event.is_set():
                    break
                if self.touch_hook is not None:
                    self.touch_hook("touch", arena.action, group_idx)
                # 2x gather per call keeps the chunker's 2-slot CPU/GPU
                # overlap alive while bounding stop latency to ~two batches.
                n = min(item.budget - done, 2 * gather)
                ran = int(chunker.run(st.tree, arena.cid, arena.cboard, n))
                if ran <= 0:
                    break  # solved/terminal below the candidate root
                done += ran
            return done
        finally:
            self._release(arena.action, group_idx)

    def _expand_candidate(
        self,
        evaluator: Any,
        arena: _CandidateArena,
        st: _SearchState,
    ) -> int:
        """First touch of a candidate: push its move, then either mark the
        terminal value or evaluate + expand the child node (this eval is the
        classic first forced sim, so it consumes 1 sim of the item budget)."""
        cb = st.root_cb.copy()
        cb.push_index(int(arena.action))
        arena.cboard = cb
        if cb.is_game_over():
            arena.terminal_value = float(cb.terminal_value())
            arena.expanded = True
            return 0
        enc = encode_cboard(
            cb,
            input_history_encoding=self._gcfg.input_history_encoding,
            input_extra_features=self._gcfg.input_extra_features,
        )
        buf = evaluator.get_input_buffer(1, slot=0)
        buf_np = buf.numpy() if hasattr(buf, "numpy") else np.asarray(buf)
        buf_np.reshape(1, int(self._cfg.input_planes), 8, 8)[0] = enc
        if self._cfg.compute_relations:
            rel = cb.compute_relations()[None, ...]
            pol_t, wdl_t, event = evaluator.evaluate_inplace_async(
                1, slot=0, relations=rel,
            )
        else:
            pol_t, wdl_t, event = evaluator.evaluate_inplace_async(1, slot=0)
        if event is not None:
            event.synchronize()
        pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
        wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
        legal = cb.legal_move_indices()
        if legal.size > 0:
            st.tree.expand_from_logits(
                arena.cid,
                legal.astype(np.int32),
                np.ascontiguousarray(pol[0], dtype=np.float32),
            )
        q = float(_wdl_to_q(np.asarray(wdl, dtype=np.float32)[0].reshape(-1)))
        st.tree.backprop(np.array([arena.cid], dtype=np.int32), q)
        arena.expanded = True
        return 1
