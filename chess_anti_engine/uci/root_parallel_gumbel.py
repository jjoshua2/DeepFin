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
  (unvisited ones contribute the mixed value).

  The root value fed to the mix is the FRESH network root value
  (``root_q_init``), matching the C path's ``root_qs``, mctx's
  ``qtransform_completed_by_mix_value``, and ``gumbel.py``'s reference
  ``_halve_remaining_for_board``. It is fixed for the whole root search; child
  visits change the prior-weighted ``weighted_q``, but they do not mutate the
  raw-value baseline that completes unvisited actions.

  ``q_scale`` reuses
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
  for the outcomes produced. Pending accounting follows the UCI
  ``PUCVPendingMode`` option and defaults to legacy (audit W3).
- ``s = 1`` evaluator groups only; late-phase budget splitting when
  survivors < groups is ON by default (``split_idle_groups``; co-owned
  survivor arenas under VL — not bit-identical to exclusive serial
  interleaving); multi-GPU may run a no-halve **open** scout
  (``open_vpa``) and floor survivors at ``min_keep`` during SH so device
  count stays fed — quality-sensitive schedule, not classic bit-identity;
  no cross-move tree reuse (fresh per-candidate arenas each move); no TB
  probing inside candidate subtrees.

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

from chess_anti_engine.async_safety import drain_failed_receipts, ensure_usable, note_cleanup, submit_async

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.headroom import reserve_concurrent_headroom
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _policy_logits_to_full,
    _root_sigma_scale,
    _select_top_m_with_gumbel,
    halving_keep_count,
    halving_visits_per_action,
)
from chess_anti_engine.mcts.puct import _value_scalar_from_wdl_logits as _wdl_to_q
from chess_anti_engine.mcts.puct_history import puct_history_args
from chess_anti_engine.mcts.puct_vl import _PLANES, PucvChunker
from chess_anti_engine.mcts.puct_tb import override_leaf_values
from chess_anti_engine.tablebase import SyzygyProbe
from chess_anti_engine.mcts.root_tactics import (
    forced_draw_actions,
    immediate_terminal_cboard_policy_or_draws,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci import wedge
from chess_anti_engine.uci.protocol import uci_debug_enabled
from chess_anti_engine.uci.startup_progress import StartupContext, startup_context

_log = logging.getLogger(__name__)

# How often the orchestrator re-checks for a stuck group while a phase runs.
_PHASE_POLL_S = 0.05
# After a group fails, how long the orchestrator waits for the OTHER groups'
# in-flight items to stop (they poll the job stop between gather batches).
_FAILURE_DRAIN_S = 2.0
# How often the constructor re-checks the startup no-progress clocks.
_STARTUP_POLL_S = 0.5


class RPGDeviceFailure(RuntimeError):
    """One root-parallel Gumbel group (device) raised or stopped answering.

    The pool cannot continue without that group (its candidates' subtrees are
    half-updated), so the caller retires the pool; ``SearchWorker`` then
    continues the same search on the classic single-evaluator path."""

    def __init__(self, label: str, what: str, reason: str) -> None:
        super().__init__(f"device {label} {what} ({reason})")
        self.label = label
        self.what = what
        self.reason = reason


class _JobStop:
    """The engine's stop event OR'd with the pool's own failure stop.

    A failing group used to ``set()`` the ENGINE's stop event, which ended the
    whole ``go``; now it stops only this pool's job, and the search can carry
    on elsewhere."""

    def __init__(self, outer: threading.Event) -> None:
        self.outer = outer
        self.inner = threading.Event()

    def is_set(self) -> bool:
        return self.inner.is_set() or self.outer.is_set()

    def set(self) -> None:
        self.inner.set()


# --- pure halving-schedule arithmetic (mirrors _mcts_tree.c) ------------------
#
# ⚑ `halving_visits_per_action` / `halving_keep_count` MOVED to `mcts/gumbel.py`
# and are imported above, so this module's public surface (and its tests') is
# unchanged. They moved because the Python reference search in that file carried
# a second, hardcoded div-2 copy of this arithmetic and never read
# `cfg.halving_div`; two implementations of one schedule is how a knob ends up
# honoured on one path and dropped on the other. `halving_rounds_left` lives
# there too -- import it from `mcts.gumbel`, which is now its only home.


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


def resolve_rpg_schedule_knobs(
    *,
    n_groups: int,
    gather: int,
    open_vpa: int = -1,
    min_vpa: int = -1,
    min_keep: int = -1,
    open_budget_frac: float = 0.50,
) -> tuple[int, int, int, float]:
    """Map UCI schedule knobs to concrete ``RootParallelGumbelConfig`` values.

    Convention (UCI spins, multi-GPU only):
      -1 = auto (open/min_vpa → gather, min_keep → n_groups)
       0 = off
      >0 = fixed value

    Single-group installs always resolve to all-off so g=1 bit-identity
    tests stay independent of UCI defaults.
    """
    gath = max(1, int(gather))
    groups = max(1, int(n_groups))
    frac = float(open_budget_frac)
    if not (0.0 < frac <= 1.0):
        frac = 0.50
    if groups <= 1:
        return 0, 0, 0, frac

    def _resolve(raw: int, auto: int) -> int:
        v = int(raw)
        if v < 0:
            return int(auto)
        return max(0, v)

    return (
        _resolve(open_vpa, gath),
        _resolve(min_vpa, gath),
        _resolve(min_keep, groups),
        frac,
    )


# --- config / stats -----------------------------------------------------------


@dataclass
class RootParallelGumbelConfig:
    n_groups: int
    # Per-candidate leaf gather for the batched-VL descents inside one
    # candidate's subtree (the PucvChunker submit size).
    gather: int = 512
    # In-tree descent selection below the candidate root (PUCT+VL regime).
    c_puct: float = 2.5
    # Lc0-style visit-dependent Cpuct on intra-candidate PUCT (factor<=0 fixed).
    cpuct_factor: float = 0.0
    cpuct_base: float = 38739.0
    fpu_at_root: float = 0.0
    fpu_reduction: float = 1.2
    vloss_weight: int = 3
    # Pending accounting for the intra-candidate batched descents. Legacy (0),
    # matching every other batched-VL path's default and the UCI
    # `PUCVPendingMode` option that now feeds this field
    # (`SearchWorker.install_root_parallel_gumbel`).
    # Virtual-mean (1) gives a pending, zero-visit child q_parent == parent_Q
    # exactly — no pessimism at all — and counts its prior into
    # `visited_policy`, which raises the FPU penalty on its untouched
    # siblings (`_mcts_tree.c` tree_select_child, VLOSS_MODE_VIRTUAL_MEAN).
    # Both push the next descent back onto the leaf already in flight: at this
    # config's own shape, 2048 sims explored 15,865 distinct nodes under mode 1
    # against 42,020 under mode 0 for identical evaluator rows (audit W3).
    vloss_mode: int = 0
    # When True (default), late phases with fewer candidates than groups
    # shard survivor budgets so idle GPUs keep working (design §2). Root
    # decisions stay sequential-halving; only the concurrent VL interleaving
    # inside a shared candidate changes. Disable for bit-identity tests
    # against serial g=1 under a deterministic evaluator.
    split_idle_groups: bool = True
    # Floor on visits_per_action when n_groups > 1 (0 = off). Lifts tiny
    # early-phase vpa (e.g. 5 at short budgets) toward gather-sized GPU
    # batches. Still sequential-halving; only spends more of the budget
    # earlier. Bit-identity / pure-schedule tests leave this 0; multi-GPU
    # install sets gather so util stays fat *after* open too (NPS gap vs
    # PUCV is not only cold-start / first phase).
    min_vpa: int = 0
    # Multi-GPU "phase A" open: before sequential-halving, give every root
    # candidate this many sims (capped by open_budget_frac and remaining
    # budget). No halving after open — all candidates stay for SH. Opens
    # enough live trees with gather-scale batches to keep devices busy.
    # 0 = off (bit-identity / single-group). Multi-GPU install sets gather.
    open_vpa: int = 0
    # Cap total open spend at this fraction of the chunk's target_sims.
    # Prevents a short movetime chunk from becoming 100% scout.
    open_budget_frac: float = 0.50
    # Floor on survivors after each SH halving (0 = classic keep only).
    # Multi-GPU install sets n_groups so the candidate set does not collapse
    # below device count until forced (keep <= n-1). "Stay somewhat in
    # phase A" for util without inventing illegal root moves.
    min_keep: int = 0
    # Per-group PucvChunker eval-cache capacity (0 = off). Same knobs as the
    # multi-GPU PUCV pool / single-thread UseVL path; each group owns its own
    # cache (subtrees are disjoint so a shared cache would rarely hit).
    eval_cache_entries: int = 0
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
    # "open" = multi-GPU scout (no halving); "halve" = sequential-halving.
    kind: str = "halve"


@dataclass(frozen=True)
class RootParallelGumbelStats:
    target_sims: int = 0
    elapsed_seconds: float = 0.0
    phases: tuple[RootParallelPhase, ...] = ()
    group_sims: tuple[int, ...] = ()
    # Pending accounting each group's chunker actually descended with, read
    # off the live chunkers (-1 = group not yet constructed). `nodes`/`nps`
    # are simulation counts and are invariant to this, so without it the
    # difference between exploring 42k and 16k positions for the same budget
    # is invisible from outside the search (audit W3).
    vloss_mode: tuple[int, ...] = ()
    # Distinct tree nodes at the end of the chunk — the quantity `nodes` is
    # routinely mistaken for.
    tree_nodes: int = 0
    # Root oracle / single-action shortcut completed without a simulation.
    # Separate completion from work accounting so open-ended searches stop.
    root_finished: bool = False


@dataclass
class _CandidateArena:
    """One root candidate's subtree, rooted at its root-child node ``cid``."""
    action: int
    cid: int
    cboard: Any = None  # CBoard after the candidate move; set on first touch
    terminal_value: float | None = None  # natural terminal or proven TB value
    expanded: bool = False
    # Serializes first expand when late-phase split lets multiple groups
    # co-own this arena (VL descents themselves are tree-mutexed).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class _RowsJob:
    """One evaluation of arbitrary rows on a healthy group's OWN thread
    (``RootParallelGumbelPool.evaluate_rows``): compiled / CUDA-graph
    evaluators stay on the thread that built them."""

    xs: np.ndarray
    relations: np.ndarray | None
    exclude: frozenset[int]
    done: threading.Event = field(default_factory=threading.Event)
    taken_by: int = -1
    since: float = 0.0
    result: tuple[np.ndarray, np.ndarray] | None = None
    error: BaseException | None = None


@dataclass
class _WorkItem:
    arena: _CandidateArena
    budget: int
    phase_index: int
    # True when this shard co-owns the arena with other groups in the same
    # phase (design §2 late-phase split). Exclusive ownership is relaxed;
    # the C tree's virtual-loss path is the concurrency contract (same as
    # MultiGpuPucvPool on a shared tree).
    shared: bool = False


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
    tb_probe: SyzygyProbe | None = None
    # A winning root's terminal-draw moves were left out of the root's
    # children: a root whose remaining children are all proven lost is
    # still a proven DRAW, not a loss (see `_resolve_root_solved`).
    pruned_root_draws: bool = False


def root_forced_draw_actions(board: chess.Board, actions: Sequence[int]) -> list[int]:
    """The root ``actions`` (policy indices) whose move ENDS the game drawn on
    the board; see ``mcts.root_tactics.forced_draw_actions`` (shared with the
    classic search's pruned-draw fallback)."""
    return forced_draw_actions(board, actions)


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
        startup_timeout_s: float | None = None,
    ) -> None:
        puct_history_args(gumbel_cfg.input_history_encoding)
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
        self._work_q: queue.SimpleQueue[_WorkItem | _RowsJob | None] = queue.SimpleQueue()
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._phase_done = threading.Event()
        self._job_stop: _JobStop | None = None

        # Ownership: exclusive one-group-per-candidate by default (design §2).
        # Late-phase split uses a refcount so multiple groups may co-own one
        # survivor arena under virtual-loss. `_claim` / `_claim_shared` assert
        # the right contract; `touch_hook` (tests) sees claim/touch/release.
        self._owner_lock = threading.Lock()
        self._owned: dict[int, int] = {}  # action -> exclusive owner group
        self._owned_shared: dict[int, set[int]] = {}  # action -> co-owners
        self.owner_history: list[tuple[int, int, int]] = []  # (phase, action, group)
        self.touch_hook: Callable[[str, int, int], None] | None = None

        self._stats_lock = threading.Lock()
        self._group_sims = [0] * cfg.n_groups
        self._phase_sims = 0
        self._last_stats = RootParallelGumbelStats(
            group_sims=tuple(0 for _ in range(cfg.n_groups)),
        )

        self._errors: list[BaseException] = []
        self._error_groups: list[int] = []
        self._rows_failed: set[int] = set()
        self._rows_notes: list[str] = []
        self._last_failure: RPGDeviceFailure | None = None
        self._init_errors: list[BaseException] = []
        self._init_done = threading.Barrier(cfg.n_groups + 1)
        # Startup: per-group "built" flags and progress contexts, so a group
        # whose factory hangs is timed out (``startup_timeout_s``) instead of
        # holding the constructor at the barrier forever.
        self._startup_timeout_s = startup_timeout_s
        self._built = [threading.Event() for _ in range(cfg.n_groups)]
        names = [
            str(self._devices[k]) if self._devices is not None else f"#{k}"
            for k in range(cfg.n_groups)
        ]
        # Repeated device names (``--devices cpu,cpu``) get ``#k`` so every
        # announcement names exactly one group.
        self._labels = tuple(
            n if names.count(n) == 1 else f"{n}#{k}" for k, n in enumerate(names)
        )
        self._startup_ctx = [StartupContext(self._labels[k]) for k in range(cfg.n_groups)]
        # Stuck-group watch: when each group entered its current device-bound
        # step (None = not in one), its slowest completed step, and the groups
        # abandoned inside a step (their threads may never come back).
        self._busy_since: list[float | None] = [None] * cfg.n_groups
        self._slowest_step_s = [0.0] * cfg.n_groups
        self._in_item = [False] * cfg.n_groups
        self._abandoned: set[int] = set()
        self._watch_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._chunkers: list[PucvChunker | None] = [None] * cfg.n_groups
        self._threads = [
            threading.Thread(
                target=wedge.parking(self._worker_loop), args=(k,),
                name=f"rpg-group-{k}", daemon=True,
            )
            for k in range(cfg.n_groups)
        ]
        self._started_threads: list[threading.Thread] = []
        try:
            for th in self._threads:
                th.start()
                self._started_threads.append(th)
        except BaseException as error:
            self._init_errors.insert(0, error)
            self._init_done.abort()
            try:
                self.close()
            except BaseException as cleanup:
                note_cleanup(error, f"partial group startup cleanup failed: {cleanup!r}")
            raise
        self._await_built()
        try:
            self._init_done.wait()
        except threading.BrokenBarrierError:
            if not self._init_errors:
                raise
            # ``_await_built`` timed a group out and broke the barrier.
        if self._init_errors:
            self.close()
            raise self._init_errors[0]

    def _await_built(self) -> None:
        """Wait for every group's factory, timing out one that makes no
        progress for ``startup_timeout_s``. RPG start is strict: a timed-out
        group is a startup failure (the barrier is broken, every group retires
        and the constructor raises), never a hang."""
        timeout = self._startup_timeout_s
        if timeout is None:
            return
        while True:
            pending = [k for k, ev in enumerate(self._built) if not ev.is_set()]
            if not pending:
                return
            now = time.monotonic()
            for k in pending:
                idle = self._startup_ctx[k].idle_for(now)
                if idle <= timeout or self._built[k].is_set():
                    continue
                self._init_errors.insert(0, TimeoutError(
                    f"root-parallel Gumbel device {self._labels[k]}: evaluator build/warm-up "
                    f"made no progress for {idle:.0f}s (device startup timeout {timeout:g}s)"
                ))
                self._abandoned.add(k)
                wedge.mark_wedged(self._threads[k], f"RPG device {self._labels[k]} startup",
                                  device=self._labels[k])
                self._startup_ctx[k].revoke_turns()
                self._init_done.abort()
                return
            self._built[pending[0]].wait(min(_STARTUP_POLL_S, max(0.01, timeout / 4)))

    def __enter__(self) -> RootParallelGumbelPool:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def n_groups(self) -> int:
        return self._cfg.n_groups

    @property
    def gather(self) -> int:
        """Per-candidate leaf gather used by each group's PucvChunker."""
        return max(1, int(self._cfg.gather))

    def close(self, *, abandoned_join_s: float = 1.0) -> None:
        """Stop every group. ``abandoned_join_s``: the shared wait for groups
        abandoned inside a device step (0 when retiring the pool mid-search:
        the move's clock must not wait for a device that is known stuck)."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for _ in self._threads:
            self._work_q.put(None)
        abandoned_deadline = time.monotonic() + max(0.0, abandoned_join_s)
        started = [k for k, th in enumerate(self._threads) if th in self._started_threads]
        for k in started:
            if k in self._abandoned:
                # Abandoned in a device step: a device fault, after which
                # ending groups park (``wedge``), so nothing to join.
                wedge.join_unless_wedged(
                    self._threads[k], max(0.0, abandoned_deadline - time.monotonic()))
        lingering = [
            self._threads[k] for k in started
            # After a device fault, ending groups park: no join.
            if k not in self._abandoned and not wedge.join_unless_wedged(self._threads[k], 2.0)
        ]
        wedge.warn_outlived_close(_log, "root-parallel Gumbel pool", lingering, 2.0)
        # Unblock any orchestrator stuck at a phase barrier mid-close.
        self._phase_done.set()

    def live_worker_threads(self) -> list[threading.Thread]:
        """Group threads still alive (after ``close``: wedged in a device step)."""
        return [th for th in self._started_threads if th.is_alive()]

    def last_stats(self) -> RootParallelGumbelStats:
        with self._stats_lock:
            return self._last_stats

    def realized_vloss_mode(self) -> tuple[int, ...]:
        """Pending accounting per group, read off the live ``PucvChunker``s.

        ``-1`` for a group whose chunker has not been built (only possible
        after a failed init). This is the observation that distinguishes
        "the knob was set" from "the descent used it".
        """
        return tuple(
            -1 if ch is None else int(ch.vloss_mode) for ch in self._chunkers
        )

    def set_vloss_mode(self, mode: int) -> None:
        """Re-point every group's chunker at ``mode`` (0 legacy, 1 virtual-mean).

        Mutating ``self._cfg`` alone would not reach the search: each group
        builds its ``PucvChunker`` once, in its own worker thread, at pool
        construction. Caller holds the search barrier.
        """
        mode = 1 if int(mode) == 1 else 0
        self._cfg.vloss_mode = mode
        for chunker in self._chunkers:
            if chunker is not None:
                chunker.set_vloss_mode(mode)

    # --- root preparation (single-threaded, before any chunk) ----------------

    def prepare_root(
        self,
        *,
        tree: MCTSTree,
        board: chess.Board,
        pol_logits: np.ndarray,
        wdl_logits: np.ndarray,
        allow_terminal_shortcuts: bool = True,
    ) -> int:
        """Install the search root in ``tree`` and return its node id.

        Idempotent for the same (tree, position): repeated ``go`` on one
        position keeps the arenas. Mirrors the classic root init in
        ``run_gumbel_root_many_c`` — legal-masked softmax priors from the
        (policy_temp'd, full-space) cached root logits, ``add_root(1,
        root_q)``, and (when ``allow_terminal_shortcuts``) mate shortcut +
        terminal-draw pruning at a winning root. Single-legal-move finish
        is NOT gated (matches classic Gumbel).
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

        pruned_root_draws = False
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
            terminal_mate = None
            terminal_draws: set[int] = set()
            if allow_terminal_shortcuts:
                # Classic: draw scan only on winning multi-legal roots; mate
                # always detected when shortcuts are allowed.
                want_draws = root_q > 0.0 and legal_idx.size > 1
                terminal_mate, terminal_draws = immediate_terminal_cboard_policy_or_draws(
                    root_cb, legal_idx, detect_draws=want_draws,
                )
            if root_q > 0.0 and legal_idx.size > 1 and terminal_draws:
                draw_arr = np.fromiter(terminal_draws, dtype=np.int32)
                keep = ~np.isin(legal_idx, draw_arr)
                # Prune only while at least two moves survive (same rule as
                # classic `gumbel_c`): a single survivor is not a forced move,
                # so the drawing moves stay in the search and compete on their
                # exact value 0.
                if np.count_nonzero(keep) > 1:
                    # The draw scan also returns moves after which the
                    # OPPONENT may claim a draw (50-move at 99 + a reversible
                    # reply, threefold on its reply). That is the opponent's
                    # option, not a draw the root can force: only a pruned
                    # move that ends the game drawn keeps a draw available.
                    pruned_root_draws = bool(root_forced_draw_actions(
                        board, legal_idx[~keep].tolist(),
                    ))
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
            elif search_legal.size == 1 and not pruned_root_draws:
                # Single-legal finish (classic). Pruning never leaves a single
                # survivor (above), so this is a genuinely forced root; the
                # flag check is kept as a guard.
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
            pruned_root_draws=pruned_root_draws,
        )
        return rid

    # --- chunk entry point ----------------------------------------------------

    def run(
        self,
        *,
        target_sims: int,
        stop_event: threading.Event,
        tb_probe: SyzygyProbe | None = None,
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
        ensure_usable(self)
        if self._abandoned:
            # A group was abandoned inside a device step: this pool's tree and
            # that group are gone for good. The caller retires the pool.
            k = min(self._abandoned)
            raise RPGDeviceFailure(self._labels[k], "stopped answering", "abandoned earlier")
        for evaluator in self._evals:
            ensure_usable(evaluator)
        st = self._state
        if st is None:
            raise RuntimeError("prepare_root() must be called before run()")
        if self._shutdown.is_set():
            raise RuntimeError("RootParallelGumbelPool is closed")
        st.tb_probe = tb_probe
        if st.finished_action is not None:
            assert st.finished_value is not None
            # No phase or group ran. Signal completion independently of the
            # requested budget; reporting that budget would invent nodes/nps.
            with self._stats_lock:
                self._last_stats = RootParallelGumbelStats(
                    target_sims=int(target_sims),
                    elapsed_seconds=0.0,
                    phases=(),
                    group_sims=tuple(0 for _ in range(self._cfg.n_groups)),
                    vloss_mode=self.realized_vloss_mode(),
                    tree_nodes=int(st.tree.node_count()),
                    root_finished=True,
                )
            return float(st.finished_value), int(st.finished_action)

        started = time.perf_counter()
        budget = max(1, int(target_sims))
        # The groups descend their candidate subtrees with the GIL released
        # while another group's expansion can grow the ONE shared arena: grow
        # it for the whole run now, before any group starts, as the other
        # shared-tree pools do (a no-op after the UCI search's own reserve).
        reserve_concurrent_headroom(st.tree, budget)
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
        self._error_groups = []
        job_stop = _JobStop(stop_event)
        self._job_stop = job_stop

        phases: list[RootParallelPhase] = []
        phase_index = 0

        # Phase A (multi-GPU open): fat first visits on ALL root candidates,
        # no halving. Opens live trees at gather-scale so every group has
        # real work before sequential-halving starts thinning the set.
        open_vpa = self._open_phase_vpa(len(remaining), budget)
        if open_vpa > 0 and remaining:
            n_cands = len(remaining)
            items = [
                _WorkItem(self._arena_for(st, a), open_vpa, phase_index)
                for a in remaining
            ]
            sims_done = self._dispatch_phase(items)
            self._raise_group_failure()
            self._resolve_root_solved(st)
            stopped = stop_event.is_set()
            budget = max(0, budget - open_vpa * n_cands)
            # Keep every candidate — open is scout, not a SH round.
            phases.append(RootParallelPhase(
                index=phase_index,
                candidates=n_cands,
                visits_per_action=open_vpa,
                sims_completed=sims_done,
                survivor_actions=tuple(remaining),
                kind="open",
            ))
            if self._info_cb is not None and uci_debug_enabled():
                self._info_cb(
                    f"rpg phase={phase_index} open cands={n_cands} "
                    f"vpa={open_vpa} sims={sims_done} "
                    f"survivors={len(remaining)} budget_left={budget} "
                    f"vloss_mode={self.realized_vloss_mode()} "
                    f"tree_nodes={st.tree.node_count()}"
                    f"{' STOPPED' if stopped else ''}"
                )
            phase_index += 1
            if stopped:
                best = remaining[0] if remaining else first_candidate
                value = self._final_value(st, best)
                with self._stats_lock:
                    self._last_stats = RootParallelGumbelStats(
                        target_sims=int(target_sims),
                        elapsed_seconds=time.perf_counter() - started,
                        phases=tuple(phases),
                        group_sims=tuple(self._group_sims),
                        vloss_mode=self.realized_vloss_mode(),
                        tree_nodes=int(st.tree.node_count()),
                    )
                return value, int(best)

        # Phase B: sequential-halving on remaining budget. min_vpa keeps
        # per-candidate jobs fat; min_keep delays collapse below n_groups.
        while budget > 0 and remaining:
            n_cands = len(remaining)
            # Exhaust a short remainder with an ordered partial phase.
            n_dispatched = min(n_cands, budget)
            vpa = self._phase_vpa(n_dispatched, budget)
            items = [
                _WorkItem(self._arena_for(st, a), vpa, phase_index)
                for a in remaining[:n_dispatched]
            ]
            sims_done = self._dispatch_phase(items)
            self._raise_group_failure()
            self._resolve_root_solved(st)
            stopped = stop_event.is_set()
            remaining = self._score_and_halve(st, remaining, gumbels)
            budget -= vpa * n_dispatched
            phases.append(RootParallelPhase(
                index=phase_index,
                candidates=n_cands,
                visits_per_action=vpa,
                sims_completed=sims_done,
                survivor_actions=tuple(remaining),
                kind="halve",
            ))
            # Per-phase diagnostics: UCI `debug on` only (otherwise every move
            # printed several of these). Keep the string compact.
            if self._info_cb is not None and uci_debug_enabled():
                self._info_cb(
                    f"rpg phase={phase_index} cands={n_cands} "
                    f"dispatched={n_dispatched} vpa={vpa} "
                    f"sims={sims_done} survivors={len(remaining)} "
                    f"budget_left={budget} "
                    f"vloss_mode={self.realized_vloss_mode()} "
                    f"tree_nodes={st.tree.node_count()}"
                    f"{' STOPPED' if stopped else ''}"
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
                vloss_mode=self.realized_vloss_mode(),
                tree_nodes=int(st.tree.node_count()),
            )
        return value, int(best)

    # --- orchestrator internals -------------------------------------------------

    def _open_phase_vpa(self, n_cands: int, budget: int) -> int:
        """Per-candidate sims for the multi-GPU open scout (0 = skip).

        Caps by remaining budget and ``open_budget_frac`` so a short chunk
        cannot become entirely open.
        """
        target = int(self._cfg.open_vpa)
        if (
            target <= 0
            or int(self._cfg.n_groups) <= 1
            or n_cands <= 0
            or int(budget) <= 0
        ):
            return 0
        max_by_budget = max(0, int(budget) // int(n_cands))
        frac = float(self._cfg.open_budget_frac)
        if 0.0 < frac < 1.0:
            max_by_frac = max(0, int(float(budget) * frac) // int(n_cands))
        else:
            max_by_frac = max_by_budget
        return int(max(0, min(target, max_by_budget, max_by_frac)))

    def _phase_vpa(self, n_cands: int, budget: int) -> int:
        """Classic sequential-halving vpa, optionally floored for multi-GPU."""
        classic = int(
            halving_visits_per_action(n_cands, budget, self._gcfg.halving_div)
        )
        floor = int(self._cfg.min_vpa)
        if floor <= 0 or int(self._cfg.n_groups) <= 1 or n_cands <= 0:
            return classic
        # Never overspend the remaining budget on this phase alone.
        max_vpa = max(1, int(budget) // max(1, int(n_cands)))
        return max(classic, min(floor, max_vpa))

    def _halving_keep(self, n_cands: int) -> int:
        """Classic keep count with optional multi-GPU survivor floor."""
        classic = int(halving_keep_count(n_cands, self._gcfg.halving_div))
        floor = int(self._cfg.min_keep)
        if floor <= 0 or n_cands <= 1:
            return classic
        # Cannot keep everyone (must eliminate at least one) or more than n.
        return max(classic, min(floor, int(n_cands) - 1))

    def _arena_for(self, st: _SearchState, action: int) -> _CandidateArena:
        arena = st.arenas.get(action)
        if arena is None:
            cid = int(st.tree.find_child(st.root_id, int(action)))
            assert cid >= 0, f"candidate action {action} missing at the root"
            arena = _CandidateArena(action=int(action), cid=cid)
            st.arenas[action] = arena
        return arena

    def _shard_phase_items(self, items: list[_WorkItem]) -> list[_WorkItem]:
        """When candidates < groups, split large survivor budgets so idle
        GPUs stay busy (design §2 late-phase split).

        Total phase sims are preserved (shards sum to original budgets).
        Tiny budgets stay unsharded — splitting a 20-sim job into 2×10 just
        adds queue overhead without filling gather.
        """
        n_groups = max(1, int(self._cfg.n_groups))
        if (
            not bool(self._cfg.split_idle_groups)
            or len(items) >= n_groups
            or not items
        ):
            return items
        min_shard = max(1, int(self._cfg.gather) // 2)
        total = sum(max(0, int(it.budget)) for it in items)
        if total < n_groups * min_shard:
            return items

        # How many shards per original item: spread work so we produce at
        # least n_groups tasks, preferring more shards on larger budgets.
        shards_out: list[_WorkItem] = []
        # First pass: proportional shard counts that sum to >= n_groups.
        budgets = [max(0, int(it.budget)) for it in items]
        n_items = len(items)
        # At least 1 shard each, then distribute extra among largest.
        counts = [1] * n_items
        extra = n_groups - n_items
        order = sorted(range(n_items), key=lambda i: -budgets[i])
        i_ord = 0
        while extra > 0 and order:
            idx = order[i_ord % len(order)]
            # Don't create shards smaller than min_shard when avoidable.
            if budgets[idx] // (counts[idx] + 1) < min_shard and counts[idx] >= 1:
                # Skip this item if every remaining item is also too small.
                if all(
                    budgets[j] // (counts[j] + 1) < min_shard
                    for j in order
                ):
                    break
                i_ord += 1
                continue
            counts[idx] += 1
            extra -= 1
            i_ord += 1

        for it, n_shards, b in zip(items, counts, budgets):
            if n_shards <= 1 or b <= 0:
                shards_out.append(it)
                continue
            base = b // n_shards
            rem = b - base * n_shards
            for s in range(n_shards):
                piece = base + (1 if s < rem else 0)
                if piece <= 0:
                    continue
                shards_out.append(_WorkItem(
                    arena=it.arena,
                    budget=piece,
                    phase_index=it.phase_index,
                    shared=True,
                ))
        return shards_out if shards_out else items

    def _dispatch_phase(self, items: list[_WorkItem]) -> int:
        work = self._shard_phase_items(items)
        with self._stats_lock:
            self._phase_sims = 0
        with self._pending_lock:
            self._pending = len(work)
        self._phase_done.clear()
        for it in work:
            self._work_q.put(it)
        while not self._phase_done.wait(_PHASE_POLL_S):
            if self._shutdown.is_set():
                break
            stuck = self._find_stuck_group()
            if stuck is not None:
                self._abandon_group(*stuck)
                break
        with self._stats_lock:
            return self._phase_sims

    def _stall_limit_s(self, k: int, *, stopped: bool, idle_other: bool) -> float:
        from chess_anti_engine.uci.device_guard import (
            idle_survivor_threshold_s,
            mid_search_threshold_s,
        )
        slowest = self._slowest_step_s[k]
        if stopped or idle_other:
            return idle_survivor_threshold_s(slowest)
        return mid_search_threshold_s(slowest)

    def _find_stuck_group(self) -> tuple[int, float, float] | None:
        """A group inside one device-bound step for too long, or None.

        The pool's rule: while another group still works the patient
        mid-search threshold (>= 5 s); once the job is stopped or another
        group sits idle, the 1 s idle-survivor threshold."""
        job = self._job_stop
        stopped = job is not None and job.is_set()
        now = time.monotonic()
        with self._watch_lock:
            busy = list(self._busy_since)
            in_item = list(self._in_item)
        idle_other = any(not x for k, x in enumerate(in_item) if k not in self._abandoned)
        for k, since in enumerate(busy):
            if since is None or k in self._abandoned:
                continue
            limit = self._stall_limit_s(k, stopped=stopped, idle_other=idle_other)
            if now - since > limit:
                return k, now - since, limit
        return None

    def _abandon_group(self, k: int, took: float, limit: float) -> None:
        """Give up on group ``k`` stuck in a device step; let the others stop."""
        with self._watch_lock:
            self._abandoned.add(k)
            if self._busy_since[k] is not None:
                wedge.mark_wedged(self._threads[k], f"RPG device {self._labels[k]} step",
                                  device=self._labels[k])
        self._errors.append(RPGDeviceFailure(
            self._labels[k], "stopped answering",
            f"one device step blocked for {took:.1f}s > {limit:.1f}s",
        ))
        self._error_groups.append(k)
        job = self._job_stop
        if job is not None:
            job.set()
        # The stuck group's own item never completes; wait (bounded) for the
        # others to drain theirs before the caller drops this pool's tree.
        deadline = time.monotonic() + _FAILURE_DRAIN_S
        while time.monotonic() < deadline:
            with self._pending_lock:
                if self._pending <= 1:
                    return
            time.sleep(_PHASE_POLL_S / 5)

    def _raise_group_failure(self) -> None:
        """Re-raise a group's failure. A group that RAISED re-raises its own
        exception (the pool's public contract); ``last_group_failure`` then
        names the device. A stuck group raises ``RPGDeviceFailure``."""
        if not self._errors:
            return
        first = self._errors[0]
        if isinstance(first, RPGDeviceFailure):
            self._last_failure = first
            raise first
        k = self._error_groups[0] if self._error_groups else -1
        label = self._labels[k] if 0 <= k < len(self._labels) else "?"
        text = " ".join(f"{type(first).__name__}: {first}".split())[:160]
        self._last_failure = RPGDeviceFailure(label, "failed mid-search", text)
        raise first

    def _run_rows_job(self, idx: int, evaluator: Any, job: _RowsJob) -> None:
        if idx in job.exclude or idx in self._abandoned or job.done.is_set():
            if not job.done.is_set():
                self._work_q.put(job)  # another group's job: hand it back
                time.sleep(_PHASE_POLL_S / 10)
            return
        with self._watch_lock:
            if job.taken_by >= 0:
                return
            job.taken_by = idx
            job.since = time.monotonic()
        self._device_step_begin(idx)
        try:
            ensure_usable(evaluator)
            if job.relations is not None:
                out = evaluator.evaluate_encoded(job.xs, relations=job.relations)
            else:
                out = evaluator.evaluate_encoded(job.xs)
            job.result = (np.asarray(out[0]), np.asarray(out[1]))
        except BaseException as exc:
            job.error = exc
        finally:
            self._device_step_end(idx)
            job.done.set()

    def usable_groups(self) -> list[int]:
        """Groups that neither raised nor were abandoned (rows can run there)."""
        bad = set(self._abandoned) | set(self._error_groups) | set(self._rows_failed)
        return [k for k in range(len(self._threads))
                if k not in bad and self._threads[k].is_alive()]

    @property
    def n_devices(self) -> int:
        return len(self.usable_groups())

    def worker_row_cap(self) -> int:
        """Rows ``evaluate_rows`` may submit: the gather every group warmed."""
        return max(1, int(self._cfg.gather))

    def evaluate_rows(
        self, xs: np.ndarray, relations: np.ndarray | None = None, *,
        idle_limit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``xs`` on a usable group's own evaluator and thread.

        Serves the classic path after the pool is retired from search when
        the primary's device is the one that failed (``RPGRowsEvaluator``),
        and root evaluations once the primary evaluator failed. A group that
        raises is set aside and the next one takes the batch; one stuck past
        the stall rule (``idle_limit``: the 1 s idle-survivor threshold) is
        abandoned (and marked wedged) the same way. Raises when no group is
        left."""
        from chess_anti_engine.uci.device_guard import (
            idle_survivor_threshold_s,
            mid_search_threshold_s,
        )
        if self._shutdown.is_set():
            raise RuntimeError("root-parallel Gumbel pool is closed")
        last: BaseException | None = None
        tried: set[int] = set()
        while True:
            usable = [k for k in self.usable_groups() if k not in tried]
            if not usable:
                break
            excluded = frozenset(k for k in range(len(self._threads)) if k not in usable)
            job = _RowsJob(xs=xs, relations=relations, exclude=excluded)
            self._work_q.put(job)
            stuck_k = -1
            while not job.done.wait(_PHASE_POLL_S):
                with self._watch_lock:
                    k = job.taken_by
                    since = job.since
                if k < 0:
                    if not any(self._threads[j].is_alive() for j in usable):
                        break
                    continue
                slowest = self._slowest_step_s[k]
                limit = (idle_survivor_threshold_s(slowest) if idle_limit and len(usable) > 1
                         else mid_search_threshold_s(slowest))
                if time.monotonic() - since > limit:
                    stuck_k = k
                    break
            if stuck_k >= 0:
                with self._watch_lock:
                    self._abandoned.add(stuck_k)
                wedge.mark_wedged(self._threads[stuck_k],
                                  f"RPG device {self._labels[stuck_k]} rows",
                                  device=self._labels[stuck_k])
                self._note_rows_failure(stuck_k, "stopped answering",
                                        "an evaluation blocked past the stall limit")
                tried.add(stuck_k)
                last = TimeoutError(f"device {self._labels[stuck_k]} evaluation stuck")
                continue
            if not job.done.is_set():
                break
            if job.error is not None:
                k = job.taken_by
                tried.add(k)
                self._rows_failed.add(k)
                self._note_rows_failure(k, "failed", " ".join(
                    f"{type(job.error).__name__}: {job.error}".split())[:160])
                last = job.error
                continue
            assert job.result is not None
            return job.result
        raise RuntimeError(
            "multi-GPU: every device failed (root-parallel Gumbel has no usable group left)"
        ) from last

    def _note_rows_failure(self, k: int, what: str, reason: str) -> None:
        wedge.note_device_fault(f"RPG device {self._labels[k]} {what}")
        msg = (f"WARNING multi-GPU DEGRADED: root-parallel Gumbel device {self._labels[k]} "
               f"{what} ({reason}); {self.n_devices} usable device(s) left")
        _log.error("%s", msg)
        with self._watch_lock:
            self._rows_notes.append(msg)

    def drain_rows_notes(self) -> list[str]:
        with self._watch_lock:
            notes, self._rows_notes = self._rows_notes, []
        return notes

    def label(self, k: int) -> str:
        return self._labels[k]

    def last_group_failure(self) -> RPGDeviceFailure | None:
        """The device failure behind the last ``run`` that raised, if any."""
        return self._last_failure

    def _device_step_begin(self, k: int) -> None:
        with self._watch_lock:
            self._busy_since[k] = time.monotonic()

    def _device_step_end(self, k: int) -> None:
        with self._watch_lock:
            since = self._busy_since[k]
            self._busy_since[k] = None
            if k in self._abandoned:
                wedge.clear_wedged(threading.current_thread())  # the step came back
            if since is not None:
                took = time.monotonic() - since
                if took > self._slowest_step_s[k]:
                    self._slowest_step_s[k] = took

    def _score_and_halve(
        self,
        st: _SearchState,
        remaining: list[int],
        gumbels: dict[int, float],
    ) -> list[int]:
        """Mirror of ``gss_score_and_halve``.

        The completed-Q transform uses the FRESH network root value as its
        raw-value baseline, matching the classic C path and the Python reference.
        """
        if len(remaining) <= 1:
            return remaining
        actions, visits, qs = st.tree.get_children_q(st.root_id, 0.0)
        visits_f = visits.astype(np.float64)
        visited = visits_f > 0.0
        n_total = float(visits_f.sum())
        # mctx completed-by-mix-value takes the FRESH network root value as
        # raw_value. Keep it fixed across halving rounds; child visits affect
        # weighted_q below, not the baseline that completes unvisited actions.
        root_q = float(st.root_q_init)
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
        keep = self._halving_keep(len(remaining))
        return [remaining[i] for i in order[:keep]]

    def _resolve_root_solved(self, st: _SearchState) -> int:
        """Resolve the REAL root's solved status from its children's.

        Each candidate's chunker descends from the candidate node, so a proof
        found inside a candidate subtree propagates up to that candidate and
        no further: the C propagation walks only the descent path. Without
        this the root stays unsolved forever, and every consumer of the root's
        status (``go mate``'s proven-mate and ``mate_unreachable`` stops, the
        solved-root ``go depth`` stop, ``_final_value``) is unreachable on this
        path. Same rule as ``tree_resolve_from_children`` in ``_mcts_tree.c``:
        one child lost for its mover makes the root won; otherwise every child
        must be proven, and any drawn child makes it a draw. Statuses are
        monotonic, so reading them while no phase runs is exact. Returns the
        root's status after resolution."""
        tree, root_id = st.tree, st.root_id
        status = int(tree.get_solved_status(root_id))
        if status != 0 or st.search_legal.size == 0:
            return status
        has_draw = st.pruned_root_draws
        all_proven = True
        for action in st.search_legal.tolist():
            cid = int(tree.find_child(root_id, int(action)))
            child = int(tree.get_solved_status(cid)) if cid >= 0 else 0
            if child == -1:
                status = 1  # one reply lost for the opponent suffices
                break
            if child == 0:
                all_proven = False  # keep scanning: a later child may win
            elif child == 2:
                has_draw = True
        else:
            if not all_proven:
                return 0
            status = 2 if has_draw else -1
        # A one-node path marks the root itself; the root has no parent.
        tree.mark_solved_path(np.array([root_id], dtype=np.int32), status)
        return int(tree.get_solved_status(root_id))

    def _final_value(self, st: _SearchState, best: int) -> float:
        """Exact solved root value, otherwise classic survivor-child Q."""
        if st.tree.get_solved_status(st.root_id) != 0:
            return st.tree.node_q(st.root_id)
        actions, visits, qs = st.tree.get_children_q(st.root_id, st.root_q_init)
        mask = actions == int(best)
        if bool(mask.any()) and int(visits[mask][0]) > 0:
            return float(qs[mask][0])
        return float(st.root_q_init)

    # --- ownership ---------------------------------------------------------------

    def _claim(self, action: int, group: int, phase: int, *, shared: bool) -> None:
        with self._owner_lock:
            if shared:
                assert action not in self._owned, (
                    f"shared claim on exclusively-owned candidate {action} "
                    f"(owner={self._owned[action]})"
                )
                owners = self._owned_shared.setdefault(action, set())
                assert group not in owners, (
                    f"group {group} double-claimed shared candidate {action}"
                )
                owners.add(group)
            else:
                assert action not in self._owned_shared, (
                    f"exclusive claim on shared candidate {action}"
                )
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

    def _release(self, action: int, group: int, *, shared: bool) -> None:
        with self._owner_lock:
            if shared:
                owners = self._owned_shared.get(action)
                if owners is not None:
                    owners.discard(group)
                    if not owners:
                        self._owned_shared.pop(action, None)
            else:
                self._owned.pop(action, None)
        if self.touch_hook is not None:
            self.touch_hook("release", action, group)

    # --- group worker ---------------------------------------------------------------

    def _worker_loop(self, idx: int) -> None:
        evaluator: Any = None
        try:
            # Ordinary threads do not inherit Inductor cudagraph TLS.
            try:
                from chess_anti_engine.inference_dispatcher import (
                    bootstrap_cudagraph_tls,
                )
                bootstrap_cudagraph_tls()
            except Exception:
                pass
            if self._factories is not None:
                if self._devices is not None:
                    device = self._devices[idx]
                    if device.startswith("cuda"):
                        # cudagraph-TLS trap: bind the device on THIS thread
                        # before the factory compiles/captures anything.
                        import torch
                        torch.cuda.set_device(device)
                earlier = wedge.wedged_earlier(self._labels[idx])
                if earlier is not None:
                    # Never build on a device that wedged earlier this session.
                    raise RuntimeError(
                        f"root-parallel Gumbel device {self._labels[idx]} left out: "
                        f"it wedged earlier this session ({earlier})"
                    )
                with startup_context(self._startup_ctx[idx]):
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
                cpuct_factor=float(cfg.cpuct_factor),
                cpuct_base=float(cfg.cpuct_base),
                vloss_weight=cfg.vloss_weight,
                vloss_mode=cfg.vloss_mode,
                eval_cache_entries=int(cfg.eval_cache_entries),
                input_planes=cfg.input_planes,
                compute_relations=cfg.compute_relations,
                input_history_encoding=self._gcfg.input_history_encoding,
            )
        except BaseException as exc:
            wedge.note_device_fault(f"RPG device {self._labels[idx]} failed at startup")
            self._init_errors.append(exc)
        wedge.clear_wedged(threading.current_thread())  # a timed-out build came back
        self._built[idx].set()
        try:
            self._init_done.wait()
        except threading.BrokenBarrierError:
            pass  # Constructor startup failure publishes _init_errors first.
        if self._init_errors:
            # Shutdown can follow a successful transfer to the caller; only
            # published startup errors leave these helpers owned by the group.
            # No constructor transfer occurred: every factory-created helper
            # is still owned by its group, including successful startup peers.
            self._chunkers[idx] = None
            if self._factories is not None:
                self._evals[idx] = None
                try:
                    close = getattr(evaluator, "close", None)
                    if callable(close):
                        close()
                except BaseException as cleanup:
                    note_cleanup(self._init_errors[0], f"startup evaluator cleanup failed: {cleanup!r}")
                evaluator = None
            return

        chunker = self._chunkers[idx]
        assert chunker is not None
        while not self._shutdown.is_set():
            item = self._work_q.get()
            if item is None or self._shutdown.is_set():
                return
            if isinstance(item, _RowsJob):
                self._run_rows_job(idx, evaluator, item)
                continue
            stop_event = self._job_stop
            with self._watch_lock:
                self._in_item[idx] = True
            try:
                sims = self._run_item(idx, evaluator, chunker, item, stop_event)
                with self._stats_lock:
                    self._group_sims[idx] += sims
                    self._phase_sims += sims
            except BaseException as exc:  # surfaced by run() after the barrier
                _log.exception("rpg-group-%d raised; requesting stop", idx)
                wedge.note_device_fault(f"RPG device {self._labels[idx]} raised")
                self._errors.append(exc)
                self._error_groups.append(idx)
                if stop_event is not None:
                    stop_event.set()
            finally:
                with self._watch_lock:
                    self._in_item[idx] = False
                    self._busy_since[idx] = None
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
        stop_event: _JobStop | threading.Event | None,
    ) -> int:
        if stop_event is not None and stop_event.is_set():
            return 0
        ensure_usable(evaluator)
        st = self._state
        assert st is not None
        arena = item.arena
        shared = bool(item.shared)
        self._claim(arena.action, group_idx, item.phase_index, shared=shared)
        try:
            done = 0
            # Expand exactly once even when split shards race the first touch.
            with arena.lock:
                if stop_event is not None and stop_event.is_set():
                    return 0
                if not arena.expanded:
                    self._device_step_begin(group_idx)
                    try:
                        done += self._expand_candidate(evaluator, arena, st)
                    finally:
                        self._device_step_end(group_idx)
            if arena.terminal_value is not None:
                # Classic semantics: every forced sim into a terminal child
                # backprops the terminal value (visit counts keep accruing so
                # schedule/scoring arithmetic matches the serial run).
                # Chunk + poll stop the same way non-terminal work does: a lone
                # survivor can receive the whole remaining phase budget, and a
                # single backprop_many([path]*n) would ignore stop_event and
                # allocate an O(n) Python list (hundreds of ms at 1M+ nodes).
                gather = max(1, int(self._cfg.gather))
                batch = max(1, 2 * gather)
                path = np.array([arena.cid], dtype=np.int32)
                tv = float(arena.terminal_value)
                while done < item.budget:
                    if stop_event is not None and stop_event.is_set():
                        break
                    n = min(item.budget - done, batch)
                    st.tree.backprop_many([path] * n, [tv] * n)
                    done += n
                return done
            chunker.set_policy_temp(float(self._gcfg.policy_temp))
            gather = max(1, int(self._cfg.gather))
            while done < item.budget:
                if stop_event is not None and stop_event.is_set():
                    break
                if self.touch_hook is not None:
                    self.touch_hook("touch", arena.action, group_idx)
                # 2x gather per call keeps the chunker's 2-slot CPU/GPU
                # overlap alive while bounding stop latency to ~two batches.
                # When remaining < gather the chunker already densifies to
                # remaining (min(gather, target)); no extra adaptive path.
                n = min(item.budget - done, 2 * gather)
                self._device_step_begin(group_idx)
                try:
                    if st.tb_probe is None:
                        ran = int(chunker.run(st.tree, arena.cid, arena.cboard, n))
                    else:
                        # No root prefix on the proof paths: a TB proof stops at
                        # the candidate and `_resolve_root_solved` decides the
                        # root, applying the pruned-terminal-draw rule.
                        ran = int(chunker.run(
                            st.tree, arena.cid, arena.cboard, n, tb_probe=st.tb_probe,
                        ))
                finally:
                    self._device_step_end(group_idx)
                if ran <= 0:
                    break  # solved/terminal below the candidate root
                done += ran
            return done
        finally:
            self._release(arena.action, group_idx, shared=shared)

    def _expand_candidate(
        self,
        evaluator: Any,
        arena: _CandidateArena,
        st: _SearchState,
    ) -> int:
        """First touch of a candidate: push its move, then either mark the
        terminal value or evaluate + expand the child node (this eval is the
        classic first forced sim, so it consumes 1 sim of the item budget)."""
        ensure_usable(self)
        ensure_usable(evaluator)
        cb = st.root_cb.copy()
        cb.push_index(int(arena.action))
        arena.cboard = cb
        if cb.is_game_over():
            arena.terminal_value = float(cb.terminal_value())
            # Record the terminal as a proven child (checkmate = mover lost,
            # every other terminal a draw: `cboard_terminal_solved_status`) so
            # `_resolve_root_solved` can count it.
            # Mark the candidate only: `_resolve_root_solved` decides the
            # root (it knows about pruned terminal draws; C propagation does not).
            st.tree.mark_solved_path(
                np.array([arena.cid], dtype=np.int32),
                -1 if arena.terminal_value < 0.0 else 2,
            )
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
        receipt = None
        try:
            if self._cfg.compute_relations:
                rel = cb.compute_relations()[None, ...]
                receipt = submit_async(evaluator,
                    1, slot=0, relations=rel,
                )
            else:
                receipt = submit_async(evaluator, 1, slot=0)
            pol_t, wdl_t, event = receipt
            if event is not None:
                event.synchronize()
        except BaseException as error:
            drain_failed_receipts(evaluator, self, [] if receipt is None else [receipt], error)
            raise
        pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
        wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
        # This forced first NN result is outside PucvChunker. Apply the same
        # post-NN/reset-clock rule here, without changing evaluator/cache data.
        snapshots = ((np.array([0], dtype=np.int32), [cb])
                     if st.tb_probe is not None else None)
        wdl, solved = override_leaf_values(st.tb_probe, snapshots, wdl)
        legal = cb.legal_move_indices()
        if legal.size > 0:
            st.tree.expand_from_logits(
                arena.cid,
                legal.astype(np.int32),
                np.ascontiguousarray(
                    _policy_logits_to_full(pol, cfg=self._gcfg)[0], dtype=np.float32,
                ),
            )
        q = float(_wdl_to_q(np.asarray(wdl, dtype=np.float32)[0].reshape(-1)))
        st.tree.backprop(np.array([arena.cid], dtype=np.int32), q)
        if solved is not None and solved[0]:
            status = int(solved[0])
            # Candidate only; `_resolve_root_solved` decides the root.
            st.tree.mark_solved_path(np.array([arena.cid], dtype=np.int32), status)
            # Proven TB values share the existing forced-terminal budget path.
            arena.terminal_value = 0.0 if status == 2 else float(status)
        arena.expanded = True
        return 1
