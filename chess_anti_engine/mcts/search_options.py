"""The search-parameter surface: one registry, shared by the UCI engine and
every offline tool that has to print "what the search actually used".

Why a registry instead of a list of ``setoption`` handlers:

This codebase's signature defect is a value that is accepted and then silently
ignored. A UCI option surface mass-produces it — ``c_puct`` and the ``fpu_*``
family were printed as realized Gumbel search configuration for months while
``GumbelConfig.full_tree=True`` made the PUCT descent they drive unreachable
(``INERT_GUMBEL_KNOBS``, play-path audit 2026-08-03 F2). An operator could
sweep them all day against a number that never moved.

So every option here carries the ONE fact a handler cannot infer: the set of
search paths it can actually reach, and — for the knobs whose reachability
depends on other knobs — a predicate that says so. ``inert_reason`` is the
only place that decides, and both the ``setoption`` warning and the
``searchconfig`` readback ask it. A knob that cannot take effect is still
accepted (a GUI must not see an error for a legal option), but it is never
silent and never reported as realized.

Search paths (``SearchWorker.realized_search_path()`` names the LIVE one — it
reads which pool is installed, not which option was requested, because
``UseVL=true`` against an evaluator without the slot API is accepted and then
falls through to classic Gumbel):

``gumbel``     classic single-thread Gumbel C path (Threads=1, no UseVL,
               no multi-GPU pool). The tuned play shape lives here.
``rpg``        root-parallel Gumbel over >=2 devices (SearchParallel=gumbel).
               Root decisions are Gumbel; the intra-candidate descent is PUCT.
``walker``     PUCT walker pool (Threads>1). Plain PUCT — no Gumbel knob.
``pucv``       single-thread batched-VL PUCT (UseVL=true). Plain PUCT.
``pucv_pool``  multi-GPU shared-tree PUCT pool (UseMultiGpuPUCV). Plain PUCT.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from chess_anti_engine.mcts.gumbel import (
    PLAY_PUCT_DEFAULTS,
    PLAY_SEARCH_DEFAULTS,
    PLAY_SEARCH_VLOSS_WEIGHT,
)

# Every search path the UCI engine can dispatch to. Ordered most-to-least
# Gumbel-shaped so a table printed in this order reads sensibly.
SEARCH_PATHS: tuple[str, ...] = ("gumbel", "rpg", "walker", "pucv", "pucv_pool")

# Human-facing one-liners, used verbatim in the info-string warnings so an
# operator reading a match log knows which engine mode swallowed their knob.
PATH_DESCRIPTIONS: Mapping[str, str] = {
    "gumbel": "classic single-thread Gumbel (Threads=1, UseVL off, 1 device)",
    "rpg": "root-parallel Gumbel (SearchParallel=gumbel, >=2 devices)",
    "walker": "PUCT walker pool (Threads>1)",
    "pucv": "single-thread batched-VL PUCT (UseVL=true)",
    "pucv_pool": "multi-GPU PUCT pool (UseMultiGpuPUCV=true)",
}

_GUMBEL_PATHS = frozenset({"gumbel", "rpg"})
_PUCT_PATHS = frozenset({"walker", "pucv", "pucv_pool", "rpg"})
_ALL_PATHS = frozenset(SEARCH_PATHS)


@dataclass(frozen=True)
class SearchOption:
    """One UCI-visible search parameter.

    ``field`` is the key ``SearchWorker.realized_search_values()`` reports it
    under — a ``GumbelConfig`` field name for the search-shape knobs, or one of
    the worker-level pseudo-fields (``chunk_sims`` / ``vloss_weight`` /
    ``minibatch_size`` / ``root_noise_scale``) for the knobs that are not
    ``GumbelConfig`` fields at all. The C-path pair ``vloss_weight`` /
    ``target_batch`` are function arguments of ``run_gumbel_root_many_c``, which
    is exactly how they went missing from every arena before 2026-07-28; they
    get real entries here rather than being left to a ``replace``-based surface
    that structurally cannot reach them.

    ``kind`` follows lc0: ``spin`` for ints, ``check`` for bools, ``string``
    for floats (UCI has no float type), ``combo`` for enums.

    ``live_in`` is the set of search paths on which the option is capable of
    changing the search at all. ``inert_reason`` narrows that further for the
    knobs whose reachability depends on OTHER knobs' values.
    """

    name: str
    kind: str
    field: str
    default: float | int | bool
    live_in: frozenset[str]
    lo: float | None = None
    hi: float | None = None
    doc: str = ""

    @property
    def lower(self) -> str:
        return self.name.lower()

    def declaration(self, current: float | int | bool) -> str:
        """The ``option name ...`` handshake line, defaulted to ``current``.

        Taking the CURRENT value rather than ``self.default`` is deliberate:
        ``--c-scale`` and friends can move the startup value, and a handshake
        that advertises a default the engine is not running is the same lie in
        a smaller font.

        The guard is ``test_the_printed_handshake_advertises_the_live_worker``
        in ``tests/test_uci_search_options.py``, which parses the PRINTED
        ``option name`` lines. An earlier revision of this docstring named a
        test asserting the same thing; that test compared
        ``EngineOptions.search_value`` against the worker and never called
        ``emit_handshake``, so reverting this line to ``self.default`` passed
        the whole file. A comment naming a guard that does not exist is worse
        than no comment — the printed line is the artifact an operator reads,
        so the test reads the printed line.
        """
        if self.kind == "check":
            return f"option name {self.name} type check default {'true' if current else 'false'}"
        if self.kind == "spin":
            return (
                f"option name {self.name} type spin default {int(current)} "
                f"min {int(self.lo or 0)} max {int(self.hi or 0)}"
            )
        # Floats ride `type string` (lc0's PolicyTemperature convention); the
        # range is advertised in the docs and enforced by the handler.
        return f"option name {self.name} type string default {_fmt(current)}"


def _fmt(v: float | int | bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return repr(float(v))


_PLAY = PLAY_SEARCH_DEFAULTS
_PUCT = PLAY_PUCT_DEFAULTS

# The default chunk budget, mirrored from `uci/__main__.py --chunk-sims`.
DEFAULT_CHUNK_SIMS = 2048

SEARCH_OPTIONS: tuple[SearchOption, ...] = (
    # ── Gumbel search shape ────────────────────────────────────────────────
    SearchOption(
        "PolicyTemperature", "string", "policy_temp", float(_PLAY["policy_temp"]),
        _GUMBEL_PATHS, 0.5, 5.0,
        "Prior temperature on the policy logits before they seed the tree "
        "(logits/T). T>1 softens, T<1 sharpens, 1.0 = no-op. != 1.0 disables "
        "the compact-legal bf16 leaf transport (the C bf16 leaf softmax has no "
        "temperature hook), so leaves fall back to dense float32 4672-wide. "
        "The ~1.9x long quoted for that was measured on the DIRECT evaluator, "
        "which has no bf16 transport to lose, and did NOT reproduce on the "
        "broker path (0.87-1.01x, inside the instrument's noise) — measure on "
        "your own transport rather than budgeting a slowdown.",
    ),
    SearchOption(
        "CScale", "string", "c_scale", float(_PLAY["c_scale"]),
        _GUMBEL_PATHS, 1e-4, 100.0,
        "Gumbel value-transform scale in sigma(q) for the DESCENT: lower leans "
        "on the prior, higher trusts search Q. On rpg the descent is PUCT, so "
        "this is only read as the root fallback when CScaleRoot < 0.",
    ),
    SearchOption(
        "CVisit", "string", "c_visit", float(_PLAY["c_visit"]),
        _GUMBEL_PATHS, 0.0, 100000.0,
        "Gumbel descent value-transform floor constant. On rpg, only the root "
        "fallback when CVisitRoot < 0.",
    ),
    SearchOption(
        "CScaleRoot", "string", "c_scale_root", float(_PLAY["c_scale_root"]),
        _GUMBEL_PATHS, -1.0, 1000.0,
        "ROOT-only c_scale (descent keeps CScale). Pairs with "
        "QVisitExpRoot<0 for the log root transform. <0 = use CScale at the "
        "root too (legacy linear).",
    ),
    SearchOption(
        "CVisitRoot", "string", "c_visit_root", float(_PLAY["c_visit_root"]),
        _GUMBEL_PATHS, -1.0, 100000.0,
        "ROOT-only c_visit (descent keeps CVisit). <0 = use CVisit at both "
        "sites (legacy, no split).",
    ),
  # ⚑ QVisitExp / QVisitFloor / QGlobalScale were REMOVED from this registry
  # along with the GumbelConfig fields behind them (never promoted; no config in
  # this repo set one). A GUI that still sends `setoption name QVisitExp ...`
  # now gets the registry's normal unknown-option handling instead of an option
  # that pretended to tune a descent transform nobody had measured.
    SearchOption(
        "QVisitExpRoot", "string", "q_visit_exp_root", float(_PLAY["q_visit_exp_root"]),
        _GUMBEL_PATHS, -10.0, 99.0,
        "ROOT-only value-transform exponent. <0 = LOG root (sim-invariant "
        "q_scale from 256 to millions of nodes). >=90 = LINEAR root "
        "(exponent 1.0), which is what selfplay runs.",
    ),
    SearchOption(
        "HalvingDiv", "spin", "halving_div", 2,
        _GUMBEL_PATHS, 2, 8,
        "Sequential-halving divisor: each round keeps ceil(n_cands/div). "
        "2 = standard halving.",
    ),
    SearchOption(
        "Topk", "spin", "topk", int(_PLAY["topk"]),
        _GUMBEL_PATHS, 2, 256,
        "Gumbel root candidates m = min(Topk, n_legal, max(2,(budget+1)//2)). "
        "The budget cap means Topk stops binding once it exceeds half the "
        "chunk budget -- a Topk sweep at a small ChunkSims measures nothing.",
    ),
    SearchOption(
        "GumbelScale", "string", "root_noise_scale", 0.0,
        _GUMBEL_PATHS, 0.0, 5.0,
        "Root Gumbel-noise scale for MATCH play. 0.0 (default) = deterministic "
        "search, which is what every prior UCI build did: the worker forced "
        "add_noise=False and GumbelConfig.gumbel_scale was unreachable. >0 "
        "re-enables root noise at that strength for opening diversity.",
    ),
    # ── budget / batching ──────────────────────────────────────────────────
    SearchOption(
        "ChunkSims", "spin", "chunk_sims", DEFAULT_CHUNK_SIMS,
        _ALL_PATHS, 32, 1048576,
        "Sims per search chunk. Higher = fuller GPU batches (more nps) at the "
        "cost of coarser stop/abort latency, and it changes the search: the "
        "Gumbel halving schedule and the Topk budget cap are both per-chunk.",
    ),
    SearchOption(
        "VLossWeight", "spin", "vloss_weight", int(PLAY_SEARCH_VLOSS_WEIGHT),
        _ALL_PATHS, 0, 64,
        "Virtual loss applied to in-flight leaves. 0 reproduces the pre-C17 "
        "duplicate-leaf search, which cost 7.4cp; every arena Elo recorded "
        "before 2026-07-28 was measured at 0 because no override surface "
        "reached it.",
    ),
    SearchOption(
        "MinibatchSize", "spin", "minibatch_size", 0,
        frozenset({"gumbel"}), 0, 8192,
        "Leaves the C Gumbel state machine accumulates before flushing to the "
        "GPU. 0 = C-side default (GSS_GPU_BATCH). Passed as "
        "run_gumbel_root_many_c(target_batch=...), so it exists only on the "
        "classic Gumbel path.",
    ),
    # ── PUCT descent ───────────────────────────────────────────────────────
    SearchOption(
        "CPuct", "string", "c_puct", float(_PUCT["c_puct"]),
        _PUCT_PATHS, 1e-4, 100.0,
        "PUCT exploration constant. With CPuctFactor>0 this is the init term "
        "in c(N)=CPuct+CPuctFactor*log((N+CPuctBase)/CPuctBase).",
    ),
    SearchOption(
        "CPuctFactor", "string", "cpuct_factor", float(_PUCT["cpuct_factor"]),
        _PUCT_PATHS, 0.0, 100.0,
        "Lc0 CPuctFactor. 0 = fixed CPuct (no log ramp).",
    ),
    SearchOption(
        "CPuctBase", "string", "cpuct_base", float(_PUCT["cpuct_base"]),
        _PUCT_PATHS, 1.0, 1e9,
        "Lc0 CPuctBase in log((N+base)/base).",
    ),
    SearchOption(
        "FpuReduction", "string", "fpu_reduction", float(_PUCT["fpu_reduction"]),
        _PUCT_PATHS, -10.0, 10.0,
        "First-play-urgency reduction subtracted from an unvisited child's "
        "assumed value in the PUCT descent.",
    ),
)

OPTIONS_BY_NAME: Mapping[str, SearchOption] = {o.lower: o for o in SEARCH_OPTIONS}


def branch_note(
    option: SearchOption, path: str, values: Mapping[str, object],
) -> str | None:
    """Why the CURRENT value cannot be varied without leaving its branch.

    **This is not inertness, and the difference is load-bearing.** A knob with a
    branch note DOES reach the search and DOES change it — but only by crossing
    a boundary, not by moving within the branch it currently sits in. Conflating
    the two produced a real defect: `inert_reason` answered "are all values
    inside this arm equivalent?" while `Engine._set_search_option` used the
    answer to assert "this setoption had no effect", so at `CScaleRoot=0.05`
    moving `QVisitExpRoot` from -1.0 to 1.0 printed `NO EFFECT on the live
    search` while the best root action went 306 -> 553 and the tree 8037 ->
    7764. The mirror printed "Only crossing to >= 0 can change anything" having
    just crossed from >= 0.

    Every arm here is **exact and read off the C source** — no thresholds, no
    saturation heuristic, nothing fitted to a sweep. The previous attempt used a
    `q_scale`-vs-prior-span bound and claimed it "under-fires rather than
    over-fires"; that claim was false. Two root transforms differ only by the
    scalar `q_scale` (the normalized `u = (q-min)/(max-min)` is identical), so a
    pair of candidates flips whenever `-dlog_prior/du` falls between the two
    `q_scale` values. `du` can be arbitrarily small on near-tied Q, so no
    readback-time bound can rule a flip out. The heuristic was removed rather
    than retuned.
    """
    if path not in option.live_in or option.field != "q_visit_exp_root":
        return None
    raw = _num(values, "q_visit_exp_root")
    if raw >= 90.0:
        return (
            "every value in [90, 99] is the same search: >= 90 is the LINEAR "
            "root sentinel -- the root exponent resolves to 1.0 "
            "(_mcts_tree.c:3947, which reads the literal 1.0 gumbel_c now "
            "passes for the deleted q_visit_exp). Setting it below 90 gives "
            "the root its own exponent and DOES change the search"
        )
    if raw < 0.0:
        return (
            "every value < 0 is the same search: the LOG root transform is "
            "q_scale = CScaleRoot*log1p(CVisitRoot+max_visit) "
            "(_mcts_tree.c:1498), which does not contain the exponent at all — "
            "only its SIGN chose the branch. Crossing to >= 0 selects the power "
            "branch and CAN change the search"
        )
    return (
        "the power branch: the exponent enters as max_visit**exp ADDED to "
        "CVisitRoot (_mcts_tree.c:1500-1504), so at CVisitRoot >> max_visit**exp "
        "it moves q_scale very little. Crossing to < 0 selects the log branch "
        "and CAN change the search"
    )


def inert_reason(option: SearchOption, path: str, values: Mapping[str, object]) -> str | None:
    """Why ``option`` cannot affect a search on ``path``, or None if it can.

    ``values`` is a realized-value mapping keyed by ``SearchOption.field`` —
    normally ``SearchWorker.realized_search_values()``. The conditional arms
    below are the knobs whose reachability depends on ANOTHER knob's value, so
    a static per-path table would be wrong in exactly the direction that
    matters (it would call a dead knob live).

    Every arm is a fact about what the C **reads**, so a non-None answer means
    NO value of the option changes the search. That is the question
    `Engine._set_search_option` asks before printing `NO EFFECT`, and it is why
    "all values within the current branch are equivalent" belongs in
    `branch_note` instead — see its docstring for the defect that came of
    answering the wrong one.
    """
    if path not in option.live_in:
        return (
            f"{option.name} does not reach the {path} path "
            f"({PATH_DESCRIPTIONS.get(path, path)})"
        )
    if path != "rpg":
        return None
    # Root-parallel Gumbel runs Gumbel ONLY at the root: the intra-candidate
    # descent is a PUCT chunker. So the descent-site transform knobs are read
    # only through `_root_sigma_scale`'s fallback, i.e. only while the
    # corresponding root-site knob is at its "unset" sentinel.
    if option.field == "c_scale" and _num(values, "c_scale_root") >= 0.0:
        return "rpg reads c_scale only as the root fallback, and CScaleRoot >= 0"
    if option.field == "c_visit" and _num(values, "c_visit_root") >= 0.0:
        return "rpg reads c_visit only as the root fallback, and CVisitRoot >= 0"
    return None


def _num(values: Mapping[str, object], key: str) -> float:
    v = values.get(key)
    return float(v) if isinstance(v, (int, float)) else 0.0


def realized_rows(
    path: str, values: Mapping[str, object],
) -> list[tuple[str, str, str, str]]:
    """``(name, value, LIVE|INERT|BRANCH, reason)`` for every registered option.

    This is the readback an operator uses to prove what the engine ran. It
    reports a row for every option rather than omitting the dead ones — a
    missing row is indistinguishable from an unsupported build — and it never
    prints a value as realized when the path could not act on it.

    Three statuses, because two conflated them into a wrong answer:

    ``INERT``   the path cannot read this option at all. ``CPuct`` under Gumbel.
                No value of it, and no change to any companion knob short of
                switching search path, makes it matter.
    ``BRANCH``  the option DOES reach the search and DOES change it, but the
                current value is inside a branch whose members are all
                equivalent. Filing this under ``INERT`` told an operator that a
                load-bearing parameter was doing nothing — ``QVisitExpRoot =
                -1.0`` is what selects the shipped log root transform.
    ``LIVE``    varying it can change the search.
    """
    rows: list[tuple[str, str, str, str]] = []
    for opt in SEARCH_OPTIONS:
        raw = values.get(opt.field, None)
        shown = "n/a" if raw is None else _fmt(raw)  # pyright: ignore[reportArgumentType]
        why = inert_reason(opt, path, values)
        if why is not None:
            rows.append((opt.name, shown, "INERT", why))
            continue
        note = branch_note(opt, path, values)
        rows.append((opt.name, shown, "BRANCH" if note else "LIVE", note or ""))
    return rows
