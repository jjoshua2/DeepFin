#!/usr/bin/env python3
"""Standardized paired-opening arena for strength measurement.

This is THE script every architecture / training-target candidate is judged
with (see docs/eval_protocol.md). It plays a candidate checkpoint against a
reference checkpoint over paired openings — each opening is played twice with
colors swapped, and the PAIR (not the game) is the unit of analysis. Results
are summarized as pentanomial counts with an Elo estimate and 95% CI computed
from the pentanomial variance (paired games are correlated, so the trinomial
W/D/L variance would understate the CI), then appended as one JSON line to
``runs/arena_results.jsonl``.

Two budget modes:

- ``matched_sims``: in-process batched MCTS via the same gumbel path as
  selfplay matches (``chess_anti_engine/selfplay/match.py``). Isolates policy
  /value quality at a fixed search budget — model latency does not matter.
- ``matched_time``: each side runs as a real UCI engine subprocess
  (``python -m chess_anti_engine.uci``), i.e. the same batched-inference path
  the engine ships with, at a fixed per-move wall clock. Model latency
  differences ARE part of the result.

Usage::

    PYTHONPATH=. python3 scripts/arena_standard.py \\
        --candidate runs/candidate/trainer.pt --reference runs/ref/trainer.pt \\
        --games 1000 --mode matched_sims --sims 64

    PYTHONPATH=. python3 scripts/arena_standard.py \\
        --candidate ... --reference ... --mode matched_time --ms-per-move 100
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.eval.arena_pgn import (
    ArenaGame,
    ArenaPgnWriter,
    engine_name_from_checkpoint,
)

# Called once per FINISHED game when --pgn-out is on, else None. Keyword-only so
# the three play loops (rolling / chunked / matched_time) cannot silently pass
# these positionally in different orders.
PgnSink = Callable[..., None]

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CONFIG = REPO_ROOT / "configs" / "pbt2_small.yaml"
DEFAULT_RESULTS_PATH = REPO_ROOT / "runs" / "arena_results.jsonl"

# matched_sims --compile=auto threshold: compile only when the run does enough
# inference work to amortize the ~2-4 min torch.compile cost. games*sims is a
# proxy for total forward passes; 12800 ~= 50 games at 256 sims. Below this the
# compile overhead dwarfs the per-call speedup, so eager is faster wall-clock.
AUTO_COMPILE_WORK_THRESHOLD = 12800

# Pentanomial bins from the candidate's point of view, by pair score
# (candidate points over the two games of one opening pair).
PAIR_SCORES = (2.0, 1.5, 1.0, 0.5, 0.0)
PAIR_LABELS = ("WW", "WD_DW", "DD_WL", "LD_DL", "LL")


# ---------------------------------------------------------------------------
# Search shape — WHICH search this arena measures
# ---------------------------------------------------------------------------
#
# There are two real search shapes in this repo and they are deliberately
# different (mcts/gumbel.py, tests/test_selfplay_gumbel_c_scale.py):
#
#   play      the tuned UCI/match shape — c_scale 0.025, topk 32, LOG root
#             (c_scale_root 7 / q_visit_exp_root -1), vloss_weight 3. These are
#             constants in mcts/gumbel.py, so they are safe to quote here.
#   training  what production selfplay actually runs. NOT quoted here on
#             purpose: every value is READ FROM `PRODUCTION_CONFIG` at call
#             time (see `resolve_search_shape`), so it tracks the yaml. Fixed
#             is the LINEAR root — the training shape leaves the root-transform
#             sentinels at their GumbelConfig defaults and never takes play's
#             log root. Run `--search-shape training` and read the realized
#             values the script prints if you need today's numbers.
#
# An earlier revision of this comment quoted "c_scale 0.1, topk 16". Commit
# ed9de8ee9 (2026-08-06) moved production selfplay to topk 32 and the comment
# silently became false. The shape logic never did — it reads the yaml. Do not
# reintroduce literals for the training shape.
#
# Until 2026-07-29 this script picked `play` for every run and never said so:
# `_parse_gumbel_overrides` seeded itself from PLAY_SEARCH_DEFAULTS even with no
# flag, and vloss_weight/target_batch were passed nowhere at all so both sat at
# 0 (the pre-C17 duplicate-leaf search) with no way to reach them from the CLI.
# So every Elo in docs/experiment_ledger.md was measured on a search selfplay
# does not run, and a sims ladder run that way was invalidated outright (ledger
# 2026-07-28: the "-52.5 Elo, 256 v 32" rung became +5.8 under the training
# shape). The shape is now an explicit REQUIRED choice, realized values are
# printed at startup and stored in the JSONL record, and there is no default.

# The training shape used to RESTATE three knobs by hand (`c_scale`, `topk`,
# `policy_temp`). Restating is why it drifted: `c62eb8ff2` (2026-08-10 23:28)
# promoted `gumbel_target_max_visit_cap: 5` and
# `gumbel_target_untempered_prior: true` into the production yaml, the hand-written
# list did not grow, and `--search-shape training` went on announcing itself as
# production's search while running a GumbelConfig production does not build.
# The designed guard
# (`tests/test_arena_search_shape_plumbing.py::test_every_config_driven_knob_...`)
# detected it on the first run after the promotion and nobody read the failure.
#
# So the training shape is no longer a list of knobs to remember. It is the
# COMPLEMENT of two small, explicitly justified sets: every GumbelConfig field
# that is not owned by the arena and not owned by the checkpoint is read straight
# off the GumbelConfig `build_selfplay_gumbel_config` — the production mapping
# itself — returns. A newly promoted knob is therefore carried with no edit here,
# and a new GumbelConfig FIELD fails the partition pin in the test by name until
# somebody classifies it.

# Fields the ARENA sets from its own flags. The sim budget, the move-selection
# temperature and the root-noise policy are properties of the MATCH, not of the
# training config: an arena at production's `mcts_simulations` and
# `add_noise=True` would be measuring selfplay's exploration, not playing strength.
ARENA_OWNED_GUMBEL_FIELDS: frozenset[str] = frozenset(
    {"simulations", "temperature", "add_noise", "gumbel_scale"}
)

# Fields the CHECKPOINT owns. `selfplay/match.pick_moves_for_boards` reads all
# four off the loaded model (`model.input_history_encoding` etc.), because an
# older checkpoint may genuinely need `v1` planes or a different policy encoding.
# Forcing production's value here would mis-encode the very cross-era arenas the
# script exists for, so these are excluded ON PURPOSE and not by omission.
CHECKPOINT_OWNED_GUMBEL_FIELDS: frozenset[str] = frozenset(
    {
        "input_history_encoding",
        "input_extra_features",
        "policy_encoding",
        "compute_relations",
    }
)

SEARCH_SHAPES = ("play", "training")

_GUMBEL_INT_KEYS = {"topk", "halving_div", "simulations"}


def training_shape_carried_fields() -> tuple[str, ...]:
    """GumbelConfig fields the ``training`` shape must carry from production.

    The complement, computed — never a hand-maintained list. A field added to
    ``GumbelConfig`` lands here automatically, so the arena cannot silently fail
    to carry it; the test pins this tuple by NAME so the addition still has to be
    looked at.
    """
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import GumbelConfig

    excluded = ARENA_OWNED_GUMBEL_FIELDS | CHECKPOINT_OWNED_GUMBEL_FIELDS
    return tuple(f.name for f in _dc.fields(GumbelConfig) if f.name not in excluded)


@dataclass(frozen=True)
class SideSearch:
    """The complete realized search configuration for ONE side of the arena.

    ``gumbel`` are GumbelConfig field overrides (applied by
    ``dataclasses.replace``); ``vloss_weight`` / ``target_batch`` are the two
    C-path controls that are function arguments rather than GumbelConfig
    fields, which is exactly why an override surface built on
    ``dataclasses.replace`` could never reach them.
    """

    shape: str
    source: str
    gumbel: dict[str, float]
    vloss_weight: int
    target_batch: int
  # Whether the search tree is carried across plies. EVERY arena move is a
  # COLD search: `selfplay/match.pick_moves_for_boards` -- the only entry point
  # this script and `chess_anti_engine/arena.py` use -- passes neither `tree`
  # nor `root_node_ids`, while production selfplay
  # (`selfplay/network_turn.py:799-801`) passes both and advances the root with
  # `find_child` after each ply. Measured (play-path audit 2026-08-03 F1,
  # `scratchpad/code_audit_20260803/repro_tree_reuse.py`, 256 nominal sims):
  # cold roots see 256 visits / max_visit 60 on every ply, warm ones 315-363 /
  # 77-108, i.e. the root value-transform scale q_scale=c_scale*(c_visit+
  # max_visit) is up to +44% sharper in production than in any arena. So
  # `matched_sims` matches the nominal budget, NOT the visit counts the policy
  # is built from. Recorded rather than fixed: giving `pick_moves_for_boards` a
  # tree carry changes arena behaviour and needs its own pre-registered
  # readout. Constant today; a field so the JSONL record dates from before the
  # fix rather than being silent about it.
  #
  # ⚑ ABSENT != "cold". Every JSONL row written before 2026-08-03 lacks this
  # key. A reader must treat a missing `tree_reuse` as UNKNOWN, never default
  # it to "cold" -- those rows were cold in fact, but a default that silently
  # answers for rows the field never covered is the `reco_diff misses absent
  # keys` failure mode, and it would keep answering after tree carry lands.
    tree_reuse: str = "cold"

    def realized_gumbel(self) -> dict[str, float | int]:
        """Every shape-defining knob's REALIZED value, overrides or not.

        Keys not overridden fall back to the GumbelConfig dataclass default,
        which is the selfplay/training shape by construction. Printing this
        rather than the sparse override dict is the point: a `training` run
        must be able to show that it is NOT running c_scale 0.025.

        Knobs in ``INERT_GUMBEL_KNOBS`` are excluded unconditionally: printing
        a value the search cannot act on as "realized" is what made a c_puct
        Swiss look like a measurement (audit 2026-08-03, F2).
        """
        from chess_anti_engine.mcts.gumbel import (
            INERT_GUMBEL_KNOBS,
            PLAY_SEARCH_DEFAULTS,
            GumbelConfig,
        )

        base = GumbelConfig()
        out: dict[str, float | int] = {}
        keys = (set(PLAY_SEARCH_DEFAULTS) | set(self.gumbel)) - INERT_GUMBEL_KNOBS
        for key in sorted(keys):
            value = self.gumbel[key] if key in self.gumbel else getattr(base, key, None)
            if not isinstance(value, (int, float)):
                raise SystemExit(
                    f"gumbel knob {key!r} is not a numeric GumbelConfig field "
                    f"(got {value!r}); --*-gumbel keys must be replaceable fields. "
                    "If this is a NEW GumbelConfig field, the training shape "
                    "carries it by default (the complement rule) and a "
                    "non-numeric one has to be classified into "
                    "CHECKPOINT_OWNED_GUMBEL_FIELDS or ARENA_OWNED_GUMBEL_FIELDS "
                    "first. Refusing rather than dropping it is deliberate."
                )
            out[key] = value
        return out

    def as_record(self) -> dict:
        return {
            "shape": self.shape,
            "source": self.source,
            "gumbel": dict(self.realized_gumbel()),
            "vloss_weight": self.vloss_weight,
            "target_batch": self.target_batch,
            "tree_reuse": self.tree_reuse,
        }

    def describe(self) -> str:
        knobs = " ".join(f"{k}={v}" for k, v in self.realized_gumbel().items())
        return (
            f"shape={self.shape} vloss_weight={self.vloss_weight} "
            f"target_batch={self.target_batch} tree_reuse={self.tree_reuse} "
            f"{knobs} [{self.source}]"
        )


def production_selfplay_configs() -> dict:
    """The selfplay config bundle a distributed worker actually builds.

    Runs the whole real channel — yaml -> live-yaml validator ->
    ``build_recommended_worker`` (the ONLY way a knob reaches a worker) ->
    ``WorkerSession._build_selfplay_configs`` — instead of reading yaml keys
    directly, so the arena's training shape cannot drift from the search
    production runs. A knob the reco does not publish resolves to the worker's
    own default HERE TOO, which is the honest answer: an unpublished knob is
    not what production runs, whatever the yaml says.

    Reads the config in THIS tree; run the arena from the live tree to price
    the live run. NOT the ``TrialConfig``/``_play_batch_kwargs`` path: that one
    silently drops ``gumbel_vloss_weight`` (no such TrialConfig field), which
    is the in-process-selfplay half of the same defect family.
    """
    import logging
    import threading
    from types import SimpleNamespace

    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
    from chess_anti_engine.utils.config_yaml import (
        flatten_run_config_defaults,
        load_yaml_file,
    )
    from chess_anti_engine.worker import WorkerSession

    flat = flatten_run_config_defaults(load_yaml_file(str(PRODUCTION_CONFIG)))
    # sf_nodes / mcts_simulations are supplied by the publisher (PID budget and
    # sim ramp), not read from the config; mirror the config's own values so
    # nothing here depends on the live controller state.
    reco = build_recommended_worker(
        config=flat,
        model_cfg=ModelConfig(),
        sf_nodes=int(flat.get("sf_nodes", 5000) or 5000),
        mcts_simulations=int(flat.get("mcts_simulations", 32) or 32),
    )
    # `_build_selfplay_configs` reads only these session fields (see
    # tests/test_selfplay_gumbel_c_scale.py); a full session would need a
    # broker, a model and a Stockfish binary.
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("arena.production_search_config")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, reco)
    return cfgs


def production_selfplay_search_config():
    """The selfplay ``SearchConfig`` half of :func:`production_selfplay_configs`."""
    return production_selfplay_configs()["search"]


def production_selfplay_gumbel_config():
    """The ``GumbelConfig`` production selfplay actually searches with.

    Built by ``selfplay.network_turn.build_selfplay_gumbel_config`` — THE
    mapping the production path calls, not a re-implementation of it. This is
    the whole structural fix: the arena's training shape is now read off this
    object, so "which knobs does production set" stopped being a question the
    arena answers from memory. ``simulations`` is arena-owned and passed here
    only because the mapping requires it; it is excluded from what is carried.
    """
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config

    cfgs = production_selfplay_configs()
    return build_selfplay_gumbel_config(
        search=cfgs["search"], game=cfgs["game"], simulations=1,
    )


def resolve_search_shape(shape: str) -> SideSearch:
    """Turn ``play``/``training`` into the concrete knobs each one means."""
    from chess_anti_engine.mcts.gumbel import (
        PLAY_SEARCH_DEFAULTS,
        PLAY_SEARCH_TARGET_BATCH,
        PLAY_SEARCH_VLOSS_WEIGHT,
    )

    if shape == "play":
        return SideSearch(
            shape="play",
            source="mcts.gumbel PLAY_SEARCH_DEFAULTS + PLAY_SEARCH_VLOSS_WEIGHT",
            gumbel={
                k: (int(v) if k in _GUMBEL_INT_KEYS else float(v))
                for k, v in PLAY_SEARCH_DEFAULTS.items()
            },
            vloss_weight=int(PLAY_SEARCH_VLOSS_WEIGHT),
            target_batch=int(PLAY_SEARCH_TARGET_BATCH),
        )
    if shape == "training":
        search = production_selfplay_search_config()
        prod = production_selfplay_gumbel_config()
        # DERIVED, never restated. Every GumbelConfig field that is neither
        # arena-owned nor checkpoint-owned is copied from the config production
        # itself builds, so a knob promoted into the yaml reaches the arena with
        # no edit here. Restating three knobs by hand is exactly what let
        # `target_max_visit_cap` / `target_untempered_prior` drift for five days.
        side = SideSearch(
            shape="training",
            source=f"{PRODUCTION_CONFIG.name} -> reco -> worker -> GumbelConfig",
            gumbel={
                name: getattr(prod, name) for name in training_shape_carried_fields()
            },
            vloss_weight=int(search.gumbel_vloss_weight),
            target_batch=int(search.gumbel_target_batch),
        )
        _assert_training_shape_matches_production(side, prod)
        return side
    raise SystemExit(f"--search-shape must be one of {SEARCH_SHAPES}, got {shape!r}")


def _assert_training_shape_matches_production(side: SideSearch, prod) -> None:
    """Refuse to run a `training` arena that is not production's search.

    ⚑ Not redundant with the test suite, and deliberately at RUN time. CI checks
    the COMMITTED `configs/pbt2_small.yaml`; this script reads whichever yaml is
    in the tree it runs from, and the live yaml is edited in place and lags the
    commit. The gate that matters therefore has to fire in the live tree, on the
    file the arena actually read — which is the shape of the original defect
    (the guard existed, CI was not where it needed to fire, the failure went
    unread for five days).

    Structurally this cannot trip while the shape stays DERIVED; it exists so
    that reintroducing a hand-written knob list fails loudly at the top of an
    arena instead of silently mislabelling its Elo.

    ⚑ The field list is recomputed HERE from ``dataclasses.fields`` and the two
    ownership sets rather than taken from ``training_shape_carried_fields()``.
    Sharing that helper with the builder is what would make this vacuous: a knob
    dropped from the helper would vanish from the constructed dict AND from the
    comparison, and the check would pass on a shape that lost it. A guard whose
    field list comes from the thing it is guarding cannot see that thing narrow.
    """
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import GumbelConfig

    excluded = ARENA_OWNED_GUMBEL_FIELDS | CHECKPOINT_OWNED_GUMBEL_FIELDS
    # Exactly how `selfplay/match.pick_moves_for_boards` applies the dict.
    applied = _dc.replace(GumbelConfig(), **side.gumbel)
    drift = {
        f.name: (getattr(prod, f.name), getattr(applied, f.name))
        for f in _dc.fields(GumbelConfig)
        if f.name not in excluded
        and getattr(prod, f.name) != getattr(applied, f.name)
    }
    if drift:
        raise SystemExit(
            "--search-shape training does not reproduce production selfplay's "
            f"GumbelConfig. Drifted (production, arena): {drift}. Running this "
            "arena would measure a search production does not run and would "
            "label the result 'training'. Fix the shape derivation in "
            f"{Path(__file__).name}; do NOT relax this check."
        )


def apply_search_overrides(
    base: SideSearch,
    *,
    spec: str | None = None,
    vloss_weight: int | None = None,
    target_batch: int | None = None,
) -> SideSearch:
    """Layer per-side CLI overrides on top of a resolved shape."""
    import dataclasses

    from chess_anti_engine.mcts.gumbel import INERT_GUMBEL_KNOBS, GumbelConfig

    fields = {f.name for f in dataclasses.fields(GumbelConfig)}
    gumbel = dict(base.gumbel)
    extra: list[str] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--*-gumbel: expected k=v pairs, got {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in fields:
            # Caught here rather than by `dataclasses.replace` several minutes
            # into the run, after both checkpoints have loaded and compiled.
            raise SystemExit(
                f"--*-gumbel: {k!r} is not a GumbelConfig field. Valid keys: "
                f"{', '.join(sorted(fields))}"
            )
        if k in INERT_GUMBEL_KNOBS:
            # Accepting it would produce a perfectly reproducible null that
            # reads as a measurement: the PUCT descent these drive is
            # unreachable while full_tree is True (audit 2026-08-03, F2).
            raise SystemExit(
                f"--*-gumbel: {k!r} cannot affect a Gumbel search and is refused. "
                "It drives the PUCT descent, which GumbelConfig.full_tree=True "
                "makes unreachable (play-path audit 2026-08-03 F2; repro "
                "scratchpad/code_audit_20260803/repro_inert_knobs.py). A Swiss "
                "over it would return a flat null and read as a measurement."
            )
        gumbel[k] = int(v) if k in _GUMBEL_INT_KEYS else float(v)
        extra.append(part)
    if vloss_weight is not None:
        extra.append(f"vloss_weight={int(vloss_weight)}")
    if target_batch is not None:
        extra.append(f"target_batch={int(target_batch)}")
    source = base.source if not extra else f"{base.source} + CLI({','.join(extra)})"
    return SideSearch(
        shape=base.shape,
        source=source,
        gumbel=gumbel,
        vloss_weight=base.vloss_weight if vloss_weight is None else int(vloss_weight),
        target_batch=base.target_batch if target_batch is None else int(target_batch),
        tree_reuse=base.tree_reuse,
    )


# ---------------------------------------------------------------------------
# Pentanomial bookkeeping + Elo math
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PentanomialSummary:
    """Pentanomial match summary from the candidate's point of view."""

    counts: tuple[int, int, int, int, int]  # WW, WD/DW, DD+WL, LD/DL, LL
    pairs: int
    games: int
    score: float           # mean per-game score in [0, 1]
    score_se: float        # standard error of the mean per-game score
    elo: float | None
    elo_ci95: tuple[float | None, float | None]


def game_scores_to_pair_scores(game_scores: list[float]) -> list[float]:
    """Collapse per-game candidate scores (1/0.5/0) into per-pair scores.

    Games must be ordered so that ``game_scores[2*i]`` and
    ``game_scores[2*i + 1]`` are the two colorings of opening pair ``i``.
    """
    if len(game_scores) % 2 != 0:
        raise ValueError(f"need an even number of games, got {len(game_scores)}")
    for s in game_scores:
        if s not in (0.0, 0.5, 1.0):
            raise ValueError(f"per-game score must be 0, 0.5 or 1, got {s}")
    return [
        game_scores[i] + game_scores[i + 1]
        for i in range(0, len(game_scores), 2)
    ]


def complete_pair_scores(game_scores: list[float | None]) -> list[float]:
    """Pair scores for the openings where BOTH colorings finished.

    ``None`` means "this game did not finish" — an unstarted queue entry or a
    game still in flight when a ``--max-seconds`` deadline stopped the loop.
    Such a pair is DROPPED, never imputed: scoring an unfinished game as a draw
    would let a truncated run claim pairs it never played, which is precisely
    the "a number that does not mean what its name says" defect the arena's own
    games column exists to avoid.
    """
    out: list[float] = []
    for i in range(0, len(game_scores) - len(game_scores) % 2, 2):
        w, b = game_scores[i], game_scores[i + 1]
        if w is not None and b is not None:
            out.append(w + b)
    return out


def pentanomial_counts(pair_scores: list[float]) -> tuple[int, int, int, int, int]:
    """Bin pair scores into (WW, WD/DW, DD+WL, LD/DL, LL) counts."""
    counts = [0, 0, 0, 0, 0]
    for s in pair_scores:
        try:
            counts[PAIR_SCORES.index(s)] += 1
        except ValueError:
            raise ValueError(f"pair score must be one of {PAIR_SCORES}, got {s}") from None
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def _elo_from_score(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0) + 0.0  # +0.0 normalizes -0.0


def summarize_pentanomial(
    counts: tuple[int, int, int, int, int], *, z: float = 1.96,
) -> PentanomialSummary:
    """Elo point estimate + CI from pentanomial pair counts.

    The pair is the sampling unit: per-pair normalized scores are
    x in {1, 0.75, 0.5, 0.25, 0} and the CI uses the empirical variance of x
    across pairs. This correctly accounts for the within-pair correlation
    that a trinomial (per-game W/D/L) variance would miss.
    """
    n = sum(counts)
    if n <= 0:
        raise ValueError("no pairs")
    xs = tuple(s / 2.0 for s in PAIR_SCORES)
    mu = sum(c * x for c, x in zip(counts, xs)) / n
    var = sum(c * (x - mu) ** 2 for c, x in zip(counts, xs)) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    lo = mu - z * se
    hi = mu + z * se
    return PentanomialSummary(
        counts=counts,
        pairs=n,
        games=2 * n,
        score=mu,
        score_se=se,
        elo=_elo_from_score(mu),
        elo_ci95=(_elo_from_score(lo), _elo_from_score(hi)),
    )


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def default_openings_path() -> Path:
    """The 8-move UHO book from the production config (opening_book_path_2)."""
    import yaml

    cfg = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    selfplay = cfg.get("selfplay", {}) if isinstance(cfg, dict) else {}
    book = selfplay.get("opening_book_path_2") or selfplay.get("opening_book_path")
    if not book:
        raise SystemExit(
            f"no opening_book_path(_2) in {PRODUCTION_CONFIG}; pass --openings"
        )
    return Path(str(book))


def _find_nested(cfg: object, key: str) -> object | None:
    """Return the first value of ``key`` anywhere in a nested dict, else None."""
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            found = _find_nested(v, key)
            if found is not None:
                return found
    return None


def default_compile_cache_dir() -> Path:
    """The production worker's shared torch.compile/triton cache root.

    Mirrors ``tune.distributed_runtime._resolve_shared_cache_root``: the
    explicit ``distributed_worker_shared_cache_dir`` from the production config
    wins (that is exactly the dir the live workers populate), otherwise we fall
    back to ``<work_dir>/server/worker_cache``. Pointing the arena's compile at
    this reuses the autotuned kernels + FX graphs that training already baked,
    so an arena run skips most of its cold-compile cost.

    Falls back to the documented literal ``runs/pbt2_small/server/worker_cache``
    (the production worker cache) if the config can't be read; an absent dir is
    harmless (just no reuse that run). Always returns an absolute path.
    """
    fallback = REPO_ROOT / "runs" / "pbt2_small" / "server" / "worker_cache"
    try:
        import yaml

        cfg = yaml.safe_load(PRODUCTION_CONFIG.read_text())
    except (OSError, ValueError):
        return fallback
    explicit = _find_nested(cfg, "distributed_worker_shared_cache_dir")
    if explicit and str(explicit).strip():
        return Path(str(explicit).strip()).expanduser().resolve()
    work_dir = _find_nested(cfg, "work_dir")
    if work_dir and str(work_dir).strip():
        root = Path(str(work_dir).strip()).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        return (root / "server" / "worker_cache").resolve()
    return fallback


def _configure_shared_compile_cache(cache_dir: Path) -> None:
    """Point TorchInductor/Triton at ``cache_dir`` for cross-run kernel reuse.

    Mirrors ``worker._configure_shared_compile_cache`` so the arena hits the
    same on-disk cache layout the training workers populate. Uses
    ``os.environ.setdefault`` so an explicitly-exported env var still wins, and
    must be called BEFORE any ``torch.compile`` (i.e. before run_arena loads
    torch). Creating the dirs is harmless when they don't exist yet.
    """
    import os

    compile_cache_root = cache_dir / "compile_cache"
    inductor_dir = compile_cache_root / "torchinductor"
    triton_dir = compile_cache_root / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_dir))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    print(
        f"[arena] compile cache: {inductor_dir} "
        f"(reusing training kernels if present)",
        flush=True,
    )


def load_paired_openings(
    path: Path, *, n_pairs: int, max_plies: int, rng: np.random.Generator,
) -> list[chess.Board]:
    """Sample ``n_pairs`` opening boards from a PGN(.zip)/Polyglot book.

    Sampling is weighted by book frequency (same as selfplay) and seeded, so
    a fixed --seed reproduces the exact opening set. Duplicate positions are
    rejected until the book runs out of variety.
    """
    from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board

    cfg = OpeningConfig(
        opening_book_path=str(path),
        opening_book_max_plies=int(max_plies),
        opening_book_prob=1.0,
    )
    boards: list[chess.Board] = []
    seen: set[str] = set()
    attempts = 0
    duplicates = 0
    max_attempts = max(1000, 50 * n_pairs)
    while len(boards) < n_pairs:
        board = sample_starting_board(rng=rng, cfg=cfg).board
        attempts += 1
        key = board.epd()
        if key in seen:
            if attempts < max_attempts:
                continue
            duplicates += 1
        seen.add(key)
        boards.append(board)
    if duplicates:
        print(
            f"[arena] WARNING: opening book ran out of unique openings; "
            f"{duplicates}/{n_pairs} pairs reuse an opening. Duplicate pairs "
            f"are correlated samples, so the reported CI is overconfident."
        )
    return boards


def _load_unique_fen_boards(path: Path) -> list[chess.Board]:
    """Validated, deduplicated seed boards from a seed list.

    Reuses selfplay's ``_load_fen_list`` + ``seed_board_from_line`` (the two
    consume the SAME seed files): illegal/terminal/forced seeds are SKIPPED with
    a logged warning — not played as phantom draws or crashed on mid-arena — and
    a zero-usable-seed file fails fast (its ValueError is surfaced as a clean
    SystemExit here, so both this and the compile-heuristic count path exit
    cleanly). ``<start_fen> | <moves>`` lines replay to the terminal seed with
    real history. Dedup is on the python-chess NORMALIZED terminal FEN, so two
    equivalent spellings (e.g. a redundant en-passant square that normalizes
    away) collapse to one board while rows that genuinely differ in
    halfmove/fullmove counters are both kept.
    """
    from chess_anti_engine.selfplay.opening import _load_fen_list, seed_board_from_line

    try:
        fens = _load_fen_list(str(path))
    except ValueError as exc:  # zero usable seeds
        raise SystemExit(f"[arena] {exc}") from exc
    seen: set[tuple[str, tuple[str, ...]]] = set()
    boards: list[chess.Board] = []
    for f in fens:
        board = seed_board_from_line(f)  # already validated by _load_fen_list
        # Key on the normalized terminal FEN AND the replayed history: two seeds
        # with the SAME terminal but DIFFERENT preceding plies are distinct model
        # inputs (the encoder consumes move_stack), so dedup must not collapse
        # them; ep-equivalent same-history spellings still merge (Codex review).
        key = (board.fen(), tuple(m.uci() for m in board.move_stack))
        if key not in seen:
            seen.add(key)
            boards.append(board)
    return boards


def load_fen_seed_count(path: Path) -> int:
    """Deduplicated usable-seed count (matches what the arena actually plays)."""
    return len(_load_unique_fen_boards(path))


def load_fen_openings(
    path: Path, *, n_pairs: int, rng: np.random.Generator,
) -> list[chess.Board]:
    """Load opening boards from a plain FEN file (one per line, ``#`` comments).

    For blind-spot / seed-list play-outs (e.g. ``data/blindspot_fens_v1.txt``):
    each FEN is played as a color-swapped pair exactly like book openings. If
    the file holds more usable rows than ``n_pairs`` a seeded subsample is drawn
    (reproducible via --seed) and the drop is logged; if fewer, ALL rows are
    used and the arena shrinks to ``2 * len(rows)`` games. Boards start with an
    empty move stack, so history planes repeat-fill — the encoding selfplay FEN
    seeds get. Validation/dedup live in ``_load_unique_fen_boards``.
    """
    boards = _load_unique_fen_boards(path)
    n_usable = len(boards)
    if n_usable > n_pairs:
        idx = rng.choice(n_usable, size=n_pairs, replace=False)
        boards = [boards[int(i)] for i in sorted(idx)]
        print(
            f"[arena] WARNING: FEN list has {n_usable} usable rows > {n_pairs} "
            f"requested pairs; seeded-subsampling to {n_pairs} and DROPPING "
            f"{n_usable - n_pairs} curated seeds — raise --games to 2x the row "
            f"count for full coverage"
        )
    elif n_usable < n_pairs:
        print(
            f"[arena] FEN list has {len(boards)} usable rows < {n_pairs} requested "
            f"pairs; using all rows ({2 * len(boards)} games)"
        )
    return boards


# ---------------------------------------------------------------------------
# Game play — matched sims (in-process batched MCTS)
# ---------------------------------------------------------------------------

def play_paired_games_matched_sims(
    model_candidate,
    model_reference,
    openings: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    sims_candidate: int,
    sims_reference: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    search_candidate: SideSearch,
    search_reference: SideSearch,
    volatility_candidate: dict[str, float] | None = None,
    syzygy_tablebase: object | None = None,
    tb_max_pieces: int = 6,
    pgn_sink: PgnSink | None = None,
    pair_id_offset: int = 0,
) -> list[float]:
    """Play each opening twice (colors swapped) and return per-pair scores.

    ``search_candidate`` / ``search_reference`` are REQUIRED and carry the full
    realized search shape per side (see ``SideSearch``). There is deliberately
    no default: a silent default is what made every arena in the ledger measure
    the UCI/play search instead of the training search.

    ``volatility_candidate`` (volatility_q_scale / volatility_fpu /
    volatility_anchor) applies volatility-aware Gumbel search to the
    CANDIDATE side only — the reference keeps today's search, which is the
    A/B the experiment needs. Non-zero flags force the Python search path
    (mcts/gumbel.py), so matched_sims is the honest mode.

    Reuses the selfplay match helpers so search behavior (gumbel MCTS, the
    model's own input_history_encoding / policy_encoding, including the
    compact lc0_1858 path) is identical to ``play_match_batch``; this loop
    only differs in pinning the starting boards and keeping per-game results.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        result_from_a_pov,
        split_active_by_side_to_move,
    )
    from scripts.match_vs_uci import _tb_adjudicate_result

    boards: list[chess.Board] = []
    a_plays_white: list[bool] = []
    start_fens: list[str] = []
    start_offsets: list[int] = []
    for opening in openings:
        for a_white in (True, False):
            boards.append(opening.copy())
            a_plays_white.append(a_white)
            # Captured BEFORE play so the PGN can replay from the book position
            # rather than from the standard start: move_stack carries the book's
            # own moves too, and slicing at this offset is what separates them.
            start_fens.append(opening.fen())
            start_offsets.append(len(opening.move_stack))

    g = len(boards)
    done = [False] * g
    adjudicated: list[str | None] = [None] * g  # Syzygy-adjudicated result per game
    emitted = [False] * g
    t0 = time.time()

    def _emit(i: int, termination: str, result_override: str | None = None) -> None:
        if pgn_sink is None or emitted[i]:
            return
        emitted[i] = True
        res = result_override or adjudicated[i] or boards[i].result(claim_draw=True)
        mv = tuple(boards[i].move_stack[start_offsets[i]:])
        pgn_sink(
            pair_id=pair_id_offset + i // 2,
            half=i % 2,
            a_is_white=bool(a_plays_white[i]),
            start_fen=start_fens[i],
            moves=mv,
            result=res,
            termination=termination,
            plies=len(mv),
            duration_s=time.time() - t0,
        )
    for ply in range(int(max_plies)):
        for i in range(g):
            if done[i]:
                continue
            if boards[i].is_game_over(claim_draw=True):
                done[i] = True
                _emit(i, "rules")
            elif syzygy_tablebase is not None:
                # Adjudicate the instant a game reaches a covered (<=N-man) position
                # — kills long endgame tails. Reuses match_vs_uci's WDL probe.
                _tb = _tb_adjudicate_result(boards[i], syzygy_tablebase, max_pieces=tb_max_pieces)
                if _tb is not None:
                    adjudicated[i] = _tb
                    done[i] = True
                    _emit(i, "syzygy")
        active = [i for i in range(g) if not done[i]]
        if not active:
            break
        if ply and ply % 20 == 0:
            print(
                f"[arena] ply {ply}: {g - len(active)}/{g} games finished "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        a_to_move, b_to_move = split_active_by_side_to_move(
            active, boards, a_plays_white,
        )
        vol_kwargs = dict(volatility_candidate or {})
        for model, idxs, sims, extra, side in (
            (model_candidate, a_to_move, sims_candidate, vol_kwargs, search_candidate),
            (model_reference, b_to_move, sims_reference, {}, search_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=side.gumbel,
                gumbel_vloss_weight=side.vloss_weight,
                gumbel_target_batch=side.target_batch,
                **extra,
            )
            apply_actions_to_boards(boards, idxs, actions)

    def _game_score(i: int) -> float:
        res = adjudicated[i] or boards[i].result(claim_draw=True)
        if res == "*":  # unfinished at max_plies, not TB-covered: adjudicate as draw
            return 0.5
        return {1: 1.0, 0: 0.5, -1: 0.0}[
            result_from_a_pov(res, a_is_white=bool(a_plays_white[i]))
        ]

    game_scores = [_game_score(i) for i in range(g)]
    # Sweep up whatever the ply loop did not already emit. Two DIFFERENT cases
    # live here and conflating them corrupts the file:
    #
    #  * genuinely unfinished ("*"): SCORED 0.5 above, so it must be WRITTEN as
    #    a draw. Emitting "*" would be silently lossy — Ordo maps it to DISCARD
    #    (pgnget.c) and drops the game, so the pooled fit would run on a
    #    different population than the pentanomial summary it is compared to.
    #  * DECISIVE on the very last ply: the loop tests for game-over at the TOP
    #    of each iteration, so a game that ends on the move played at ply
    #    max_plies-1 is never re-tested and reaches here undelivered. It has a
    #    real result and `_game_score` already counted it as a win/loss —
    #    blanket-overriding it to a draw made the PGN disagree with the
    #    pentanomial, which is the exact cross-check the agreement claim rests
    #    on. Rolling never had this because it reaps before refilling.
    #
    # `adjudicated[i]` is ALWAYS None here, so "rules" is accurate and there is
    # deliberately no "syzygy" branch: adjudication sets `adjudicated[i]` and
    # calls `_emit(i, "syzygy")` in the same block above, so such a game is
    # already emitted and `_emit`'s `emitted[i]` guard makes this a no-op for
    # it. A `"syzygy" if adjudicated[i] else "rules"` here would be a branch
    # that cannot be reached or tested. A position that only becomes TB-covered
    # on the FINAL ply is never probed (the loop exits first) and lands in the
    # "*" case, scored 0.5 by `_game_score` and written as a draw — the two
    # still agree, which is the property that matters.
    for i in range(g):
        res = adjudicated[i] or boards[i].result(claim_draw=True)
        if res == "*":
            _emit(i, "max_plies", result_override="1/2-1/2")
        else:
            _emit(i, "rules")
    return game_scores_to_pair_scores(game_scores)


def play_paired_games_matched_sims_rolling(
    model_candidate,
    model_reference,
    openings: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    sims_candidate: int,
    sims_reference: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    search_candidate: SideSearch,
    search_reference: SideSearch,
    volatility_candidate: dict[str, float] | None = None,
    syzygy_tablebase: object | None = None,
    tb_max_pieces: int = 6,
    pool_size: int = 256,
    report_every: int = 64,
    deadline: float | None = None,
    pgn_sink: PgnSink | None = None,
) -> list[float]:
    """Rolling-pool variant: keep ``pool_size`` games active at all times, starting
    a fresh game the instant one finishes (like production selfplay), instead of
    playing fixed chunks to completion.

    Two wins over the chunked path: (1) the GPU never drains until the very end
    (no per-chunk tail), and (2) the active-game count — hence the batched-inference
    shape — stays FIXED at ``pool_size`` while the queue lasts, so torch.compile
    compiles once and reuses it (no per-shape recompile thrash). Only the final
    drain (last ~pool_size games) has shrinking shapes.

    Each opening is still played twice (colors swapped) and scored as a pair;
    game_id ``2k``/``2k+1`` are the white/black halves of opening ``k``, so the
    flat ``game_scores`` reassemble into the same pairs as the lockstep path.

    ``deadline`` (``time.time()`` epoch seconds) stops the loop between plies
    and returns whatever COMPLETE pairs exist. Only complete pairs are ever
    returned: a half-played game contributes nothing, because filling it in as
    a draw would let a truncated run report pairs it never finished.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        result_from_a_pov,
        split_active_by_side_to_move,
    )
    from scripts.match_vs_uci import _tb_adjudicate_result

    queue: list[tuple[int, chess.Board, bool]] = []
    for k, opening in enumerate(openings):
        queue.append((2 * k, opening, True))
        queue.append((2 * k + 1, opening, False))
    n_games = len(queue)
    queue.reverse()  # pop() from the end
    game_scores: list[float | None] = [None] * n_games

    boards: list[chess.Board] = []
    gids: list[int] = []
    awhite: list[bool] = []
    gplies: list[int] = []
    gfens: list[str] = []
    goffs: list[int] = []
    gt0: list[float] = []

    def _refill() -> None:
        while len(boards) < pool_size and queue:
            gid, opening, aw = queue.pop()
            boards.append(opening.copy())
            gids.append(gid)
            awhite.append(aw)
            gplies.append(0)
            # Book position + how many of move_stack belongs to the book, so the
            # PGN starts where PLAY started rather than replaying the opening.
            gfens.append(opening.fen())
            goffs.append(len(opening.move_stack))
            gt0.append(time.time())

    def _record(j: int, res: str, termination: str) -> None:
        if res == "*":
            game_scores[gids[j]] = 0.5
        else:
            game_scores[gids[j]] = {1: 1.0, 0: 0.5, -1: 0.0}[
                result_from_a_pov(res, a_is_white=bool(awhite[j]))
            ]
        if pgn_sink is not None:
            # "*" is SCORED 0.5, so it is WRITTEN as a draw: Ordo drops "*"
            # (pgnget.c DISCARD), which would make the pooled fit and the
            # pentanomial summary disagree about which games exist.
            mv = tuple(boards[j].move_stack[goffs[j]:])
            pgn_sink(
                pair_id=gids[j] // 2,
                half=gids[j] % 2,
                a_is_white=bool(awhite[j]),
                start_fen=gfens[j],
                moves=mv,
                result="1/2-1/2" if res == "*" else res,
                termination=termination,
                plies=len(mv),
                duration_s=time.time() - gt0[j],
            )

    t0 = time.time()
    done = 0
    last_report = 0
    while queue or boards:
        # Stop on our OWN clock rather than waiting to be SIGKILLed by the
        # caller's `timeout`. A killed process returns nothing at all; stopping
        # here returns every pair that finished and lets run_arena print the
        # final summary and append the JSONL record like any other run.
        _refill()
        # Reap finished / adjudicated / over-cap games, compacting the pool.
        kb: list[chess.Board] = []
        kg: list[int] = []
        ka: list[bool] = []
        kp: list[int] = []
        kf: list[str] = []
        ko: list[int] = []
        kt: list[float] = []
        for j in range(len(boards)):
            b = boards[j]
            res: str | None = None
            termination = "rules"
            if b.is_game_over(claim_draw=True):
                res = b.result(claim_draw=True)
            elif syzygy_tablebase is not None:
                res = _tb_adjudicate_result(b, syzygy_tablebase, max_pieces=tb_max_pieces)
                if res is not None:
                    termination = "syzygy"
            if res is None and gplies[j] >= int(max_plies):
                res = "*"  # not naturally decided and not TB-covered: adjudicate draw
                termination = "max_plies"
            if res is not None:
                _record(j, res, termination)
                done += 1
            else:
                kb.append(b)
                kg.append(gids[j])
                ka.append(awhite[j])
                kp.append(gplies[j])
                kf.append(gfens[j])
                ko.append(goffs[j])
                kt.append(gt0[j])
        boards[:], gids[:], awhite[:], gplies[:] = kb, kg, ka, kp
        gfens[:], goffs[:], gt0[:] = kf, ko, kt
        # Deadline check goes AFTER the reap, not before it. Checking first
        # discarded every game that had finished on the ply we just played —
        # up to pool_size of them, and measurably: the 2026-07-31 proof run
        # banked 100 games but scored 96, and 118 but scored 114. Reaping first
        # costs nothing (no ply is played below this point) and cannot fabricate
        # anything, because `_record` still only runs for a game with a result.
        if deadline is not None and time.time() >= deadline:
            print(
                f"[arena] max-seconds reached after {time.time() - t0:.0f}s: "
                f"stopping with {done}/{n_games} games finished — "
                f"scoring COMPLETE PAIRS only",
                flush=True,
            )
            break
        _refill()  # backfill the slots the reaped games freed — keep the pool full
        if not boards:
            break
        if done - last_report >= report_every:
            print(
                f"[arena] rolling: {done}/{n_games} games done, "
                f"{len(boards)} active ({time.time() - t0:.0f}s)",
                flush=True,
            )
            # Running Elo over the pairs that have BOTH colorings finished so far,
            # so the standings stream in instead of only printing at the end.
            ready = complete_pair_scores(game_scores)
            if ready:
                print(f"[arena] RUNNING Elo after {len(ready)} complete pairs:", flush=True)
                print_summary(summarize_pentanomial(pentanomial_counts(ready)))
            last_report = done
        active = list(range(len(boards)))
        a_to_move, b_to_move = split_active_by_side_to_move(active, boards, awhite)
        for model, idxs, sims, extra, side in (
            (model_candidate, a_to_move, sims_candidate,
             dict(volatility_candidate or {}), search_candidate),
            (model_reference, b_to_move, sims_reference, {}, search_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=side.gumbel,
                gumbel_vloss_weight=side.vloss_weight,
                gumbel_target_batch=side.target_batch,
                **extra,
            )
            apply_actions_to_boards(boards, idxs, actions)
        for i in active:
            gplies[i] += 1

    return complete_pair_scores(game_scores)


# ---------------------------------------------------------------------------
# Game play — matched time (real UCI engine subprocesses)
# ---------------------------------------------------------------------------

def play_paired_games_matched_time(
    candidate_ckpt: str,
    reference_ckpt: str,
    openings: list[chess.Board],
    *,
    device: str,
    ms_per_move: int,
    max_plies: int,
    uci_args: str,
    deadline: float | None = None,
    pgn_sink: PgnSink | None = None,
) -> list[float]:
    """Pair-by-pair UCI match using the production engine inference path.

    ``deadline`` stops between PAIRS. A wall-clock budget is not a search knob
    — it changes nothing about the ruler — so unlike ``--search-shape`` and the
    vloss/target-batch family it is honoured here rather than refused. Pair
    granularity is the natural unit: this loop only ever appends a score once
    both colorings of an opening are played, so a truncated run drops the
    in-progress pair by construction.
    """
    import chess.engine

    from scripts.match_vs_uci import _open_engine, _score_for_a, play_one_game

    limit = chess.engine.Limit(time=float(ms_per_move) / 1000.0)

    def engine_cmd(ckpt: str) -> str:
  # _open_engine shlex-splits the command, so quote anything path-like.
        cmd = (
            f"{shlex.quote(sys.executable)} -m chess_anti_engine.uci "
            f"--checkpoint {shlex.quote(ckpt)} --device {shlex.quote(device)}"
        )
        if uci_args:
            cmd = f"{cmd} {uci_args}"
        return cmd

    eng_a = eng_b = None
    pair_scores: list[float] = []
    try:
        print(f"[arena] starting candidate engine: {engine_cmd(candidate_ckpt)}")
        eng_a = _open_engine(engine_cmd(candidate_ckpt), cwd=str(REPO_ROOT))
        print(f"[arena] starting reference engine: {engine_cmd(reference_ckpt)}")
        eng_b = _open_engine(engine_cmd(reference_ckpt), cwd=str(REPO_ROOT))
        for pair_idx, opening in enumerate(openings):
            if deadline is not None and time.time() >= deadline:
                print(
                    f"[arena] max-seconds reached: stopping before pair "
                    f"{pair_idx + 1}/{len(openings)} with {len(pair_scores)} "
                    f"pairs complete",
                    flush=True,
                )
                break
            scores: list[float] = []
            for a_is_white in (True, False):
                eng_w, eng_b_side = (eng_a, eng_b) if a_is_white else (eng_b, eng_a)
                _g_t0 = time.time()
                record = play_one_game(
                    eng_w, eng_b_side,
                    limit_w=limit, limit_b=limit,
                    enforce_nodes_w=False, enforce_nodes_b=False,
                    max_plies=int(max_plies),
                    start_board=opening,
                    game=(pair_idx, a_is_white),
                )
                scores.append(_score_for_a(record.result, a_is_white=a_is_white))
                if pgn_sink is not None:
                    pgn_sink(
                        pair_id=pair_idx,
                        half=0 if a_is_white else 1,
                        a_is_white=a_is_white,
                        start_fen=record.start_board.fen(),
                        moves=tuple(record.moves),
                        result=record.result,
                        termination=record.termination,
                        plies=record.plies,
                        duration_s=time.time() - _g_t0,
                    )
            pair_scores.append(scores[0] + scores[1])
            print(
                f"[arena] pair {pair_idx + 1}/{len(openings)}: "
                f"pair_score={pair_scores[-1]:.1f} "
                f"running_score={sum(pair_scores) / (2 * len(pair_scores)):.3f}",
                flush=True,
            )
    finally:
        for eng in (eng_a, eng_b):
            if eng is not None:
                try:
                    eng.quit()
                except chess.engine.EngineError:
                    pass  # already dead (e.g. crashed mid-match); keep closing the other
    return pair_scores


# ---------------------------------------------------------------------------
# Result record + JSONL log
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def production_config_hash() -> str:
    try:
        return hashlib.sha256(PRODUCTION_CONFIG.read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def build_result_record(
    summary: PentanomialSummary,
    *,
    mode: str,
    candidate: str,
    reference: str,
    openings_path: str,
    opening_plies: int | None,
    sims_candidate: int | None,
    sims_reference: int | None,
    ms_per_move: int | None,
    temperature: float,
    gumbel_add_noise: bool,
    max_plies: int,
    seed: int,
    device: str,
    duration_s: float,
    openings_kind: str = "book",
    label: str | None = None,
    volatility_candidate: dict[str, float] | None = None,
    search_candidate: SideSearch | None = None,
    search_reference: SideSearch | None = None,
    games_requested: int | None = None,
    max_seconds: float | None = None,
    truncated: bool = False,
) -> dict:
    elo_lo, elo_hi = summary.elo_ci95
    return {
        # WHICH search produced this Elo. Rows without these keys predate
        # 2026-07-29 and were all measured on the play shape at vloss_weight=0
        # regardless of what their argv suggests.
        "search_candidate": None if search_candidate is None else search_candidate.as_record(),
        "search_reference": None if search_reference is None else search_reference.as_record(),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config_hash": production_config_hash(),
        "mode": mode,
        "label": label,
        "volatility_candidate": volatility_candidate,
        "candidate": candidate,
        "reference": reference,
        # `games`/`pairs` are what was PLAYED AND SCORED. `games_requested` and
        # `truncated` are what makes a wall-clock-capped row readable: a 40-game
        # row that asked for 200 is a valid small sample, not a 200-game claim.
        "games": summary.games,
        "pairs": summary.pairs,
        "games_requested": games_requested,
        "max_seconds": max_seconds,
        "truncated": bool(truncated),
        "openings": openings_path,
        "openings_kind": openings_kind,
        "opening_plies": opening_plies,
        "sims_candidate": sims_candidate,
        "sims_reference": sims_reference,
        "ms_per_move": ms_per_move,
        "temperature": temperature,
        "gumbel_add_noise": gumbel_add_noise,
        "max_plies": max_plies,
        "seed": seed,
        "device": device,
        "pentanomial": dict(zip(PAIR_LABELS, summary.counts)),
        "score": round(summary.score, 5),
        "score_se": round(summary.score_se, 5),
        "elo": None if summary.elo is None else round(summary.elo, 2),
        "elo_ci95": [
            None if elo_lo is None else round(elo_lo, 2),
            None if elo_hi is None else round(elo_hi, 2),
        ],
        "duration_s": round(duration_s, 1),
        "argv": sys.argv,
    }


def append_result(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def print_summary(summary: PentanomialSummary) -> None:
    counts = dict(zip(PAIR_LABELS, summary.counts))
    elo_lo, elo_hi = summary.elo_ci95

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:+.1f}"

    print()
    print(f"[arena] {summary.games} games ({summary.pairs} opening pairs)")
    print(f"[arena] pentanomial (candidate POV): {counts}")
    print(f"[arena] score: {summary.score:.4f} +/- {summary.score_se:.4f} (SE)")
    # flush is LOAD-BEARING, not cosmetic, and this ONE line is by itself the
    # whole fix for the 2026-07-30/31 ratchet outage. Every caller redirects
    # stdout to a file, so it is block-buffered, and the ratchet runs this
    # script under `timeout -k 20` — SIGKILL discards the buffer.
    #
    # Buffering is not all-or-nothing: each flush=True print pushes everything
    # written before it, so the loss is always exactly the LAST block. The
    # RUNNING-Elo header above each block flushed while these five lines did
    # not, so the block was only ever saved by the NEXT header. A run that
    # printed several blocks kept all but the last (2026-07-28/29 recorded rows
    # this way); a run slow enough to print only one lost everything, and its
    # log ends EXACTLY at
    #   [arena] RUNNING Elo after 6 complete pairs:
    # so daily_gate_ratchet.sh's `grep '^\[arena\] Elo:'` found nothing and
    # wrote no CSV row. See data/ratchet/arena_2026-07-3*_vs_prev.log
    # .broken_flush_evidence, whose last byte is that header.
    print(f"[arena] Elo: {fmt(summary.elo)}  95% CI: [{fmt(elo_lo)}, {fmt(elo_hi)}]",
          flush=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_arena(
    *,
    candidate: str,
    reference: str,
    games: int,
    openings_path: Path | None,
    opening_plies: int,
    openings_fen: Path | None = None,
    mode: str,
    sims_candidate: int,
    sims_reference: int,
    ms_per_move: int,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    device: str,
    seed: int,
    out_path: Path | None,
    uci_args: str = "",
    label: str | None = None,
    volatility_candidate: dict[str, float] | None = None,
    max_concurrent_games: int = 128,
    report_every: int = 64,
    max_seconds: float | None = None,
    syzygy_path: str | None = None,
    tb_max_pieces: int = 6,
    compile_models: bool = True,
    rolling: bool = True,
    search_candidate: SideSearch | None = None,
    search_reference: SideSearch | None = None,
    pgn_out: Path | None = None,
    pgn_candidate_name: str | None = None,
    pgn_reference_name: str | None = None,
) -> dict:
    """Run one standardized arena and return (and optionally log) the record.

    ``search_candidate`` / ``search_reference`` are mandatory for
    ``matched_sims`` — this is the single funnel every arena entry point goes
    through (arena_standard, elo_vs_sims, anything future), so refusing to run
    without an explicit shape here is what makes "which search did I measure?"
    unanswerable-by-omission impossible. ``matched_time`` runs real UCI
    subprocesses, which carry their own play shape; pass search knobs there via
    ``--uci-args`` instead.
    """
    if games < 2 or games % 2 != 0:
        raise SystemExit("--games must be even and >= 2 (paired openings)")
    if mode == "matched_sims" and (search_candidate is None or search_reference is None):
        raise SystemExit(
            "matched_sims needs an explicit search shape: pass --search-shape "
            f"{{{'|'.join(SEARCH_SHAPES)}}}. 'training' is what production selfplay "
            f"runs, read from {PRODUCTION_CONFIG.name} at run time (linear root; "
            "c_scale/topk/vloss_weight/target_batch all come from that yaml, so "
            "they are not quoted here); 'play' is the tuned UCI/match shape "
            "(c_scale 0.025, topk 32, log root, vloss_weight 3). There is no "
            "default because the silent one was wrong for every arena in "
            "docs/experiment_ledger.md."
        )
    if mode == "matched_time" and (search_candidate is not None or search_reference is not None):
        raise SystemExit(
            "matched_time plays through UCI engine subprocesses, which use their "
            "own play shape; --search-shape cannot apply. Use --uci-args."
        )
    n_pairs = games // 2
    rng = np.random.default_rng(seed)
    # Anchored HERE, not at the play loop, so opening sampling and the two
    # ~700MB checkpoint loads are inside the budget the caller granted — they
    # are what the caller's `timeout` counts, and on a contended box they are
    # not small (data/ratchet/arena_2026-07-31_vs_boot512.log spent its whole
    # 880s window before the play loop started).
    deadline = None if max_seconds is None else time.time() + float(max_seconds)

    if (openings_path is None) == (openings_fen is None):
        raise SystemExit("exactly one of openings_path / openings_fen is required")
    if openings_fen is not None:
        print(f"[arena] loading FEN openings from {openings_fen} (seed={seed})")
        openings = load_fen_openings(openings_fen, n_pairs=n_pairs, rng=rng)
    else:
        assert openings_path is not None
        print(f"[arena] sampling {n_pairs} openings from {openings_path} (seed={seed})")
        openings = load_paired_openings(
            openings_path, n_pairs=n_pairs, max_plies=opening_plies, rng=rng,
        )

    pgn_writer: ArenaPgnWriter | None = None
    pgn_sink: PgnSink | None = None
    if pgn_out is not None:
        cand_name = pgn_candidate_name or engine_name_from_checkpoint(
            candidate, fallback="candidate")
        ref_name = pgn_reference_name or engine_name_from_checkpoint(
            reference, fallback="reference")
        if cand_name == ref_name:
            # Two players that share a name become ONE player in a pooled fit,
            # and the resulting rating is a silent average of both. Refuse
            # rather than emit a PGN that fits cleanly and means nothing.
            raise SystemExit(
                f"--pgn-out: candidate and reference both resolve to the engine "
                f"name {cand_name!r}. Pass --pgn-candidate-name / "
                f"--pgn-reference-name to distinguish them."
            )
        no_shape = "n/a (matched_time: UCI subprocess play shape)"
        cand_search = no_shape if search_candidate is None else search_candidate.describe()
        ref_search = no_shape if search_reference is None else search_reference.describe()
        base_tags = {
            "ConfigHash": production_config_hash(),
            "GitSha": git_sha(),
            "ArenaMode": mode,
        }
        if label:
            base_tags["ArenaLabel"] = label
        pgn_writer = ArenaPgnWriter(pgn_out, event=label or "arena", base_tags=base_tags)
        print(f"[arena] PGN output -> {pgn_out} "
              f"(White/Black = {cand_name} / {ref_name})", flush=True)

        def _emit_pgn(
            *,
            pair_id: int,
            half: int,
            a_is_white: bool,
            start_fen: str,
            moves: tuple[chess.Move, ...],
            result: str,
            termination: str,
            plies: int,
            duration_s: float,
        ) -> None:
            assert pgn_writer is not None
            pgn_writer.write_game(ArenaGame(
                white=cand_name if a_is_white else ref_name,
                black=ref_name if a_is_white else cand_name,
                result=result,
                moves=moves,
                start_fen=start_fen,
                pair_id=pair_id,
                pair_half=half,
                extra={
                    "WhiteSearch": cand_search if a_is_white else ref_search,
                    "BlackSearch": ref_search if a_is_white else cand_search,
                    "Termination": termination,
                    # For the informative-missingness check: if pairs complete
                    # at different rates across matchups, the pairs a partial
                    # run HAS from a slow matchup are its FAST ones, which are
                    # systematically more decisive. No bootstrap fixes that, so
                    # the evidence to detect it has to be IN the file.
                    "Plies": str(int(plies)),
                    "GameDurationSec": f"{float(duration_s):.2f}",
                },
            ))

        pgn_sink = _emit_pgn

    t0 = time.time()
    if mode == "matched_sims":
        from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

  # `require_complete` is left at its auto default, which is per-side, not
  # blanket: a side whose checkpoint embeds its own `arch` must load exactly
  # or raise here, so a partially fresh-initialised net cannot quietly produce
  # a lopsided Elo. A side WITHOUT embedded arch (older checkpoints, resolved
  # via params.json) still loads tolerantly -- the arch is a guess there, and
  # demanding exactness would break legitimate reads of the pre-`arch` era. So
  # this enforces audit method rule 7 in code for arch-bearing checkpoints and
  # leaves it a human habit for the rest; check for a `Tolerant load` line on
  # the console before believing a lopsided result off an old checkpoint.
        assert search_candidate is not None
        assert search_reference is not None
        print(f"[arena] SEARCH candidate: {search_candidate.describe()}", flush=True)
        print(f"[arena] SEARCH reference: {search_reference.describe()}", flush=True)
        # Flushed, and timed. data/ratchet/arena_2026-07-31_vs_boot512.log spent
        # its entire 880s window somewhere in here and the log could not say
        # whether loading had even started, because these two lines were
        # buffered behind the next flushed print. A load that takes minutes is a
        # real signal (GPU/disk contention) and has to be visible while it is
        # happening, not only if the process survives to the next flush.
        print(f"[arena] loading candidate: {candidate}", flush=True)
        _t_load = time.time()
        model_candidate = load_model_from_checkpoint(candidate, device=device)
        print(f"[arena] loading reference: {reference} "
              f"(candidate loaded in {time.time() - _t_load:.0f}s)", flush=True)
        _t_load = time.time()
        model_reference = load_model_from_checkpoint(reference, device=device)
        print(f"[arena] both checkpoints loaded "
              f"(reference in {time.time() - _t_load:.0f}s)", flush=True)
        if compile_models:
            import torch
            # Plain inductor compile (NOT reduce-overhead/cudagraphs, which recompile
            # per batch shape and OOM'd us). Auto-dynamic batch: a couple of warmup
            # recompiles in chunk 1, then cached + reused across the shrinking batch
            # sizes and into chunk 2.
            model_candidate = torch.compile(model_candidate)
            model_reference = torch.compile(model_reference)
            print("[arena] torch.compile ON (inductor, auto-dynamic batch)", flush=True)
        print(
            f"[arena] matched_sims: candidate={sims_candidate} sims/move, "
            f"reference={sims_reference} sims/move, temp={temperature}, "
            f"noise={gumbel_add_noise}"
        )
        # Syzygy adjudication: end each game the instant it reaches a covered
        # (<=N-man) position, so long endgame tails don't dominate the wall clock
        # (reuses match_vs_uci's WDL probe). Opened once, shared across chunks.
        syzygy_tb = None
        if syzygy_path:
            from scripts.match_vs_uci import _open_syzygy_tablebase
            try:
                syzygy_tb = _open_syzygy_tablebase(syzygy_path)
            except Exception as exc:
                syzygy_tb = None
                print(f"[arena] WARNING: syzygy open failed ({exc})", flush=True)
            print(
                f"[arena] syzygy adjudication {'ON' if syzygy_tb is not None else 'OFF'} "
                f"(<={tb_max_pieces}-man, {syzygy_path})",
                flush=True,
            )
        if rolling:
            # Rolling pool: fixed active-game count => fixed batch shape => compile
            # reuses one graph (no per-shape thrash), and the GPU never drains until
            # the very end (no per-chunk tail).
            print(
                f"[arena] ROLLING pool: keep {max_concurrent_games} games active, "
                f"start a fresh one as each finishes",
                flush=True,
            )
            pair_scores = play_paired_games_matched_sims_rolling(
                model_candidate, model_reference, openings,
                device=device, rng=rng,
                sims_candidate=sims_candidate, sims_reference=sims_reference,
                max_plies=max_plies, temperature=temperature,
                gumbel_add_noise=gumbel_add_noise,
                volatility_candidate=volatility_candidate,
                syzygy_tablebase=syzygy_tb, tb_max_pieces=tb_max_pieces,
                pool_size=int(max_concurrent_games),
                search_candidate=search_candidate, search_reference=search_reference,
                report_every=int(report_every),
                deadline=deadline,
                pgn_sink=pgn_sink,
            )
        else:
            # Chunked: plays each chunk of `max_concurrent_games` to completion
            # (drains per chunk). Numerically identical (pair scores concatenate).
            chunk_pairs = max(1, int(max_concurrent_games) // 2)
            n_chunks = (len(openings) + chunk_pairs - 1) // chunk_pairs
            pair_scores = []
            for ci in range(0, len(openings), chunk_pairs):
                # Chunk granularity: a chunk plays to completion, so this stops
                # BEFORE starting one that would run past the budget rather
                # than mid-chunk. Rolling (the default, and what the ratchet
                # runs) stops per ply.
                if deadline is not None and time.time() >= deadline:
                    print(
                        f"[arena] max-seconds reached: stopping before chunk "
                        f"{ci // chunk_pairs + 1}/{n_chunks} with "
                        f"{len(pair_scores)} pairs complete",
                        flush=True,
                    )
                    break
                sub = openings[ci:ci + chunk_pairs]
                print(
                    f"[arena] matched_sims chunk {ci // chunk_pairs + 1}/{n_chunks}: "
                    f"{len(sub)} pairs ({2 * len(sub)} games)",
                    flush=True,
                )
                pair_scores.extend(play_paired_games_matched_sims(
                    model_candidate, model_reference, sub,
                    device=device, rng=rng,
                    sims_candidate=sims_candidate, sims_reference=sims_reference,
                    max_plies=max_plies, temperature=temperature,
                    gumbel_add_noise=gumbel_add_noise,
                    volatility_candidate=volatility_candidate,
                    syzygy_tablebase=syzygy_tb, tb_max_pieces=tb_max_pieces,
                    search_candidate=search_candidate,
                    search_reference=search_reference,
                    pgn_sink=pgn_sink,
                    pair_id_offset=ci,
                ))
                print(f"[arena] RUNNING Elo after {2 * len(pair_scores)} games:", flush=True)
                print_summary(summarize_pentanomial(pentanomial_counts(pair_scores)))
        if syzygy_tb is not None:
            syzygy_tb.close()
    elif mode == "matched_time":
        print(f"[arena] matched_time: {ms_per_move}ms/move per side")
        pair_scores = play_paired_games_matched_time(
            candidate, reference, openings,
            device=device, ms_per_move=ms_per_move, max_plies=max_plies,
            uci_args=uci_args, deadline=deadline, pgn_sink=pgn_sink,
        )
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    duration_s = time.time() - t0
    if pgn_writer is not None:
        # Every game was already flushed as it finished, so a crash before this
        # point still leaves a complete, parseable PGN of the games that ended —
        # closing here just releases the handle on the normal path.
        n_written = pgn_writer.games_written
        pgn_writer.close()
        print(f"[arena] wrote {n_written} games to {pgn_out}", flush=True)

    # Against the openings ACTUALLY loaded, not the requested `n_pairs`:
    # load_fen_openings uses every row of a short FEN file rather than padding
    # to games//2, so comparing with n_pairs stamped truncated=True on every
    # complete --openings-fen run with a small seed list.
    truncated = bool(len(pair_scores) < len(openings))
    if not pair_scores:
        # Nothing finished. Say so on stdout and exit non-zero instead of
        # letting summarize_pentanomial raise "no pairs" from deep in the
        # stack: the caller (daily_gate_ratchet.sh) distinguishes "ran and
        # found nothing" from "crashed" only by what reaches the log.
        print(
            f"[arena] NO COMPLETE PAIRS in {duration_s:.0f}s — nothing to score. "
            f"Raise --max-seconds, lower --games/--sims, or check GPU contention.",
            flush=True,
        )
        raise SystemExit(3)
    if truncated:
        print(
            f"[arena] TRUNCATED: {len(pair_scores)}/{len(openings)} opening pairs "
            f"completed in {duration_s:.0f}s",
            flush=True,
        )
    summary = summarize_pentanomial(pentanomial_counts(pair_scores))
    print_summary(summary)

    record = build_result_record(
        summary,
        mode=mode,
        candidate=candidate,
        reference=reference,
        # Keep openings_path a real filesystem path; record the source kind in
        # its own field so downstream readers can tell FEN-list runs from book
        # runs without the path string changing shape. opening_plies is a book
        # concept, so it is null on the FEN path (it is never applied there).
        openings_path=str(openings_fen if openings_fen is not None else openings_path),
        openings_kind="fen" if openings_fen is not None else "book",
        opening_plies=None if openings_fen is not None else opening_plies,
        sims_candidate=sims_candidate if mode == "matched_sims" else None,
        sims_reference=sims_reference if mode == "matched_sims" else None,
        ms_per_move=ms_per_move if mode == "matched_time" else None,
        temperature=temperature,
        gumbel_add_noise=gumbel_add_noise,
        max_plies=max_plies,
        seed=seed,
        device=device,
        duration_s=duration_s,
        label=label,
        volatility_candidate=volatility_candidate,
        search_candidate=search_candidate,
        search_reference=search_reference,
        games_requested=games,
        max_seconds=None if max_seconds is None else float(max_seconds),
        truncated=truncated,
    )
    if out_path is not None:
        append_result(record, out_path)
        print(f"[arena] result appended to {out_path}")
    return record


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arena knobs shared with scripts/elo_vs_sims.py."""
    # No default, on purpose. The silent default (`play`) is what made every
    # arena Elo in the ledger a measurement of a search selfplay never runs;
    # run_arena refuses matched_sims without it.
    p.add_argument("--search-shape", choices=SEARCH_SHAPES, default=None,
                   help="REQUIRED for matched_sims: which search to measure. "
                        "'training' = what production selfplay runs: linear root, "
                        "with c_scale/topk/vloss_weight/target_batch read from "
                        f"{PRODUCTION_CONFIG.name} at run time (deliberately not "
                        "quoted here — they change with the config; the realized "
                        "values are printed at startup and stored in the result "
                        "record). 'play' = the tuned UCI/match shape (c_scale "
                        "0.025, topk 32, log root, vloss_weight 3). Use 'training' "
                        "to judge anything about the training loop.")
    p.add_argument("--games", type=int, default=1000,
                   help="total games; must be even — games/2 opening pairs (default: 1000)")
    p.add_argument("--openings", type=Path, default=None,
                   help="PGN(.zip)/Polyglot opening book "
                   "(default: the 8-move UHO book from configs/pbt2_small.yaml)")
    p.add_argument("--opening-plies", type=int, default=16,
                   help="book plies to apply per opening (default: 16 = 8 moves)")
    p.add_argument("--max-plies", type=int, default=300,
                   help="adjudicate as draw after this many plies (default: 300)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="move-selection temperature for matched_sims (default: 0.1)")
    p.add_argument("--no-gumbel-noise", action="store_true",
                   help="disable root Gumbel noise in matched_sims (fully deterministic; "
                   "self-play pairs then carry zero information)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS_PATH,
                   help=f"JSONL results log (default: {DEFAULT_RESULTS_PATH})")


def _volatility_kwargs_from_args(args) -> dict[str, float] | None:
    """CANDIDATE-side volatility search kwargs, or None when all flags are off."""
    if float(args.volatility_q_scale) == 0.0 and float(args.volatility_fpu) == 0.0:
        return None
    if args.mode != "matched_sims":
        raise SystemExit(
            "--volatility-* flags require --mode matched_sims (the Python "
            "search path is slower, so matched_time would under-credit it)"
        )
    out = {
        "volatility_q_scale": float(args.volatility_q_scale),
        "volatility_fpu": float(args.volatility_fpu),
    }
    if args.volatility_anchor is not None:
        out["volatility_anchor"] = float(args.volatility_anchor)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candidate", required=True, help="candidate checkpoint (trainer.pt or dir)")
    p.add_argument("--reference", required=True, help="reference checkpoint (trainer.pt or dir)")
    p.add_argument("--pgn-out", type=Path, default=None,
                   help="ALSO append every finished game to this PGN, for a "
                        "pooled multi-player rating fit (Ordo/BayesElo). Default "
                        "OFF; when unset nothing about the run changes. Games are "
                        "flushed as they finish, so a killed run still leaves a "
                        "valid PGN. Tags carry the engine names, ConfigHash, "
                        "GitSha and BOTH sides' realized search shape, plus "
                        "PairId/PairHalf so a pair-level block bootstrap can "
                        "recover the pairing Ordo itself ignores.")
    p.add_argument("--pgn-candidate-name", default=None,
                   help="stable engine identity for the candidate in --pgn-out "
                        "(default: last two checkpoint path components, "
                        "sanitized). Set it explicitly when pooling across runs "
                        "— two arms that share a name become ONE player.")
    p.add_argument("--pgn-reference-name", default=None,
                   help="stable engine identity for the reference in --pgn-out")
    p.add_argument("--mode", choices=["matched_sims", "matched_time"],
                   default="matched_sims")
    p.add_argument("--sims", type=int, default=64,
                   help="MCTS sims/move for both sides in matched_sims (default: 64)")
    p.add_argument("--sims-candidate", type=int, default=None,
                   help="override candidate sims/move (defaults to --sims)")
    p.add_argument("--sims-reference", type=int, default=None,
                   help="override reference sims/move (defaults to --sims)")
    p.add_argument("--max-concurrent-games", type=int, default=128,
                   help="matched_sims: cap simultaneous games per batch to bound "
                        "GPU memory; total --games still played in chunks "
                        "(default: 128). Lower if you OOM on a small card.")
    p.add_argument("--report-every", type=int, default=64,
                   help="rolling mode: print a RUNNING Elo block every N finished "
                        "games (default: 64). Lower it when the run is under a "
                        "wall-clock cap — a capped run that never reaches the "
                        "first report yields NO reading at all and the GPU time "
                        "is simply lost. 2026-07-26: a 32-sim rung was stopped at "
                        "18 min having printed nothing, because the first block "
                        "only lands at 64 games.")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop after this many wall-clock seconds and score the "
                        "COMPLETE PAIRS finished so far, printing the summary and "
                        "writing the JSONL record normally. Use this instead of "
                        "relying on an external `timeout`: a SIGKILLed arena "
                        "returns NOTHING (the ratchet lost every reading it "
                        "computed on 2026-07-30/31 that way). The budget covers "
                        "opening sampling and checkpoint loading too.")
    p.add_argument("--syzygy", default=None,
                   help="matched_sims: colon-separated Syzygy dir(s) to adjudicate "
                        "games the instant they reach a covered position (kills "
                        "long endgame tails). e.g. data/syzygy_3-4-5")
    p.add_argument("--syzygy-max-pieces", type=int, default=6,
                   help="adjudicate positions with <= this many men (default: 6)")
    p.add_argument("--compile", choices=["auto", "on", "off"], default="auto",
                   help="matched_sims torch.compile policy (default: auto). "
                        "on=always compile; off=eager; auto=compile only when "
                        f"games*sims >= {AUTO_COMPILE_WORK_THRESHOLD} (else the "
                        "~2-4 min compile cost isn't worth it for a quick run).")
    p.add_argument("--no-compile", action="store_true",
                   help="matched_sims: back-compat alias for --compile off "
                        "(disables torch.compile regardless of --compile).")
    p.add_argument("--compile-cache-dir", type=Path, default=None,
                   help="matched_sims: torch.compile/triton cache root to reuse "
                        "(default: the production worker cache, so arena reuses "
                        "the kernels training already baked). Overridable; an "
                        "absent dir just means no reuse that run.")
    p.add_argument("--no-rolling", action="store_true",
                   help="matched_sims: disable the rolling pool and use fixed chunks "
                        "instead. Rolling (on by default) keeps a fixed pool of "
                        "--max-concurrent-games active, starting a fresh game as each "
                        "finishes => fixed batch shape so compile reuses one graph + "
                        "no per-chunk drain tail.")
    p.add_argument("--ms-per-move", type=int, default=100,
                   help="per-move wall clock for matched_time (default: 100)")
    p.add_argument("--uci-args", default="",
                   help="extra args appended to both UCI engine commands in matched_time "
                   "(e.g. '--no-compile')")
    p.add_argument("--label", default=None, help="free-form tag stored in the JSONL record")
    p.add_argument("--openings-fen", type=Path, default=None,
                   help="plain FEN file (one per line, # comments) used as paired "
                        "openings instead of a PGN/Polyglot book — for blind-spot "
                        "seed-list play-outs; mutually exclusive with --openings")
    p.add_argument("--volatility-q-scale", type=float, default=0.0,
                   help="CANDIDATE-side volatility-aware sigma(q) exponent "
                        "(matched_sims only; forces the Python search path)")
    p.add_argument("--volatility-fpu", type=float, default=0.0,
                   help="CANDIDATE-side pessimistic FPU coefficient (matched_sims only)")
    p.add_argument("--volatility-anchor", type=float, default=None,
                   help="dataset-mean volatility anchor override (see exp_volatility_search.yaml)")
    p.add_argument("--cand-gumbel", default=None,
                   help="candidate gumbel knob overrides as k=v,k=v "
                        "(c_scale,c_visit,c_visit_root,c_scale_root,topk,halving_div,"
                        "policy_temp). Use the SAME checkpoint for "
                        "--candidate/--reference + differing gumbel here = a pure "
                        "search-config Swiss (matched_sims). c_puct/fpu_reduction/"
                        "cpuct_factor/cpuct_base are REJECTED: they cannot affect a "
                        "Gumbel search (audit 2026-08-03 F2).")
    p.add_argument("--ref-gumbel", default=None,
                   help="reference gumbel knob overrides (same k=v,k=v format as --cand-gumbel)")
    p.add_argument("--cand-vloss-weight", type=int, default=None,
                   help="candidate virtual-loss weight override (C path). NOT a "
                        "GumbelConfig field, so --cand-gumbel cannot carry it; "
                        "defaults to the --search-shape value.")
    p.add_argument("--ref-vloss-weight", type=int, default=None,
                   help="reference virtual-loss weight override (see --cand-vloss-weight)")
    p.add_argument("--cand-target-batch", type=int, default=None,
                   help="candidate leaf-accumulation target override (C path); "
                        "defaults to the --search-shape value")
    p.add_argument("--ref-target-batch", type=int, default=None,
                   help="reference leaf-accumulation target override (see --cand-target-batch)")
    add_common_args(p)
    args = p.parse_args()

    if args.openings_fen is not None and args.openings is not None:
        raise SystemExit("--openings-fen and --openings are mutually exclusive")
    openings_path = (
        None if args.openings_fen is not None
        else (args.openings if args.openings is not None else default_openings_path())
    )
    # Detect an EXPLICIT --opening-plies (any value, incl. the default 16) via
    # argv, since comparing to the default can't tell "passed 16" from "unset".
    opening_plies_passed = any(
        a == "--opening-plies" or a.startswith("--opening-plies=") for a in sys.argv[1:]
    )
    if args.openings_fen is not None and opening_plies_passed:
        raise SystemExit(
            "--opening-plies has no effect with --openings-fen (FEN seeds are "
            "whole positions, not book truncations); drop it")

    # The FEN path caps the game count at 2*usable_rows, which main() must know
    # for the compile heuristic below — a small seed file over-triggers compile
    # if we size the work off the requested --games. _load_fen_list is cached,
    # so run_arena's later load reuses this read.
    effective_games = int(args.games)
    if args.openings_fen is not None:
        effective_games = min(effective_games, 2 * load_fen_seed_count(args.openings_fen))

    # Resolve the matched_sims torch.compile decision (matched_time is a UCI
    # subprocess and unaffected). --no-compile is the back-compat "off" alias and
    # wins over --compile. auto compiles only when the run does enough inference
    # work to amortize the ~2-4 min compile cost.
    sims_cand = int(args.sims if args.sims_candidate is None else args.sims_candidate)
    sims_ref = int(args.sims if args.sims_reference is None else args.sims_reference)
    effective_sims = max(sims_cand, sims_ref)
    compile_mode = "off" if args.no_compile else args.compile
    if compile_mode == "off":
        compile_models = False
        if args.mode == "matched_sims":
            print("[arena] compile=off -> EAGER", flush=True)
    elif compile_mode == "on":
        compile_models = True
        if args.mode == "matched_sims":
            print("[arena] compile=on -> COMPILE", flush=True)
    else:  # auto
        work = int(effective_games) * int(effective_sims)
        compile_models = work >= AUTO_COMPILE_WORK_THRESHOLD
        if args.mode == "matched_sims":
            if compile_models:
                print(
                    f"[arena] compile=auto -> COMPILE "
                    f"(games*sims={work} >= {AUTO_COMPILE_WORK_THRESHOLD})",
                    flush=True,
                )
            else:
                print(
                    f"[arena] compile=auto -> EAGER (games*sims={work} < "
                    f"{AUTO_COMPILE_WORK_THRESHOLD}; compile overhead not worth it)",
                    flush=True,
                )

    # Point inductor/triton at the shared training cache BEFORE torch is imported
    # in run_arena, so the compile (when on) reuses already-baked kernels.
    if compile_models and args.mode == "matched_sims":
        cache_dir = (
            args.compile_cache_dir.expanduser().resolve()
            if args.compile_cache_dir is not None
            else default_compile_cache_dir()
        )
        _configure_shared_compile_cache(cache_dir)

    # Resolve the search shape BEFORE any model load, so a missing/bad
    # --search-shape fails in a second rather than after a 4-minute compile.
    # matched_time carries no in-process search: leave both sides None and let
    # run_arena reject the combination.
    side_candidate = side_reference = None
    if args.mode == "matched_sims":
        if args.search_shape is None:
            raise SystemExit(
                "--search-shape is required for matched_sims "
                f"({'|'.join(SEARCH_SHAPES)}); see the flag help. Use 'training' "
                "for anything judging the training loop."
            )
        base = resolve_search_shape(args.search_shape)
        side_candidate = apply_search_overrides(
            base, spec=args.cand_gumbel,
            vloss_weight=args.cand_vloss_weight,
            target_batch=args.cand_target_batch,
        )
        side_reference = apply_search_overrides(
            base, spec=args.ref_gumbel,
            vloss_weight=args.ref_vloss_weight,
            target_batch=args.ref_target_batch,
        )
    else:
        # matched_time launches UCI subprocesses, which carry their own search.
        # EVERY in-process search flag is inert here, not just --search-shape:
        # accepting `--cand-vloss-weight 5` and running something else is the
        # exact accepted-then-ignored defect this script was just fixed for, and
        # it would be newly introduced by the fix. Refuse the whole family.
        inert = [
            flag for flag, value in (
                ("--search-shape", args.search_shape),
                ("--cand-gumbel", args.cand_gumbel),
                ("--ref-gumbel", args.ref_gumbel),
                ("--cand-vloss-weight", args.cand_vloss_weight),
                ("--ref-vloss-weight", args.ref_vloss_weight),
                ("--cand-target-batch", args.cand_target_batch),
                ("--ref-target-batch", args.ref_target_batch),
            ) if value is not None
        ]
        if inert:
            raise SystemExit(
                f"{', '.join(inert)} cannot apply to matched_time: it plays through "
                "UCI engine subprocesses, which build their own search from their "
                "own flags. Pass engine search settings via --uci-args, or use "
                "--mode matched_sims."
            )

    run_arena(
        candidate=args.candidate,
        reference=args.reference,
        games=args.games,
        max_concurrent_games=args.max_concurrent_games,
        report_every=args.report_every,
        max_seconds=args.max_seconds,
        syzygy_path=args.syzygy,
        tb_max_pieces=args.syzygy_max_pieces,
        compile_models=compile_models,
        rolling=not args.no_rolling,
        openings_path=openings_path,
        openings_fen=args.openings_fen,
        opening_plies=args.opening_plies,
        mode=args.mode,
        sims_candidate=sims_cand,
        sims_reference=sims_ref,
        ms_per_move=args.ms_per_move,
        max_plies=args.max_plies,
        temperature=args.temperature,
        gumbel_add_noise=not args.no_gumbel_noise,
        device=args.device,
        seed=args.seed,
        out_path=args.out,
        pgn_out=args.pgn_out,
        pgn_candidate_name=args.pgn_candidate_name,
        pgn_reference_name=args.pgn_reference_name,
        uci_args=args.uci_args,
        label=args.label,
        volatility_candidate=_volatility_kwargs_from_args(args),
        search_candidate=side_candidate,
        search_reference=side_reference,
    )


if __name__ == "__main__":
    main()
