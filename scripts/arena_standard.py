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

Fixed-N and final-only is the DEFAULT and stays that way: reading a rolling
arena and stopping when it looked good manufactured +112 Elo out of a true null.
``--sprt`` is the opt-in alternative — a pentanomial GSPRT against a boundary
declared before the first game, where ``--games`` becomes a hard cap and the
deliverable is an H1/H0/INCONCLUSIVE verdict::

    PYTHONPATH=. python3 scripts/arena_standard.py \\
        --candidate ... --reference ... --games 1000 \\
        --sprt 'elo0=0,elo1=5,alpha=0.05,beta=0.05'
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import chess
import numpy as np

if TYPE_CHECKING:
    from chess_anti_engine.eval.production_shape import LiveConfig

from chess_anti_engine.eval.arena_pgn import (
    ArenaGame,
    ArenaPgnWriter,
    engine_name_from_checkpoint,
)
from chess_anti_engine.eval.sprt import SprtMonitor, SprtSpec
from chess_anti_engine.moves import ActionDecodeError
from chess_anti_engine.utils.game_log import (
    GameLogWriter,
    default_game_log_path,
    fingerprint_differences,
    latest_rows_by_key,
    read_game_log,
    refuse_existing_log_message,
    refuse_settings_mismatch_message,
    settings_fingerprint,
    take_over_header_only_log,
)

# Called once per FINISHED game. Keyword-only so the three play loops (rolling /
# chunked / matched_time) cannot silently pass these positionally in different
# orders. It used to be installed only for --pgn-out; it is now ALWAYS
# installed, because the per-game JSONL that makes a crashed arena resumable
# rides the same hook.
PgnSink = Callable[..., None]

REPO_ROOT = Path(__file__).resolve().parent.parent


# The FALLBACK file, not the answer. `production_config()` below is the single
# resolution; this is only what it lands on when the live config cannot be
# resolved. Named for what it is so no caller mistakes it for "the production
# config" and reads it directly — that mistake is the whole finding.
IN_TREE_CONFIG = REPO_ROOT / "configs" / "pbt2_small.yaml"

# GumbelConfig fields on which an ARENA legitimately differs from production
# selfplay. Reasons, not just names — an undocumented deviation has nowhere to
# go. The two target_* knobs are the interesting entries: they shape the
# STORED TRAINING TARGET only and have no effect on which move is played, so
# an arena that omits them is still measuring production's PLAY behaviour.
# (`scripts/audit_targets.py` scores the target itself, and there they are
# mandatory — same knobs, opposite verdicts, because the instruments measure
# different objects.)
ARENA_SHAPE_DEVIATIONS: dict[str, str] = {
    "simulations": "the arena's own budget (--sims / matched_time), not the yaml's",
    "temperature": "arena move selection, not selfplay's stored-target temperature",
  # ⚑ THE REASON THAT USED TO SIT HERE WAS FALSE, and the falsehood was the
  # load-bearing part: it read "no root Gumbel noise: an arena must be
  # deterministic per seed" while the CLI passes
  # `gumbel_add_noise=not args.no_gumbel_noise`, i.e. noise is ON by default.
  # Worse, the arena's scale is the `GumbelConfig` default 1.0, while
  # production runs 0.75 selfplay / 0.25 curriculum and DECAYS both to 0 after
  # move 12 — a per-ply schedule (`network_turn._scheduled_gumbel_scale`)
  # applied outside `build_selfplay_gumbel_config`, so no field comparison can
  # express it and this guard cannot see it. An exemption whose reason is
  # wrong is worse than no exemption: it answers the question a reader would
  # otherwise ask.
  #
  # Kept as an exemption rather than a refusal because the arena genuinely
  # cannot reproduce a per-ply schedule through a flat override dict, and
  # refusing would take `--search-shape training` away entirely. What changed
  # is that `_warn_noise_schedule_deviation` now PRINTS the divergence with
  # production's actual numbers on every noisy training arena. The JSONL record
  # already banks `gumbel_add_noise` (`_result_record`), so the artifact carries
  # which of the two regimes a row was measured in.
    "add_noise": (
        "arena-level flag (--no-gumbel-noise), not the yaml's. Production "
        "selfplay always enables noise and modulates it through the per-ply "
        "gumbel_scale schedule instead"
    ),
    "gumbel_scale": (
        "production's scale is a per-PLY schedule (0.75 selfplay / 0.25 "
        "curriculum, decaying to 0 after move 12) that a flat override dict "
        "cannot express; the arena runs the flat GumbelConfig default. "
        "Reported by _warn_noise_schedule_deviation rather than hidden here"
    ),
    "target_max_visit_cap": (
        "TARGET-only knob — shapes the stored policy target, never the played "
        "move, and an arena stores no targets"
    ),
    "target_untempered_prior": (
        "TARGET-only knob — undoes policy_temp on the stored target's prior "
        "term only, so it cannot change which move the arena plays"
    ),
    "input_history_encoding": "read off the loaded checkpoint, not the yaml",
    "input_extra_features": "read off the loaded checkpoint, not the yaml",
    "policy_encoding": "read off the loaded checkpoint, not the yaml",
    "compute_relations": "read off the loaded checkpoint, not the yaml",
}
DEFAULT_RESULTS_PATH = REPO_ROOT / "runs" / "arena_results.jsonl"
# Per-run game logs. NOT derived from --out: that is a shared, append-only
# aggregate (every arena ever run appends to runs/arena_results.jsonl, and the
# ratchet never passes --out at all), so `<out stem>.games.jsonl` would be ONE
# file for every arena in history — which either mixes every run's games or
# makes the "already exists" guard fire on every run. The default log name
# carries the settings fingerprint instead; see default_game_log_path.
DEFAULT_GAME_LOG_DIR = REPO_ROOT / "runs" / "arena_games"

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
#             purpose: every value is READ FROM `production_config()` at call
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

    def __post_init__(self) -> None:
        """Refuse a side carrying a knob value the search will not run.

        Here rather than in ``apply_search_overrides`` because EVERY side is
        built through this constructor -- the resolved shape, the CLI overrides
        layered on top of it, and any programmatic caller (``elo_vs_sims.py``)
        -- so "a SideSearch that exists is one the search will actually run" is
        structural instead of being true of the two call sites someone
        remembered. It still fires minutes early: ``main()`` resolves both sides
        before any checkpoint is loaded or compiled.

        The check itself is ``mcts.gumbel.validate_gumbel_config``, the ONE
        home of the bands, applied to the config exactly as ``match.py`` will
        (``dataclasses.replace`` onto a ``GumbelConfig``) so the guard shares
        the criterion's instrument.

        The hole it closes: ``--cand-gumbel policy_temp=1e300`` reached
        ``dataclasses.replace`` without passing the yaml loader's validator, so
        the search ran UNTEMPERED (``policy_temp_active(1e300)`` is False) while
        ``realized_gumbel`` banked 1e300 into the JSONL as this side's realized
        setting. A sweep over out-of-band temperatures is then a set of
        IDENTICAL arms recorded as different ones -- the c_puct Swiss (audit
        2026-08-03 F2) with a live knob instead of a dead one.

        ⚑ ``frozen=True`` freezes the ATTRIBUTES, not the ``gumbel`` dict they
        point at: ``side.gumbel["policy_temp"] = 1e300`` after construction
        mutates in place and never re-enters this check. Nothing in this script
        does that (every layer builds a new ``SideSearch``), and copying the
        dict would only move the hole one alias further out, so this is recorded
        rather than defended against -- but a future in-place edit is the way
        past the guard, and it should build a new side instead.
        """
        import dataclasses as _dc

        from chess_anti_engine.mcts.gumbel import GumbelConfig, validate_gumbel_config

        try:
            validate_gumbel_config(
                _dc.replace(GumbelConfig(), **self.gumbel), where=f"[shape] {self.source}",
            )
        except ValueError as exc:
            raise SystemExit(
                f"{exc}. Refusing rather than dropping it is deliberate: the "
                "value does reach GumbelConfig, so nothing downstream would "
                "notice, and realized_gumbel() would bank it as this side's "
                "realized search."
            ) from exc

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


def production_config() -> LiveConfig:
    """THE resolution. Every consumer in this file goes through it.

    ⚑ There used to be two: a module constant ``PRODUCTION_CONFIG`` resolved at
    IMPORT time, which backed the openings default, the banked config digest
    and every provenance string, and a separate call-time
    ``load_live_config()`` that backed the search shape. The commit that
    introduced the second one claimed it had removed the first. It had not, and
    a divergence between them is not cosmetic: the arena would have banked a
    digest of file A into a result record describing a search built from file
    B, and the record is the artifact every later reading is joined against.

    So the constant is gone. What remains is this function, resolved at CALL
    time from ``$CHESS_ANTI_ENGINE_LIVE_CONFIG`` with the in-tree copy as a
    reported, NON-authoritative fallback. Call time rather than import time
    because the env var is the mechanism production actually uses (see
    ``scripts/train.sh``) and an import-time read is decided by whichever
    module happened to import first. It is deliberately NOT memoized either:
    a cache is a second source of truth with extra steps, and the resolution
    is a `stat` plus a yaml parse.
    """
    from chess_anti_engine.eval.production_shape import (
        load_config_file,
        load_live_config_or_reason,
    )

    live, reason = load_live_config_or_reason()
    if live is not None:
        return live
    fallback, fallback_reason = load_config_file(
        IN_TREE_CONFIG,
        provenance=f"in-tree fallback; the live config is unavailable ({reason})",
        authoritative=False,
    )
    if fallback is None:
        raise SystemExit(
            f"[shape] no production config could be read: {reason}; "
            f"{fallback_reason}"
        )
    return fallback


def production_config_path() -> Path:
    """The file ``production_config()`` resolved. Same resolution, path only."""
    return production_config().path


def production_config_flat() -> tuple[dict, Path, bool]:
    """``(flat config, path it came from, is it the LIVE file)``."""
    cfg = production_config()
    return cfg.flat, cfg.path, cfg.authoritative


def production_selfplay_configs(flat: dict | None = None) -> dict:
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
    from chess_anti_engine.worker import WorkerSession

    if flat is None:
        flat, _path, _live = production_config_flat()
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


def production_selfplay_search_config(flat: dict | None = None):
    """The selfplay ``SearchConfig`` half of :func:`production_selfplay_configs`.

    ⚑ Takes ``flat`` so a caller that already resolved the config does not
    resolve it a second time -- ``production_config()`` is deliberately not
    memoized, so calling the no-arg form twice is two stats and two yaml parses
    of a file that can change between them.
    """
    return production_selfplay_configs(flat)["search"]


def production_selfplay_gumbel_config(cfgs: dict | None = None):
    """The ``GumbelConfig`` production selfplay actually searches with.

    Built by ``selfplay.network_turn.build_selfplay_gumbel_config`` — THE
    mapping the production path calls, not a re-implementation of it. This is
    the whole structural fix: the arena's training shape is now read off this
    object, so "which knobs does production set" stopped being a question the
    arena answers from memory. ``simulations`` is arena-owned and passed here
    only because the mapping requires it; it is excluded from what is carried.

    ``cfgs`` lets a caller that already built the bundle pass it in rather than
    re-running the yaml -> reco -> worker channel, which is not cheap.
    """
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config

    if cfgs is None:
        cfgs = production_selfplay_configs()
    return build_selfplay_gumbel_config(
        search=cfgs["search"], game=cfgs["game"], simulations=1,
    )


def _assert_training_shape_is_production(
    gumbel: dict[str, float],
    flat: dict,
    cfg: LiveConfig | None = None,
    *,
    vloss_weight: int,
    target_batch: int,
) -> None:
    """Prove `--search-shape training` reproduces production selfplay's search.

    ``flat`` is the config the arena ITSELF read — passed in rather than
    re-resolved, so the guard cannot end up checking against a different file
    than the shape was built from.

    Note the guard deliberately crosses production paths: the arena builds its
    shape through the DISTRIBUTED worker channel (``build_recommended_worker``
    -> ``WorkerSession._build_selfplay_configs``), while the reference comes
    from the in-process channel (``TrialConfig`` -> ``_play_batch_kwargs`` ->
    ``build_selfplay_gumbel_config``). Both are production; a knob published by
    one and dropped by the other shows up here as a diff, which is a defect
    worth stopping on rather than a false alarm to suppress.

    The instrument being shared with the criterion is the point: ``gumbel`` is
    the dict ``match.py`` will ``dataclasses.replace`` onto a ``GumbelConfig``,
    so this applies it to a ``GumbelConfig`` exactly as the arena will and
    compares the RESULT against the config production's own builder produces.
    It never asks "is this key present" — presence proved nothing when the
    value was a stale literal.

    ``vloss_weight`` / ``target_batch`` are required keyword arguments rather
    than optional extras: they reach the C runner as function arguments, they
    have no ``GumbelConfig`` field, and a guard that compared only the dict
    would be blind to exactly the pair this arena has always carried and the
    audit did not. Making them mandatory means a caller cannot half-check.

    The failing input is easy to name: set any move-affecting search key in the
    live yaml that this dict does not carry (or leave one stale), and the diff
    is non-empty. ``tests/test_production_shape_guard.py`` produces exactly
    that.
    """
    import dataclasses as _dc

    from chess_anti_engine.eval.production_shape import (
        assert_matches_production,
        format_shape_table,
        production_search_shape,
    )
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape

  # `cfg` is the resolution the CALLER already made, so the header names the
  # file `flat` came out of. Falling back to `production_config()` when a test
  # drives the guard directly is safe for the same reason the constant was not:
  # it is the same resolver, so it cannot land on a different file.
    resolved = cfg if cfg is not None else production_config()
    print(resolved.header(), flush=True)
    if not resolved.authoritative:
        print(
            "[shape] WARNING: the config above is NOT the live one "
            f"({resolved.provenance}); it is stale by construction outside the "
            "live working tree, so the comparison below proves only that this "
            "arena agrees with THAT file.",
            flush=True,
        )
    prod = production_search_shape(
        flat, simulations=int(GumbelConfig().simulations),
    )
    realized = SelfplaySearchShape(
        cfg=_dc.replace(GumbelConfig(), **gumbel),
        vloss_weight=int(vloss_weight),
        target_batch=int(target_batch),
    )
    print(
        "[shape] --search-shape training: realized vs production selfplay\n"
        + format_shape_table(realized, prod, exempt=ARENA_SHAPE_DEVIATIONS),
        flush=True,
    )
    assert_matches_production(
        realized, prod, exempt=ARENA_SHAPE_DEVIATIONS,
        where="--search-shape training",
    )


def _warn_noise_schedule_deviation(base: SideSearch, *, add_noise: bool) -> None:
    """Say what a training arena does about production's per-ply noise schedule.

    ⚑ THE ONE THING THE FIELD DIFF STRUCTURALLY CANNOT CHECK. Production's root
    noise is ``_scheduled_gumbel_scale``: 0.75 for selfplay, 0.25 for
    curriculum, decaying to ``gumbel_scale_after`` over
    ``gumbel_scale_decay_moves`` from ``gumbel_scale_decay_start_move``. It is
    applied as a per-GAME, per-PLY ``per_game_gumbel_scale`` list OUTSIDE
    ``build_selfplay_gumbel_config``, so ``GumbelConfig.gumbel_scale`` — the
    only thing a shape diff can see — is the flat literal 1.0 on both sides
    and the comparison passes while the arena perturbs roots on nearly every
    move at a scale production only uses before move 12.

    This does not refuse. A flat override dict cannot express a schedule, so
    refusing would remove ``--search-shape training`` rather than fix it, and
    ``value_regret.py`` sets the precedent for reporting an unrepresentable
    divergence instead of pretending it is absent. What it must not do is stay
    silent, which is what an exemption reason reading "an arena must be
    deterministic per seed" achieved while noise was on by default.
    """
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    if base.shape != "training":
        return
    flat, _path, _live = production_config_flat()
    search = production_selfplay_search_config(flat)
    if not add_noise:
        print(
            "[shape] --search-shape training with --no-gumbel-noise: root "
            "noise OFF, production runs it ON (selfplay scale "
            f"{float(search.gumbel_scale)}, curriculum "
            f"{float(search.curriculum_gumbel_scale)}). DELIBERATE deviation "
            "— a deterministic-per-seed arena. Not production play behaviour.",
            flush=True,
        )
        return
    print(
        "[shape] ⚑ --search-shape training with root noise ON: the arena uses "
        f"the FLAT GumbelConfig scale {float(GumbelConfig().gumbel_scale)} on "
        "every ply, while production uses a per-ply SCHEDULE it cannot "
        f"express — selfplay {float(search.gumbel_scale)} -> "
        f"{float(search.gumbel_scale_after)} over "
        f"{int(search.gumbel_scale_decay_moves)} moves from move "
        f"{int(search.gumbel_scale_decay_start_move)}, curriculum "
        f"{float(search.curriculum_gumbel_scale)} -> "
        f"{float(search.curriculum_gumbel_scale_after)}. The shape table above "
        "CANNOT check this: the schedule is applied outside "
        "build_selfplay_gumbel_config, so no GumbelConfig field carries it. "
        "This run's root perturbations differ from production's on nearly "
        "every move. Pass --no-gumbel-noise for a deterministic arena, or read "
        "the result as 'the training shape with unscheduled root noise'.",
        flush=True,
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
      # ⚑ ONE resolution and ONE bundle build, shared by the shape, by the
      # `search` half and by the guard that checks them. Both halves used to be
      # taken through their no-arg forms, which re-`stat`ed and re-parsed the
      # yaml and re-ran the (not cheap) yaml -> reco -> worker channel a second
      # time -- so the dict the arena SEARCHES with came from a different read
      # of the file than the digest, header and `search` it is reported under.
      # A yaml edited between the two reads makes the guard refuse a shape that
      # was never inconsistent. `production_selfplay_search_config`'s own
      # docstring states the rule; this is the call site that broke it.
        cfg = production_config()
        flat, config_path = cfg.flat, cfg.path
        cfgs = production_selfplay_configs(flat)
        search = cfgs["search"]
        # DERIVED, never restated. Every GumbelConfig field that is neither
        # arena-owned nor checkpoint-owned is copied from the config production
        # itself builds, so a knob promoted into the yaml reaches the arena with
        # no edit here.
        #
        # ⚑⚑ A HAND-WRITTEN LIST HERE IS CONFIG-DEPENDENT, AND THAT IS WHY IT
        # CANNOT SHIP ON THIS BRANCH. The six-key literal this replaces is
        # correct only while every OTHER carried field already equals production
        # at its `GumbelConfig` default. That holds on main's yaml and is FALSE
        # on the live one: production here sets `target_max_visit_cap` 5 (default
        # 0) and `target_untempered_prior` True (default False), so the literal
        # would run `--search-shape training` at 0/False and measure a search
        # production does not run. Restating knobs by hand is exactly what let
        # those two drift for five days already.
        prod = production_selfplay_gumbel_config(cfgs)
        gumbel = {
            name: getattr(prod, name) for name in training_shape_carried_fields()
        }
        _assert_training_shape_is_production(
            gumbel, flat, cfg,
            vloss_weight=int(search.gumbel_vloss_weight),
            target_batch=int(search.gumbel_target_batch),
        )
        return SideSearch(
            shape="training",
            source=f"{config_path.name} -> reco -> worker SearchConfig",
          # The SAME object the guard above checked. It was briefly a second
          # copy of the literal, which meant the guard verified a dict the
          # arena did not use — this repo's signature defect, reintroduced by
          # the commit that was fixing it.
            gumbel=gumbel,
            vloss_weight=int(search.gumbel_vloss_weight),
            target_batch=int(search.gumbel_target_batch),
        )
    raise SystemExit(f"--search-shape must be one of {SEARCH_SHAPES}, got {shape!r}")



def overrides_with_volatility(
    side: SideSearch, vol: dict[str, float] | None,
) -> dict[str, float]:
    """The shape's knobs with an EXPLICIT ``--volatility-*`` request layered ON TOP.

    ⚑ Precedence, and it is not academic. ``pick_moves_for_boards`` builds its
    ``GumbelConfig`` from the dedicated volatility ARGUMENTS first and applies
    ``gumbel_overrides`` afterwards via ``dataclasses.replace``, so the override
    dict WINS. While the training shape carried three knobs that was harmless.
    The moment it became exhaustive it started carrying
    ``volatility_q_scale`` / ``volatility_fpu`` / ``volatility_anchor`` at
    production's values (0.0 today), which would silently reset an explicit
    ``--volatility-q-scale 0.5`` back to zero, keep the run on the C path, and
    report a volatility arena that never ran a volatility search.

    That is the accepted-then-ignored defect this file exists to prevent,
    reintroduced by the fix for a different instance of it. So the two are merged
    in ONE place, explicit request last, and both play loops call this rather
    than passing the shape and the flags down separate parameters where only
    their arrival order decides the winner.
    """
    return {**side.gumbel, **(vol or {})}


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
        try:
            gumbel[k] = int(v) if k in _GUMBEL_INT_KEYS else float(v)
        except ValueError:
            # `int("2.5")` and `float("abc")` both land here. Refused in this
            # function's own style rather than as a raw traceback: an int knob
            # given 2.5 would otherwise have been truncated to 2 by the
            # consumer if it parsed, which is the same silent-value defect.
            raise SystemExit(
                f"--*-gumbel: {k}={v!r} is not "
                f"{'an integer' if k in _GUMBEL_INT_KEYS else 'a number'}"
            ) from None
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
# Optional GSPRT early stop
# ---------------------------------------------------------------------------
#
# OFF unless --sprt is given, and off means OFF: no monitor is built, no play
# loop takes a different branch, and the JSONL record grows no key. This is a
# standing instrument with banked readings, so the fixed-N default has to stay
# byte-comparable with every row already in runs/arena_results.jsonl.
#
# The math lives in chess_anti_engine/eval/sprt.py (Van den Bergh's pentanomial
# GSPRT, the formulation fishtest uses); what lives HERE is only the wiring —
# WHERE the boundary is checked, and what is printed and banked when it fires.
#
# ⚑ The check granularity differs by loop and is recorded rather than smoothed
# over, because it is a property of the reading:
#
#   rolling      after every reap, i.e. as soon as a pair's SECOND coloring
#                finishes. Never mid-pair: the loop scores only pairs whose two
#                colorings are both on file (`complete_pair_scores`), so a
#                half-played pair contributes nothing to the statistic. Looking
#                at a half-pair would leak the opening's colour bias straight
#                into the stopping decision, which is the one thing the paired
#                design exists to remove.
#   chunked      between chunks. `play_paired_games_matched_sims` plays a whole
#                chunk lockstep and can only score it once every game in it is
#                over, so a mid-chunk stop would have to impute the unfinished
#                games as draws — fabricating pairs to decide a test with. A
#                chunk boundary is a set of COMPLETE pairs, so it is a legal
#                (merely coarser) stopping time: Wald's inequality bounds the
#                error rates for any stopping rule measurable at the look, and
#                looking less often only costs power. `--no-rolling --sprt` is
#                therefore allowed and prints its granularity, not refused.
#   matched_time after every pair. That loop already appends one score per
#                completed pair, so pair granularity is its natural unit.
SPRT_GRANULARITY_PAIR = "pair"
SPRT_GRANULARITY_CHUNK = "chunk"


def sprt_should_stop(
    sprt: SprtMonitor | None, new_pair_scores: Sequence[float], *, where: str,
) -> bool:
    """Fold ``new_pair_scores`` into the running GSPRT; True means stop now.

    ``new_pair_scores`` is what THIS invocation has completed; the monitor holds
    the resumed pairs and re-fits the statistic over the union, so a resumed run
    decides on the whole match rather than on its own tail.
    """
    if sprt is None:
        return False
    if sprt.update(new_pair_scores) is None:
        return False
    print(
        f"[arena] SPRT boundary CROSSED in the {where} loop after "
        f"{sprt.pairs} complete pair(s): LLR {sprt.llr:+.4f} "
        f"(H0 <= {sprt.spec.bound_h0:+.4f}, H1 >= {sprt.spec.bound_h1:+.4f}) "
        f"-> {sprt.verdict}. Stopping; remaining pairs are NOT played.",
        flush=True,
    )
    return True


def announce_sprt_armed(sprt: SprtMonitor | None, *, where: str) -> None:
    """Print the boundary the LOOP received, from the loop's own object.

    Read off the monitor the play loop was handed rather than off the CLI
    string, so a --sprt that never reached the consumer cannot print as though
    it had. That distinction is this repo's signature defect, and a flag that is
    accepted, echoed at startup and then never consulted would look identical
    from the console without this line.
    """
    if sprt is None:
        return
    print(
        f"[arena] SPRT ARMED in the {where} loop "
        f"(checked at {sprt.granularity} granularity, "
        f"{sprt.pairs} pair(s) already banked): {sprt.spec.describe()}",
        flush=True,
    )


# The header key the spec is recorded under, inside the game log's
# NON-fingerprinted `info` block. ⚑ It must stay out of `arena_game_log_settings`:
# everything there is hashed into the resume fingerprint, and putting the
# hypothesis in it would (a) refuse the legitimate "resume a crashed fixed-N
# arena as a sequential test" and (b) make every pre-branch log unresumable,
# since their fingerprint was computed without the key.
SPRT_LOG_INFO_KEY = "sprt"
SPRT_SPEC_FIELDS = ("elo0", "elo1", "alpha", "beta")


def describe_recorded_sprt_spec(recorded: Mapping[str, Any] | None) -> str:
    """The four DECLARED numbers of a spec read back out of a log header.

    Only the four: ``s0``/``s1``/``bound_h0``/``bound_h1`` are banked beside them
    but are functions of them, and a message that repeats a derived value invites
    the reader to compare the wrong pair of numbers.
    """
    if recorded is None:
        return "<none: that run was fixed-N>"
    return ", ".join(f"{k}={recorded.get(k, '<absent>')}" for k in SPRT_SPEC_FIELDS)


def _is_the_number(recorded: object, want: float) -> bool:
    """``recorded`` IS the number ``want``.

    A missing key, a null, or a string that happens to spell the number are all
    DIFFERENCES: a header can be hand-edited, and coercing whatever it holds into
    a float here would let a malformed spec read as a match. Bools are excluded
    because ``True == 1.0`` in Python and a boolean elo is not a match either.
    """
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        return False
    return float(recorded) == float(want)


def sprt_spec_carryover_warning(
    recorded: Mapping[str, Any] | None, current: SprtSpec | None,
) -> str | None:
    """Warn when a resume decides a log's pairs against a DIFFERENT hypothesis.

    A warning and never a refusal: the spec does not change how a game is
    played, so the pairs are one population however they are judged, and
    carrying them across specs is deliberate — a crashed fixed-N arena resumed
    as a sequential test is a supported thing to do, and so is tightening a
    boundary that the first segment did not reach. What it is NOT is free:
    alpha and beta are defined for ONE preregistered boundary, so a sample
    collected under one and decided against another realizes neither.

    Returns None when the log records no spec at all. That is the fixed-N ->
    sequential case, where there is no earlier hypothesis to contradict; warning
    there would put a line on every legitimate first sequential resume and teach
    the operator to skip reading it.
    """
    if recorded is None:
        return None
    if current is not None:
        want = current.as_record()
        if all(_is_the_number(recorded.get(k), want[k]) for k in SPRT_SPEC_FIELDS):
            return None
    now = (
        "<none: --sprt not given, this run is fixed-N>" if current is None
        else describe_recorded_sprt_spec(current.as_record())
    )
    return (
        "[arena] WARNING: this resume is judging the log's pairs against a "
        "DIFFERENT SPRT hypothesis than the one they were collected under.\n"
        f"  recorded in the log: {describe_recorded_sprt_spec(recorded)}\n"
        f"  this invocation:     {now}\n"
        "  The spec is deliberately OUTSIDE the resume fingerprint (it does not "
        "change how a game is played, so it cannot mix two populations), which "
        "is what makes this allowed rather than refused. But alpha and beta are "
        "the crossing probabilities of ONE preregistered boundary: pairs "
        "collected under the recorded spec and decided against another realize "
        "neither run's error rates. The log header keeps the ORIGINAL spec — a "
        "resume does not rewrite it — so the record of what was preregistered "
        "survives this."
    )


# ---------------------------------------------------------------------------
# Crash-resilient game log + resume
# ---------------------------------------------------------------------------
#
# Every FINISHED game is appended to a JSONL log and flushed before the next
# ply is played, so a crash loses only the games still in flight. 2026-08-21: a
# 128-game compiled arena OOMed at ply 20 with ZERO games persisted, and the
# relaunch lost its first minutes the same way — the run's only durable output
# was written after the last game.
#
# --resume replays nothing that finished. The unit it keeps is the PAIR, not
# the game: pentanomial scoring is pair-based, so a pair with only one coloring
# played is DISCARDED and replayed in full. Keeping the orphan half would fold
# a single-color result into a pair-score bin and quietly bias the very
# color-balance the paired design exists to remove.

def score_from_result(result: str, *, a_is_white: bool) -> float:
    """Candidate-POV score from a PGN result string.

    ``"*"`` maps to 0.5 to match the play loops, which score an unfinished
    max-plies game as a draw and WRITE it as ``1/2-1/2`` (Ordo discards ``*``,
    which would give the pooled fit a different population than the
    pentanomial).
    """
    if result in ("1/2-1/2", "*"):
        return 0.5
    win = "1-0" if a_is_white else "0-1"
    return 1.0 if result == win else 0.0


COMPILE_UNKNOWN = "unknown"
HOIST_UNKNOWN = "unknown"


def compile_tag(compile_models: bool, *, mode: str) -> str:
    """What a game row records about the invocation's ``torch.compile`` state.

    ``matched_time`` plays through UCI subprocesses that build their own
    engine, so ``compile_models`` never reaches those games: it records
    ``"n/a"`` rather than the resolved flag, which would otherwise flip with
    free VRAM between two attempts and report a mix that cannot exist.
    """
    if mode != "matched_sims":
        return "n/a"
    return "on" if compile_models else "off"


def row_compile_tag(row: Mapping[str, Any]) -> str:
    """The compile tag of one game row; ``"unknown"`` for pre-2026-08 logs."""
    value = row.get("compile", COMPILE_UNKNOWN)
    return COMPILE_UNKNOWN if value is None else str(value)


def hoist_tag(
    eval_max_batch: int, *, mode: str, no_hoist: str | None, uncapped_leaf_rows: int,
) -> str:
    """What a game row records about the invocation's evaluator hoist.

    Recorded for the same reason ``compile`` is: it is deliberately OUTSIDE the
    resume fingerprint (a pre-hoist log must stay resumable under post-hoist
    code), and it can change the arithmetic that plays the games. Unlike
    compile, it can change the SEARCH: below the uncapped leaf-buffer size the
    C tree absorbs surplus leaves as root-Q pseudo-terminals instead of
    evaluating them.

    So the tag is the effective configuration, not the raw flag:
    ``"n/a"`` off the matched_sims path, ``"off"`` when no evaluator is hoisted,
    ``"4096"`` when the cap cannot bind, and ``"128<4096"`` when it does — the
    one form that has to be distinguishable in a log years later.
    """
    if mode != "matched_sims":
        return "n/a"
    if no_hoist is not None:
        return "off"
    cap = int(eval_max_batch)
    if cap < int(uncapped_leaf_rows):
        return f"{cap}<{int(uncapped_leaf_rows)}"
    return str(cap)


def row_hoist_tag(row: Mapping[str, Any]) -> str:
    """The hoist tag of one game row; ``"unknown"`` for rows written before it.

    A missing key is UNKNOWN, never "off": those games were pre-hoist in fact,
    but a default that answers for rows the field never covered is how a
    resumed splice stops being visible. Unknown joining a real tag is a mix,
    which is the honest reading.
    """
    value = row.get("eval_hoist", HOIST_UNKNOWN)
    return HOIST_UNKNOWN if value is None else str(value)


def arena_game_log_settings(
    *,
    mode: str,
    candidate: str,
    reference: str,
    games: int,
    seed: int,
    openings_path: str,
    openings_kind: str,
    opening_plies: int | None,
    sims_candidate: int | None,
    sims_reference: int | None,
    ms_per_move: int | None,
    max_plies: int,
    temperature: float,
    gumbel_add_noise: bool,
    search_candidate: SideSearch | None,
    search_reference: SideSearch | None,
    volatility_candidate: dict[str, float] | None,
    uci_args: str,
    syzygy_path: str | None,
    tb_max_pieces: int,
) -> dict:
    """The settings a resume must MATCH — the population and the ruler.

    Deliberately EXCLUDED: ``max_concurrent_games``, ``rolling``, ``compile``,
    ``device``, ``max_seconds``, ``report_every``, ``label``, ``out``. The
    reason is NOT that all of them are outcome-neutral — ``compile`` and
    ``device`` change the arithmetic that plays the games, and a compiled
    segment and an eager one are not bit-identical. It is that the motivating
    crash was an OOM, whose whole remedy is to retry at a lower
    ``--max-concurrent-games`` or without compile: a fingerprint covering them
    would refuse the one resume this was built for. ``daily_gate_ratchet.sh``
    re-derives ``--compile on|off`` from free VRAM on EVERY attempt while
    always passing ``--resume``, so that mix is routine rather than
    hypothetical. It is not hidden: every game row records the compile setting
    it was played under, and a resume that spans two of them prints a warning
    and sets ``mixed_compile`` in the result record.
    """
    return {
        "mode": mode,
        "candidate": candidate,
        "reference": reference,
        "games": int(games),
        "seed": int(seed),
        "openings": openings_path,
        "openings_kind": openings_kind,
        "opening_plies": None if opening_plies is None else int(opening_plies),
        "sims_candidate": None if sims_candidate is None else int(sims_candidate),
        "sims_reference": None if sims_reference is None else int(sims_reference),
        "ms_per_move": None if ms_per_move is None else int(ms_per_move),
        "max_plies": int(max_plies),
        "temperature": float(temperature),
        "gumbel_add_noise": bool(gumbel_add_noise),
        "search_candidate": None if search_candidate is None else search_candidate.as_record(),
        "search_reference": None if search_reference is None else search_reference.as_record(),
        "volatility_candidate": volatility_candidate,
        "uci_args": uci_args,
        "syzygy": syzygy_path or "",
        "syzygy_max_pieces": int(tb_max_pieces),
    }


@dataclass(frozen=True)
class ArenaResume:
    """What a game log contributes to a resumed arena."""

    path: Path
    pair_scores: list[float]          # COMPLETE pairs only, by pair id
    complete_pair_ids: list[int]
    orphan_pair_ids: list[int]        # one coloring played: discarded, replayed
    games_loaded: int
    truncated_tail: bool
    # Distinct compile tags of the games this resume KEEPS (orphan halves are
    # replayed, so their tag never reaches the score). "unknown" for rows from
    # a log written before the field existed.
    compile_tags: list[str] = field(default_factory=list)
    # Same contract as compile_tags, for the evaluator hoist.
    hoist_tags: list[str] = field(default_factory=list)
    # The SPRT spec the log's header records, or None when it records none —
    # a fixed-N run, or any log written before the field existed. NOT part of
    # the fingerprint, so it never refuses a resume; the caller warns.
    sprt_spec: dict[str, Any] | None = None


def load_arena_resume(
    path: Path, *, settings: dict, openings: list[chess.Board],
) -> ArenaResume:
    """Load finished pairs from ``path``, refusing anything that does not match.

    Three refusals, all loud, because each one would otherwise produce a number
    that reads as a clean arena result:

    * settings fingerprint differs -> two populations averaged into one Elo;
    * a recorded ``opening_fen`` differs from the opening the schedule
      regenerated at that pair id -> the schedule is not reproducible from the
      seed, so the pairs cannot be matched up at all;
    * ``half`` disagrees with ``a_is_white`` -> the file is not what it says.

    The second is the one that matters most: it verifies AT RUN TIME the
    property resume rests on (openings are a pure function of seed + book +
    games) rather than assuming it.
    """
    log = read_game_log(path)
    recorded_sprt = log.info.get(SPRT_LOG_INFO_KEY)
    diffs = fingerprint_differences(log.settings, settings)
    if diffs:
        raise SystemExit(refuse_settings_mismatch_message(
            path, differences=diffs, resume_flag="--resume",
        ))
    rows = latest_rows_by_key(
        log.games, key=lambda r: (int(r["pair_id"]), int(r["half"])),
    )
    halves: dict[int, dict[int, float]] = {}
    tags: dict[int, set[str]] = {}
    htags: dict[int, set[str]] = {}
    for (pair_id, half), row in sorted(rows.items()):
        if not 0 <= pair_id < len(openings):
            raise SystemExit(
                f"--resume: {path} has pair_id {pair_id}, but this invocation "
                f"schedules only {len(openings)} pairs"
            )
        a_is_white = bool(row["a_is_white"])
        if a_is_white != (half == 0):
            raise SystemExit(
                f"--resume: {path} pair {pair_id} half {half} records "
                f"a_is_white={a_is_white}; half 0 is always the candidate as "
                "White. The log does not describe the schedule it claims to."
            )
        want = openings[pair_id].fen()
        got = str(row.get("opening_fen", ""))
        if got != want:
            raise SystemExit(
                f"--resume: {path} pair {pair_id} was played from\n"
                f"    {got}\n  but this invocation's schedule regenerates\n"
                f"    {want}\n  The opening schedule is NOT reproducible from "
                "the recorded settings, so resumed and replayed pairs would be "
                "different openings. Refusing."
            )
        halves.setdefault(pair_id, {})[half] = score_from_result(
            str(row["result"]), a_is_white=a_is_white,
        )
        tags.setdefault(pair_id, set()).add(row_compile_tag(row))
        htags.setdefault(pair_id, set()).add(row_hoist_tag(row))
    complete: list[int] = []
    orphans: list[int] = []
    scores: list[float] = []
    kept_tags: set[str] = set()
    kept_htags: set[str] = set()
    for pair_id in sorted(halves):
        by_half = halves[pair_id]
        if 0 in by_half and 1 in by_half:
            complete.append(pair_id)
            scores.append(by_half[0] + by_half[1])
            kept_tags |= tags[pair_id]
            kept_htags |= htags[pair_id]
        else:
            orphans.append(pair_id)
    return ArenaResume(
        path=path,
        pair_scores=scores,
        complete_pair_ids=complete,
        orphan_pair_ids=orphans,
        games_loaded=len(rows),
        truncated_tail=log.truncated_tail,
        compile_tags=sorted(kept_tags),
        hoist_tags=sorted(kept_htags),
        sprt_spec=dict(recorded_sprt) if isinstance(recorded_sprt, dict) else None,
    )


def _first_few(values: Sequence[object], limit: int = 8) -> str:
    """``[a, b, c] (+N more)`` — a diagnostic that stays readable at 500 pairs."""
    head = ", ".join(repr(v) for v in values[:limit])
    extra = len(values) - limit
    return f"[{head}]" + (f" (+{extra} more)" if extra > 0 else "")


def verify_game_log_on_disk(
    path: Path, *, settings: dict, openings: list[chess.Board],
    expected_pair_scores: Sequence[float],
    expected_pair_ids: Sequence[int] | None = None,
) -> tuple[bool, str]:
    """Re-READ the finished log and check it holds what was just scored.

    This has to touch the disk to mean anything. The check it replaced compared
    two tallies built at the same call site, so a write that never reached the
    file agreed with it perfectly — a guard that cannot fail, which is this
    repo's signature defect wearing the name of the thing it does not do.

    Deliberately routed through ``load_arena_resume``, the loader ``--resume``
    itself uses, so the check also answers the question the field exists for:
    is this log resumable, or has the run left something a later resume will
    reject?

    ``expected_pair_ids`` is passed only when the run KNOWS which pairs it
    scored — a wall-clock-truncated run does not, because the play loops return
    scores rather than ids, and inventing the set there would make this fire on
    a healthy run.

    Every failure is reported, never raised — including a malformed row, which
    is precisely the input this exists to survive: the games have already been
    played, and losing the summary to a bookkeeping fault would be a worse
    outcome than a flagged record. ``KeyError``/``TypeError`` are in the catch
    because a row missing ``pair_id``, or holding a string where an int
    belongs, reaches the loader as an unhandled exception rather than a
    refusal.
    """
    try:
        reloaded = load_arena_resume(path, settings=settings, openings=openings)
    except (SystemExit, ValueError, TypeError, KeyError, OSError) as exc:
        return False, (
            f"the log cannot be re-read as a resumable arena: "
            f"{type(exc).__name__}: {exc}"
        )
    on_disk_ids = sorted(reloaded.complete_pair_ids)
    on_disk = sorted(reloaded.pair_scores)
    expected = sorted(expected_pair_scores)
    ids_agree = (
        expected_pair_ids is None or on_disk_ids == sorted(expected_pair_ids)
    )
    if on_disk == expected and ids_agree:
        return True, ""
    parts = [
        f"the log holds {len(on_disk)} complete pairs on disk, this run scored "
        f"{len(expected)}"
    ]
    if expected_pair_ids is not None:
        wanted = set(expected_pair_ids)
        missing = sorted(wanted - set(on_disk_ids))
        extra = sorted(set(on_disk_ids) - wanted)
        if missing:
            parts.append(
                f"pair ids scored but NOT complete on disk: {_first_few(missing)}"
            )
        if extra:
            parts.append(
                f"pair ids complete on disk but not scored: {_first_few(extra)}"
            )
    if on_disk != expected:
        # Counts alone are silent when the two differ by VALUE at equal length.
        parts.append(
            f"pair scores on disk {_first_few(on_disk)} vs scored "
            f"{_first_few(expected)}"
        )
    return False, "; ".join(parts)


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

DEFAULT_EVAL_MAX_BATCH = 4096
"""Forward-batch cap for the hoisted arena evaluators; 0 disables the hoist.

4096 is what production selfplay runs (``worker.py`` builds its
``DirectGPUEvaluator`` with ``max_batch=4096, n_slots=2``), and at the default
``--max-concurrent-games 128`` it is at or above every batch the C search
already builds, so the cap does not reshape the search at the default settings
-- it only binds once concurrency is raised past that. Raising concurrency
without raising this is what the cap exists to stop.
"""


def build_arena_evaluator(model: Any, *, device: str, max_batch: int, n_slots: int = 2) -> Any:
    """One LONG-LIVED evaluator for one arena side.

    Without this, ``pick_moves_for_boards`` passes no evaluator and every C
    search entry point builds a THROWAWAY ``LocalModelEvaluator`` per call --
    per side, per ply. Each one lazily creates its own CUDA stream on first use;
    torch hands streams out of a fixed round-robin pool of 32 per device and the
    caching allocator partitions its segments BY STREAM, so a two-model arena
    cycles the entire pool in 16 plies and every stream ends up retaining a full
    forward's working set. Reserved VRAM inflates by up to the pool size and
    OOMs a 32G card well before the game count does.

    ``DirectGPUEvaluator`` (not ``LocalModelEvaluator``) for two reasons beyond
    lifetime: it implements the pinned slot API, so ``supports_inplace_api`` is
    true and the C search writes encodes straight into reused pinned buffers
    instead of allocating a fresh numpy batch per rep; and it carries
    ``_max_batch``, which is the ONLY thing that caps the leaf batch --
    ``mcts/gumbel_c.py`` mins its leaf cap against ``getattr(eval_impl,
    "_max_batch", <uncapped>)``, so a ``LocalModelEvaluator`` leaves the forward
    batch growing with concurrency without bound.

    ``n_slots=2`` because the C search's 2-group eval pipeline (any call with
    >= 64 boards) needs two independent output slots; with one slot it silently
    falls back off the in-place path.

    ``legal_bf16=False`` deliberately. It defaults True, and turning it on would
    switch the non-pipelined leaf transport to compact BF16 logits softmaxed in
    C -- a real numerics change against every arena already in the ledger.
    ``LocalModelEvaluator`` has no ``evaluate_legal_bf16`` at all, so today's
    arena runs dense float32; keeping it dense is what makes this change a
    memory fix rather than a new instrument.
    """
    from chess_anti_engine.inference import DirectGPUEvaluator

    return DirectGPUEvaluator(
        model,
        device=str(device),
        max_batch=int(max_batch),
        n_slots=int(n_slots),
        legal_bf16=False,
    )


def realized_topk(side: SideSearch) -> int:
    """The ``topk`` this side's search will actually run.

    Mirrors how ``pick_moves_for_boards`` builds its config: the GumbelConfig
    default unless the side overrides it. Read through ``.gumbel`` rather than
    ``realized_gumbel()`` because that one filters to the printable knob set.
    """
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    return int(side.gumbel.get("topk", GumbelConfig().topk))


def arena_pool_size(*, max_concurrent_games: int, n_pairs: int) -> int:
    """Concurrent games this arena can ACTUALLY have in flight.

    ``--max-concurrent-games`` is a CEILING, not the pool. Both matched_sims
    loops feed from a queue of ``2 * n_pairs`` games — the rolling loop refills
    to ``pool_size`` only while the queue lasts, and the chunked loop plays
    ``2 * len(chunk)`` — so a 2-game smoke arena at the default concurrency 128
    never has more than 2 boards alive, and neither does a 400-game run whose
    ``--openings-fen`` file yielded 2 usable rows.

    ⚑ This is the number the leaf-buffer bound and the root-submit refusal have
    to be computed from, and taking the ceiling instead is not merely
    conservative. At mcg 128 / topk 32 the ceiling claims 4096 leaf rows for a
    2-game arena whose search asks for 512: ``--eval-max-batch 512`` then draws
    the "NOT COMPARABLE" warning, is recorded as ``eval_leaf_cap_bound=true``,
    and smaller caps are refused against a root submit that carries two boards.
    A provenance field that is wrong in the safe direction is still wrong.

    ⚑ Derived from the LOADED openings, not from ``--games``: the two differ
    exactly when a FEN list is short, which is the case the ``--games`` figure
    cannot see. A resumed run plays a subset of those pairs and is therefore
    still bounded from above by this — the recorded fields then describe the
    match's schedule rather than one process's tail, which is the same scope the
    pentanomial summary is computed over.
    """
    return max(1, min(int(max_concurrent_games), 2 * max(0, int(n_pairs))))


def arena_uncapped_leaf_rows(
    *, max_concurrent_games: int, sides: Sequence[SideSearch | None],
    relations: Sequence[bool] | None = None,
) -> int:
    """Largest leaf buffer this arena's search will ask for, before any cap.

    ⚑ The number ``--eval-max-batch`` has to be compared against, and the reason
    that flag is a SEARCH-SHAPE knob rather than a memory one: ``gumbel_c`` mins
    its leaf buffer against the evaluator's ``_max_batch``, and when the buffer
    fills the C tree does not flush -- it absorbs the leaf as a SOLVED_UNKNOWN
    pseudo-terminal carrying the ROOT's Q. Below this value, leaves stop being
    evaluated and moves change.

    Computed from ``mcts/gumbel_c.leaf_buffer_rows`` -- the search's own
    expression -- over every board count either loop can hand one side (1 ..
    ``max_concurrent_games``; rolling passes up to ``pool_size``, chunked up to
    ``2 * chunk_pairs``, and either can put every active game on one side at a
    ply). Both regimes are checked because they are not ordered in ``n``: the
    single-buffer path applies below 64 boards and at topk 32 wants 4032 rows at
    63 boards, more than the pipelined path's 2048 at 64.

    ``relations[i]`` says whether side ``i``'s MODEL computes dynamic relations
    (``use_dynamic_relations``, `configs/exp_dynamic_relations.yaml`, default
    off). It is per side because the two sides are different checkpoints and
    only one of them may have it. Relations force ``_use_pipeline`` False at
    every board count, so the single-buffer path then runs at the REAL n rather
    than only below 64 -- which is larger: at mcg 128 / topk 32 it is 8192
    against the 4096 a relations-off model asks for.

    ⚑ Omitting ``relations`` assumes OFF and therefore returns a FLOOR, not the
    exact figure. That is deliberate for the launch-time check, which runs
    before the checkpoints are loaded so a refusal beats a multi-minute compile
    and cannot read the flag. The caller must re-derive with the real flags once
    the models exist -- otherwise a relations-on model with a cap in
    [4096, 8192) is bound in fact while every recorded field says it is not.

    ⚑ ``max_concurrent_games`` is the top of the board-count RANGE ONE SEARCH
    CALL CAN BE HANDED, which is not the same quantity for every caller and is
    deliberately not renamed:

    * an ARENA hands one call a whole side of its pool, so it passes the pool
      the schedule can actually fill (``arena_pool_size``) — NOT the raw
      ``--max-concurrent-games`` ceiling, since bounding a range the search can
      never enter reports an inert cap as binding;
    * a SEQUENTIAL match hands one call a single board, so it passes 1 —
      ``scripts/match_vs_handicapped_sf.py``'s ``resolve_eval_leaf_cap`` does
      exactly that, and takes no concurrency argument at all, because nothing
      about its cap depends on how many games are in flight.

    Either way the rule is the same: pass what ONE call can carry, never a
    ceiling that no call reaches.
    """
    from chess_anti_engine.mcts.gumbel_c import leaf_buffer_rows

    n_max = max(1, int(max_concurrent_games))
    rels = list(relations) if relations is not None else [False] * len(sides)
    if len(rels) != len(sides):
        raise ValueError(
            f"relations has {len(rels)} entries for {len(sides)} sides"
        )
    rows = 0
    for side, rel in zip(sides, rels, strict=True):
        if side is None:
            continue
        topk = realized_topk(side)
        if rel:
            # Pipeline unreachable: the single path runs at every n, and it is
            # monotone, so its value at the top of the range subsumes the <64
            # term below.
            rows = max(rows, leaf_buffer_rows(n_max, topk=topk, pipelined=False))
            continue
        rows = max(rows, leaf_buffer_rows(min(63, n_max), topk=topk, pipelined=False))
        if n_max >= 64:
            rows = max(rows, leaf_buffer_rows(n_max, topk=topk, pipelined=True))
    return rows


def no_hoist_reason(
    *, mode: str, device: str, eval_max_batch: int,
    volatility_candidate: dict[str, float] | None,
) -> str | None:
    """Why this invocation builds no hoisted evaluator, or None if it does.

    ONE source of truth, because three separate things key off it: whether the
    launch-time cap checks apply at all (they are meaningless on a path that
    never builds an evaluator -- matched_time runs UCI subprocesses and a CPU
    arena is excluded by design), what the console prints, and what the result
    record and every game row store.
    """
    if mode != "matched_sims":
        return f"--mode {mode} builds no in-process evaluator"
    if not eval_max_batch:
        return "--eval-max-batch 0"
    if volatility_candidate is not None:
        return "--volatility-* on the candidate"
    if not str(device).startswith("cuda"):
        return f"device={device} is not CUDA"
    return None


def _warn_leaf_cap_binds(
    eval_max_batch: int, uncapped_leaf_rows: int, *, late: bool,
) -> None:
    """The one warning text, printed from both the pre-load and post-load checks.

    Shared rather than duplicated because the post-load check exists precisely
    to catch what the pre-load floor missed, and two copies of a warning that
    must say the same thing is how one of them ends up saying less.
    """
    when = (
        "after loading the checkpoints (a dynamic-relations model routes every "
        "call to the single-buffer path, which the pre-load estimate could not "
        "know): "
        if late else ""
    )
    print(
        f"[arena] ⚑ WARNING: {when}--eval-max-batch {eval_max_batch} is BELOW "
        f"this arena's uncapped leaf-buffer size {uncapped_leaf_rows}, so it is "
        f"acting as a SEARCH-SHAPE knob, not a memory knob. gumbel_c mins its "
        f"leaf buffer against the evaluator's cap, and when that buffer fills "
        f"the C tree does NOT flush and retry — it absorbs the leaf as a "
        f"SOLVED_UNKNOWN pseudo-terminal carrying the ROOT's Q. Leaves beyond "
        f"{eval_max_batch} are therefore never evaluated: measured on CPU, 128 "
        f"vs 4096 dropped 57-75% of leaf evaluations and changed the chosen "
        f"move on 53 of 64 boards. THIS ARENA'S SEARCH IS NOT COMPARABLE TO AN "
        f"UNCAPPED ONE. Pass --eval-max-batch {uncapped_leaf_rows} (or 0) if "
        f"that was not intended; the result record and every game row store "
        f"which it was.",
        file=sys.stderr, flush=True,
    )


def _free_cached_vram(device: str) -> None:
    """Return the caching allocator's idle segments to the driver. No-op off CUDA.

    Cheap insurance at the two points where the batch shape shrinks for good
    (a finished chunk, the rolling pool's drain): the segments cached for the
    wide shape are dead weight against the next arena stage or a concurrent
    trainer, and nothing else in this process will reclaim them.
    """
    if not str(device).startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def default_openings_path() -> Path:
    """The 8-move UHO book from the production config (opening_book_path_2)."""
    import yaml

    path = production_config_path()
    cfg = yaml.safe_load(path.read_text())
    selfplay = cfg.get("selfplay", {}) if isinstance(cfg, dict) else {}
    book = selfplay.get("opening_book_path_2") or selfplay.get("opening_book_path")
    if not book:
        raise SystemExit(
            f"no opening_book_path(_2) in {path}; pass --openings"
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

        cfg = yaml.safe_load(production_config_path().read_text())
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
    pair_ids: Sequence[int] | None = None,
    chunk: int | None = None,
    evaluator_candidate: Any = None,
    evaluator_reference: Any = None,
) -> list[float]:
    """Play each opening twice (colors swapped) and return per-pair scores.

    ``pair_ids[k]`` is the GLOBAL pair id of ``openings[k]`` (default: ``k``).
    It replaced a plain ``pair_id_offset`` because a resumed run plays a
    NON-CONTIGUOUS subset of the schedule — the pairs the crash left unfinished
    — and an offset can only describe a contiguous slice. The ids are what the
    game log and the PGN's ``PairId`` are keyed on, so getting them wrong
    merges unrelated pairs into one block.

    ``search_candidate`` / ``search_reference`` are REQUIRED and carry the full
    realized search shape per side (see ``SideSearch``). There is deliberately
    no default: a silent default is what made every arena in the ledger measure
    the UCI/play search instead of the training search.

    ``volatility_candidate`` (volatility_q_scale / volatility_fpu /
    volatility_anchor) applies volatility-aware Gumbel search to the
    CANDIDATE side only — the reference keeps today's search, which is the
    A/B the experiment needs. Non-zero flags force the Python search path
    (mcts/gumbel.py), so matched_sims is the honest mode.

    ``evaluator_candidate`` / ``evaluator_reference`` are the per-side
    long-lived evaluators (``build_arena_evaluator``). ``None`` on both is
    today's behaviour: each search call then builds its own throwaway
    ``LocalModelEvaluator`` and its own CUDA stream. They are per SIDE, not
    shared -- an evaluator is bound to one model, and handing the candidate's
    evaluator to the reference would silently play the candidate's weights for
    both sides.

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

    ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
    if len(ids) != len(openings):
        raise ValueError(
            f"pair_ids has {len(ids)} entries for {len(openings)} openings"
        )
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
            pair_id=ids[i // 2],
            half=i % 2,
            a_is_white=bool(a_plays_white[i]),
            start_fen=start_fens[i],
            moves=mv,
            result=res,
            termination=termination,
            plies=len(mv),
            duration_s=time.time() - t0,
            chunk=chunk,
            loop="chunked",
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
        for model, idxs, sims, extra, side, ev in (
            (model_candidate, a_to_move, sims_candidate, vol_kwargs, search_candidate,
             evaluator_candidate),
            (model_reference, b_to_move, sims_reference, {}, search_reference,
             evaluator_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=overrides_with_volatility(side, extra),
                gumbel_vloss_weight=side.vloss_weight,
                gumbel_target_batch=side.target_batch,
                evaluator=ev,
            )
            # strict: this is the deciding Elo instrument. Substituting a legal
            # move for an id that decoded to nothing would keep the arena
            # scoring games under a broken action space.
            apply_actions_to_boards(boards, idxs, actions, strict=True)

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
    pair_ids: Sequence[int] | None = None,
    prior_pair_scores: Sequence[float] | None = None,
    sprt: SprtMonitor | None = None,
    evaluator_candidate: Any = None,
    evaluator_reference: Any = None,
    free_cached_vram: bool = True,
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

    ``sprt`` (default None = today's fixed-N behaviour) adds a GSPRT boundary
    check after every reap, on the same COMPLETE-pairs-only set the summary
    scores. It stops the loop exactly the way ``deadline`` does; what it returns
    is unchanged.

    ``evaluator_candidate`` / ``evaluator_reference`` are the per-side
    long-lived evaluators (``build_arena_evaluator``). ``None`` on both is
    today's behaviour: each search call then builds its own throwaway
    ``LocalModelEvaluator`` and its own CUDA stream. They are per SIDE, not
    shared -- an evaluator is bound to one model, and handing the candidate's
    evaluator to the reference would silently play the candidate's weights for
    both sides.
    """
    from chess_anti_engine.selfplay.match import (
        apply_actions_to_boards,
        pick_moves_for_boards,
        result_from_a_pov,
        split_active_by_side_to_move,
    )
    from scripts.match_vs_uci import _tb_adjudicate_result

    # ``pair_ids[k]`` is the GLOBAL pair id of ``openings[k]`` (default: k) —
    # a resumed run plays a non-contiguous subset of the schedule, and the
    # PairId in the game log / PGN must stay the schedule's own numbering.
    # ``gids`` remain LOCAL (2k / 2k+1) because they index ``game_scores``.
    ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
    if len(ids) != len(openings):
        raise ValueError(
            f"pair_ids has {len(ids)} entries for {len(openings)} openings"
        )
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
                pair_id=ids[gids[j] // 2],
                half=gids[j] % 2,
                a_is_white=bool(awhite[j]),
                start_fen=gfens[j],
                moves=mv,
                result="1/2-1/2" if res == "*" else res,
                termination=termination,
                plies=len(mv),
                duration_s=time.time() - gt0[j],
                chunk=None,
                loop="rolling",
            )

    t0 = time.time()
    done = 0
    last_report = 0
    drain_freed = False
    announce_sprt_armed(sprt, where="rolling")
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
        # The SPRT look sits HERE — after the reap, so the pairs this ply
        # completed are in the sample, and BEFORE the deadline check, so a run
        # that crosses on its last affordable ply reports the VERDICT rather
        # than an INCONCLUSIVE-at-the-clock. The set it sees is
        # `complete_pair_scores`, i.e. pairs with both colorings on file, which
        # is what makes this a pair-granularity look and not a mid-pair one.
        #
        # ⚑ `sprt is not None` is checked HERE and not only inside
        # `sprt_should_stop`: Python evaluates arguments eagerly, so the bare
        # call rescans all `n_games` scores every ply of every FIXED-N run —
        # the default path, which must pay nothing for a feature it did not
        # ask for. The helper keeps its own None guard for its other callers.
        if sprt is not None and sprt_should_stop(
            sprt, complete_pair_scores(game_scores), where="rolling",
        ):
            break
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
        if free_cached_vram and not drain_freed and not queue and len(boards) < pool_size:
            # Drain has begun: the queue is empty, so the pool only shrinks from
            # here and the allocator's full-width segments will never be reused.
            # Freed ONCE, at the transition — doing it per ply would sync the
            # device on each of the last ~pool_size plies for no further gain.
            _free_cached_vram(device)
            drain_freed = True
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
            # Resumed pairs are part of the sample, so the RUNNING block must
            # count them: a run killed again would otherwise be read off a
            # block that understates its own n (the ratchet reads these blocks
            # when the process does not survive to the final summary).
            ready = list(prior_pair_scores or []) + complete_pair_scores(game_scores)
            if ready:
                print(f"[arena] RUNNING Elo after {len(ready)} complete pairs:", flush=True)
                print_summary(summarize_pentanomial(pentanomial_counts(ready)))
            last_report = done
        active = list(range(len(boards)))
        a_to_move, b_to_move = split_active_by_side_to_move(active, boards, awhite)
        for model, idxs, sims, extra, side, ev in (
            (model_candidate, a_to_move, sims_candidate,
             dict(volatility_candidate or {}), search_candidate, evaluator_candidate),
            (model_reference, b_to_move, sims_reference, {}, search_reference,
             evaluator_reference),
        ):
            if not idxs:
                continue
            actions = pick_moves_for_boards(
                model, [boards[i] for i in idxs],
                device=device, rng=rng,
                mcts_type="gumbel", mcts_simulations=int(sims),
                temperature=float(temperature), c_puct=2.5,
                gumbel_add_noise=bool(gumbel_add_noise),
                gumbel_overrides=overrides_with_volatility(side, extra),
                gumbel_vloss_weight=side.vloss_weight,
                gumbel_target_batch=side.target_batch,
                evaluator=ev,
            )
            # strict: same instrument as the chunked path above.
            apply_actions_to_boards(boards, idxs, actions, strict=True)
        for i in active:
            gplies[i] += 1

    if free_cached_vram:
        _free_cached_vram(device)
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
    pair_ids: Sequence[int] | None = None,
    sprt: SprtMonitor | None = None,
) -> list[float]:
    """Pair-by-pair UCI match using the production engine inference path.

    ``deadline`` stops between PAIRS. A wall-clock budget is not a search knob
    — it changes nothing about the ruler — so unlike ``--search-shape`` and the
    vloss/target-batch family it is honoured here rather than refused. Pair
    granularity is the natural unit: this loop only ever appends a score once
    both colorings of an opening are played, so a truncated run drops the
    in-progress pair by construction.

    ``sprt`` (default None = today's fixed-N behaviour) looks at that same pair
    boundary, immediately after each pair's score is appended.
    """
    import chess.engine

    from scripts.match_vs_uci import _open_engine, _score_for_a, play_one_game

    limit = chess.engine.Limit(time=float(ms_per_move) / 1000.0)
    # Global pair ids (see the chunked loop): a resumed run plays a
    # non-contiguous subset of the schedule.
    ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
    if len(ids) != len(openings):
        raise ValueError(
            f"pair_ids has {len(ids)} entries for {len(openings)} openings"
        )

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
    announce_sprt_armed(sprt, where="matched_time")
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
                        pair_id=ids[pair_idx],
                        half=0 if a_is_white else 1,
                        a_is_white=a_is_white,
                        start_fen=record.start_board.fen(),
                        moves=tuple(record.moves),
                        result=record.result,
                        termination=record.termination,
                        plies=record.plies,
                        duration_s=time.time() - _g_t0,
                        chunk=None,
                        loop="matched_time",
                    )
            pair_scores.append(scores[0] + scores[1])
            print(
                f"[arena] pair {pair_idx + 1}/{len(openings)}: "
                f"pair_score={pair_scores[-1]:.1f} "
                f"running_score={sum(pair_scores) / (2 * len(pair_scores)):.3f}",
                flush=True,
            )
            if sprt_should_stop(sprt, pair_scores, where="matched_time"):
                break
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


def production_config_record() -> dict:
    """The banked IDENTITY of the config this run's search shape came from.

    One resolution, three fields, written together. A bare digest was enough
    while the config was a fixed in-tree path; it is not now that the file can
    be either the live yaml or the in-tree fallback, because a reader joining
    two rows cannot tell an unrecognised digest ("a config I have not seen")
    from a non-authoritative one ("a config that was never production's").
    ``config_authoritative`` is the field that answers the second question, and
    it is the one that decides whether the row belongs in a table at all.
    """
    try:
        cfg = production_config()
    except (OSError, SystemExit):
        return {
            "config_hash": "unknown",
            "config_name": "unknown",
            "config_authoritative": False,
        }
    return {
        "config_hash": cfg.sha256[:12],
        "config_name": cfg.path.name,
        "config_authoritative": bool(cfg.authoritative),
    }


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
    game_log: str | None = None,
    game_log_fingerprint: str | None = None,
    game_log_agrees: bool = True,
    resumed_pairs: int = 0,
    resumed_orphan_pairs: int = 0,
    compile_setting: str = COMPILE_UNKNOWN,
    compile_values: Sequence[str] = (),
    hoist_setting: str = HOIST_UNKNOWN,
    hoist_values: Sequence[str] = (),
    eval_max_batch: int | None = None,
    eval_leaf_cap_uncapped: int | None = None,
    eval_leaf_cap_bound: bool = False,
    max_concurrent_games: int | None = None,
    arena_pool: int | None = None,
    sprt: dict[str, Any] | None = None,
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
        **production_config_record(),
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
        # Crash-resilience / resume provenance. `resumed_pairs` > 0 means this
        # row is a SPLICE of two processes' games — same schedule and same
        # settings (the fingerprint is checked before a single pair is
        # reused), but not one continuous run.
        "game_log": game_log,
        "game_log_fingerprint": game_log_fingerprint,
        "game_log_agrees": bool(game_log_agrees),
        "resumed_pairs": int(resumed_pairs),
        "resumed_orphan_pairs": int(resumed_orphan_pairs),
        # torch.compile is NOT in the resume fingerprint (see
        # arena_game_log_settings), so a resumed row can span a compiled
        # segment and an eager one — daily_gate_ratchet.sh re-derives the flag
        # from free VRAM on every retry. `compile` is what THIS process ran;
        # `compile_values` is every setting the scored games were played under,
        # and `mixed_compile` says they are not all the same.
        "compile": compile_setting,
        "compile_values": list(compile_values) or [compile_setting],
        "mixed_compile": len(set(compile_values)) > 1,
        # The evaluator hoist, recorded exactly like compile and for the same
        # reason: outside the resume fingerprint, and able to change the
        # arithmetic. `eval_leaf_cap_bound` is the field that makes a shrunken
        # search identifiable afterwards -- `eval_max_batch` alone cannot say
        # whether it bound, because that depends on topk, concurrency AND
        # whether a side's model computes dynamic relations.
        #
        # ⚑ These two are POST-LOAD EXACT, not the launch-time floor: run_arena
        # re-derives them once the checkpoints are in hand and can be asked for
        # `use_dynamic_relations`. The launch check necessarily runs earlier
        # (its refusal has to beat the compile) and under-reports for a
        # relations-on model, so a record written from that estimate would say
        # bound=False about a search that was bound.
        "eval_hoist": hoist_setting,
        "eval_hoist_values": list(hoist_values) or [hoist_setting],
        "mixed_eval_hoist": len(set(hoist_values)) > 1,
        "eval_max_batch": eval_max_batch,
        "eval_leaf_cap_uncapped": eval_leaf_cap_uncapped,
        "eval_leaf_cap_bound": bool(eval_leaf_cap_bound),
        # ⚑ THE INPUTS the two fields above are a FUNCTION OF, banked so a row
        # can be re-derived instead of only read. `eval_leaf_cap_uncapped` is
        # computed from the POOL — the concurrency ceiling capped by the loaded
        # opening pairs (`arena_pool_size`) — and at mcg 128 / topk 32 that is
        # 4096 rows over a full schedule, 2560 over 40 games and 512 over 2. A
        # row that banked only the answer would be unreproducible from its own
        # fields, and `games` cannot substitute: a short --openings-fen list
        # shrinks the pool below what --games asked for.
        #
        # `arena_pool` is what the bound was ACTUALLY taken over;
        # `max_concurrent_games` is the ceiling the operator asked for, kept
        # separately because the gap between them is the thing worth seeing.
        # Both null on rows written before this field existed — never 0, which
        # would be a claim about a run the field never covered.
        "max_concurrent_games": max_concurrent_games,
        "arena_pool_size": arena_pool,
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
        # ⚑ ABSENT, not null, on a run without --sprt. runs/arena_results.jsonl
        # is a shared append-only aggregate with years of rows in it, and the
        # fixed-N default has to keep producing byte-identical records; a
        # `"sprt": null` on every row would be a schema change bought for
        # nothing. Present means the run was sequential, and then `elo`/`elo_ci95`
        # ABOVE are conditioned on a stopping rule (see the record's `caveat`)
        # rather than being fixed-N estimates.
        **({"sprt": sprt} if sprt is not None else {}),
        "duration_s": round(duration_s, 1),
        "argv": sys.argv,
    }


def append_result(record: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _abort_void(exc: ActionDecodeError, *, completed_pairs: int) -> NoReturn:
    """End the run VOID: named line on stderr, non-zero exit, no result row.

    A corrupted instrument must not bank a reading — `append_result` is never
    reached, so the JSONL keeps no row for this arena and no downstream fit can
    pick one up. The pairs that DID complete are named rather than left as a
    bare traceback: the operator needs to know how much of the budget burned.
    ``completed_pairs`` is what the driver had banked, so the rolling path
    reports 0 (it returns only on success) while the chunked path reports the
    chunks that finished.
    """
    print(
        f"[arena] VOID — action id {exc.index} decoded to no legal move "
        f"({exc.detail}); side to move {'white' if exc.turn else 'black'}, "
        f"fen {exc.fen!r}. {completed_pairs} complete pair(s) had been banked "
        "by the driver; NO result row was written. The action space and the "
        "checkpoints disagree, so every game in this run is unscorable — "
        "rerun after fixing the encoding, do not salvage the partial score.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(2)


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
    resume: bool = False,
    game_log_path: Path | None = None,
    eval_max_batch: int = DEFAULT_EVAL_MAX_BATCH,
    sprt: SprtSpec | None = None,
) -> dict:
    """Run one standardized arena and return (and optionally log) the record.

    ``search_candidate`` / ``search_reference`` are mandatory for
    ``matched_sims`` — this is the single funnel every arena entry point goes
    through (arena_standard, elo_vs_sims, anything future), so refusing to run
    without an explicit shape here is what makes "which search did I measure?"
    unanswerable-by-omission impossible. ``matched_time`` runs real UCI
    subprocesses, which carry their own play shape; pass search knobs there via
    ``--uci-args`` instead.

    ``sprt`` (default None) turns the run into a sequential test against a
    PREREGISTERED boundary: ``--games`` becomes a hard CAP rather than the sample
    size, and the deliverable is the H1/H0/INCONCLUSIVE verdict. None leaves
    every byte of the fixed-N path, and of its JSONL record, unchanged.
    """
    if games < 2 or games % 2 != 0:
        raise SystemExit("--games must be even and >= 2 (paired openings)")
    if eval_max_batch < 0:
        raise SystemExit("--eval-max-batch must be >= 0 (0 disables the hoist)")
    if mode == "matched_sims" and (search_candidate is None or search_reference is None):
        raise SystemExit(
            "matched_sims needs an explicit search shape: pass --search-shape "
            f"{{{'|'.join(SEARCH_SHAPES)}}}. 'training' is what production selfplay "
            f"runs, read from {production_config_path().name} at run time (linear root; "
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

    # ---- evaluator cap checks --------------------------------------------
    # These live HERE: after the shape refusals (so both sides' topk are
    # resolved), after the openings are loaded (so the POOL the schedule can
    # fill is known, which `--games` alone cannot say for a short FEN list),
    # and still before any checkpoint load or compile — a refusal must beat a
    # multi-minute compile, and opening sampling creates no files, so nothing
    # is left behind. ONLY on a path that will actually build an evaluator:
    # matched_time plays through UCI subprocesses and a CPU arena is excluded
    # by design, so a cap that can never bind must not refuse either of them.
    _no_hoist = no_hoist_reason(
        mode=mode, device=device, eval_max_batch=eval_max_batch,
        volatility_candidate=volatility_candidate,
    )
    pool_size = arena_pool_size(
        max_concurrent_games=max_concurrent_games, n_pairs=len(openings),
    )
    uncapped_leaf_rows = (
        arena_uncapped_leaf_rows(
            max_concurrent_games=pool_size,
            sides=(search_candidate, search_reference),
        )
        if _no_hoist is None else 0
    )
    leaf_cap_bound = _no_hoist is None and eval_max_batch < uncapped_leaf_rows
    if _no_hoist is None and eval_max_batch < pool_size:
        # Refused HERE, not on the ply that trips it. The C gumbel ROOT submit
        # is handed every board on one side at once and is NOT bucketed against
        # the evaluator's cap, so `get_input_buffer` would raise `batch N > max
        # M` mid-arena -- after the checkpoint load and a multi-minute compile,
        # and after the PGN/game log already exist.
        #
        # ⚑ Compared against the POOL, not --max-concurrent-games: the root
        # submit carries one side of the games actually in flight, so a 2-game
        # arena at mcg 128 can only ever hand it two boards and refusing a cap
        # of 32 there would reject a configuration the search can serve.
        #
        # ⚑ The remedy deliberately does NOT say "raise it to the pool size".
        # That is the value at which the ROOT submit stops raising and the LEAF
        # cap binds HARDEST: at a pool of 128 the search asks for 4096 leaf
        # rows, so a cap of 128 would run and absorb most of the leaves. Naming
        # the bare minimum would trade a loud crash for a quiet search change.
        raise SystemExit(
            f"--eval-max-batch {eval_max_batch} is below this arena's pool of "
            f"{pool_size} concurrent game(s) (--max-concurrent-games "
            f"{max_concurrent_games} against {len(openings)} loaded opening "
            f"pair(s)): the search's root batch is up to one whole side of the "
            f"pool and the hoisted evaluator would refuse it.\n"
            f"  Use --eval-max-batch {uncapped_leaf_rows} — this arena's uncapped "
            f"leaf-buffer size, at which the search runs unchanged — or "
            f"--eval-max-batch 0 to keep the per-call evaluators.\n"
            f"  Anything between {pool_size} and {uncapped_leaf_rows} "
            f"runs, but SHRINKS THE SEARCH rather than just the memory."
        )
    if leaf_cap_bound:
        # Allowed, warned, and recorded — the way compile is. It changes the
        # arithmetic, there are legitimate reasons to want it (a smaller card),
        # and refusing would take the option away. What is not allowed is it
        # being quiet.
        _warn_leaf_cap_binds(eval_max_batch, uncapped_leaf_rows, late=False)

    # ---- crash-resilient game log + resume -------------------------------
    # Decided before any file is created and before either checkpoint is
    # loaded: a refusal here must not leave a stray PGN behind, and a
    # fully-resumed schedule must not pay for two ~700MB loads and a 4-minute
    # compile to play zero games.
    openings_source = str(openings_fen if openings_fen is not None else openings_path)
    openings_kind = "fen" if openings_fen is not None else "book"
    log_settings = arena_game_log_settings(
        mode=mode,
        candidate=candidate,
        reference=reference,
        games=games,
        seed=seed,
        openings_path=openings_source,
        openings_kind=openings_kind,
        opening_plies=None if openings_fen is not None else opening_plies,
        sims_candidate=sims_candidate if mode == "matched_sims" else None,
        sims_reference=sims_reference if mode == "matched_sims" else None,
        ms_per_move=ms_per_move if mode == "matched_time" else None,
        max_plies=max_plies,
        temperature=temperature,
        gumbel_add_noise=gumbel_add_noise,
        search_candidate=search_candidate,
        search_reference=search_reference,
        volatility_candidate=volatility_candidate,
        uci_args=uci_args,
        syzygy_path=syzygy_path,
        tb_max_pieces=tb_max_pieces,
    )
    fingerprint = settings_fingerprint(log_settings)
    log_path = (
        Path(game_log_path) if game_log_path is not None
        else default_game_log_path(
            DEFAULT_GAME_LOG_DIR, label=label, fingerprint=fingerprint,
        )
    )
    had_log = log_path.exists() and log_path.stat().st_size > 0
    resumed: ArenaResume | None = None
    if had_log and not resume:
        # A crash during the two ~700MB checkpoint loads or the ~4-minute
        # compile leaves a log with a header and no games. Refusing the retry
        # then protects games that do not exist.
        taken_over, note = take_over_header_only_log(
            log_path, fingerprint=fingerprint,
        )
        if taken_over:
            print(f"[arena] {note}", flush=True)
            had_log = False
        else:
            message = refuse_existing_log_message(
                log_path, resume_flag="--resume", out_flag="--games-out",
                fingerprint_keyed=game_log_path is None,
            )
            raise SystemExit(message + ("\n" + note if note else ""))
    if had_log:
        resumed = load_arena_resume(
            log_path, settings=log_settings, openings=openings,
        )
    elif resume:
        # Not an error: wrappers pass --resume unconditionally (the ratchet
        # does, so its retries continue rather than abort). Never silent
        # either — the default log name is keyed on --label and on the settings
        # fingerprint, so a person who expected to continue a specific crashed
        # run needs to see WHICH path was looked for before watching it replay
        # from zero.
        print(
            f"[arena] --resume: no game log at {log_path} — nothing to resume, "
            f"this run starts from scratch. That path is keyed on --label and "
            f"on the settings fingerprint ({fingerprint}); if the crashed run "
            f"used a different label or any different setting, point "
            f"--games-out at its log.",
            flush=True,
        )
    done_pair_ids = set(resumed.complete_pair_ids) if resumed is not None else set()
    orphan_pair_ids = set(resumed.orphan_pair_ids) if resumed is not None else set()
    loaded_pair_scores = list(resumed.pair_scores) if resumed is not None else []
    remaining_ids = [i for i in range(len(openings)) if i not in done_pair_ids]
    openings_to_play = [openings[i] for i in remaining_ids]
    if resumed is not None:
        print(
            f"[arena] RESUMED {len(resumed.complete_pair_ids)} complete pairs "
            f"({resumed.games_loaded} games on file) from {log_path}; "
            f"{len(remaining_ids)}/{len(openings)} pairs left to play",
            flush=True,
        )
        if resumed.truncated_tail:
            print(
                "[arena] resume: the log's last line was truncated (the crash "
                "caught a write mid-flight); that game is replayed",
                flush=True,
            )
        if orphan_pair_ids:
            print(
                f"[arena] resume: {len(orphan_pair_ids)} pair(s) had only ONE "
                f"coloring played and are DISCARDED and replayed in full "
                f"(pentanomial scoring is pair-based): "
                f"{sorted(orphan_pair_ids)}",
                flush=True,
            )
    # `compile` is excluded from the resume fingerprint on purpose (a ratchet
    # retry at a lower concurrency or without compile must stay resumable), so
    # the mix it permits is surfaced here instead of being refused.
    this_compile = compile_tag(compile_models, mode=mode)
    this_hoist = hoist_tag(
        eval_max_batch, mode=mode, no_hoist=_no_hoist,
        uncapped_leaf_rows=uncapped_leaf_rows,
    )
    kept_hoist_tags = sorted(resumed.hoist_tags) if resumed is not None else []
    scored_hoist_tags = set(kept_hoist_tags)
    predicted_hoist_tags = scored_hoist_tags | (
        {this_hoist} if remaining_ids else set()
    )
    if len(predicted_hoist_tags) > 1:
        # Same forecast/fact split as the compile warning below: printed before
        # play where it can still change an operator's mind, while
        # `mixed_eval_hoist` in the record is what actually happened. "unknown"
        # here means the kept rows predate the field, so sameness cannot be
        # shown — which is a mix, not a match.
        print(
            f"[arena] WARNING: this resume is ABOUT TO MIX evaluator-hoist "
            f"settings. The pairs kept from {log_path} were played under "
            f"{kept_hoist_tags} and this process plays the remaining "
            f"{len(remaining_ids)} pair(s) under {this_hoist!r}. A tag of the "
            f"form 'N<M' means the leaf buffer was CAPPED BELOW what the search "
            f"asked for, so those games ran a different search, not merely a "
            f"different memory budget. eval_max_batch is deliberately outside "
            f"the resume fingerprint so a pre-hoist log stays resumable; the "
            f"result record's mixed_eval_hoist says whether the mix happened.",
            file=sys.stderr, flush=True,
        )
    kept_compile_tags = sorted(resumed.compile_tags) if resumed is not None else []
    # The tags of the games that end up SCORED. `this_compile` joins it after
    # the play loop, once it is known that this process scored a pair at all.
    scored_compile_tags = set(kept_compile_tags)
    predicted_tags = scored_compile_tags | (
        {this_compile} if remaining_ids else set()
    )
    if len(predicted_tags) > 1:
        # Printed BEFORE play, where it can still change an operator's mind, so
        # it is a forecast: `mixed_compile` in the result record is the fact.
        print(
            f"[arena] WARNING: this resume is ABOUT TO MIX torch.compile. The "
            f"pairs kept from {log_path} were played under "
            f"{kept_compile_tags} and this process plays the remaining "
            f"{len(remaining_ids)} pair(s) under {this_compile!r}. compile is "
            f"deliberately outside the resume fingerprint "
            f"(daily_gate_ratchet.sh re-derives it from free VRAM on every "
            f"retry), so this is allowed and the pooled Elo will span two "
            f"inference paths. The result record's mixed_compile says whether "
            f"it did: a run that scores no new pair mixes nothing.",
            file=sys.stderr, flush=True,
        )
    if resumed is not None:
        # Same shape as the two warnings above: the hypothesis is outside the
        # fingerprint on purpose, so the mix it permits is surfaced rather than
        # refused. Silence here means the log recorded no spec at all.
        spec_warning = sprt_spec_carryover_warning(resumed.sprt_spec, sprt)
        if spec_warning is not None:
            print(spec_warning, file=sys.stderr, flush=True)
    if (
        resumed is not None
        and resumed.games_loaded > 0
        and pgn_out is not None
        and not (pgn_out.exists() and pgn_out.stat().st_size > 0)
    ):
        # The PGN is opened in APPEND mode, so a resume writes only the games
        # this process plays. When the earlier segment's PGN is gone (or was
        # never asked for), the file that appears is a partial record of a run
        # whose JSONL is complete — and nothing downstream can tell.
        print(
            f"[arena] WARNING: --pgn-out {pgn_out} is missing or empty, but "
            f"{log_path} already holds {resumed.games_loaded} finished game(s). "
            f"The PGN will contain ONLY the games this process plays, not the "
            f"whole match the summary scores. Point --pgn-out at the earlier "
            f"segment's file, or treat this one as a partial record.",
            file=sys.stderr, flush=True,
        )
    pgn_writer: ArenaPgnWriter | None = None
    pgn_sink: PgnSink | None = None
    cand_name = ref_name = ""
    cand_search = ref_search = ""
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
            "ConfigHash": production_config_record()["config_hash"],
            "GitSha": git_sha(),
            "ArenaMode": mode,
        }
        if label:
            base_tags["ArenaLabel"] = label
        pgn_writer = ArenaPgnWriter(pgn_out, event=label or "arena", base_tags=base_tags)
        print(f"[arena] PGN output -> {pgn_out} "
              f"(White/Black = {cand_name} / {ref_name})", flush=True)

    # Created LAST, after every check that can still raise (--pgn-out's engine
    # name collision is one), because creating it writes the header: a run that
    # aborts after this point leaves a log behind, and the fixed re-run would
    # then hit the "log already exists" refusal for a reason unrelated to it.
    game_log = GameLogWriter(
        log_path, driver="arena_standard", settings=log_settings,
        resuming=had_log,
        # Recorded, NOT fingerprinted: a log has to say which hypothesis its
        # games were collected under — a verdict is unreadable a month later
        # without it — while a resume must never be refused over a spec, which
        # is what putting it in `log_settings` would do. None on a fixed-N run,
        # and then the header keeps its pre-branch shape exactly.
        info=(
            None if sprt is None
            else {SPRT_LOG_INFO_KEY: sprt.as_record()}
        ),
    )
    print(f"[arena] game log -> {log_path} (fingerprint {fingerprint})", flush=True)

    def _on_game(
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
        chunk: int | None = None,
        loop: str | None = None,
    ) -> None:
        """Persist ONE finished game. Called by all three play loops.

        WRITE ORDER IS THE INVARIANT: JSONL is written LAST; it is the commit
        record — anything not in it is replayed on resume. So the PGN (and its
        flush) goes first, and a crash in the window between the two costs a
        DUPLICATE PGN game tagged ``ResumeReplay`` on the next resume, which is
        visible and recoverable. The other order loses the PGN game outright
        with nothing to detect it: the resume reads a complete pair in the
        JSONL and never replays it. match_vs_uci.py orders its writes the same
        way.
        """
        score = score_from_result(result, a_is_white=a_is_white)
        if pgn_writer is not None:
            extra = {
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
                # Provenance for a POOLED fit. Two runs with the same engine
                # names, ConfigHash, GitSha and SideSearch but different
                # evaluator caps played DIFFERENT searches -- a binding cap
                # makes the C tree absorb leaves instead of evaluating them --
                # and Ordo would fit them as ONE player. The value is the
                # effective hoist state ("4096", "512<4032", "off", "n/a"),
                # the same string the game-log row carries, and its ABSENCE
                # identifies a PGN written by pre-hoist code: that is what makes
                # a banked file attributable without git archaeology.
                #
                # ⚑ Read from `this_hoist` HERE rather than declared in
                # `base_tags`: the writer is built before the checkpoints load,
                # so a base tag would freeze the pre-load FLOOR and a
                # relations-on side would stamp every game "4096" for a search
                # that was bound against 8192. Same by-reference closure the
                # game-log row's `eval_hoist` relies on.
                "EvaluatorHoist": this_hoist,
            }
            if int(pair_id) in orphan_pair_ids:
                # This pair is being REPLAYED because the crash left it half
                # played, and the PGN already holds that orphan game
                # (append-only: it cannot be unwritten). Tag the replacements
                # so a pooled fit can drop the stale row — for any PairId
                # carrying ResumeReplay, the games WITHOUT the tag are the
                # discarded orphans.
                extra["ResumeReplay"] = "1"
            pgn_writer.write_game(ArenaGame(
                white=cand_name if a_is_white else ref_name,
                black=ref_name if a_is_white else cand_name,
                result=result,
                moves=moves,
                start_fen=start_fen,
                pair_id=pair_id,
                pair_half=half,
                extra=extra,
            ))
        game_log.write_game({
            "pair_id": int(pair_id),
            "half": int(half),
            "opening_index": int(pair_id),
            # The SCHEDULE's opening, which is what a resume matches against;
            # `start_fen` is what the loop actually played from, kept as
            # evidence that the two agree.
            "opening_fen": openings[int(pair_id)].fen(),
            "start_fen": start_fen,
            "a_is_white": bool(a_is_white),
            "result": result,
            "score_candidate": score,
            "plies": int(plies),
            "termination": termination,
            "seed": int(seed),
            # What played THIS game. compile is outside the resume
            # fingerprint, so a resumed log can hold both values.
            "compile": this_compile,
            "eval_hoist": this_hoist,
            "chunk": None if chunk is None else int(chunk),
            "loop": loop,
            "duration_s": round(float(duration_s), 2),
        })

    pgn_sink = _on_game

    # ---- optional GSPRT early stop --------------------------------------
    # Built AFTER the resume load, and seeded with the resumed pairs: the
    # statistic is a function of the whole match, so an SPRT arena that crashed
    # and was resumed must decide on loaded + new, never on its own tail alone.
    # Built BEFORE play so its `pairs` count and boundary are printed by every
    # play loop that receives it.
    sprt_monitor: SprtMonitor | None = None
    if sprt is not None:
        sprt_monitor = SprtMonitor(
            sprt,
            prior_pair_scores=loaded_pair_scores,
            pairs_cap=len(openings),
            granularity=(
                SPRT_GRANULARITY_CHUNK
                if mode == "matched_sims" and not rolling
                else SPRT_GRANULARITY_PAIR
            ),
        )
        print(
            f"[arena] SPRT ON — --games {games} is now a HARD CAP "
            f"({len(openings)} pairs), not the sample size. "
            f"{sprt_monitor.spec.describe()}",
            flush=True,
        )

    t0 = time.time()
    if not openings_to_play:
        print(
            f"[arena] resume: all {len(openings)} pairs are already complete "
            f"in {log_path} — nothing left to play, scoring the log",
            flush=True,
        )
        pair_scores: list[float] = []
    elif sprt_monitor is not None and sprt_monitor.crossed():
        # The resumed pairs alone already decide it. Playing the remainder
        # would spend GPU hours to answer a question that is closed, and would
        # push the sample past the stopping time the error rates are defined
        # at — a sequential test that keeps going after it crosses is not the
        # test whose alpha was preregistered.
        print(
            f"[arena] SPRT: the boundary was ALREADY crossed by the "
            f"{len(loaded_pair_scores)} resumed pair(s) (LLR "
            f"{sprt_monitor.llr:+.4f} -> {sprt_monitor.verdict}); this "
            f"invocation plays ZERO games and scores the log",
            flush=True,
        )
        pair_scores = []
    elif mode == "matched_sims":
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
  # RE-DERIVE the uncapped leaf size now that the checkpoints exist. The launch
  # check had to run before this point so a refusal beats a multi-minute
  # compile, and there it could only assume relations OFF -- a FLOOR. A model
  # with `use_dynamic_relations` (configs/exp_dynamic_relations.yaml, default
  # off) forces `_use_pipeline` False at every board count, so the
  # single-buffer path runs at the real n and asks for 8192 rows at mcg 128 /
  # topk 32 where the floor said 4096. Without this, a cap in [4096, 8192) is
  # bound IN FACT while nothing warns and every recorded field says
  # bound=False -- a value accepted and then silently misreported, which is the
  # defect class this whole branch is about.
  #
  # Still BEFORE torch.compile, so a late warning can still change an
  # operator's mind rather than only explaining the number afterwards.
        if _no_hoist is None:
            _pre_load_leaf_rows = uncapped_leaf_rows
            uncapped_leaf_rows = arena_uncapped_leaf_rows(
                # The same pool the launch check used: only `relations` is new
                # here, so a difference between the two figures can only ever
                # be the flag this re-derivation exists to read.
                max_concurrent_games=pool_size,
                sides=(search_candidate, search_reference),
                relations=(
                    bool(getattr(model_candidate, "use_dynamic_relations", False)),
                    bool(getattr(model_reference, "use_dynamic_relations", False)),
                ),
            )
            _was_bound, leaf_cap_bound = (
                leaf_cap_bound, eval_max_batch < uncapped_leaf_rows,
            )
  # `this_hoist` is read by `_on_game` (defined above, called during play), so
  # reassigning it here is what puts the corrected tag on every game ROW too --
  # the same by-reference closure the neighbouring `this_compile` relies on.
            this_hoist = hoist_tag(
                eval_max_batch, mode=mode, no_hoist=_no_hoist,
                uncapped_leaf_rows=uncapped_leaf_rows,
            )
            if uncapped_leaf_rows != _pre_load_leaf_rows:
                print(
                    f"[arena] uncapped leaf-buffer size re-derived after load: "
                    f"{_pre_load_leaf_rows} -> {uncapped_leaf_rows} "
                    f"(dynamic relations on a side disable the eval pipeline)",
                    flush=True,
                )
            if leaf_cap_bound and not _was_bound:
                _warn_leaf_cap_binds(eval_max_batch, uncapped_leaf_rows, late=True)
  # Search reads exactly two heads -- `policy_own` (the prior) and `wdl` --
  # while `ChessNet.forward` otherwise computes ten. Its `_inference_only`
  # branch returns those two from the SAME expressions the full branch uses
  # (`policy_own(_policy_tokens(t), ft_bias=...)`, and `value_wdl(t)`, which is
  # what the coupled branch's `head_from_hidden(hidden(t))` evaluates to), so
  # this drops work without moving a number the search reads. Set BEFORE
  # torch.compile -- the same point worker.py and SlotBroker set it -- because
  # after compile it is a guard change on an already-traced graph.
  #
  # ⚑ NOT under --volatility-*. That search runs the PYTHON path and reads the
  # `volatility` head through `evaluate_encoded_with_volatility`, which
  # substitutes ZEROS when the key is absent rather than raising. Setting
  # `_inference_only` there would leave a volatility arena searching with vol=0
  # on every node and reporting it as a volatility result -- a value accepted
  # and then silently ignored, which is the defect class this change removes.
  # ⚑ ALSO gated on --eval-max-batch: 0 is documented as restoring the
  # pre-hoist arena, and a 2-head forward is not what a pre-hoist arena ran.
  # Under torch.compile the two-head branch is a DIFFERENT traced graph, so
  # "bit-identical in eager" does not carry to a compiled run -- which makes 0
  # a real escape hatch only if it turns this off as well.
        _full_heads = volatility_candidate is not None or not eval_max_batch
        if not _full_heads:
            for _m in (model_candidate, model_reference):
                if hasattr(_m, "_inference_only"):
                    setattr(_m, "_inference_only", True)
  # Read BACK off the models rather than echoing the intent: `hasattr` is the
  # gate above, so a model that never had the attribute must show as absent
  # here instead of being reported as configured.
        print(
            "[arena] inference-only heads: candidate={} reference={}{}".format(
                getattr(model_candidate, "_inference_only", "absent"),
                getattr(model_reference, "_inference_only", "absent"),
                (
                    " (forced full: --volatility-* needs the volatility head)"
                    if volatility_candidate is not None
                    else " (forced full: --eval-max-batch 0 restores the "
                         "pre-hoist 10-head forward)"
                    if not eval_max_batch
                    else " (policy_own + wdl only)"
                ),
            ),
            flush=True,
        )
        if compile_models:
            import torch
            # Plain inductor compile (NOT reduce-overhead/cudagraphs, which recompile
            # per batch shape and OOM'd us). Auto-dynamic batch: a couple of warmup
            # recompiles in chunk 1, then cached + reused across the shrinking batch
            # sizes and into chunk 2.
            model_candidate = torch.compile(model_candidate)
            model_reference = torch.compile(model_reference)
            print("[arena] torch.compile ON (inductor, auto-dynamic batch)", flush=True)
  # ONE evaluator per side for the whole run, built AFTER compile so it holds
  # the module the forwards actually go through. See `build_arena_evaluator`
  # for what the per-call evaluators cost. Three conditions, each PRINTED when
  # it turns the hoist off, because a knob that silently does nothing is the
  # defect this whole change is about:
  #  * CUDA only. The defect is the CUDA stream pool and the VRAM the allocator
  #    strands per stream; a CPU arena has neither, and would pay ~1.05 GB of
  #    host buffers at the default cap for nothing (262 MB per slot x 2 slots
  #    x 2 sides; page-locked on CUDA).
  #  * not under --volatility-*. That side searches on the PYTHON path, whose
  #    leaf batches are not sized against any evaluator cap, and hoisting only
  #    the side that stayed on the C path would put the two halves of the A/B
  #    on different transports.
  #  * --eval-max-batch 0, the same opt-out by hand, for reproducing a
  #    pre-hoist arena exactly.
        evaluator_candidate = None
        evaluator_reference = None
        if _no_hoist is None:
            from chess_anti_engine.inference_dispatcher import supports_inplace_api

            evaluator_candidate = build_arena_evaluator(
                model_candidate, device=device, max_batch=eval_max_batch,
            )
            evaluator_reference = build_arena_evaluator(
                model_reference, device=device, max_batch=eval_max_batch,
            )
  # Announced off the OBJECTS, through the very expressions mcts/gumbel_c.py
  # uses to read them (`getattr(eval_impl, "_max_batch", ...)`,
  # `supports_inplace_api`), so a hoist that produced an evaluator the search
  # would not actually use in-place cannot print as if it had. The two ids
  # differ iff the sides really got separate evaluators.
            print(
                "[arena] evaluator HOISTED (one per side, whole run): "
                f"{type(evaluator_candidate).__name__} "
                f"max_batch={getattr(evaluator_candidate, '_max_batch', 'absent')} "
                f"n_slots={getattr(evaluator_candidate, 'n_slots', 'absent')} "
                f"inplace={supports_inplace_api(evaluator_candidate)} "
                f"cand=0x{id(evaluator_candidate):x} ref=0x{id(evaluator_reference):x}",
                flush=True,
            )
        else:
            print(
                f"[arena] evaluator hoist OFF ({_no_hoist}): every search call "
                "builds its own LocalModelEvaluator, and on CUDA its own stream",
                flush=True,
            )
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
        pair_scores = []
        try:
            if rolling:
                # Rolling pool: fixed active-game count => fixed batch shape => compile
                # reuses one graph (no per-shape thrash), and the GPU never drains until
                # the very end (no per-chunk tail).
                #
                # ⚑ The POOL, not --max-concurrent-games. `_refill` tops the
                # board list up to its `pool_size` argument only WHILE THE
                # QUEUE LASTS, so what the loop can actually keep active is
                # bounded by the pairs still to play. Announcing the ceiling is
                # the same falsehood the leaf-row bound used to be built on:
                # "keep 128 games active" over a 2-game arena.
                #
                # ⚑ THREE different quantities are called some form of "pool"
                # in this neighbourhood and they are not interchangeable:
                #   * `rolling_pool` here — bounded by the REMAINDER this
                #     invocation plays, which is what this banner describes;
                #   * the `pool_size` local from the cap checks — bounded by
                #     the whole loaded SCHEDULE, because it scopes the recorded
                #     leaf-cap fields and those describe the MATCH, resumed
                #     pairs included. The two coincide except on a resume;
                #   * the callee's `pool_size=` parameter below, left as the
                #     raw ceiling on purpose — the loop's own refill is already
                #     capped by the queue length, so lowering it would change
                #     nothing except when the drain-time cache free fires.
                rolling_pool = arena_pool_size(
                    max_concurrent_games=max_concurrent_games,
                    n_pairs=len(openings_to_play),
                )
                print(
                    f"[arena] ROLLING pool: keep {rolling_pool} games active, "
                    f"start a fresh one as each finishes",
                    flush=True,
                )
                pair_scores = play_paired_games_matched_sims_rolling(
                    model_candidate, model_reference, openings_to_play,
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
                    pair_ids=remaining_ids,
                    prior_pair_scores=loaded_pair_scores,
                    sprt=sprt_monitor,
                    evaluator_candidate=evaluator_candidate,
                    evaluator_reference=evaluator_reference,
                    free_cached_vram=bool(eval_max_batch),
                )
            else:
                # Chunked: plays each chunk of `max_concurrent_games` to completion
                # (drains per chunk). Numerically identical (pair scores concatenate).
                chunk_pairs = max(1, int(max_concurrent_games) // 2)
                n_chunks = (len(openings_to_play) + chunk_pairs - 1) // chunk_pairs
                # The chunked SPRT look lives HERE rather than inside
                # `play_paired_games_matched_sims`: that function scores a chunk
                # only once every game in it is over, so stopping inside it would
                # mean imputing the unfinished games as draws. A chunk boundary
                # is a set of complete pairs; see SPRT_GRANULARITY_CHUNK.
                announce_sprt_armed(sprt_monitor, where="chunked")
                for ci in range(0, len(openings_to_play), chunk_pairs):
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
                    sub = openings_to_play[ci:ci + chunk_pairs]
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
                        pair_ids=remaining_ids[ci:ci + chunk_pairs],
                        chunk=ci // chunk_pairs,
                        evaluator_candidate=evaluator_candidate,
                        evaluator_reference=evaluator_reference,
                    ))
                    # The chunk has drained; its widest batch shapes are gone for
                    # good until the next chunk rebuilds them. Skipped under
                    # --eval-max-batch 0, which restores the pre-hoist arena whole.
                    if eval_max_batch:
                        _free_cached_vram(device)
                  # ⚑ loaded + played, not played alone. On a resumed run the
                  # running line is the only Elo an operator sees until the end,
                  # and printing just this invocation's pairs reports a number
                  # for a fraction of the match under a header that does not say
                  # so. The final fold at the bottom already works this way.
                    _so_far = loaded_pair_scores + pair_scores
                    print(
                        f"[arena] RUNNING Elo after {2 * len(_so_far)} games:",
                        flush=True,
                    )
                    print_summary(summarize_pentanomial(pentanomial_counts(_so_far)))
                    if sprt_should_stop(sprt_monitor, pair_scores, where="chunked"):
                        break
        except ActionDecodeError as exc:
            _abort_void(exc, completed_pairs=len(pair_scores))
        finally:
            if syzygy_tb is not None:
                syzygy_tb.close()
    elif mode == "matched_time":
        print(f"[arena] matched_time: {ms_per_move}ms/move per side")
        pair_scores = play_paired_games_matched_time(
            candidate, reference, openings_to_play,
            device=device, ms_per_move=ms_per_move, max_plies=max_plies,
            uci_args=uci_args, deadline=deadline, pgn_sink=pgn_sink,
            pair_ids=remaining_ids, sprt=sprt_monitor,
        )
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    duration_s = time.time() - t0
    # Fold the resumed pairs in. Order is irrelevant to the pentanomial (it
    # bins pair scores), so a resumed run and an uninterrupted one with the
    # same schedule produce the same counts, the same Elo and the same CI.
    played_pair_scores = list(pair_scores)
    pair_scores = loaded_pair_scores + played_pair_scores
    # What this invocation's compile setting ACTUALLY contributed. Gated on
    # pairs scored, not on pairs scheduled: a --max-seconds deadline that lands
    # before the first pair finishes adds no games, so flagging a mix there
    # would report a splice that never happened.
    if played_pair_scores:
        scored_compile_tags.add(this_compile)
        scored_hoist_tags.add(this_hoist)
    # Which pair ids the log must hold complete — knowable only when every
    # scheduled pair finished, since the play loops return scores, not ids.
    expected_pair_ids = (
        sorted(done_pair_ids | set(remaining_ids))
        if len(played_pair_scores) == len(remaining_ids) else None
    )
    # Did what we PERSISTED match what we SCORED? Answered off the DISK, after
    # the writer is closed: a game log that disagrees with the summary is not a
    # cosmetic bug — every future resume is built on it, and it would look
    # perfectly healthy right up to the wrong Elo.
    game_log.close()
    game_log_agrees, disagreement = verify_game_log_on_disk(
        log_path, settings=log_settings, openings=openings,
        expected_pair_scores=pair_scores,
        expected_pair_ids=expected_pair_ids,
    )
    if not game_log_agrees:
        print(
            f"[arena] WARNING: {log_path} does not hold what this run scored: "
            f"{disagreement}. The summary below uses the play loop; do NOT "
            f"--resume that log until this is understood.",
            file=sys.stderr, flush=True,
        )
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
            f"completed in {duration_s:.0f}s — "
            + (
                "the SPRT boundary was crossed, so this is a COMPLETED "
                "sequential test, not a short run; `--resume` would replay "
                "nothing and stop again on the same verdict"
                if sprt_monitor is not None and sprt_monitor.crossed()
                else "`--resume` with the same settings plays only the remainder"
            )
            + f" ({log_path})",
            flush=True,
        )
    summary = summarize_pentanomial(pentanomial_counts(pair_scores))
    print_summary(summary)
    sprt_record: dict[str, Any] | None = None
    if sprt_monitor is not None:
        # Settle the verdict against the pre-committed rule. `finalize` only
        # writes when no boundary was crossed, so an uncrossed run lands on
        # INCONCLUSIVE and is REPORTED as such — never quietly re-read as a
        # fixed-N result, which is the optional-stopping fallacy running in
        # reverse.
        sprt_monitor.finalize(
            stop_reason=(
                "max_seconds"
                if deadline is not None and time.time() >= deadline
                else "cap" if len(pair_scores) >= len(openings)
                else "incomplete"
            ),
        )
        sprt_record = sprt_monitor.as_record()
        print(f"[arena] {sprt_monitor.verdict_line()}", flush=True)

    record = build_result_record(
        summary,
        mode=mode,
        candidate=candidate,
        reference=reference,
        # Keep openings_path a real filesystem path; record the source kind in
        # its own field so downstream readers can tell FEN-list runs from book
        # runs without the path string changing shape. opening_plies is a book
        # concept, so it is null on the FEN path (it is never applied there).
        openings_path=openings_source,
        openings_kind=openings_kind,
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
        game_log=str(log_path),
        game_log_fingerprint=fingerprint,
        game_log_agrees=game_log_agrees,
        resumed_pairs=len(loaded_pair_scores),
        resumed_orphan_pairs=len(orphan_pair_ids),
        compile_setting=this_compile,
        compile_values=sorted(scored_compile_tags),
        hoist_setting=this_hoist,
        hoist_values=sorted(scored_hoist_tags),
        eval_max_batch=int(eval_max_batch),
        eval_leaf_cap_uncapped=(uncapped_leaf_rows or None),
        eval_leaf_cap_bound=leaf_cap_bound,
        # Banked UNCONDITIONALLY, unlike the leaf-cap fields above: those are
        # meaningless off the hoisted path and go null there, but the pool is a
        # property of the schedule and is just as true for a matched_time or
        # CPU row. Making it conditional would mean a null that says "not
        # applicable" on some rows and "written before the field" on others.
        max_concurrent_games=int(max_concurrent_games),
        arena_pool=int(pool_size),
        sprt=sprt_record,
    )
    if out_path is not None:
        if resumed is not None and not played_pair_scores:
            # A no-op resume recomputes and prints the same summary the
            # finished run already appended. Appending it again would put TWO
            # rows for one arena in a shared aggregate that other tools
            # average over — and the ratchet passes --resume unconditionally,
            # so re-running a completed series does exactly this. Gated on
            # pairs PLAYED, not on openings scheduled: an SPRT log whose
            # resumed pairs already crossed the boundary plays zero games
            # while openings_to_play still holds the unplayed remainder, and
            # that resume must not append a second row either.
            print(
                f"[arena] resume played ZERO new pairs "
                f"({len(loaded_pair_scores)} already in {log_path}), so "
                f"nothing is appended to {out_path} — a second row for one "
                f"arena would double-count it in a shared aggregate. This "
                f"process did NOT read {out_path}: confirm it holds the "
                f"earlier run's row (a crash between the last game and that "
                f"append would have left none).",
                flush=True,
            )
        else:
            append_result(record, out_path)
            print(f"[arena] result appended to {out_path}")
    return record


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arena knobs shared with scripts/elo_vs_sims.py."""
    # ⚑ In the SHARED set, unlike --sprt (declared in main(), for the reason
    # stated there): scripts/elo_vs_sims.py FORWARDS this one into every
    # run_arena call it makes, so the flag it advertises takes effect. Without
    # it a sims ladder was pinned to the default cap — no way to pick a smaller
    # one on a constrained card, and no way to use the documented 0 escape
    # hatch to reproduce a pre-hoist rung.
    #
    # ⚑ THE RULE FOR THIS FUNCTION IS "FORWARDED OR REFUSED", NEVER SILENTLY
    # DROPPED — and it is a rule about BOTH callers, not just this file. A knob
    # that only one of them can honour either stays out of the shared set
    # (--sprt) or is rejected by the script that cannot honour it: `--games` is
    # declared below and elo_vs_sims sizes its rungs from --games-per-rung, so
    # it refuses an explicit --games rather than accept a number it will not
    # read. Before adding anything here, check what the OTHER script does with
    # it.
    p.add_argument("--eval-max-batch", type=int, default=DEFAULT_EVAL_MAX_BATCH,
                   help="matched_sims: forward-batch cap for the ONE long-lived "
                        f"evaluator built per side (default: {DEFAULT_EVAL_MAX_BATCH}, "
                        "which is what production selfplay runs, and at the "
                        "default --max-concurrent-games is at or above every "
                        "batch the search asks for). ⚑ BELOW that it is a "
                        "SEARCH-SHAPE knob, not a memory knob: gumbel_c mins its "
                        "leaf buffer against this, and a full buffer makes the C "
                        "tree ABSORB surplus leaves as root-Q pseudo-terminals "
                        "instead of evaluating them, so the moves change. Values "
                        "below the arena's POOL SIZE (--max-concurrent-games "
                        "capped by the loaded opening pairs; elo_vs_sims runs "
                        "the 128 default) are refused — the root submit would "
                        "raise; values between that and the "
                        "uncapped leaf-buffer size run but print a loud warning "
                        "and are recorded in the result record and every game "
                        "row. 0 disables the hoist and restores the pre-hoist "
                        "arena exactly — per-call evaluators, 10-head forward, "
                        "no cache frees — for reproduction, not normal use.")
    # No default, on purpose. The silent default (`play`) is what made every
    # arena Elo in the ledger a measurement of a search selfplay never runs;
    # run_arena refuses matched_sims without it.
    p.add_argument("--search-shape", choices=SEARCH_SHAPES, default=None,
                   help="REQUIRED for matched_sims: which search to measure. "
                        "'training' = what production selfplay runs: linear root, "
                        "with c_scale/topk/vloss_weight/target_batch read from "
                        f"{production_config_path().name} at run time (deliberately not "
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
    p.add_argument("--resume", action="store_true",
                   help="continue the run recorded in this invocation's game "
                        "log: its COMPLETE pairs are kept and only the "
                        "remaining pairs are played, then everything is scored "
                        "as one pentanomial. A pair with only one coloring "
                        "played is discarded and replayed (pair-based "
                        "scoring). REFUSES if the log's recorded settings "
                        "(nets, seed, games, mode, sims, openings, search "
                        "shape, ...) differ from this invocation. Without this "
                        "flag an existing log for the same settings is an "
                        "error, so two runs can never be mixed by accident. "
                        "A resumed run is statistically valid under those "
                        "settings but is NOT bit-identical to an uninterrupted "
                        "one: the post-opening RNG stream restarts for the "
                        "remainder pairs, so the individual games differ even "
                        "though the population does not. The fingerprint also "
                        "stores checkpoint PATHS, not content hashes — "
                        "overwriting a weights file in place between segments "
                        "is undetectable here, so don't.")


def resolve_sides_from_args(args) -> tuple[SideSearch, SideSearch]:
    """The complete ``matched_sims`` search resolution, exactly as ``main()`` does it.

    Module-level rather than a block inside ``main()`` so the ORDER below is
    drivable by a test: shape -> per-side overrides -> refuse the flags the run
    would discard -> print. A guard whose invocation only ``main()`` performs is
    a guard nothing can prove runs, which is the defect this whole change is
    about; ``main()`` needs two checkpoints, so nothing was ever going to drive
    it there.

    Resolves BEFORE any model load, so a bad shape or an unrunnable knob costs a
    second rather than a four-minute compile.
    """
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
    refuse_flags_the_arena_would_discard(base, args)
  # AFTER the overrides, and after the shape is final: the sides are what will
  # actually be searched with. `describe()` therefore reports the realized
  # values including every CLI override, which is what makes the printed record
  # downstream of every override application site.
    for label, side in (("candidate", side_candidate), ("reference", side_reference)):
        print(f"[shape] {label}: {side.describe()}", flush=True)
    _warn_noise_schedule_deviation(base, add_noise=not args.no_gumbel_noise)
    return side_candidate, side_reference


def refuse_flags_the_arena_would_discard(base: SideSearch, args) -> None:
    """Refuse flags this run would accept, print, bank -- and then not use.

    Both cases below are the PR's own defect one level out: the value is not
    out of band, it is perfectly legal, and the arena simply never applies it.

    1. ``--volatility-*`` under ``--search-shape training``. ``match.py`` builds
       the config with the volatility kwargs and THEN applies ``side.gumbel``
       over it (``dataclasses.replace``, match.py:140-152). The training shape's
       ``gumbel`` dict carries ``volatility_q_scale`` / ``volatility_fpu`` /
       ``volatility_anchor`` read from production (0.0 / 0.0), so the replace
       overwrites the CLI value with production's zero and the Python volatility
       path never switches on -- while ``volatility_candidate`` is banked into
       the result record naming the operator's number. The PLAY shape carries no
       volatility keys, so there the flags survive; that combination stays legal.
    2. A non-finite ``--temperature``. ``sample_action_with_temperature`` gates
       on ``temperature > 0``, which is False for ``nan``, so the arena plays
       pure argmax while the JSONL records ``temperature: nan``.
    """
    if base.shape == "training" and _volatility_kwargs_from_args(args) is not None:
        raise SystemExit(
            "--volatility-* with --search-shape training: the training shape's "
            "gumbel dict carries volatility_q_scale/volatility_fpu from "
            "production (both 0.0) and match.py applies it AFTER the volatility "
            "kwargs, so your value is overwritten before the search sees it -- "
            "while the result record banks it as volatility_candidate. Use "
            "--search-shape play for a volatility A/B, or drop the flags."
        )
    temperature = float(args.temperature)
    if not math.isfinite(temperature):
        raise SystemExit(
            f"--temperature {temperature!r}: sample_action_with_temperature "
            "gates on `temperature > 0`, which is False for a non-finite value, "
            "so the arena would play pure argmax and record your value as the "
            "temperature it played at."
        )


def _volatility_kwargs_from_args(args) -> dict[str, float] | None:
    """CANDIDATE-side volatility search kwargs, or None when all flags are off.

    ⚑ The all-off early return happens BEFORE validation, deliberately: with
    both scales at 0.0 there is no volatility search to configure, and
    ``volatility_anchor`` alone cannot switch one on (this same predicate is
    what ``volatility_search_enabled`` asks). Validating a dict that is never
    built would refuse a run over a knob nothing reads.
    """
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
  # These are GumbelConfig fields that ride as kwargs instead of through
  # `SideSearch.gumbel`, so `SideSearch.__post_init__` never sees them -- and
  # they are banked into the result record (`volatility_candidate`) exactly as
  # given. `--volatility-q-scale nan` reads as ENABLED (nan != 0.0), forces the
  # Python path, and makes every sigma nan.
    import dataclasses as _dc

    from chess_anti_engine.mcts.gumbel import GumbelConfig, validate_gumbel_config

    try:
        validate_gumbel_config(
            _dc.replace(GumbelConfig(), **out), where="--volatility-*",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
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
                        "GitSha, BOTH sides' realized search shape and the "
                        "effective EvaluatorHoist state, plus "
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
    p.add_argument("--games-out", type=Path, default=None,
                   help="per-game JSONL written AS EACH GAME FINISHES (and read "
                        "back by --resume). Default: "
                        f"{DEFAULT_GAME_LOG_DIR}/<label>.<settings-fingerprint>"
                        ".games.jsonl. It is NOT derived from --out, which is a "
                        "shared append-only aggregate every arena writes to; "
                        "the fingerprint in the name is what keeps two "
                        "different runs out of one file. Pass this to resume a "
                        "log whose label has changed.")
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
    # ⚑ Declared HERE and not in add_common_args, which scripts/elo_vs_sims.py
    # also calls: that script builds its own run_arena calls and would not pass
    # the flag on, so sharing it would produce a --sprt that parses, prints and
    # then decides nothing. A knob offered on a path that ignores it is this
    # repo's signature defect; the fix is to not offer it there.
    p.add_argument("--sprt", default=None, metavar="elo0=E,elo1=E,alpha=A,beta=B",
                   help="OPT-IN sequential test (pentanomial GSPRT, fishtest's "
                        "stop rule). Default OFF, and off changes nothing: no "
                        "stop check runs and the JSONL record is byte-identical "
                        "to today's. When given, ALL FOUR of elo0, elo1, alpha, "
                        "beta are REQUIRED (no defaults — an unstated hypothesis "
                        "is not a hypothesis), e.g. "
                        "--sprt 'elo0=0,elo1=5,alpha=0.05,beta=0.05'. The LLR is "
                        "recomputed from every COMPLETE pair (resumed ones "
                        "included) and checked at pair boundaries — rolling and "
                        "matched_time after each pair, --no-rolling between "
                        "chunks; never mid-pair, which would leak opening bias "
                        "into the stop. --games becomes a HARD CAP: reaching it "
                        "without crossing is INCONCLUSIVE and is reported as "
                        "that, never as a fixed-N verdict. ⚑ The VERDICT is the "
                        "deliverable — a sequentially stopped Elo point estimate "
                        "is biased away from zero and its CI has no nominal "
                        "coverage; both are printed and banked as descriptive "
                        "only. This exists because ad-hoc peeking at a rolling "
                        "arena manufactured +112 Elo from a true null; a "
                        "preregistered boundary is the principled alternative to "
                        "'never look'.")
    add_common_args(p)
    args = p.parse_args()

    sprt_spec: SprtSpec | None = None
    if args.sprt is not None:
        try:
            sprt_spec = SprtSpec.from_cli(args.sprt)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None

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
        side_candidate, side_reference = resolve_sides_from_args(args)
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
        eval_max_batch=args.eval_max_batch,
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
        resume=bool(args.resume),
        game_log_path=args.games_out,
        uci_args=args.uci_args,
        label=args.label,
        volatility_candidate=_volatility_kwargs_from_args(args),
        search_candidate=side_candidate,
        search_reference=side_reference,
        sprt=sprt_spec,
    )


if __name__ == "__main__":
    main()
