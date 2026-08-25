#!/usr/bin/env python3
"""CPU generation-zero selfplay shard generator (uniform prior, no teacher).

Warms a replay window with real game data for the AZ-purity loop experiment,
before any net exists. At random initialisation a net's policy and value heads
are noise, so a Gumbel search driven by a UNIFORM prior over the legal moves and
a cheap position-independent value is the same search the real net would run --
and it needs no GPU. Running it on the CPU keeps the accelerator free while the
first iterations get real games instead of inference against a meaningless net.

What it emits
-------------
Production-shaped replay shards (``shard_NNNNNN.zarr``) through the repo's own
writer, ``replay.shard.save_local_shard_arrays`` -- so the schema is not
reimplemented here and cannot drift from the loader:

* ``x``              (175, 8, 8) float32 ``v2_threats`` planes, encoded by the
                     same ``encoding.cboard_encode.encode_cboard`` selfplay uses.
* ``policy_target``  the search's improved policy -- NOT a visit distribution;
                     Gumbel stores ``softmax(log_prior + sigma*Qbar)``, which is
                     dense over the legal moves by construction. Projected into
                     compact ``lc0_1858`` train space by
                     ``policy_vector_to_encoding`` exactly as
                     ``selfplay/finalize.py`` does. Read the next section before
                     forming any expectation about its shape.
* ``wdl_target``     the game outcome through ``selfplay.game._result_to_wdl``,
                     POV = the side to move AT THAT PLY (so it alternates down a
                     selfplay game). The convention is INHERITED from that
                     function, never restated here.
* ``legal_mask``, ``moves_left``, ``game_id``, ``ply_index``,
  ``opening_source_code``, ``is_selfplay``, ``is_network_turn``, ``has_policy``.

⚑ NO Stockfish fields and NO ``search_wdl``. Every ``has_sf_*`` flag is 0, which
is how the loader spells "absent" (``prune_storage_arrays`` drops an all-zero
optional pair entirely). Writing the stub's value into ``search_wdl`` was
considered and REJECTED: it would launder a constant -- or a hash of the piece
placement -- into a field every reader treats as a search's value estimate,
which is this codebase's signature defect verbatim.

⚑ WHAT THE ABSENT LABELS COST, stated at the right size. An earlier revision of
this note claimed the value TARGET silently changes with the yaml's fracs. It
does not, and the overstatement mattered: ``train/losses.py`` sets
``blend_fallback_target = game_oh`` and, on rows where ``has_sf_wdl`` and
``has_search_wdl`` are both 0, ``sf_component`` and ``search_component`` each
collapse to exactly ``game_oh``. With no SF fields there is also no
``has_adjusted_wdl``, so ``game_target`` is ``game_oh`` too, and the blend is
``(game_frac + sf_wdl_frac + search_wdl_frac) * game_oh == game_oh`` -- BIT-
IDENTICAL for any fracs, taper branch included.

The real cost is TELEMETRY, not targets. Every SF-conditioned column computed
over these rows has an identically-zero denominator: the orphaned-label rate
(``replay/shard.py::sf_eval_pv_orphan_flags`` -- ``checked`` is 0, so the rate
reads 0.000000, which is the value that means HEALTHY), the policy-side
no-MultiPV fingerprint, and the realized ``sf_wdl_frac`` reported next to a
component that is doing nothing. A window mixing gen-0 rows with live rows
therefore dilutes exactly the detectors that exist to catch a Stockfish desync.

⇒ Set ``sf_wdl_frac: 0.0``, ``sf_wdl_frac_floor: 0.0`` and ``search_wdl_frac:
0.0`` so the config STATES the value target it is training rather than arriving
there by fallback, and read the SF detectors' denominators, never just their
rates. ⚑ This requirement is UNENFORCED -- nothing in this tool or the trainer
checks it. A trainer-side refusal is the real end state and it is deferred, by
name, to the AZ-purity prereg (``scratchpad/az_purity/prereg_draft.md``), because
it is a training-affecting change that needs its own ledger entry rather than
being smuggled into an offline data tool. ``required_run_config`` in the sidecar
records the requirement so a run cannot claim it was never told.

What the policy target actually contains
----------------------------------------
⚑ MEASURED, and it is the property most likely to surprise a reader: under the
default ``--value-source zero`` the stored ``policy_target`` is uniform over the
legal moves on **99.32 %** of rows (total variation < 1e-3), and only **0.63 %**
are SHARP (TV > 0.1) -- the ones where sequential halving met a terminal inside
the tree. Measured over 488,249 rows on this head; an independent review measured
99.08 % / 0.77 % on an earlier one, so read the third digit off the sidecar
rather than off this paragraph. ``--value-source material`` is a different corpus
on this axis entirely: 39.70 % sharp against 44.58 % uniform.

That is CORRECT, not a bug, and the extraction is deliberately unchanged. Gumbel
stores ``softmax(log_prior + sigma*Qbar)``. At generation zero the prior is
uniform by construction and ``Qbar`` is a constant everywhere the tree did not
reach a terminal, so both terms are flat and the improved policy is flat with
them. A generation-zero target that claimed to know which move is better would be
knowledge the run does not have -- it would be the "sharp and wrong" failure the
ledger already owns, manufactured on purpose.

So be precise about where the gen-0 signal lives:
  * the TRAJECTORIES are diverse (Gumbel root noise on a uniform prior selects
    candidates at random, so the games are genuinely varied -- 138k games/h of
    them, ~75 % ending in checkmate);
  * ``wdl_target`` -- the game OUTCOME -- is the real learning signal;
  * ``policy_target`` contributes only near in-tree terminals, where mate
    detection sharpens it.
Every run measures this rather than assuming it: the sidecar's
``policy_target_shape`` block reports mean/median TV-to-uniform and the sharp-row
fraction, per run, so a change in the search that flattened or sharpened the
target is visible without re-deriving anything.

The evaluator stub
------------------
``UniformPriorEvaluator`` implements the ``inference.BatchEvaluator`` protocol,
so the production search calls it exactly where it would call the net.

* POLICY: full-width (4672) ZERO logits. ``_policy_logits_to_full`` dispatches on
  SHAPE, so a full-width vector passes through untouched and every legal move
  reaches ``_masked_priors`` with the same logit -- a provably uniform prior over
  the legal moves. (Compact 1858 logits would be scattered with ``fill_value =
  -1e9``, which is right for a real compact net and would silently zero the prior
  of any legal move outside the 1858 vocabulary.)
* VALUE: ``--value-source``, DEFAULT ``zero``.
    ``zero``     q = 0 everywhere. Pure: no handcrafted chess knowledge at all.
    ``material`` q = tanh(cp / 400) where cp is the standard piece-value balance
                 P=100 N=320 B=330 R=500 Q=900, us minus them, read from the
                 ENCODED planes (``plane_decode.decode_step0_bitboards``, which
                 returns the step-0 bitboards in the stored side-to-move frame),
                 so the value is already side-to-move POV by construction.
    ``random``   ⚑ STRUCTURED NOISE -- PLUMBING TESTS ONLY, never a training
                 corpus. q = blake2b over those same bitboards plus ``--seed``,
                 mapped to [-1, 1]: a fixed function of the position, which is
                 what a randomly-initialised net is, and NOT fresh noise per
                 visit. That fidelity is exactly the hazard. Because the value is
                 position-dependent, ``sigma*Qbar`` is non-constant everywhere:
                 MEASURED over 72,317 rows, **99.74 %** of its targets are SHARP
                 (TV > 0.1) and 0.25 % uniform, against 0.63 % / 99.32 % for
                 ``zero``. It emits a corpus of confident targets whose
                 confidence encodes a hash of the piece placement.
                 That is the ledger's "sharp and wrong" trap manufactured at
                 scale, and it is strictly worse than ``zero``, which is honestly
                 flat. Use it to prove the value plumbing moves the search, then
                 throw the shards away.

  q reaches the search as WDL LOGITS, because that is what the protocol returns.
  The mapping is exact, not approximate: ``q_to_wdl_logits`` emits
  ``log(p + 1e-9)`` for ``p = (max(q, 0), 1 - |q|, max(-q, 0))`` -- the minimum-
  information triple whose W-L margin is q -- and both search paths turn WDL
  logits into a value with ``softmax([w, d, l])`` then ``p_w - p_l``
  (``mcts/puct.py::_value_scalar_from_wdl_logits``; ``mcts/_mcts_tree.c:42``), so
  the value the search sees is q back to a relative 3e-9.

The native NNUE arms
--------------------
``--value-source nnue-static`` and ``--value-source nnue-qsearch`` replace the
stub's cheap value with the two race arms of ``chess_anti_engine/nnue``: a
Stockfish-quality evaluation with **no Stockfish process**. Both resolve check
nodes recursively (mandatory shared infrastructure -- the NNUE evaluation is
undefined in check); the second adds a tactical quiescence beyond that. The
prior stays uniform: no net is involved anywhere, at any point.

⚑ **The arm reads the POSITION, not the planes.** Check resolution needs the
legal evasions, the fifty-move terminal needs the halfmove clock, and the
two-fold terminal that makes a perpetual check finite needs the repetition
history -- none of which the 175-plane encoding carries. So the evaluator asks
the searching tree for its pending leaf ``CBoard``s
(``MCTSTree.pending_leaf_cboards``) and evaluates those; ``x`` is used only to
check that answer, once per worker, by re-encoding the tree's own first leaf and
demanding the planes match. Reconstructing a board from planes would hand the
resolver a position that merely looks right.

⚑ **The value scale is a free parameter, and it is a flag.** The arms return
Stockfish INTERNAL units, not centipawns; the banked UCI anchors were labelled
from Stockfish SEARCH cp through production's cp-logistic. ``--nnue-cp-per-unit``
is the bridge, its default was measured rather than assumed (see
``NNUE_CP_PER_INTERNAL_UNIT``), and it is recorded in the sidecar -- because
getting it wrong makes every stored policy target uniformly sharper or flatter
with no other symptom, which is precisely what a label-quality readout measures.

⚑ **``--all-root-moves`` defaults ON for the arms** and off for the pure
sources. Under a uniform prior, sequential halving picks its root candidates by
the Gumbel draw alone; with a real value that ranks a RANDOM SUBSET sharply and
never looks at the rest. The banked UCI anchors were generated with it, so a
native cell without it would differ from the anchor in search shape rather than
in value source. See ``MAX_LEGAL_MOVES``.

  PYTHONPATH=. nice -n 10 python3 scripts/gen_random_selfplay_shards.py \\
      --out-dir data/gen0_nnue --games 200 --workers 4 \\
      --value-source nnue-static --nnue-pack big.pack --sims 32 \\
      --run-id gen0_nnue_static

Search
------
The C tree (``mcts.gumbel_c.run_gumbel_root_many_c``) -- production's own search,
and it accepts a ``BatchEvaluator`` directly, so wiring the stub to it was one
argument. Measured on this box at ``--sims 32``: 1190 plies/s against 49 plies/s
for the Python path, for a bit-identical game at the same seed. Every search knob
defaults to the LIVE production value (see the defaults block); ``--sims`` is the
one deliberate departure -- live runs 100, this runs 32 for CPU throughput.

⚑ ``gumbel_scale`` is held at its PRE-decay value (1.0) for the whole game.
Production decays it to ``gumbel_scale_after`` 0.5 over 3 moves from move 12,
because by then it trusts a TRAINED prior. At generation zero the prior is
uniform, so decaying the noise would only make the games less diverse. Stated as
a deviation rather than absorbed. (An earlier revision of this line said
production decays to 0; the live yaml's own corrected comment says 0.5.)

Terminal handling mirrors production: the game state is a ``CBoard``, ends on
``CBoard.is_game_over()`` (50-move, threefold, insufficient material, no legal
moves) and is labelled from ``CBoard.result()``. A game truncated at
``--max-plies`` returns ``"*"``, which ``_result_to_wdl`` labels a draw --
production would adjudicate that with Stockfish, and there is no Stockfish here.
⚑ That draw label is a per-ROW cost, not a per-game one: truncated games are the
LONGEST games, so their share of rows is several times their share of games. Both
are reported (``terminations`` and ``terminations_by_rows``).

Usage
-----
  PYTHONPATH=. python3 scripts/gen_random_selfplay_shards.py \\
      --out-dir data/gen0_shards --games 2000 --workers 8

  PYTHONPATH=. python3 scripts/gen_random_selfplay_shards.py \\
      --out-dir data/gen0_shards --games 2000 --workers 8 \\
      --value-source material --openings data/opening_books/book.pgn.zip

Re-running against a populated ``--out-dir`` APPENDS: shard numbering starts
above the highest index already there, and an occupied index is skipped rather
than overwritten.

⚑ That is FILENAME safety, not CONTENT safety. The games a worker plays are a
pure function of ``--seed``, so re-running with the same seed -- after a crash,
or to top the corpus up -- replays the SAME games into fresh shard indices and
silently duplicates them in the window. Duplicate rows are not a loud failure:
they reweight the replay draw and inflate the effective epoch count over exactly
the positions that were played twice. **Every re-run into the same ``--out-dir``
must use a fresh ``--seed``.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V2_THREATS
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.encoding.plane_decode import decode_step0_bitboards
from chess_anti_engine.mcts import _mcts_tree as _mcts_tree_ext
from chess_anti_engine.mcts._mcts_tree import MCTSTree
from chess_anti_engine.mcts.gumbel import (
    SELFPLAY_GUMBEL_C_SCALE,
    GumbelConfig,
    assert_c_path_can_run,
    validate_gumbel_config,
)
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    index_to_move_strict,
    policy_mask_to_encoding,
    policy_vector_to_encoding,
)
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    iter_shard_paths,
    load_shard_arrays,
    local_shard_path,
    samples_to_arrays,
    save_local_shard_arrays,
    shard_index,
)
from chess_anti_engine.selfplay.finalize import _stable_game_id
from chess_anti_engine.selfplay.game import _result_to_wdl
from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board
from chess_anti_engine.selfplay.seed_manifest import opening_source_code
from chess_anti_engine.stockfish.wdl import cp_to_wdl_array, mate_to_effective_cp_array
from chess_anti_engine.utils.atomic import atomic_write_text
from chess_anti_engine.utils.git_meta import git_sha

_LOG = logging.getLogger("gen0")

# ── defaults ─────────────────────────────────────────────────────────────────
# ⚑ Every value below was read off the LIVE production yaml (the one the running
# trial re-reads each iteration), NOT off `main`'s copy of it -- those two differ
# in ~55 keys and `main` is the stale one. An earlier revision of this block
# sourced four numbers from `main` and every one of them was wrong: sims 256
# (live: 100), gumbel_scale 0.75 (live: 1.0 pre-decay / 0.5 after), policy_temp
# 1.0 (live: 1.5), and it claimed gumbel_scale "decays to 0 after move 12" when
# the yaml's own corrected comment says it decays to 0.5. Re-read the live file
# before changing any of them; do not infer a production value from `main`.
DEFAULT_SIMS = 32              # ⚑ the ONE deliberate departure; live runs 100
DEFAULT_TOPK = 16              # selfplay.gumbel_topk
# selfplay.gumbel_scale, the PRE-decay value. Live decays it to
# gumbel_scale_after 0.5 over gumbel_scale_decay_moves 3 from move 12; this tool
# holds the pre-decay value all game (see the module docstring).
DEFAULT_GUMBEL_SCALE = 1.0
# selfplay.gumbel_policy_temp. PROVABLY INERT at a uniform prior -- the stub's
# logits are all zero and 0/T == 0 for any finite T -- so this matches production
# rather than inventing a third value, and stays correct if the prior ever stops
# being uniform. Pinned by test_policy_temp_is_inert_at_a_uniform_prior.
DEFAULT_POLICY_TEMP = 1.5
DEFAULT_TEMPERATURE = 0.0      # selfplay.selfplay_temperature
DEFAULT_MAX_PLIES = 450        # selfplay.max_plies
DEFAULT_SHARD_SIZE = 2000      # distributed.shard_size
DEFAULT_OPENING_PLIES = 16     # selfplay.opening_book_max_plies_2
DEFAULT_OPENING_MAX_GAMES = 200_000  # selfplay.opening_book_max_games_2
# ── the two target-decoupling knobs, pinned to live rather than left at the
# library default. Both are inert at generation zero (they reshape the STORED
# target's sigma*Qbar and log_prior terms, and at a uniform prior with a constant
# value both terms are flat) -- which is exactly why they must be set NOW: the
# moment the values stop being constant, an unset knob is a silent departure from
# the shape the loop was calibrated on, and nothing would flag it.
DEFAULT_TARGET_MAX_VISIT_CAP = 5        # selfplay.gumbel_target_max_visit_cap
DEFAULT_TARGET_UNTEMPERED_PRIOR = True  # selfplay.gumbel_target_untempered_prior
# ── the two C-search controls that are NOT GumbelConfig fields ───────────────
# `dataclasses.replace` cannot reach them, so `run_gumbel_root_many_c` takes them
# as arguments and mcts/gumbel.py's standing comment requires every call site to
# pass them EXPLICITLY. That comment exists because they went missing once: every
# arena Elo in the ledger was measured at vloss_weight=0 while production selfplay
# ran 1. This single-board call shape is also the measured C17 duplicate-leaf
# worst case, so the default is production's, not the C fallback's.
DEFAULT_VLOSS_WEIGHT = 1       # selfplay.gumbel_vloss_weight
DEFAULT_TARGET_BATCH = 0       # SearchConfig.gumbel_target_batch (absent from the yaml)
DEFAULT_NICE = 10
DEFAULT_RUN_ID = "gen0_random_selfplay"
# Total-variation distance from uniform-over-legal at which a stored row counts
# as carrying search signal rather than restating the prior. See the module
# docstring's "What the policy target actually contains".
SHARP_ROW_TV_THRESHOLD = 0.1
_TV_HISTOGRAM_BINS = 1000

VALUE_SOURCE_ZERO = "zero"
VALUE_SOURCE_RANDOM = "random"
VALUE_SOURCE_MATERIAL = "material"
# ── the two native NNUE race arms ────────────────────────────────────────────
# ⚑ The strings are the provider names `_nnue_ext` publishes, NOT local labels:
# they are passed straight to `arm_open`, so a typo is a load failure at startup
# rather than a corpus labelled with an arm that never ran. Keeping a separate
# spelling here and mapping between them is how a value gets accepted and then
# quietly turned into the other one.
VALUE_SOURCE_NNUE_STATIC = "nnue-static"
VALUE_SOURCE_NNUE_QSEARCH = "nnue-qsearch"
NNUE_ARM_SOURCES: tuple[str, ...] = (
    VALUE_SOURCE_NNUE_STATIC, VALUE_SOURCE_NNUE_QSEARCH,
)
VALUE_SOURCES: tuple[str, ...] = (
    VALUE_SOURCE_ZERO, VALUE_SOURCE_RANDOM, VALUE_SOURCE_MATERIAL,
    *NNUE_ARM_SOURCES,
)

# ── the native arms' value scale ─────────────────────────────────────────────
# ⚑⚑ MEASURED, and it is a FREE PARAMETER THE PREREG DOES NOT PIN. The arms
# return Stockfish's INTERNAL units (psqt/16 + positional/16), not centipawns,
# while the banked UCI anchors were labelled from Stockfish's SEARCH cp through
# production's cp-logistic. A scale error here does not crash and does not show
# up as a wrong move: it makes every stored policy target uniformly sharper or
# flatter than the anchor's, which is exactly what the quality readout measures.
# So it is a flag, it is recorded in the sidecar, and its default was measured
# rather than assumed:
#
#   786 quiet positions reservoir-sampled from natural playouts (the resolver
#   bench's own stream, seed 20260824, capture bias 0.55), each evaluated by the
#   nnue-static arm and by `go nodes 512` on the Stockfish the production config
#   names. Least-squares slope of cp on internal units, through the origin:
#     |internal| <= 3000 (the informative band): slope 0.2784, R^2 0.877, n=641
#     median ratio over |internal| > 50:         0.2877
#     whole range:                               slope 0.2168, R^2 0.848, n=786
#   The whole-range slope is lower because Stockfish's cp normalisation
#   compresses large evaluations; 0.28 is the band the logistic actually cares
#   about. It also lands on Stockfish's own 100 / NormalizeToPawnValue ~= 0.281,
#   which is the independent confirmation, not the source.
NNUE_CP_PER_INTERNAL_UNIT = 0.28
# selfplay.sf_wdl_cp_slope / sf_wdl_cp_draw_width, read off the LIVE production
# yaml -- the same pair the banked UCI anchor shards were labelled with, so the
# two corpora differ in where the number came from and not in how it was mapped.
NNUE_CP_SLOPE = 0.006
NNUE_CP_DRAW_WIDTH = 120.0

# ── all-root-moves coverage ──────────────────────────────────────────────────
# ⚑⚑ THE TRAP THIS EXISTS FOR, and it bites the native arms exactly as it bites
# the UCI one. Sequential halving picks its root candidates by `gumbel +
# log_prior`; the prior here is UNIFORM, so `log_prior` is constant and the
# candidate set is chosen by the Gumbel draw ALONE -- a uniformly random subset
# of the legal moves. Under `--value-source zero` that is harmless (no move is
# ranked). Under a real value it is not: the search ranks a random subset
# sharply and never looks at the rest, so the stored target can be confident
# about 16 of 40 moves with the best move simply absent. The target is dense
# over the legal moves either way, so nothing downstream can see it.
#
# Full coverage needs BOTH `topk >= n_legal` AND `sims >= 2 * n_legal` (the
# candidate count is also capped at sims/2), and no constant `--sims` can
# promise that at every branching factor -- hence the per-position budget
# `max(--sims, 2 * n_legal)`. The banked UCI anchors were generated under
# exactly this rule (their sidecars read legal-moves mean 20.6 / sims mean
# 46.8 at `--sims` 32), so a native cell that dropped it would differ from the
# anchor in SEARCH SHAPE and the readout would be measuring that instead.
MAX_LEGAL_MOVES = 218  # the most legal moves any chess position admits
ROOT_SIMS_PER_LEGAL_MOVE = 2  # the sims/2 candidate cap, inverted

# ⚑⚑ AND THE PROMISE HAS A CEILING THE FLAG CANNOT RAISE. `gss_score_and_halve`
# writes into a `double scores_buf[GSS_MAX_CANDS]` and clamps `n_cands` to fit
# it, so on a root with more legal moves than the ceiling only the FIRST
# `GSS_MAX_CANDS` candidates are scored and the halving keeps the top half OF
# THOSE. Raising `topk` to 218 does not change it; the ceiling is a C buffer.
#
# ⚑ MEASURED, because the obvious reading of "dropped unscored" is wrong and
# would have gone into the readout. On the 218-legal-move position
# `R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1` at `--all-root-moves`
# (realized budget 436), every one of the 218 root children is expanded and
# visited -- `get_children_visits` reports 218 children, 218 with a non-zero
# visit -- and the stored target has non-zero mass on all 218. Budget is
# deducted for every candidate before the clamp. What the surplus does NOT get
# is a place in the RANKING: candidates past the ceiling are dropped from the
# halving without being scored, so they are never DEEPENED. The cap costs
# ranking resolution on wide roots, not coverage.
#
# DECIDED (not silently absorbed): the C limit stays at 64 and the generator
# REPORTS every root that exceeded it. Raising it to 218 would widen a stack
# buffer and the per-board candidate arrays in a state machine PRODUCTION
# SELFPLAY runs at 64+ boards per batch, to buy ranking resolution on positions
# that do not occur: over 500 measured games the widest root had 61 legal moves
# (the banked UCI anchors' widest was 65), so the ceiling binds on a tail this
# corpus has not produced. `root_budget.roots_over_candidate_cap` is 0 when that
# stays true, and is the number to read before believing it.
#
# Read off the extension rather than restated: a Python copy of a C buffer bound
# is exactly the drift this note exists about, and
# `test_the_exported_candidate_cap_is_the_compiled_one` pins the export against
# the `#define` that `gss_score_and_halve` actually sizes its buffer with.
GSS_MAX_CANDS: int = int(getattr(_mcts_tree_ext, "GSS_MAX_CANDS", 64))

#: Schema of the opt-in per-leaf observation bank (``--bank-leaf-observations``).
LEAF_BANK_SCHEMA = 1

# Standard piece values in centipawns, indexed the way decode_step0_bitboards
# returns them: columns 0-5 are "us" P/N/B/R/Q/K, 6-11 are "them". The king
# entry is 0 because both sides always have exactly one.
PIECE_VALUES_CP: tuple[float, ...] = (100.0, 320.0, 330.0, 500.0, 900.0, 0.0)
MATERIAL_CP_SCALE = 400.0
# Probability floor under the log so a certain-result triple stays finite. At
# 1e-9 the value the search reads back is q / (1 + 3e-9).
WDL_PROB_FLOOR = 1e-9

_POPCOUNT_LUT: np.ndarray = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1,
).sum(axis=1).astype(np.int64)


_CANDIDATE_CAP_WARNED = False


def _warn_candidate_cap(legal_moves: int) -> None:
    """Say ONCE per process that this root exceeded the C ranking ceiling.

    ⚑ The sidecar's ``roots_over_candidate_cap`` is the number to read; this is
    the line that makes an operator look. A clamp nobody is told about is how
    "every legal root move is a candidate" becomes a claim rather than a fact.
    ⚑ And the wording is the MEASURED effect, not the feared one -- see
    ``GSS_MAX_CANDS``: every legal move is still expanded, visited and carries
    target mass; what the surplus loses is a place in the halving's ranking.
    """
    global _CANDIDATE_CAP_WARNED
    if _CANDIDATE_CAP_WARNED:
        return
    _CANDIDATE_CAP_WARNED = True
    _LOG.warning(
        "--all-root-moves: this root has %d legal moves against the C sampler's "
        "GSS_MAX_CANDS=%d ranking ceiling, so only the first %d candidates are "
        "SCORED by sequential halving and the rest, though visited, are never "
        "deepened. Counted in the sidecar as root_budget.roots_over_candidate_cap.",
        int(legal_moves), GSS_MAX_CANDS, GSS_MAX_CANDS,
    )


def _sha256_file(path: Path) -> str:
    """SHA-256 of the bytes on disk, streamed.

    ⚑ Not the same question as ``_nnue_ext.source_sha256``, which returns the
    digest the PACKER embedded in the header: that names the ``.nnue`` the pack
    was built from and survives a truncated, corrupted or hand-edited payload
    unchanged. This one identifies the bytes the arm actually mapped.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Set-bit count of a uint64 array, without numpy-version-gated builtins."""
    arr = np.ascontiguousarray(values, dtype=np.uint64)
    as_bytes = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT_LUT[as_bytes].sum(axis=-1)


def q_to_wdl_logits(q: np.ndarray) -> np.ndarray:
    """(N,) value in [-1, 1] → (N, 3) WDL logits the search reads back as q.

    ``p = (max(q, 0), 1 - |q|, max(-q, 0))`` is the triple with the least
    win/loss mass whose ``p_w - p_l`` is q; the logits are ``log(p + floor)``,
    and both search paths recover the value as ``softmax(logits)[0] -
    softmax(logits)[2]``, i.e. ``q / (1 + 3*floor)``.
    """
    clipped = np.clip(np.asarray(q, dtype=np.float64), -1.0, 1.0)
    probs = np.stack(
        [
            np.maximum(clipped, 0.0),
            1.0 - np.abs(clipped),
            np.maximum(-clipped, 0.0),
        ],
        axis=1,
    )
    return np.log(probs + WDL_PROB_FLOOR).astype(np.float32)


def material_q(x: np.ndarray) -> np.ndarray:
    """Side-to-move material balance of encoded positions, as tanh(cp / 400)."""
    counts = _popcount64(decode_step0_bitboards(x)).astype(np.float64)
    values = np.asarray(PIECE_VALUES_CP, dtype=np.float64)
    cp = counts[:, :6] @ values - counts[:, 6:] @ values
    return np.tanh(cp / MATERIAL_CP_SCALE)


def random_q(x: np.ndarray, *, salt: int) -> np.ndarray:
    """A fixed pseudo-random value per POSITION (not per visit), seeded by salt."""
    bitboards = decode_step0_bitboards(x)
    salt_bytes = int(salt).to_bytes(8, "little", signed=False)
    out = np.empty((bitboards.shape[0],), dtype=np.float64)
    for i in range(bitboards.shape[0]):
        payload = np.ascontiguousarray(bitboards[i]).tobytes() + salt_bytes
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / float(1 << 64) * 2.0 - 1.0
    return out


# ⚑⚑ THREE KINDS OF NUMBER LIVE IN ONE `arm_stats` DICT, and merging them the
# same way is wrong for two of the three. The C context merges its own counters
# with atomic ADD and its peaks with atomic MAX; a Python merge that summed
# everything reported four workers at the depth cap 32 as `max_depth_seen: 128`
# -- a value the resolver cannot produce, above its own configured ceiling, and
# the configured ceiling itself came back as 128 too. Each key is classified
# ONCE, here, and an unclassified key is a hard error rather than a default.
#: Accumulated counts. `sum` across workers. Transcribed from
#: `arm_stats_dict` in `_nnue_ext.c`, where these are exactly the fields read
#: through `ARM_STAT_U64` and merged with `__atomic_fetch_add`.
#:
#: ⚑ THE REFUSAL IS NOT HYPOTHETICAL: `qterminal_draw` and `qply_cutoffs` were
#: missing from the first version of this set, every unit test passed (they
#: forge their own dicts), and the FIRST real two-worker qsearch run stopped
#: with "not classified" instead of publishing a merged number. That is the
#: guard doing its job, and it is why
#: `test_every_arm_stats_key_is_classified` now reads the key set off a live
#: handle rather than trusting this transcription.
_CTX_COUNTER_KEYS: frozenset[str] = frozenset({
    "calls", "calls_in_check", "nodes", "resolved_leaves", "depth_cutoffs",
    "terminal_mate", "terminal_draw", "qnodes", "qterminal_draw", "qply_cutoffs",
})
#: Peaks. `max` across workers; a sum is meaningless and a mean would hide the
#: tail these exist to expose.
_CTX_PEAK_KEYS: frozenset[str] = frozenset({"max_depth_seen", "qmax_ply_seen"})
#: Settings, not measurements. Every worker builds its context from the same
#: config, so these must AGREE; a disagreement is a real finding (a worker that
#: ran a different arm configuration) and is surfaced rather than averaged away.
_CTX_CONFIG_KEYS: frozenset[str] = frozenset({
    "resolver_max_depth", "qsearch_max_ply", "qsearch_check_plies",
})


@dataclass
class NnueArmStats:
    """What the native arm did, merged across workers, reported not implied."""

    batches: int = 0
    leaves: int = 0
    # ⚑ ROOTS ARE COUNTED SEPARATELY FROM LEAVES. One root position per ply also
    # goes through the arm, and the C context cannot tell the two apart -- so
    # every rate derived from `context` ("calls") is over roots AND leaves, and
    # is named for that. Pooling them would put a per-ply population inside a
    # per-leaf rate at a ratio (~1:45 here) that changes with the sim budget,
    # i.e. a number that moves when the ladder rung does, for no reason.
    roots: int = 0
    mate_band_leaves: int = 0
    mate_band_roots: int = 0
    binding_checks: int = 0
    # Copied out of the CONTEXT that ran (`arm_stats(handle)`), never restated
    # from the module globals -- `set_arm_config` is read at init(), so a global
    # read here would describe a configuration the batches may not have used.
    context: dict[str, int] = field(default_factory=dict)
    #: Config keys whose per-worker values did NOT agree, as {key: sorted values}.
    #: Empty is the healthy reading; non-empty means the cells are not one cell.
    context_conflicts: dict[str, list[int]] = field(default_factory=dict)

    def merge(self, other: NnueArmStats) -> None:
        self.batches += other.batches
        self.leaves += other.leaves
        self.roots += other.roots
        self.mate_band_leaves += other.mate_band_leaves
        self.mate_band_roots += other.mate_band_roots
        self.binding_checks += other.binding_checks
        for key, values in other.context_conflicts.items():
            self.context_conflicts[key] = sorted(
                set(self.context_conflicts.get(key, [])) | set(values),
            )
        for key, raw in other.context.items():
            value = int(raw)
            if key not in self.context:
                self.context[key] = value
                continue
            if key in _CTX_PEAK_KEYS:
                self.context[key] = max(self.context[key], value)
            elif key in _CTX_CONFIG_KEYS:
                if self.context[key] != value:
                    self.context_conflicts[key] = sorted(
                        set(self.context_conflicts.get(key, []))
                        | {self.context[key], value},
                    )
                    # Keep the LOWER value: for a ceiling, the run is only as
                    # deep as its shallowest worker, and a conflict is reported
                    # rather than smoothed either way.
                    self.context[key] = min(self.context[key], value)
            elif key in _CTX_COUNTER_KEYS:
                self.context[key] += value
            else:
                # ⚑ Not a default: a new C counter must be classified before it
                # can be merged, or it silently gets whichever rule happens to
                # be the fallback -- which is how `max_depth_seen` came to read
                # 128 in the first place.
                raise ValueError(
                    f"arm_stats key {key!r} is not classified as a counter, a "
                    "peak or a config value; add it to one of _CTX_*_KEYS in "
                    "gen_random_selfplay_shards.py before merging it",
                )

    def summary(self) -> dict[str, float]:
        if self.leaves == 0 and self.roots == 0:
            return {}
        ctx = self.context
        calls = float(ctx.get("calls", 0))
        resolved = float(ctx.get("resolved_leaves", 0)) or 1.0
        out: dict[str, float] = {
            "batches": float(self.batches),
            "leaves": float(self.leaves),
            "roots": float(self.roots),
            "leaves_per_batch": self.leaves / max(1, self.batches),
            "mate_band_leaves": float(self.mate_band_leaves),
            "mate_band_roots": float(self.mate_band_roots),
            "mate_band_frac": self.mate_band_leaves / max(1, self.leaves),
            "binding_checks": float(self.binding_checks),
        }
        if calls > 0:
            # The two numbers the prereg asks every cell to report. ⚑ The
            # in-check rate is over CALLS (leaves + one root per ply), because
            # that is the population the C context counts; `roots` above is what
            # a reader needs to price the difference.
            out["in_check_call_frac"] = float(ctx.get("calls_in_check", 0)) / calls
            out["resolver_expansion_factor"] = float(ctx.get("nodes", 0)) / resolved
        for key in sorted(_CTX_COUNTER_KEYS | _CTX_PEAK_KEYS):
            if key in ctx:
                out[f"ctx_{key}"] = float(ctx[key])
        # A merged run whose workers disagreed about their arm configuration is
        # not one cell; say so in the sidecar rather than publishing the min.
        out["ctx_config_conflicts"] = float(len(self.context_conflicts))
        return out


class NnueArmValueSource:
    """One of the two native NNUE race arms, as a batched value function.

    ⚑ THE ARM READS THE POSITION, NOT THE PLANES, and that is why this class
    takes ``CBoard``s rather than the encoded tensor every other value source
    here consumes. Check resolution needs the legal evasions (castling rights,
    the en-passant square), the fifty-move terminal needs the halfmove clock,
    and the two-fold terminal that makes a perpetual check finite needs the
    repetition history. None of those survive the 175-plane encoding, so a
    board rebuilt from planes would be a position that merely looks right --
    and the resolver would silently score a different tree.

    ⚑ ONE LONG-LIVED CONTEXT for the whole worker, not one per batch. The arm's
    configuration is snapshotted at ``init()``, and its counters live in the
    context; rebuilding it per call would reset every counter while the work
    kept accumulating, which is the exact defect ``nnue_resolver_bench.py``
    documents having had.

    ⚑ The realized knob values are read back out of that context via
    ``arm_stats`` and are what the sidecar reports. ``set_arm_config`` returns
    what it STORED for the next ``init()``; that is the producer's copy, and
    the producer's copy is not evidence the consumer received it.

    ⚑⚑ AND ONLY THE KNOBS THIS ARM CONSUMES ARE REPORTED. ``arm_stats`` answers
    with all three whichever arm asked, but ``cae_arm_static_eval`` reads only
    ``resolver_max_depth`` -- the static arm has no quiescence to budget. An
    earlier revision published ``qsearch_max_ply=4`` in a static run's realized
    line, which is this repo's signature defect with a correct-looking receipt
    on top: a value accepted, snapshotted, announced, and never consumed. The
    two quiescence knobs are ``None`` for ``nnue-static``, and supplying them
    explicitly there is REFUSED rather than absorbed.
    """

    def __init__(
        self,
        *,
        arm: str,
        pack: Path,
        resolver_max_depth: int,
        qsearch_max_ply: int | None,
        qsearch_check_plies: int | None,
        cp_per_internal_unit: float,
        cp_slope: float,
        cp_draw_width: float,
        leaf_bank: Path | None = None,
    ) -> None:
        from chess_anti_engine.nnue import _nnue_ext

        if arm not in NNUE_ARM_SOURCES:
            raise ValueError(f"arm must be one of {NNUE_ARM_SOURCES}, got {arm!r}")
        wants_qsearch = arm == VALUE_SOURCE_NNUE_QSEARCH
        if not wants_qsearch and (
            qsearch_max_ply is not None or qsearch_check_plies is not None
        ):
            raise ValueError(
                f"{arm} does not consume the quiescence knobs (cae_arm_static_eval "
                "reads resolver_max_depth only); pass them as None rather than "
                "letting them be stored and reported unused",
            )
        if wants_qsearch and (
            qsearch_max_ply is None or qsearch_check_plies is None
        ):
            raise ValueError(
                f"{arm} consumes qsearch_max_ply and qsearch_check_plies; both "
                "must be given",
            )
        # ⚑ isfinite, not `> 0`. Every comparison against NaN is False, so a NaN
        # slope sails through a positivity check and then propagates: the value
        # target of every row becomes NaN, the loss silences it as an absent
        # label, and nothing raises. A guard that a NaN passes is not a guard.
        for name, value in (
            ("--nnue-cp-per-unit", cp_per_internal_unit),
            ("--nnue-cp-slope", cp_slope),
            ("--nnue-cp-draw-width", cp_draw_width),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name} must be finite, got {value!r}: a non-finite value "
                    "does not fail any range check and reaches every stored "
                    "target as NaN",
                )
        if float(cp_per_internal_unit) <= 0.0:
            raise ValueError(
                "--nnue-cp-per-unit must be > 0: it is the slope of a monotone "
                f"map from internal units to cp, got {cp_per_internal_unit!r}",
            )
        if float(cp_slope) <= 0.0 or float(cp_draw_width) < 0.0:
            raise ValueError(
                "--nnue-cp-slope must be > 0 and --nnue-cp-draw-width >= 0, got "
                f"{cp_slope!r} / {cp_draw_width!r}",
            )
        self._ext = _nnue_ext
        self.arm = str(arm)
        self.pack = Path(pack)
        self.consumes_qsearch = wants_qsearch
        self.cp_per_internal_unit = float(cp_per_internal_unit)
        self.cp_slope = float(cp_slope)
        self.cp_draw_width = float(cp_draw_width)
        # ⚑ `set_arm_config` takes all three unconditionally, so the static arm
        # still has to hand it SOMETHING for the quiescence pair. It gets the
        # extension's own compiled-in defaults -- not a caller value, and not a
        # number this class then republishes. That is the whole point of P1-1:
        # the value exists in the context because the C setter demands it, and
        # `consumed_keys` keeps it out of every report because the static
        # evaluator never reads it.
        # The clamp is for the static path only, and it exists because
        # `set_arm_config` validates the triple as a chain
        # (check_plies <= max_ply <= resolver_max_depth): a static run at
        # `--nnue-resolver-max-depth 1` would otherwise be REFUSED over a
        # default it does not consume.
        qmax = (
            int(qsearch_max_ply) if qsearch_max_ply is not None
            else min(int(_nnue_ext.QSEARCH_MAX_PLY), int(resolver_max_depth))
        )
        qchk = (
            int(qsearch_check_plies) if qsearch_check_plies is not None
            else min(int(_nnue_ext.QSEARCH_CHECK_PLIES), qmax)
        )
        # Set BEFORE arm_open: the triple is read at init() and a context keeps
        # it for life, so setting it afterwards would configure the next
        # context and leave this one running the previous values.
        stored = dict(
            _nnue_ext.set_arm_config(int(resolver_max_depth), qmax, qchk),
        )
        self._handle = _nnue_ext.arm_open(self.arm, str(self.pack))
        self.stats = NnueArmStats(context=dict(_nnue_ext.arm_stats(self._handle)))
        self.consumed_keys = (
            ("resolver_max_depth", "qsearch_max_ply", "qsearch_check_plies")
            if self.consumes_qsearch else ("resolver_max_depth",)
        )
        self.requested = {k: int(stored[k]) for k in self.consumed_keys}
        self.realized = {
            key: int(self.stats.context[key]) for key in self.consumed_keys
        }
        # ⚑ TWO DIFFERENT HASHES, and the one that was here before is not the
        # one a reader assumes. `source_sha256` returns the digest the packer
        # EMBEDDED in the header -- it names the `.nnue` the pack was built
        # from, and a pack whose payload was truncated, corrupted or hand-edited
        # keeps reporting it. The identity of the bytes the arm actually mapped
        # is a hash of the file; both are reported, each under its own name.
        self.pack_source_sha256 = str(
            _nnue_ext.source_sha256(_nnue_ext.load(str(self.pack))),
        )
        self.pack_file_sha256 = _sha256_file(self.pack)
        self.kernel = "avx2" if _nnue_ext.simd_active() else "scalar"
        self._bank = None if leaf_bank is None else leaf_bank.open("a")
        self.leaf_bank_path = None if leaf_bank is None else leaf_bank
        self.bank_rows = 0
        # Above the mate band floor a value is a mate score in PLIES, not an
        # evaluation in internal units -- read off the extension's own exported
        # constants so a change to the C scale cannot leave a stale copy here.
        self.mate_band_floor = float(
            _nnue_ext.RESOLVER_MATE_BASE
            - _nnue_ext.RESOLVER_MAX_PLIES * _nnue_ext.RESOLVER_MATE_PLY_STEP,
        )
        self.mate_base = float(_nnue_ext.RESOLVER_MATE_BASE)
        self.mate_ply_step = float(_nnue_ext.RESOLVER_MATE_PLY_STEP)

    def q_from_values(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Arm values (internal units, or a mate score) → (q in [-1,1], mate mask).

        Two bands, and conflating them is the banked N1 defect in a new scale: a
        mate score run through the cp slope would land at hundreds of thousands
        of "centipawns", which is harmless only because the logistic saturates
        -- until someone changes the slope. Mates are folded through
        ``mate_to_effective_cp``, THE single mate→score home, exactly as the SF
        value source does; evaluations are scaled and only then mapped.
        """
        values = np.asarray(raw, dtype=np.float64)
        is_mate = np.abs(values) >= self.mate_band_floor
        eff_cp = np.empty_like(values)
        eff_cp[~is_mate] = values[~is_mate] * self.cp_per_internal_unit
        if bool(is_mate.any()):
            mated = values[is_mate]
            plies = (self.mate_base - np.abs(mated)) / self.mate_ply_step
            # ⚑ The SIGN IS APPLIED TO THE RESULT, never folded into the plies.
            # `mate_to_effective_cp` reads the sign off its argument, and a mate
            # AT depth 0 has plies == 0 -- so `sign * plies` is 0 for both
            # colours and every immediate mate would come back as a WIN. Caught
            # by test_mate_band_values_do_not_go_through_the_centipawn_slope.
            eff_cp[is_mate] = np.sign(mated) * np.abs(
                mate_to_effective_cp_array(plies),
            )
        wdl = cp_to_wdl_array(
            eff_cp, slope=self.cp_slope, draw_width_cp=self.cp_draw_width,
        )
        # W - L, as the SF value source does; see that mapping's note on why the
        # D channel is dropped rather than propagated.
        return (
            wdl[..., 0].astype(np.float64) - wdl[..., 2].astype(np.float64)
        ), is_mate

    def q_for_boards(
        self,
        boards: list[CBoard],
        *,
        role: str = "leaf",
        cluster: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Evaluate a batch through the arm's one live context.

        ``role`` is "leaf" for a search batch and "root" for the one position
        per ply the caller evaluates before the search starts. They are counted
        apart -- see ``NnueArmStats`` for why pooling them corrupts the rates.
        ``cluster`` is the (game index, ply index) this batch belongs to, used
        only by the leaf bank.
        """
        raw = np.asarray(
            self._ext.arm_handle_eval(self._handle, boards), dtype=np.float64,
        )
        q, is_mate = self.q_from_values(raw)
        if self._bank is not None:
            self._bank_batch(boards, raw, role=role, cluster=cluster)
        if role == "root":
            self.stats.roots += int(q.size)
            self.stats.mate_band_roots += int(is_mate.sum())
            return q
        self.stats.batches += 1
        self.stats.leaves += int(q.size)
        self.stats.mate_band_leaves += int(is_mate.sum())
        return q

    def _bank_batch(
        self,
        boards: list[CBoard],
        raw: np.ndarray,
        *,
        role: str,
        cluster: tuple[int, int] | None,
    ) -> None:
        """One JSONL row per evaluated position: the RAW arm score, and its key.

        ⚑ THE AGGREGATE IS NOT THE MEASUREMENT, and here the raw score is the
        only thing that cannot be recovered later. ``--nnue-cp-per-unit`` and
        the cp-logistic are a pure function of it; banking the raw value makes
        a scale correction, a different estimator, or a re-derived target a
        REANALYSIS rather than a regeneration of the corpus. Without it the
        arithmetic is baked into ``policy_target`` and the only way back is to
        play the games again.
        """
        assert self._bank is not None
        game, ply = (-1, -1) if cluster is None else cluster
        for board, value in zip(boards, raw.tolist(), strict=True):
            self._bank.write(
                json.dumps(
                    {
                        "schema": LEAF_BANK_SCHEMA,
                        "arm": self.arm,
                        "role": role,
                        "fen": board.fen(),
                        "value": int(value),
                        "game": int(game),
                        "ply": int(ply),
                        "pack_file_sha256": self.pack_file_sha256,
                        "cp_per_internal_unit": self.cp_per_internal_unit,
                        "cp_slope": self.cp_slope,
                        "cp_draw_width": self.cp_draw_width,
                        **self.realized,
                    },
                    sort_keys=True,
                )
                + "\n",
            )
            self.bank_rows += 1

    def refresh_context_stats(self) -> None:
        """Copy the live context's counters into the reported stats."""
        self.stats.context = dict(self._ext.arm_stats(self._handle))

    def close(self) -> None:
        """Freeze the context counters. The handle itself is refcounted in C."""
        self.refresh_context_stats()
        if self._bank is not None:
            self._bank.close()
            self._bank = None


class UniformPriorEvaluator:
    """``BatchEvaluator`` stub: uniform prior over legal moves + a cheap value.

    ``eval_calls`` / ``eval_rows`` are not decoration: they are how a test
    proves the search actually consumed ``--sims`` rather than merely storing it
    (a bigger budget must ask this object for more positions).

    ⚑ THE NATIVE ARMS TAKE A DIFFERENT ROUTE THROUGH THE SAME OBJECT. They
    cannot read ``x`` -- see ``NnueArmValueSource`` for why planes are not
    enough -- so they ask the tree that produced this batch for the leaf boards
    themselves, and ``x`` is used only to CHECK that answer. The tree is bound
    per search by ``bind_tree``; an unbound tree, or a board count that does not
    fit the batch, is a hard failure. Falling back to a plane-derived value
    there would be this repo's signature defect: a corpus labelled by an
    evaluator nobody asked for, with no symptom.
    """

    def __init__(
        self,
        *,
        value_source: str,
        expected_planes: int,
        random_salt: int = 0,
        nnue_source: NnueArmValueSource | None = None,
        input_history_encoding: str = LC0_HISTORY_ROOT_LEGACY_META,
        input_extra_features: str = EXTRA_FEATURES_V2_THREATS,
    ) -> None:
        if value_source not in VALUE_SOURCES:
            raise ValueError(
                f"value_source must be one of {VALUE_SOURCES}, got {value_source!r}",
            )
        if (value_source in NNUE_ARM_SOURCES) != (nnue_source is not None):
            raise ValueError(
                f"value_source {value_source!r} and nnue_source disagree: the two "
                "native arms REQUIRE a source and the other three refuse one, "
                "because a source that is present and unused is a value accepted "
                "and ignored",
            )
        if nnue_source is not None and nnue_source.arm != value_source:
            raise ValueError(
                f"nnue_source runs arm {nnue_source.arm!r} but value_source is "
                f"{value_source!r}: the corpus would be labelled with the arm "
                "that did not produce it",
            )
        self.value_source = str(value_source)
        self.expected_planes = int(expected_planes)
        self.random_salt = int(random_salt)
        self.nnue_source = nnue_source
        self.input_history_encoding = str(input_history_encoding)
        self.input_extra_features = str(input_extra_features)
        self.eval_calls = 0
        self.eval_rows = 0
        self._tree: MCTSTree | None = None
        self._cluster: tuple[int, int] | None = None

    def bind_tree(
        self, tree: MCTSTree | None, *, cluster: tuple[int, int] | None = None,
    ) -> None:
        """Name the tree whose pending leaves this evaluator will read.

        Called once per search by ``play_game``. ``None`` for the three
        plane-only sources, which never look at a tree. ``cluster`` is the
        (game index, ply index) the batches belong to, carried only so a banked
        leaf observation can be re-stratified by the unit it was drawn from --
        positions inside one search are correlated, and a row without its
        cluster key can only be re-analysed as an independent draw.
        """
        self._tree = tree
        self._cluster = cluster

    def close(self) -> None:
        if self.nnue_source is not None:
            self.nnue_source.close()

    def _leaf_boards(self, n_rows: int) -> list[CBoard]:
        """The pending leaf boards for this batch, checked against the planes.

        ⚑ Three checks, and each one is a real failure mode rather than
        defensive noise. No bound tree: the evaluator was handed a batch it has
        no way to interpret. An empty list: the tree has no pending batch, so
        these planes came from somewhere else. More boards than rows: the list
        is not this batch. The row count may EXCEED the board count -- the
        caller pads the encode buffer up to a bucket size and ignores the tail
        -- so only one direction is an error.
        """
        tree = self._tree
        if tree is None:
            raise RuntimeError(
                "native arm evaluator has no bound tree: call bind_tree(tree) "
                "with the tree passed to run_gumbel_root_many_c before searching",
            )
        if not hasattr(tree, "pending_leaf_cboards"):
            raise RuntimeError(
                "compiled _mcts_tree has no pending_leaf_cboards(); rebuild the "
                "C extensions: python3 scripts/build_production_extensions.py",
            )
        boards = tree.pending_leaf_cboards()
        if not boards:
            raise RuntimeError(
                "native arm evaluator got a batch with no pending leaf boards; "
                "the encoded rows it was handed were produced by a different tree",
            )
        if len(boards) > n_rows:
            raise RuntimeError(
                f"native arm evaluator got {len(boards)} pending leaf boards for "
                f"{n_rows} encoded rows: the board list is not this batch",
            )
        # ⚑⚑ AND THE COUNT, AGAINST A SECOND ACCESSOR. The row count above is a
        # one-directional check by necessity (the caller pads the encode buffer
        # up to a bucket and ignores the tail, so rows > boards is normal), and
        # that leaves a TRUNCATED batch invisible: `evaluate_encoded` fills
        # `q[:len(boards)]` and every unfilled row keeps q = 0, which is the
        # value of a drawn position. A dropped tail would therefore backprop a
        # confident draw into real leaves and change the target with nothing
        # raising -- this repo's signature defect, in the one place a plane
        # comparison cannot see it (a truncation from the END leaves every
        # remaining board on its correct row).
        #
        # `get_pending_legal_indices` reports the same batch's leaf count from a
        # DIFFERENT C entry point, so the two disagreeing is exactly the failure
        # this cannot otherwise detect. MEASURED cost, not assumed: 0.22 us per
        # call against a 418.7 us static-arm batch on the 44-legal-move root at
        # `--sims 64`, i.e. 0.05%.
        pending = len(tree.get_pending_legal_indices()[1])
        if pending != len(boards):
            raise RuntimeError(
                f"native arm evaluator got {len(boards)} pending leaf boards but "
                f"the tree reports {pending} pending leaves: the batch is "
                "truncated, and the missing rows would be scored as draws",
            )
        return boards

    def _check_binding_once(self, boards: list[CBoard], arr: np.ndarray) -> None:
        """Re-encode the first leaf and compare, ONCE per evaluator.

        The row-count checks prove the list is the right SIZE. This proves it is
        the right batch, by re-deriving row 0's planes from the board the tree
        handed back and demanding they match the planes the tree wrote. It runs
        once because it is the wiring that can be wrong, not a per-batch risk --
        and once is enough to fail every run that got it wrong.
        """
        source = self.nnue_source
        if source is None or source.stats.binding_checks:
            return
        source.stats.binding_checks += 1
        expected = encode_cboard(
            boards[0],
            input_history_encoding=self.input_history_encoding,
            input_extra_features=self.input_extra_features,
        )
        if not np.array_equal(np.asarray(expected, dtype=np.float32), arr[0]):
            raise RuntimeError(
                "native arm leaf binding check FAILED: re-encoding the tree's "
                "own first pending leaf board does not reproduce the planes the "
                "tree wrote for row 0 -- the boards and the batch are not the "
                "same positions",
            )

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # `relations` belongs to the protocol; compute_relations stays off here.
        del relations
        arr = np.asarray(x)
        if (
            arr.ndim != 4
            or int(arr.shape[1]) != self.expected_planes
            or tuple(int(d) for d in arr.shape[2:]) != (8, 8)
        ):
            # A shape mismatch here means the encoding flags did not reach the
            # search. Fail loudly: silently evaluating the wrong planes is the
            # failure this whole tool would be useless under. The board dims are
            # checked too -- a plane COUNT alone would wave through a
            # transposed or ragged batch that happened to have the right depth.
            raise ValueError(
                f"evaluator expected (N, {self.expected_planes}, 8, 8) planes, "
                f"got {tuple(arr.shape)}",
            )
        n = int(arr.shape[0])
        self.eval_calls += 1
        self.eval_rows += n
        policy_logits = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        if self.nnue_source is not None:
            boards = self._leaf_boards(n)
            self._check_binding_once(boards, arr)
            # The padded tail is never read back (the caller slices to n_leaves),
            # so it stays at the drawn-value of a position with no information.
            q = np.zeros((n,), dtype=np.float64)
            q[: len(boards)] = self.nnue_source.q_for_boards(
                boards, cluster=self._cluster,
            )
        elif self.value_source == VALUE_SOURCE_ZERO:
            q = np.zeros((n,), dtype=np.float64)
        elif self.value_source == VALUE_SOURCE_MATERIAL:
            q = material_q(arr)
        else:
            q = random_q(arr, salt=self.random_salt)
        return policy_logits, q_to_wdl_logits(q)


@dataclass(frozen=True)
class GenConfig:
    """Everything the game loop is allowed to depend on, in one picklable box."""

    out_dir: Path
    games: int = 100
    workers: int = 1
    sims: int = DEFAULT_SIMS
    topk: int = DEFAULT_TOPK
    c_scale: float = SELFPLAY_GUMBEL_C_SCALE
    policy_temp: float = DEFAULT_POLICY_TEMP
    temperature: float = DEFAULT_TEMPERATURE
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE
    target_max_visit_cap: int = DEFAULT_TARGET_MAX_VISIT_CAP
    target_untempered_prior: bool = DEFAULT_TARGET_UNTEMPERED_PRIOR
    vloss_weight: int = DEFAULT_VLOSS_WEIGHT
    target_batch: int = DEFAULT_TARGET_BATCH
    value_source: str = VALUE_SOURCE_ZERO
    all_root_moves: bool = False
    # ── the native NNUE arms ────────────────────────────────────────────────
    # ⚑ An arm knob is a RESOLVED CONCRETE NUMBER exactly when the selected arm
    # CONSUMES it -- filled from the extension's own exported defaults in
    # `config_from_args`, in the parent, so the sidecar's config block states the
    # value that ran instead of a sentinel a reader has to know how to expand.
    # It is `None` when no arm is selected AND when the selected arm does not
    # read it: `nnue-static` leaves the quiescence pair `None`, because
    # `cae_arm_static_eval` never looks at it and a number here would be
    # published as realized. See `resolve_arm_knob_defaults`.
    nnue_pack: Path | None = None
    nnue_resolver_max_depth: int | None = None
    nnue_qsearch_max_ply: int | None = None
    nnue_qsearch_check_plies: int | None = None
    nnue_cp_per_unit: float = NNUE_CP_PER_INTERNAL_UNIT
    nnue_cp_slope: float = NNUE_CP_SLOPE
    nnue_cp_draw_width: float = NNUE_CP_DRAW_WIDTH
    # Raw per-leaf observations beside the shards. OFF by default -- see
    # `_bank_batch` for what it buys and `build_parser` for what it costs.
    bank_leaf_observations: bool = False
    max_plies: int = DEFAULT_MAX_PLIES
    shard_size: int = DEFAULT_SHARD_SIZE
    seed: int = 0
    nice: int = DEFAULT_NICE
    openings: Path | None = None
    opening_plies: int = DEFAULT_OPENING_PLIES
    opening_max_games: int = DEFAULT_OPENING_MAX_GAMES
    random_start_plies: int = 0
    input_history_encoding: str = LC0_HISTORY_ROOT_LEGACY_META
    input_extra_features: str = EXTRA_FEATURES_V2_THREATS
    history_rep_fix: bool = True
    run_id: str = DEFAULT_RUN_ID


@dataclass(frozen=True)
class WorkerSpec:
    cfg: GenConfig
    worker_id: int
    games: int
    seed: int
    shard_index_start: int


@dataclass(frozen=True)
class PlyRecord:
    """One stored ply: the position BEFORE the move, and the search's answer."""

    x: np.ndarray
    policy_probs: np.ndarray  # (4672,) improved policy, search space
    legal_mask: np.ndarray    # (4672,) bool
    pov_white: bool           # side to move AT THIS PLY
    ply_index: int


@dataclass(frozen=True)
class GameOutcome:
    records: list[PlyRecord]
    result: str
    plies: int
    termination: str
    start_fen: str
    opening_source: str
    move_trace: str
    end_ply_index: int


def root_simulation_budget(legal_moves: int, *, cfg: GenConfig) -> int:
    """Simulations for THIS root, so every legal move can be a candidate.

    ``--sims`` is a FLOOR, not the value: sequential halving caps the candidate
    set at ``sims / 2``, so a fixed budget silently truncates wide positions.
    Returns ``max(--sims, 2 * n_legal)`` under ``--all-root-moves`` and
    ``--sims`` verbatim otherwise. See ``ROOT_SIMS_PER_LEGAL_MOVE``.
    """
    if not cfg.all_root_moves:
        return int(cfg.sims)
    return max(int(cfg.sims), ROOT_SIMS_PER_LEGAL_MOVE * int(legal_moves))


@dataclass
class RootBudgetStats:
    """What the root budget actually cost, so the sidecar reports it.

    ``--sims`` stops being the realized budget under ``--all-root-moves``, and a
    flag whose realized value is invisible is the thing this repo keeps getting
    caught by. Sums rather than a list: bounded at any ply count.
    """

    plies: int = 0
    legal_sum: int = 0
    legal_max: int = 0
    sims_sum: int = 0
    sims_max: int = 0
    # ⚑ Roots where the all-root-moves PROMISE could not be kept, because the C
    # sampler scores at most GSS_MAX_CANDS candidates per halving round and
    # clamps above that. 0 is the reading that says the ceiling never bound.
    roots_over_candidate_cap: int = 0
    legal_over_cap_max: int = 0

    def add(self, *, legal_moves: int, sims: int, candidate_cap: int = 0) -> None:
        self.plies += 1
        self.legal_sum += int(legal_moves)
        self.legal_max = max(self.legal_max, int(legal_moves))
        self.sims_sum += int(sims)
        self.sims_max = max(self.sims_max, int(sims))
        if candidate_cap and int(legal_moves) > int(candidate_cap):
            self.roots_over_candidate_cap += 1
            self.legal_over_cap_max = max(self.legal_over_cap_max, int(legal_moves))

    def merge(self, other: RootBudgetStats) -> None:
        self.plies += other.plies
        self.legal_sum += other.legal_sum
        self.legal_max = max(self.legal_max, other.legal_max)
        self.sims_sum += other.sims_sum
        self.sims_max = max(self.sims_max, other.sims_max)
        self.roots_over_candidate_cap += other.roots_over_candidate_cap
        self.legal_over_cap_max = max(
            self.legal_over_cap_max, other.legal_over_cap_max,
        )

    def summary(self) -> dict[str, float]:
        if self.plies == 0:
            return {}
        return {
            "plies": float(self.plies),
            "legal_moves_mean": self.legal_sum / self.plies,
            "legal_moves_max": float(self.legal_max),
            "sims_mean": self.sims_sum / self.plies,
            "sims_max": float(self.sims_max),
            # The coverage ceiling, reported whether or not it bound.
            "candidate_cap": float(GSS_MAX_CANDS),
            "roots_over_candidate_cap": float(self.roots_over_candidate_cap),
            "roots_over_candidate_cap_frac": (
                self.roots_over_candidate_cap / self.plies
            ),
            "legal_moves_over_cap_max": float(self.legal_over_cap_max),
        }


def policy_tv_to_uniform(policy: np.ndarray, legal_mask: np.ndarray) -> float:
    """Total-variation distance from uniform-over-legal, in [0, 1].

    THE instrument for "does this row carry search signal or just restate the
    prior". 0 means the stored target is exactly the uniform prior the search
    started from; 1 means it is a one-hot. Computed over the row's own legal set
    so it does not depend on how many moves happen to be legal.
    """
    legal = np.flatnonzero(np.asarray(legal_mask))
    if legal.size <= 1:
        return 0.0
    values = np.asarray(policy, dtype=np.float64)[legal]
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    return float(0.5 * np.abs(values / total - 1.0 / legal.size).sum())


@dataclass
class PolicyShapeStats:
    """Streaming mean/median TV-to-uniform and sharp-row share over a run.

    The median comes from a fixed-width histogram rather than a retained list so
    the memory is bounded at any row count; ``_TV_HISTOGRAM_BINS`` = 1000 puts
    the quantisation error at 5e-4, two orders below the sharp-row threshold.
    """

    rows: int = 0
    tv_sum: float = 0.0
    sharp_rows: int = 0
    uniform_rows: int = 0
    histogram: list[int] = field(default_factory=lambda: [0] * _TV_HISTOGRAM_BINS)

    def add(self, tv: float) -> None:
        self.rows += 1
        self.tv_sum += float(tv)
        self.sharp_rows += int(tv > SHARP_ROW_TV_THRESHOLD)
        self.uniform_rows += int(tv < 1e-3)
        idx = min(_TV_HISTOGRAM_BINS - 1, max(0, int(tv * _TV_HISTOGRAM_BINS)))
        self.histogram[idx] += 1

    def merge(self, other: PolicyShapeStats) -> None:
        self.rows += other.rows
        self.tv_sum += other.tv_sum
        self.sharp_rows += other.sharp_rows
        self.uniform_rows += other.uniform_rows
        for i, count in enumerate(other.histogram):
            self.histogram[i] += count

    def summary(self) -> dict[str, float]:
        if self.rows == 0:
            return {}
        target = self.rows / 2.0
        seen = 0
        median = 0.0
        for i, count in enumerate(self.histogram):
            seen += count
            if seen >= target:
                median = (i + 0.5) / _TV_HISTOGRAM_BINS
                break
        return {
            "rows": float(self.rows),
            "tv_to_uniform_mean": self.tv_sum / self.rows,
            "tv_to_uniform_median": median,
            "sharp_row_frac": self.sharp_rows / self.rows,
            "uniform_row_frac": self.uniform_rows / self.rows,
            "sharp_row_tv_threshold": SHARP_ROW_TV_THRESHOLD,
        }


@dataclass
class WorkerResult:
    worker_id: int
    realized: dict[str, Any]
    games: int = 0
    rows: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    plies: list[int] = field(default_factory=list)
    terminations: dict[str, int] = field(default_factory=dict)
    termination_rows: dict[str, int] = field(default_factory=dict)
    shards: list[dict[str, Any]] = field(default_factory=list)
    policy_shape: PolicyShapeStats = field(default_factory=PolicyShapeStats)
    root_budget: RootBudgetStats = field(default_factory=RootBudgetStats)
    nnue: NnueArmStats = field(default_factory=NnueArmStats)
    eval_calls: int = 0
    eval_rows: int = 0
    bank_rows: int = 0
    seconds: float = 0.0
    # Set when the worker died partway. ⚑ The stats above are then PARTIAL, not
    # absent -- see WorkerFailure.
    failed: str | None = None


class WorkerFailure(Exception):
    """A worker died, carrying the work it had already done.

    ⚑ WITHOUT THIS THE ACCOUNTING LIES IN THE ONE CASE THAT MATTERS. A worker
    that raises after evaluating for an hour used to lose its ``WorkerResult``
    entirely: the shards it had already flushed stayed on disk and turned up in
    the sidecar as ORPHANS (files present, attributed to nobody), while the run
    block reported an empty ``nnue`` section -- so the summary said the arm did
    no work next to the files it produced. The partial result travels with the
    exception and is folded in before the failure is re-raised, so a crashed run
    still states how far it got.
    """

    def __init__(self, result: WorkerResult, cause: BaseException) -> None:
        # ⚑ `args` IS the pickle. `ProcessPoolExecutor` sends the worker's
        # exception back to the parent by pickling it, and the default
        # `BaseException.__reduce__` reconstructs it as `cls(*self.args)`. An
        # `__init__` that called `super().__init__(<message string>)` would
        # therefore arrive in the parent as `WorkerFailure("worker 0 failed: ")`
        # -- a TypeError during unpickling, inside the futures machinery, in
        # place of the real error. The two payload objects are the args, and the
        # message is built in `__str__` instead.
        super().__init__(result, cause)
        self.result = result
        self.cause = cause

    def __str__(self) -> str:
        return f"worker {self.result.worker_id} failed: {self.cause}"


def build_gumbel_config(cfg: GenConfig) -> GumbelConfig:
    """The search config the game loop will hand to the C tree, and nothing else.

    Validated here, at the construction boundary, by the repo's own
    ``validate_gumbel_config``: an out-of-band knob would otherwise be recorded
    in the realized line and the sidecar while the hot path quietly ignored it,
    which is a worse outcome than a refusal for a tool whose whole product is
    data labelled with the settings that produced it.
    """
    if cfg.all_root_moves and int(cfg.topk) < MAX_LEGAL_MOVES:
        # ⚑ REFUSE, do not silently raise it. `--all-root-moves` promises every
        # legal root move is a candidate and `topk` caps that set, so honouring
        # one while ignoring the other would publish a coverage guarantee the
        # search does not keep. Same rule as the gumbel-override refusal.
        raise ValueError(
            f"--all-root-moves needs --topk >= {MAX_LEGAL_MOVES} (the most legal "
            f"moves any position admits), got {int(cfg.topk)}: topk caps the "
            "root candidate set, so a lower value would drop legal moves from "
            "the very target this flag exists to make complete",
        )
    gcfg = GumbelConfig(
        simulations=int(cfg.sims),
        topk=int(cfg.topk),
        temperature=float(cfg.temperature),
        policy_temp=float(cfg.policy_temp),
        c_scale=float(cfg.c_scale),
        add_noise=float(cfg.gumbel_scale) > 0.0,
        gumbel_scale=float(cfg.gumbel_scale),
        # Pinned to live, inert at gen-0, correct the moment values stop being
        # constant -- see DEFAULT_TARGET_* above.
        target_max_visit_cap=int(cfg.target_max_visit_cap),
        target_untempered_prior=bool(cfg.target_untempered_prior),
        input_history_encoding=str(cfg.input_history_encoding),
        input_extra_features=str(cfg.input_extra_features),
        policy_encoding=POLICY_ENCODING_LC0_1858,
    )
    validate_gumbel_config(gcfg, where="gen_random_selfplay_shards")
    assert_c_path_can_run(gcfg, where="gen_random_selfplay_shards")
    return gcfg


def build_opening_config(cfg: GenConfig) -> OpeningConfig:
    """The two opening sources this tool has: a book, and random start plies.

    ⚑ The blind-spot FEN-list branch of ``resolve_slot_opening`` is
    STRUCTURALLY UNREACHABLE from this generator. It needs both a list path and
    a positive draw probability on the shared ``OpeningConfig``, and neither is
    a flag here nor assigned below, so both stay at the dataclass defaults
    (``None`` / ``0.0``) on every run. That is deliberate: generation zero is
    pure selfplay, and seeding from harvested blind spots would be curriculum
    data smuggled into the arm that exists to have none.

    Because it cannot vary, it is NOT announced in ``realized_config`` — see
    that function's docstring for why a constant in the realized line is a
    liability rather than extra provenance. The tests reach the branch by
    building an ``OpeningConfig`` directly, which is a fixture, not a setting.
    """
    return OpeningConfig(
        opening_book_path=None if cfg.openings is None else str(cfg.openings),
        opening_book_max_plies=int(cfg.opening_plies),
        opening_book_max_games=int(cfg.opening_max_games),
        opening_book_prob=1.0 if cfg.openings is not None else 0.0,
        random_start_plies=int(cfg.random_start_plies),
    )


def build_nnue_source(
    cfg: GenConfig, *, leaf_bank: Path | None = None,
) -> NnueArmValueSource | None:
    """The native arm for this config, or None for the three pure sources.

    Built INSIDE the worker (unlike the pack path, which is resolved once in the
    parent): the arm context holds an mmap and its own counters, and a context
    pickled across a spawn boundary would be neither.
    """
    if cfg.value_source not in NNUE_ARM_SOURCES:
        return None
    if cfg.nnue_pack is None:
        raise ValueError(
            f"--value-source {cfg.value_source} needs --nnue-pack: the weight "
            "pack is a runtime artifact and nothing in this repo hardcodes one",
        )
    if cfg.nnue_resolver_max_depth is None:
        raise ValueError(
            "native arm knobs are unresolved; config_from_args fills them from "
            "the extension's exported defaults before the workers start",
        )
    # ⚑ The quiescence pair is resolved for `nnue-qsearch` and left None for
    # `nnue-static`, so "unresolved" is arm-dependent and cannot be checked as
    # `is None` across the board. `resolve_arm_knob_defaults` owns the mapping;
    # this only asserts the shape it produced still matches the arm, because a
    # None reaching the qsearch arm would silently run the extension default.
    wants_qsearch = cfg.value_source == VALUE_SOURCE_NNUE_QSEARCH
    if wants_qsearch and (
        cfg.nnue_qsearch_max_ply is None or cfg.nnue_qsearch_check_plies is None
    ):
        raise ValueError(
            "quiescence knobs are unresolved for nnue-qsearch; "
            "resolve_arm_knob_defaults fills them from the extension's own "
            "exported defaults before the workers start",
        )
    return NnueArmValueSource(
        arm=str(cfg.value_source),
        pack=Path(cfg.nnue_pack),
        resolver_max_depth=int(cfg.nnue_resolver_max_depth),
        qsearch_max_ply=(
            int(cfg.nnue_qsearch_max_ply)
            if wants_qsearch and cfg.nnue_qsearch_max_ply is not None else None
        ),
        qsearch_check_plies=(
            int(cfg.nnue_qsearch_check_plies)
            if wants_qsearch and cfg.nnue_qsearch_check_plies is not None
            else None
        ),
        cp_per_internal_unit=float(cfg.nnue_cp_per_unit),
        cp_slope=float(cfg.nnue_cp_slope),
        cp_draw_width=float(cfg.nnue_cp_draw_width),
        leaf_bank=leaf_bank if cfg.bank_leaf_observations else None,
    )


def realized_config(
    *,
    gcfg: GumbelConfig,
    evaluator: UniformPriorEvaluator,
    opening_cfg: OpeningConfig,
    cfg: GenConfig,
    worker_id: int,
) -> dict[str, Any]:
    """Read the realized settings back off the objects that will consume them.

    Deliberately NOT built from ``args``: every field here is fetched from the
    ``GumbelConfig`` the search receives, the evaluator instance it calls, the
    ``OpeningConfig`` the sampler reads, or the OS. A flag that got dropped on
    the way in is therefore visible in this line, not just in the parser.

    ⚑ And ONLY such fields. A field no flag can move is not extra provenance,
    it is a constant wearing a realized value's clothes — the mirror of the
    defect this line exists to catch, and it dilutes the one property the line
    claims (every entry here moved because something asked it to). So the
    unreachable FEN-list opening fields are absent by the same rule that keeps
    the parser's own spellings out; ``build_opening_config`` says why they
    cannot vary. Wiring a flag for one is what should put it back, and the
    deletion-annotation guard in ``tests/test_deletion_annotations.py`` will
    stop that PR to have the key re-judged, which is the right place for it.
    """
    arm = evaluator.nnue_source
    return {
        "worker_id": int(worker_id),
        "simulations": int(gcfg.simulations),
        "topk": int(gcfg.topk),
        "c_scale": float(gcfg.c_scale),
        "policy_temp": float(gcfg.policy_temp),
        "temperature": float(gcfg.temperature),
        "add_noise": bool(gcfg.add_noise),
        "gumbel_scale": float(gcfg.gumbel_scale),
        "target_max_visit_cap": int(gcfg.target_max_visit_cap),
        "target_untempered_prior": bool(gcfg.target_untempered_prior),
        # NOT GumbelConfig fields -- arguments of run_gumbel_root_many_c, read
        # back off the same GenConfig the call site passes.
        "vloss_weight": int(cfg.vloss_weight),
        "target_batch": int(cfg.target_batch),
        "input_history_encoding": str(gcfg.input_history_encoding),
        "input_extra_features": str(gcfg.input_extra_features),
        "input_planes": int(evaluator.expected_planes),
        "policy_encoding": str(gcfg.policy_encoding),
        "policy_width": int(COMPACT_POLICY_SIZE),
        "value_source": str(evaluator.value_source),
        # The coverage guarantee and the knob that implements it. `topk` above is
        # read off the GumbelConfig the search receives, so these disagreeing
        # would be visible rather than inferred.
        "all_root_moves": bool(cfg.all_root_moves),
        "root_sims_per_legal_move": (
            ROOT_SIMS_PER_LEGAL_MOVE if cfg.all_root_moves else 0
        ),
        "material_cp_scale": float(MATERIAL_CP_SCALE),
        "random_salt": int(evaluator.random_salt),
        # ── the native arm, read off the SOURCE OBJECT the evaluator calls, and
        # inside it off the CONTEXT that will run the batches. `set_arm_config`
        # returns what it stored for the next init(); that is the producer's
        # copy. `arm_stats(handle)` is the consumer's. They are both reported so
        # a divergence is a reading rather than a discovery.
        "nnue_arm": None if arm is None else str(arm.arm),
        # ⚑ TWO HASHES, NEITHER OF WHICH IS THE OTHER. `nnue_pack_sha256` is a
        # digest of the pack FILE -- the bytes the arm mapped, which is what
        # "which weights ran" means. `nnue_pack_source_sha256` is the `.nnue`
        # digest the packer EMBEDDED in the header, which names the upstream net
        # and keeps reporting it even if the payload was truncated or edited.
        # An earlier revision published the embedded digest under the file
        # digest's name, so two packs with the same provenance and different
        # bytes were indistinguishable in the manifest.
        "nnue_pack_sha256": None if arm is None else str(arm.pack_file_sha256),
        "nnue_pack_source_sha256": (
            None if arm is None else str(arm.pack_source_sha256)
        ),
        "nnue_kernel": None if arm is None else str(arm.kernel),
        # ⚑ `.get`, not `[...]`: the quiescence pair is ABSENT from `realized`
        # on the static arm, which does not consume it, so those two report
        # None. A number there would be a knob announced as realized by an arm
        # whose evaluator never reads it -- this repo's signature defect with a
        # receipt attached. The static arm still reports resolver_max_depth,
        # which it does consume.
        "nnue_resolver_max_depth": (
            None if arm is None else arm.realized.get("resolver_max_depth")
        ),
        "nnue_qsearch_max_ply": (
            None if arm is None else arm.realized.get("qsearch_max_ply")
        ),
        "nnue_qsearch_check_plies": (
            None if arm is None else arm.realized.get("qsearch_check_plies")
        ),
        "nnue_consumed_knobs": (
            None if arm is None else list(arm.consumed_keys)
        ),
        "nnue_arm_config_requested": None if arm is None else dict(arm.requested),
        "nnue_cp_per_internal_unit": (
            None if arm is None else float(arm.cp_per_internal_unit)
        ),
        "nnue_cp_slope": None if arm is None else float(arm.cp_slope),
        "nnue_cp_draw_width": None if arm is None else float(arm.cp_draw_width),
        "nnue_mate_band_floor": None if arm is None else float(arm.mate_band_floor),
        "nnue_leaf_bank": (
            None if arm is None or arm.leaf_bank_path is None
            else str(arm.leaf_bank_path.name)
        ),
        # ⚑ The all-root-moves ceiling the C sampler imposes, announced next to
        # the flag that promises coverage -- see GSS_MAX_CANDS.
        "root_candidate_cap": int(GSS_MAX_CANDS),
        # ⚑ Terminal root shortcuts are DISABLED under all-root-moves: the
        # shortcut prunes an immediately-drawing legal move from a winning root
        # before candidate selection, which is a coverage hole the flag exists
        # to close. Read off the same expression the call site passes.
        "allow_terminal_root_shortcuts": bool(not cfg.all_root_moves),
        "opening_book_path": opening_cfg.opening_book_path,
        "opening_book_prob": float(opening_cfg.opening_book_prob),
        "opening_book_max_plies": int(opening_cfg.opening_book_max_plies),
        "opening_book_max_games": int(opening_cfg.opening_book_max_games),
        "random_start_plies": int(opening_cfg.random_start_plies),
        "history_rep_fix": bool(rep_fix.current() or False),
        "max_plies": int(cfg.max_plies),
        "shard_size": int(cfg.shard_size),
        "nice": int(os.getpriority(os.PRIO_PROCESS, 0)),
        "torch_threads": int(torch.get_num_threads()),
    }


def format_realized(realized: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in realized.items())


TERMINATIONS: tuple[str, ...] = (
    "checkmate", "stalemate", "fifty_moves", "insufficient_material",
    "threefold", "max_plies", "unknown",
)


def _termination(cb: CBoard, board: chess.Board) -> str:
    """Why the game stopped, in ``CBoard.is_game_over()``'s own terms.

    ⚑ No catch-all. This feeds ``ShardMeta.checkmate_games`` /
    ``stalemate_games`` and the sidecar, and ``distributed_runtime`` aggregates
    those, so a reason folded into the wrong bucket is a wrong published number.
    The four positive branches mirror ``cboard_is_game_over`` exactly (halfmove
    clock, threefold, insufficient material, no legal moves); anything else is
    reported as ``unknown`` rather than guessed at.
    """
    if not cb.is_game_over():
        return "max_plies"
    if cb.is_checkmate():
        return "checkmate"
    if cb.is_stalemate():
        return "stalemate"
    if int(cb.halfmove_clock) >= 100:
        return "fifty_moves"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_repetition(3):
        return "threefold"
    return "unknown"


def boards_agree(cb: CBoard, board: chess.Board) -> bool:
    """Whether the CBoard and the python-chess board describe one position.

    ⚑ The en-passant field CANNOT be compared verbatim, and getting that wrong
    is a false alarm on ordinary games, not a corner case. The two encoders use
    different, individually-correct conventions and they disagree in BOTH
    directions:

      * ``CBoard.fen()`` prints the square when an enemy pawn is ADJACENT to the
        landing square, whether or not the capture is legal;
      * ``chess.Board.fen()`` defaults to ``en_passant="legal"`` and prints it
        only when the capture is actually legal -- so a pinned capturer, or a
        double push that DELIVERS MATE (no legal moves at all ⇒ no legal ep
        capture), prints ``-`` where CBoard prints the square;
      * ``chess.Board.fen(en_passant="fen")`` prints it after ANY double push,
        including with no adjacent pawn, where CBoard prints ``-``.

    Measured: ``8/8/8/8/K3p3/3Q4/3Q1Pk1/3R4 w - - 0 1`` + ``f2f4`` is mate, both
    boards agree on ``result()``, and a verbatim field-4 comparison raises.

    So compare placement / side-to-move / castling exactly, and require the ep
    squares to MATCH only when both name one. That keeps the check tight in the
    direction that matters -- CBoard's adjacency set is a subset of the
    double-push set, so whenever CBoard names a square, ``en_passant="fen"``
    names the same one and a genuine mismatch still fails.
    """
    cb_fields = cb.fen().split()
    pc_fields = board.fen(en_passant="fen").split()
    if cb_fields[:3] != pc_fields[:3]:
        return False
    cb_ep, pc_ep = cb_fields[3], pc_fields[3]
    return cb_ep == "-" or pc_ep == "-" or cb_ep == pc_ep


def native_root_logits(
    cb: CBoard, *, source: NnueArmValueSource,
    cluster: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The root's (uniform policy, arm value) as pre-computed batch logits.

    ⚑ The root is evaluated HERE rather than through ``evaluate_encoded``
    because that method's native branch reads the tree's PENDING LEAF batch,
    and at root time there is none. Handing the search pre-computed root logits
    keeps the evaluator's leaf contract exact -- an empty pending batch stays a
    hard error instead of becoming a case it has to guess about.
    """
    q = source.q_for_boards([cb], role="root", cluster=cluster)
    return (
        np.zeros((1, POLICY_SIZE), dtype=np.float32),
        q_to_wdl_logits(q),
    )


def play_game(
    *,
    cfg: GenConfig,
    gcfg: GumbelConfig,
    evaluator: UniformPriorEvaluator,
    rng: np.random.Generator,
    opening_cfg: OpeningConfig,
    budget: RootBudgetStats | None = None,
    game_index: int = 0,
) -> GameOutcome:
    """Play one complete game with the production C Gumbel search."""
    start = sample_starting_board(rng=rng, cfg=opening_cfg)
    board = start.board
    cb = CBoard.from_board(board)
    start_fen = board.fen()
    records: list[PlyRecord] = []
    actions: list[int] = []
    arm = evaluator.nnue_source

    while not cb.is_game_over() and len(actions) < int(cfg.max_plies):
        x = encode_cboard(
            cb,
            input_history_encoding=cfg.input_history_encoding,
            input_extra_features=cfg.input_extra_features,
        )
        # ⚑ PER-POSITION budget, because the candidate set is capped at sims/2
        # and the branching factor is not a constant. `per_game_simulations`
        # overrides `gcfg.simulations` for this root only; with
        # --all-root-moves off it is just `--sims` and nothing changes.
        legal_moves = board.legal_moves.count()
        sims_here = root_simulation_budget(legal_moves, cfg=cfg)
        if budget is not None:
            budget.add(
                legal_moves=legal_moves, sims=sims_here,
                candidate_cap=GSS_MAX_CANDS if cfg.all_root_moves else 0,
            )
        if cfg.all_root_moves and legal_moves > GSS_MAX_CANDS:
            _warn_candidate_cap(legal_moves)
        # A FRESH tree per ply either way -- passing one only names the object
        # the search would otherwise have made for itself, so the native arm can
        # read that same tree's pending leaves. No root is carried across plies.
        search_tree = MCTSTree() if arm is not None else None
        evaluator.bind_tree(search_tree, cluster=(game_index, len(actions)))
        try:
            pre_pol, pre_wdl = (
                native_root_logits(
                    cb, source=arm, cluster=(game_index, len(actions)),
                )
                if arm is not None else (None, None)
            )
            probs, acts, _values, masks, _tree, _root_ids = run_gumbel_root_many_c(
                None,
                [board],
                device="cpu",
                rng=rng,
                cfg=gcfg,
                evaluator=evaluator,
                cboards=[cb],
                tree=search_tree,
                pre_pol_logits=pre_pol,
                pre_wdl_logits=pre_wdl,
                per_game_simulations=[sims_here],
                # ⚑⚑ OFF UNDER --all-root-moves, and this is a COVERAGE fix, not
                # a performance one. The shortcut prunes every immediately-
                # DRAWING legal move from a root whose value is positive, before
                # candidate selection -- so that move is never evaluated, gets
                # zero target probability, and the stored legal mask advertises
                # it anyway. On a root with a repetition available that is
                # exactly the hole `--all-root-moves` exists to close, and it is
                # silent: the target is dense over the legal set either way.
                # Left ON otherwise, where production's topk-capped candidate set
                # makes the shortcut a real saving and no coverage is promised.
                allow_terminal_root_shortcuts=not bool(cfg.all_root_moves),
                # Explicit, never defaulted: these two are not GumbelConfig
                # fields, so no override surface can reach them and
                # mcts/gumbel.py's standing comment requires every call site to
                # pass them by hand.
                vloss_weight=int(cfg.vloss_weight),
                target_batch=int(cfg.target_batch),
            )
        finally:
            # Unbound in a `finally`: a search that raised would otherwise leave
            # a dead tree bound, and the NEXT ply's batch would be interpreted
            # against it. The generator keeps going on a worker error only to
            # write the partial sidecar, so "the next ply" is reachable.
            evaluator.bind_tree(None)
        action = int(acts[0])
        records.append(
            PlyRecord(
                x=x,
                policy_probs=np.asarray(probs[0]),
                legal_mask=np.asarray(masks[0]),
                pov_white=bool(cb.turn),
                ply_index=int(cb.ply),
            ),
        )
        cb.push_index(action)
        # STRICT, unlike production selfplay's lenient `index_to_move`: an
        # undecodable action there costs one played move, here it would desync
        # the two boards and mislabel every row after the split.
        board.push(index_to_move_strict(action, board))
        actions.append(action)

    # The two boards are pushed independently from one action index, exactly as
    # selfplay/network_turn.py does. If they ever disagreed, every label after
    # the split would be wrong with no shape change and no failing assertion.
    if not boards_agree(cb, board):
        raise RuntimeError(
            f"CBoard/python-chess divergence: {cb.fen()!r} vs "
            f"{board.fen(en_passant='fen')!r}",
        )
    return GameOutcome(
        records=records,
        result=cb.result(),
        plies=len(actions),
        termination=_termination(cb, board),
        start_fen=start_fen,
        opening_source=start.source,
        move_trace=",".join(str(a) for a in actions),
        end_ply_index=int(cb.ply),
    )


def rows_from_game(
    outcome: GameOutcome, *, cfg: GenConfig, shape: PolicyShapeStats | None = None,
) -> list[ReplaySample]:
    """Turn one finished game into replay rows, production conventions only."""
    game_id = _stable_game_id(
        start_fen=outcome.start_fen,
        opening_source=outcome.opening_source,
        move_trace=outcome.move_trace,
        result=outcome.result,
        total_plies_played=outcome.plies,
    )
    source_code = opening_source_code(outcome.opening_source)
    max_plies = max(1.0, float(cfg.max_plies))
    rows: list[ReplaySample] = []
    for rec in outcome.records:
        legal_mask = policy_mask_to_encoding(rec.legal_mask).astype(
            np.uint8, copy=False,
        )
        policy_target = policy_vector_to_encoding(rec.policy_probs)
        if shape is not None:
            shape.add(policy_tv_to_uniform(policy_target, legal_mask))
        rows.append(
            ReplaySample(
                x=np.asarray(rec.x, dtype=np.float32),
                policy_target=policy_target,
                wdl_target=int(
                    _result_to_wdl(outcome.result, pov_white=rec.pov_white),
                ),
                legal_mask=legal_mask,
                # Production's rule, verbatim (network_turn.py: `has_policy =
                # is_full and int(c_mask[j].sum()) > 1`): a forced move's target
                # is one-hot whatever the search did, so training CE on it is
                # gradient on nothing. Every ply here is a full ply, so only the
                # legal-count half applies.
                has_policy=int(legal_mask.sum()) > 1,
                is_selfplay=True,
                is_network_turn=True,
                moves_left=(
                    float(max(0, outcome.end_ply_index - rec.ply_index)) / max_plies
                ),
                game_id=game_id,
                ply_index=int(rec.ply_index),
                opening_source_code=source_code,
                input_history_encoding=cfg.input_history_encoding,
                history_rep_fix=bool(cfg.history_rep_fix),
            ),
        )
    return rows


def shard_digest(path: str | Path) -> str:
    """Content hash of a written shard, read back through the real loader."""
    arrs, _meta = load_shard_arrays(path)
    digest = hashlib.sha256()
    for name in sorted(arrs):
        arr = np.asarray(arrs[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def next_shard_index(out_dir: Path) -> int:
    """One past the highest shard already in ``out_dir`` (0 when empty)."""
    indices = [shard_index(p) for p in iter_shard_paths(out_dir)]
    live = [i for i in indices if i >= 0]
    return max(live) + 1 if live else 0


@dataclass
class ShardTally:
    """The per-shard game statistics ``ShardMeta`` publishes.

    Accumulated over WHOLE games -- shards close on a game boundary -- so every
    counter below is exact rather than apportioned across a split game.
    ``distributed_runtime`` aggregates these fields across a window, so leaving
    them at their ``None`` default publishes "0 checkmates" about a corpus that
    is ~75 % checkmates: an absent count and a zero count are indistinguishable
    downstream, which is why they are filled rather than skipped.
    """

    games: int = 0
    white_wins: int = 0
    draws: int = 0
    black_wins: int = 0
    plies_total: int = 0
    plies_white_win: int = 0
    plies_draw: int = 0
    plies_black_win: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0

    def add(self, outcome: GameOutcome) -> None:
        self.games += 1
        self.plies_total += int(outcome.plies)
        if outcome.result == "1-0":
            self.white_wins += 1
            self.plies_white_win += int(outcome.plies)
        elif outcome.result == "0-1":
            self.black_wins += 1
            self.plies_black_win += int(outcome.plies)
        else:
            self.draws += 1
            self.plies_draw += int(outcome.plies)
        self.checkmate_games += int(outcome.termination == "checkmate")
        self.stalemate_games += int(outcome.termination == "stalemate")


def write_shard(
    *,
    out_dir: Path,
    index: int,
    rows: list[ReplaySample],
    cfg: GenConfig,
    tally: ShardTally,
) -> dict[str, Any]:
    path = local_shard_path(out_dir, index)
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(rows),
        meta=ShardMeta(
            run_id=str(cfg.run_id),
            generated_at_unix=int(time.time()),
            input_history_encoding=str(cfg.input_history_encoding),
            history_rep_fix=bool(cfg.history_rep_fix),
            policy_encoding=POLICY_ENCODING_LC0_1858,
            policy_size=int(COMPACT_POLICY_SIZE),
            positions=len(rows),
            games=int(tally.games),
            # ⚑ WHITE-relative, NOT production's network-relative. Production
            # scores wins/losses from the net's seat (`net_col`); in net-vs-net
            # generation-zero selfplay both seats are the same (absent) net, so
            # there is no such seat and white/black is the only reading that
            # means anything. Do not pool these with curriculum shards' counters.
            wins=int(tally.white_wins),
            draws=int(tally.draws),
            losses=int(tally.black_wins),
            total_game_plies=int(tally.plies_total),
            plies_win=int(tally.plies_white_win),
            plies_draw=int(tally.plies_draw),
            plies_loss=int(tally.plies_black_win),
            checkmate_games=int(tally.checkmate_games),
            stalemate_games=int(tally.stalemate_games),
            total_draw_games=int(tally.draws),
            selfplay_games=int(tally.games),
            selfplay_draw_games=int(tally.draws),
            selfplay_adjudicated_games=0,
            adjudicated_games=0,
        ),
    )
    return {
        "index": int(index),
        "path": path.name,
        "rows": len(rows),
        "games": int(tally.games),
        "digest": shard_digest(path),
    }


def _git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes; None when git cannot say.

    ⚑ A sha without this is a claim, not provenance: the corpus a dirty tree
    produced was not produced by the code that sha names, and nothing
    downstream could tell. None (git absent, or not a checkout) is reported as
    UNKNOWN rather than as clean.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def _apply_nice(delta: int) -> None:
    if int(delta) == 0:
        return
    try:
        os.nice(int(delta))
    except OSError as exc:  # pragma: no cover - only when privileges are missing
        _LOG.warning("could not renice by %d: %s", int(delta), exc)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def run_worker(spec: WorkerSpec) -> WorkerResult:
    """Play this worker's games and write its shards. Runs in a child process."""
    setup_logging()
    cfg = spec.cfg
    _apply_nice(cfg.nice)
    torch.set_num_threads(1)
    # Before any CBoard exists: per-slot repetition flags are recorded at push
    # time and never recomputed (encoding/rep_fix.py).
    rep_fix.apply(bool(cfg.history_rep_fix))

    gcfg = build_gumbel_config(cfg)
    evaluator = UniformPriorEvaluator(
        value_source=cfg.value_source,
        expected_planes=input_plane_count(cfg.input_extra_features),
        random_salt=int(cfg.seed),
        nnue_source=build_nnue_source(
            cfg, leaf_bank=leaf_bank_path(cfg.out_dir, spec.worker_id),
        ),
        input_history_encoding=str(cfg.input_history_encoding),
        input_extra_features=str(cfg.input_extra_features),
    )
    opening_cfg = build_opening_config(cfg)
    rng = np.random.default_rng(spec.seed)

    realized = realized_config(
        gcfg=gcfg, evaluator=evaluator, opening_cfg=opening_cfg,
        cfg=cfg, worker_id=spec.worker_id,
    )
    _LOG.info("realized %s", format_realized(realized))

    result = WorkerResult(worker_id=spec.worker_id, realized=realized)
    # Shards close on a GAME boundary at or past `--shard-size` rows, so a shard
    # holds whole games and every ShardMeta game counter is exact rather than
    # apportioned across a split game.
    pending: list[ReplaySample] = []
    tally = ShardTally()
    shard_idx = int(spec.shard_index_start)
    stride = max(1, int(cfg.workers))
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending, tally, shard_idx
        if not pending:
            return
        while local_shard_path(cfg.out_dir, shard_idx).exists():
            shard_idx += stride
        result.shards.append(
            write_shard(
                out_dir=cfg.out_dir, index=shard_idx, rows=pending, cfg=cfg,
                tally=tally,
            ),
        )
        shard_idx += stride
        pending = []
        tally = ShardTally()

    # ⚑ The `finally` copies the arm's counters out of the CONTEXT that ran
    # them. A worker that raises mid-game has done real evaluation work, and
    # `generate` deliberately keeps going to build the partial sidecar -- so
    # without this the partial run would describe its arm as having evaluated
    # nothing.
    try:
        for game_index in range(int(spec.games)):
            outcome = play_game(
                cfg=cfg, gcfg=gcfg, evaluator=evaluator, rng=rng,
                opening_cfg=opening_cfg, budget=result.root_budget,
                # ⚑ Worker-local, and paired with `worker_id` wherever it is
                # written: two workers both start at game 0, so the cluster key
                # is (worker_id, game_index, ply) and the leaf bank carries the
                # worker in its FILENAME rather than in every row.
                game_index=game_index,
            )
            rows = rows_from_game(outcome, cfg=cfg, shape=result.policy_shape)
            result.games += 1
            result.rows += len(rows)
            result.plies.append(outcome.plies)
            result.terminations[outcome.termination] = (
                result.terminations.get(outcome.termination, 0) + 1
            )
            # ⚑ Rows as well as games. Truncated games are the LONGEST games, so
            # a per-game share understates what fraction of the CORPUS carries
            # the "*"-to-draw label -- measured 4 % of games and 10.6 % of rows.
            result.termination_rows[outcome.termination] = (
                result.termination_rows.get(outcome.termination, 0) + len(rows)
            )
            result.wins += int(outcome.result == "1-0")
            result.losses += int(outcome.result == "0-1")
            result.draws += int(outcome.result not in ("1-0", "0-1"))
            pending.extend(rows)
            tally.add(outcome)
            if len(pending) >= int(cfg.shard_size):
                flush()
        flush()
    except BaseException as exc:
        # ⚑ THE PARTIAL RESULT TRAVELS WITH THE EXCEPTION. Re-raising the bare
        # error discards `result` -- and with it the shards already flushed,
        # which then reappear in the sidecar as ORPHANS attributed to no worker
        # while the run's `nnue` block reads as if the arm never ran. The
        # failure is still fatal; it is now fatal WITH accounting.
        _finalize_worker(result, evaluator=evaluator, started=started)
        result.failed = f"{type(exc).__name__}: {exc}"
        raise WorkerFailure(result, exc) from exc
    finally:
        evaluator.close()

    _finalize_worker(result, evaluator=evaluator, started=started)
    return result


def _finalize_worker(
    result: WorkerResult, *, evaluator: UniformPriorEvaluator, started: float,
) -> None:
    """Copy the counters off the objects that ran them onto the result.

    Called on BOTH exits, which is why it is a function: the success path and
    the failure path reporting different fields is how a partial run acquires a
    plausible-looking but incomplete summary.
    """
    arm = evaluator.nnue_source
    if arm is not None:
        arm.refresh_context_stats()
        result.nnue = arm.stats
        result.bank_rows = int(arm.bank_rows)
    result.eval_calls = evaluator.eval_calls
    result.eval_rows = evaluator.eval_rows
    result.seconds = time.perf_counter() - started


def plies_summary(plies: list[int]) -> dict[str, float]:
    if not plies:
        return {}
    arr = np.asarray(plies, dtype=np.float64)
    quantiles = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": float(arr.size),
        "min": float(arr.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def plies_histogram(plies: list[int], *, max_plies: int) -> dict[str, int]:
    """Fixed 50-ply buckets up to ``max_plies``, so two runs are comparable.

    The top bucket is labelled ``"<max_plies>+"``, not ``"450-499"``: no game can
    exceed the cap, so a range label there would advertise a span half of which
    is unreachable and invite a reader to treat the pile-up as a tail rather than
    as the truncation it is.
    """
    hist: dict[str, int] = {}
    width = 50
    cap = int(max_plies)
    for value in plies:
        lo = min(int(value) // width * width, cap)
        label = f"{cap}+" if lo >= cap else f"{lo}-{lo + width - 1}"
        hist[label] = hist.get(label, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0].rstrip("+").split("-")[0])))


def build_worker_specs(cfg: GenConfig, *, shard_index_start: int) -> list[WorkerSpec]:
    workers = max(1, int(cfg.workers))
    base, extra = divmod(max(0, int(cfg.games)), workers)
    specs: list[WorkerSpec] = []
    for wid in range(workers):
        games = base + (1 if wid < extra else 0)
        seed = int(
            np.random.SeedSequence([int(cfg.seed), wid]).generate_state(
                1, dtype=np.uint64,
            )[0],
        )
        specs.append(
            WorkerSpec(
                cfg=cfg,
                worker_id=wid,
                games=games,
                seed=seed,
                shard_index_start=shard_index_start + wid,
            ),
        )
    return specs


def orphan_shards(
    *, out_dir: Path, shard_index_start: int, claimed: set[int],
) -> list[dict[str, Any]]:
    """Shards this run wrote that no worker result accounts for.

    A worker that dies mid-run has already written shards, and its result never
    comes back -- so without this the sidecar would describe a corpus smaller
    than what is on disk, and the NEXT invocation's ``next_shard_index`` would
    quietly fold the unaccounted files into the pool as if they had been
    inspected. Named here instead.
    """
    found: list[dict[str, Any]] = []
    for path in iter_shard_paths(out_dir):
        index = shard_index(path)
        if index < int(shard_index_start) or index in claimed:
            continue
        found.append({"index": int(index), "path": path.name})
    return sorted(found, key=lambda s: int(s["index"]))


@dataclass(frozen=True)
class RunProvenance:
    """The code identity, captured BEFORE the workers start.

    ⚑ CAPTURED AT LAUNCH, NOT AT SUMMARY TIME, and the difference is hours. The
    sidecar used to call ``git_sha()`` while building the summary -- after a
    generation run that can span a whole night -- so a branch switch, a rebase or
    a stash anywhere in that window relabelled the corpus with code that never
    produced a single row of it. Worse for ``dirty``: an edit made DURING the run
    reads as if the run had been made from a dirty tree, and a run genuinely
    started dirty then committed reads as clean. Both endpoints are recorded, so
    a tree that moved underneath the run is visible rather than silently
    resolved to whichever end happened to be read.
    """

    code_sha: str | None
    code_dirty: bool | None

    @classmethod
    def capture(cls) -> RunProvenance:
        return cls(code_sha=git_sha(), code_dirty=_git_dirty())


def leaf_bank_path(out_dir: Path, worker_id: int) -> Path:
    """Where worker ``worker_id`` banks its raw per-leaf observations.

    Beside the shards, one file per worker -- the workers are separate processes
    appending line by line, and a shared file would interleave partial writes.
    The worker id lives in the FILENAME so the JSONL rows stay small.
    """
    return Path(out_dir) / f"leaf_observations_w{int(worker_id):03d}.jsonl"


def summarize(
    *, cfg: GenConfig, results: list[WorkerResult], wall_seconds: float,
    shard_index_start: int, partial: bool = False, error: str | None = None,
    provenance: RunProvenance | None = None,
) -> dict[str, Any]:
    plies: list[int] = []
    terminations: dict[str, int] = {}
    termination_rows: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    shape = PolicyShapeStats()
    root_budget = RootBudgetStats()
    nnue = NnueArmStats()
    games = rows = wins = draws = losses = eval_calls = eval_rows = bank_rows = 0
    for res in results:
        root_budget.merge(res.root_budget)
        nnue.merge(res.nnue)
        plies.extend(res.plies)
        for key, count in res.terminations.items():
            terminations[key] = terminations.get(key, 0) + count
        for key, count in res.termination_rows.items():
            termination_rows[key] = termination_rows.get(key, 0) + count
        shards.extend(res.shards)
        shape.merge(res.policy_shape)
        games += res.games
        rows += res.rows
        wins += res.wins
        draws += res.draws
        losses += res.losses
        eval_calls += res.eval_calls
        eval_rows += res.eval_rows
        bank_rows += res.bank_rows
    hours = max(wall_seconds, 1e-9) / 3600.0
    # Captured now only when the caller did not capture it at launch, which is
    # the shape every non-`generate` caller (tests, ad-hoc summaries) has.
    prov = RunProvenance.capture() if provenance is None else provenance
    orphans = orphan_shards(
        out_dir=cfg.out_dir, shard_index_start=shard_index_start,
        claimed={int(s["index"]) for s in shards},
    )
    return {
        "run_id": str(cfg.run_id),
        # ⚑ True when at least one worker did not report. The shards it already
        # wrote are on disk and are listed under `orphan_shards`; treat the whole
        # batch as unverified rather than as a smaller successful run.
        "partial": bool(partial or orphans),
        "error": error,
        # Every Path field needs a str here or `json.dumps` refuses the whole
        # sidecar -- which, because the sidecar is written in `generate`'s
        # `finally`, would turn a completed run into a crashed one.
        "config": {**asdict(cfg), "out_dir": str(cfg.out_dir),
                   "openings": None if cfg.openings is None else str(cfg.openings),
                   "nnue_pack": (
                       None if cfg.nnue_pack is None else str(cfg.nnue_pack)
                   )},
        # ⚑ RUN-LEVEL PROVENANCE, and it is run-level ON PURPOSE. The readout
        # asked for the arm, the budget, the code sha and the seed to travel
        # with every row; a row here is a `ReplaySample`, and adding columns to
        # that would break the one property requirement 2 is about -- these
        # shards must be schema-identical to the banked UCI anchor sets so ONE
        # scoring pass can read both. Every row already carries its own cluster
        # key (`game_id` + `ply_index`), and this block plus `shards` (index →
        # digest) plus `realized_per_worker` (worker → seed → settings) is what
        # resolves a row to the run that made it.
        "provenance": {
            # ⚑ `code_sha` / `code_dirty` are the LAUNCH state -- see
            # RunProvenance. `code_sha_at_finish` / `code_dirty_at_finish` are
            # read here, so the two disagreeing says the tree moved mid-run and
            # neither end alone describes the corpus.
            "code_sha": prov.code_sha,
            "code_dirty": prov.code_dirty,
            "code_sha_at_finish": git_sha(),
            "code_dirty_at_finish": _git_dirty(),
            "seed": int(cfg.seed),
            "worker_seeds": [
                {"worker_id": int(s.worker_id), "seed": int(s.seed)}
                for s in build_worker_specs(cfg, shard_index_start=shard_index_start)
            ],
            "value_source": str(cfg.value_source),
            "nnue_pack": None if cfg.nnue_pack is None else str(cfg.nnue_pack),
        },
        "realized_per_worker": [res.realized for res in results],
        "workers_expected": max(1, int(cfg.workers)),
        "workers_reported": len(results),
        "games": games,
        "rows": rows,
        "results": {"white_wins": wins, "draws": draws, "black_wins": losses},
        "plies": plies_summary(plies),
        "plies_histogram": plies_histogram(plies, max_plies=int(cfg.max_plies)),
        "terminations": dict(sorted(terminations.items())),
        # Same events, weighted by ROWS. Truncated games are the longest games,
        # so this share is several times the per-game one and it is the one that
        # describes the corpus.
        "terminations_by_rows": dict(sorted(termination_rows.items())),
        "policy_target_shape": shape.summary(),
        # ⚑ What the root budget REALLY was. Under --all-root-moves `--sims` is
        # only a floor, so `sims_mean`/`sims_max` are the realized cost and
        # `legal_moves_max` says how wide it had to stretch.
        "root_budget": root_budget.summary(),
        # Empty on the three sources with no arm, which is the reading. Carries
        # the two numbers the prereg asks each cell for: the in-check leaf
        # fraction and the resolver's expansion factor.
        "nnue": nnue.summary(),
        # ⚑ The raw per-leaf observations, or the fact that there are none.
        # `enabled: false` is the DEFAULT, and a readout cell scored off a run
        # with it false cannot be re-derived under a corrected cp scale without
        # replaying the games -- which is why the prereg requires it on.
        "leaf_bank": {
            "enabled": bool(cfg.bank_leaf_observations),
            "rows": int(bank_rows),
            "files": (
                [
                    leaf_bank_path(cfg.out_dir, res.worker_id).name
                    for res in results if res.bank_rows
                ]
                if cfg.bank_leaf_observations else []
            ),
        },
        # Workers that died partway and still reported. Their counters ARE in
        # the totals above (partial, not absent); this names them so a reader is
        # not left inferring the shortfall from workers_reported.
        "failed_workers": [
            {"worker_id": int(res.worker_id), "error": str(res.failed)}
            for res in results if res.failed is not None
        ],
        "shards_written": len(shards),
        "shard_index_start": int(shard_index_start),
        "shards": sorted(shards, key=lambda s: int(s["index"])),
        "orphan_shards": orphans,
        "evaluator": {"calls": eval_calls, "rows": eval_rows},
        "wall_seconds": float(wall_seconds),
        "games_per_hour": games / hours,
        "rows_per_hour": rows / hours,
        # No Stockfish label and no search value estimate on any row.
        "sf_fields": "absent",
        "search_wdl": "absent",
        "required_run_config": {
            "values": {
                "sf_wdl_frac": 0.0, "sf_wdl_frac_floor": 0.0, "search_wdl_frac": 0.0,
            },
            "enforced": False,
            "why": (
                "With has_sf_wdl and has_search_wdl identically 0, losses.py's "
                "blend_fallback_target makes the value target exactly game_oh for "
                "ANY fracs, so these values do not change what is trained. They "
                "make the config STATE it, and they keep the SF-conditioned "
                "telemetry honest: every such column computed over these rows has "
                "a zero denominator, and the orphaned-label rate then reads "
                "0.000000, which is the value that means healthy."
            ),
            "enforcement_deferred_to": "scratchpad/az_purity/prereg_draft.md",
        },
    }


def summary_path_for(cfg: GenConfig, *, shard_index_start: int) -> Path:
    return cfg.out_dir / f"gen0_summary_{int(shard_index_start):06d}.json"


def _write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write the sidecar the way the shards are written: all of it, or none.

    ⚑ A plain `write_text` truncates first. The shards go through
    `save_local_shard_arrays`'s atomic path already, so the sidecar was the one
    artifact of this tool that could be left half-written -- and the realistic
    trigger is not a crash but `--summary-json` pointed at an EXISTING summary,
    where the truncation destroys a good record before the new one is serialized.
    A reader then finds a JSON file that fails to parse and no way back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(summary, indent=2, sort_keys=True) + "\n")


def generate(cfg: GenConfig, *, summary_json: Path | None = None) -> dict[str, Any]:
    """Run the whole generation, write the sidecar, and return the summary.

    ⚑ The sidecar is written in a ``finally``, not after a clean return. Workers
    write shards as they go, so a crash at game 25 of 40 leaves real files on
    disk; the previous shape re-raised straight through ``main`` and left those
    files with NO record of what produced them, at which point the next
    invocation's ``next_shard_index`` folded them into the corpus silently. Now a
    failed run still gets a sidecar, flagged ``partial``, naming the shards no
    worker accounted for.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shard_index_start = next_shard_index(cfg.out_dir)
    sidecar = (
        summary_json if summary_json is not None
        else summary_path_for(cfg, shard_index_start=shard_index_start)
    )
    specs = build_worker_specs(cfg, shard_index_start=shard_index_start)
    # ⚑ BEFORE the workers start. See RunProvenance: read at summary time this
    # is the tree as it stands after the run, not the tree that made the rows.
    provenance = RunProvenance.capture()
    started = time.perf_counter()
    results: list[WorkerResult] = []
    failure: str | None = None
    try:
        if len(specs) == 1:
            try:
                results.append(run_worker(specs[0]))
            except WorkerFailure as exc:
                results.append(exc.result)
                raise exc.cause from None
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(specs), mp_context=ctx) as pool:
                # submit + as_completed, not map: map re-raises on the first
                # failed future and discards the workers that DID finish, so the
                # partial sidecar would omit shards that are sitting on disk.
                futures = [pool.submit(run_worker, spec) for spec in specs]
                for future in as_completed(futures):
                    # Bound before the append: `future.result()` re-raises a
                    # worker's exception, and everything already appended is what
                    # the partial sidecar is built from. A comprehension here
                    # would discard them.
                    try:
                        completed = future.result()
                    except WorkerFailure as exc:
                        # ⚑ The dead worker's OWN partial counters, carried
                        # across the process boundary by the exception. Without
                        # this branch its shards are on disk and unattributed
                        # while its arm reports zero evaluations -- see
                        # WorkerFailure.
                        results.append(exc.result)
                        raise exc.cause from None
                    results.append(completed)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = summarize(
            cfg=cfg, results=results, wall_seconds=time.perf_counter() - started,
            shard_index_start=shard_index_start,
            partial=failure is not None or len(results) != len(specs),
            error=failure,
            provenance=provenance,
        )
        summary["summary_json"] = str(sidecar)
        _write_summary(summary, sidecar)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help=(
            "Where shards land. Re-running APPENDS: numbering starts above the "
            "highest index present and an occupied index is skipped. ⚑ Use a "
            "FRESH --seed for every re-run into the same directory: the games "
            "are a function of the seed, so reusing it replays the same games "
            "into new shard numbers and duplicates them in the corpus. ⚑ ONE "
            "invocation at a time per directory -- two concurrent runs both read "
            "the same starting index before either writes, so they would race "
            "for the same shard numbers. Give concurrent runs separate "
            "directories and merge afterwards."
        ),
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sims", type=int, default=DEFAULT_SIMS,
        help=(
            "Simulation budget per root. ⚑ Under --all-root-moves this is a "
            "FLOOR, not the value: the realized per-position budget is "
            f"max(--sims, {ROOT_SIMS_PER_LEGAL_MOVE} * n_legal) and the sidecar's "
            "`root_budget` block reports what it actually cost."
        ),
    )
    parser.add_argument(
        "--topk", type=int, default=None,
        help=(
            "Root candidate cap. Default: production's "
            f"{DEFAULT_TOPK}, or {MAX_LEGAL_MOVES} under --all-root-moves "
            "(where a lower value is REFUSED, not quietly raised)."
        ),
    )
    parser.add_argument("--c-scale", type=float, default=SELFPLAY_GUMBEL_C_SCALE)
    parser.add_argument("--policy-temp", type=float, default=DEFAULT_POLICY_TEMP)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gumbel-scale", type=float, default=DEFAULT_GUMBEL_SCALE)
    parser.add_argument(
        "--target-max-visit-cap", type=int, default=DEFAULT_TARGET_MAX_VISIT_CAP,
        help="selfplay.gumbel_target_max_visit_cap; inert at gen-0, pinned to live.",
    )
    parser.add_argument(
        "--no-target-untempered-prior", dest="target_untempered_prior",
        action="store_false",
        help="Leave gumbel_target_untempered_prior off (live production: on).",
    )
    parser.set_defaults(target_untempered_prior=DEFAULT_TARGET_UNTEMPERED_PRIOR)
    parser.add_argument(
        "--vloss-weight", type=int, default=DEFAULT_VLOSS_WEIGHT,
        help=(
            "C-search virtual loss; passed explicitly to run_gumbel_root_many_c "
            "(not a GumbelConfig field). Live production runs 1."
        ),
    )
    parser.add_argument(
        "--target-batch", type=int, default=DEFAULT_TARGET_BATCH,
        help="C-search leaf batch flush; passed explicitly. 0 = production.",
    )
    parser.add_argument(
        "--value-source", choices=VALUE_SOURCES, default=VALUE_SOURCE_ZERO,
        help=(
            "zero (default, pure) | material (opt-in, standard piece values) | "
            f"{VALUE_SOURCE_NNUE_STATIC} / {VALUE_SOURCE_NNUE_QSEARCH} (the two "
            "native NNUE race arms: SF-quality VALUES with no SF process, both "
            "with mandatory recursive check resolution, the second adding a "
            "tactical quiescence -- needs --nnue-pack) | "
            "random (⚑ STRUCTURED NOISE -- plumbing tests only, never a training "
            "corpus: it emits confident targets encoding a hash of the position)."
        ),
    )
    parser.add_argument(
        "--nnue-pack", type=Path, default=None,
        help=(
            "mmap-able weight pack for the native arms (scripts/nnue_pack.py). "
            "Required by them and refused by every other source. Runtime "
            "artifact: nothing in this repo hardcodes a weights path."
        ),
    )
    parser.add_argument(
        "--nnue-resolver-max-depth", type=int, default=None,
        help=(
            "Recursion cap for check resolution. Default: the extension's own "
            "CAE_RESOLVER_MAX_DEPTH. A line neither resolved nor terminal at the "
            "cap scores 0 -- the sidecar reports `ctx_depth_cutoffs` so a run "
            "that hit it says so."
        ),
    )
    parser.add_argument(
        "--nnue-qsearch-max-ply", type=int, default=None,
        help=(
            f"Quiescence ply budget ({VALUE_SOURCE_NNUE_QSEARCH} only; 0 makes "
            "that arm agree with the static one node for node, which is the "
            "arm's own negative control). Default: the extension's own value."
        ),
    )
    parser.add_argument(
        "--nnue-qsearch-check-plies", type=int, default=None,
        help=(
            "How many quiescence plies may try CHECKS as well as captures. "
            "Default: the extension's own value."
        ),
    )
    # ⚑ The three cp knobs default to None ON THE PARSER and to their constants
    # in `config_from_args`. That is not indirection for its own sake: argparse
    # cannot tell "the operator typed the default" from "the operator typed
    # nothing", and `unconsumable_flags` has to REFUSE a cp knob supplied to a
    # source that has no arm to apply it to. A sentinel default is the only way
    # to know which happened.
    parser.add_argument(
        "--nnue-cp-per-unit", type=float, default=None,
        help=(
            "Centipawns per internal NNUE unit, before production's cp-logistic. "
            "⚑ A FREE PARAMETER THE PREREG DOES NOT PIN: it sets how sharp every "
            "stored policy target is, so it must be fixed (or swept as its own "
            f"axis) before cells are compared. Default {NNUE_CP_PER_INTERNAL_UNIT} "
            "was measured against Stockfish's own search cp -- see the constant."
        ),
    )
    parser.add_argument(
        "--nnue-cp-slope", type=float, default=None,
        help=(
            "cp-logistic slope; default is live selfplay.sf_wdl_cp_slope "
            f"({NNUE_CP_SLOPE})."
        ),
    )
    parser.add_argument(
        "--nnue-cp-draw-width", type=float, default=None,
        help=(
            "cp-logistic draw half-width; live selfplay.sf_wdl_cp_draw_width "
            f"({NNUE_CP_DRAW_WIDTH})."
        ),
    )
    parser.add_argument(
        "--bank-leaf-observations", action="store_true",
        help=(
            "Write every arm evaluation as a JSONL row beside the shards: the RAW "
            "internal score, the position, and the settings it was taken under "
            "(leaf_observations_w<id>.jsonl). ⚑ OFF by default because it is one "
            "row per evaluated leaf -- millions per hour -- but a readout cell "
            "generated WITHOUT it can never be re-derived under a corrected "
            "--nnue-cp-per-unit or a different estimator: the arithmetic is baked "
            "into policy_target and the only way back is to replay the games. The "
            "prereg requires it on for every scored cell."
        ),
    )
    parser.add_argument(
        "--all-root-moves", dest="all_root_moves", action="store_true",
        default=None,
        help=(
            "Make EVERY legal root move a search candidate: topk defaults to the "
            f"{MAX_LEGAL_MOVES}-move ceiling and the per-root budget becomes "
            f"max(--sims, {ROOT_SIMS_PER_LEGAL_MOVE}*n_legal). ON by default for "
            "the native arms and off otherwise. ⚑ Without it, a uniform prior "
            "leaves sequential halving to pick candidates by the Gumbel draw "
            "alone, so on wide positions the best move is simply absent from the "
            "target -- and the banked UCI anchors were generated WITH it, so a "
            "native cell without it differs from the anchor in search shape."
        ),
    )
    parser.add_argument(
        "--no-all-root-moves", dest="all_root_moves", action="store_false",
        help="Keep production's topk-capped candidate set even under an arm.",
    )
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nice", type=int, default=DEFAULT_NICE,
        help=(
            "niceness INCREMENT applied by each worker (os.nice semantics, like "
            "the nice(1) shell builtin), NOT an absolute level: launched from an "
            "already-niced shell it ADDS to what it inherits, so `nice -n 10 ... "
            f"--nice {DEFAULT_NICE}` realizes 19 (the ceiling), not {DEFAULT_NICE}. "
            "The `nice=` field of the startup realized line is the absolute level "
            "actually in force, read back from os.getpriority — trust that over "
            "this flag."
        ),
    )
    parser.add_argument(
        "--openings", type=Path, default=None,
        help="PGN/PGN.zip/Polyglot opening book; default is the start position.",
    )
    parser.add_argument("--opening-plies", type=int, default=DEFAULT_OPENING_PLIES)
    parser.add_argument(
        "--opening-max-games", type=int, default=DEFAULT_OPENING_MAX_GAMES,
    )
    parser.add_argument(
        "--random-start-plies", type=int, default=0,
        help="Random legal plies from the start position when no book is given.",
    )
    parser.add_argument(
        "--input-history-encoding", type=str, default=LC0_HISTORY_ROOT_LEGACY_META,
    )
    parser.add_argument(
        "--input-extra-features", type=str, default=EXTRA_FEATURES_V2_THREATS,
    )
    parser.add_argument(
        "--no-history-rep-fix", dest="history_rep_fix", action="store_false",
        help="Encode without the lc0-root repetition-plane fix (production: on).",
    )
    parser.set_defaults(history_rep_fix=True)
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--summary-json", type=Path, default=None,
        help="Sidecar path; default <out-dir>/gen0_summary_<start index>.json",
    )
    return parser


def resolve_arm_knob_defaults(
    args: argparse.Namespace,
) -> tuple[int | None, int | None, int | None]:
    """The three arm knobs as concrete numbers -- each one only if it is READ.

    ⚑ Read off ``_nnue_ext``'s exported constants rather than restated here.
    A Python copy of a C default is a second home for the same number, and the
    two only have to disagree once for a sidecar to describe a configuration
    the context never ran. The import is local because the three pure value
    sources must not need the NNUE extension to exist.

    ⚑⚑ GATED ON THE ARM, NOT ON "IS THIS AN ARM AT ALL". An earlier revision
    returned all three for either arm, so a ``nnue-static`` run resolved
    ``qsearch_max_ply=4``, stored it, published it in the realized line and in
    every sidecar -- while ``cae_arm_static_eval`` reads ``resolver_max_depth``
    and nothing else. That is this repo's signature defect wearing a receipt:
    the number was accepted, snapshotted, announced, and never consumed. The
    quiescence pair is ``None`` for the static arm, and supplying it there is
    refused by ``unconsumable_flags`` rather than absorbed.
    """
    source = str(args.value_source)
    if source not in NNUE_ARM_SOURCES:
        return (None, None, None)
    from chess_anti_engine.nnue import _nnue_ext

    resolver_depth = (
        int(_nnue_ext.RESOLVER_MAX_DEPTH)
        if args.nnue_resolver_max_depth is None
        else int(args.nnue_resolver_max_depth)
    )
    if source != VALUE_SOURCE_NNUE_QSEARCH:
        return (resolver_depth, None, None)
    return (
        resolver_depth,
        int(_nnue_ext.QSEARCH_MAX_PLY)
        if args.nnue_qsearch_max_ply is None
        else int(args.nnue_qsearch_max_ply),
        int(_nnue_ext.QSEARCH_CHECK_PLIES)
        if args.nnue_qsearch_check_plies is None
        else int(args.nnue_qsearch_check_plies),
    )


# Which value sources actually READ each optional flag. ⚑ The right-hand side is
# the consumer set, established by reading the consumer: `cae_arm_static_eval`
# takes `resolver_max_depth` only, `cae_arm_qsearch_eval` takes all three, and
# the cp trio is `NnueArmValueSource.q_from_values`, which exists only under an
# arm. A flag outside its set is REFUSED, never nulled -- see `unconsumable_flags`.
_FLAG_CONSUMERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("--nnue-resolver-max-depth", "nnue_resolver_max_depth", NNUE_ARM_SOURCES),
    ("--nnue-qsearch-max-ply", "nnue_qsearch_max_ply", (VALUE_SOURCE_NNUE_QSEARCH,)),
    (
        "--nnue-qsearch-check-plies", "nnue_qsearch_check_plies",
        (VALUE_SOURCE_NNUE_QSEARCH,),
    ),
    ("--nnue-cp-per-unit", "nnue_cp_per_unit", NNUE_ARM_SOURCES),
    ("--nnue-cp-slope", "nnue_cp_slope", NNUE_ARM_SOURCES),
    ("--nnue-cp-draw-width", "nnue_cp_draw_width", NNUE_ARM_SOURCES),
    ("--nnue-pack", "nnue_pack", NNUE_ARM_SOURCES),
)


def unconsumable_flags(args: argparse.Namespace) -> list[str]:
    """Flags the operator SUPPLIED that the chosen value source cannot read.

    ⚑ THE SILENT-NULLING PATH IS THE DEFECT THIS CLOSES. Every one of these
    flags used to be replaced with ``None`` when the selected source had no use
    for it: ``--nnue-qsearch-max-ply 0`` on ``--value-source nnue-static``
    parsed, was discarded, and the run proceeded -- with a command line that
    reads like the negative control and a corpus that is the static arm at its
    default. Only ``--nnue-pack`` refused, and only because a missing pack
    would have crashed anyway.

    Each entry is a full sentence naming the flag, the source, and why the two
    do not meet; ``main`` refuses on any non-empty result.
    """
    source = str(args.value_source)
    problems: list[str] = []
    for flag, attr, consumers in _FLAG_CONSUMERS:
        if source in consumers or getattr(args, attr, None) is None:
            continue
        problems.append(
            f"{flag} is read only by {'/'.join(consumers)}; --value-source is "
            f"{source!r}, which would ignore it",
        )
    if bool(getattr(args, "bank_leaf_observations", False)) and (
        source not in NNUE_ARM_SOURCES
    ):
        problems.append(
            "--bank-leaf-observations banks the native arm's raw per-leaf "
            f"scores; --value-source is {source!r}, which has no arm and would "
            "write nothing",
        )
    return problems


def config_from_args(args: argparse.Namespace) -> GenConfig:
    # ⚑ Resolved TOGETHER, because they are one decision. `--all-root-moves`
    # defaults ON for the native arms (where an unranked candidate subset would
    # make the target sharp about a random half of the moves) and OFF elsewhere,
    # and topk follows it unless the operator named a value -- in which case a
    # contradiction is refused in `build_gumbel_config` rather than resolved
    # behind their back.
    all_root_moves = (
        str(args.value_source) in NNUE_ARM_SOURCES
        if args.all_root_moves is None
        else bool(args.all_root_moves)
    )
    topk = (
        int(args.topk) if args.topk is not None
        else (MAX_LEGAL_MOVES if all_root_moves else DEFAULT_TOPK)
    )
    resolver_depth, qsearch_ply, qsearch_checks = resolve_arm_knob_defaults(args)
    return GenConfig(
        out_dir=Path(args.out_dir),
        games=int(args.games),
        workers=int(args.workers),
        sims=int(args.sims),
        topk=topk,
        c_scale=float(args.c_scale),
        policy_temp=float(args.policy_temp),
        temperature=float(args.temperature),
        gumbel_scale=float(args.gumbel_scale),
        target_max_visit_cap=int(args.target_max_visit_cap),
        target_untempered_prior=bool(args.target_untempered_prior),
        vloss_weight=int(args.vloss_weight),
        target_batch=int(args.target_batch),
        value_source=str(args.value_source),
        all_root_moves=all_root_moves,
        # Left None for the sources with no arm, rather than resolving numbers
        # nothing reads. `build_nnue_source` refuses a native arm without them.
        nnue_pack=None if args.nnue_pack is None else Path(args.nnue_pack),
        nnue_resolver_max_depth=resolver_depth,
        nnue_qsearch_max_ply=qsearch_ply,
        nnue_qsearch_check_plies=qsearch_checks,
        # The parser's sentinel None expands to the module constant here -- see
        # the ⚑ above the three flags for why the parser cannot hold the default.
        nnue_cp_per_unit=(
            NNUE_CP_PER_INTERNAL_UNIT if args.nnue_cp_per_unit is None
            else float(args.nnue_cp_per_unit)
        ),
        nnue_cp_slope=(
            NNUE_CP_SLOPE if args.nnue_cp_slope is None else float(args.nnue_cp_slope)
        ),
        nnue_cp_draw_width=(
            NNUE_CP_DRAW_WIDTH if args.nnue_cp_draw_width is None
            else float(args.nnue_cp_draw_width)
        ),
        bank_leaf_observations=bool(args.bank_leaf_observations),
        max_plies=int(args.max_plies),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        nice=int(args.nice),
        openings=None if args.openings is None else Path(args.openings),
        opening_plies=int(args.opening_plies),
        opening_max_games=int(args.opening_max_games),
        random_start_plies=int(args.random_start_plies),
        input_history_encoding=str(args.input_history_encoding),
        input_extra_features=str(args.input_extra_features),
        history_rep_fix=bool(args.history_rep_fix),
        run_id=str(args.run_id),
    )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.games <= 0:
        raise SystemExit("--games must be > 0")
    if cfg.shard_size <= 0:
        raise SystemExit("--shard-size must be > 0")
    # ⚑ Refuse rather than ignore -- for EVERY arm flag, not just the pack. A
    # value accepted and dropped is this repo's signature defect, and the
    # command line is the artifact a reader trusts most: `--nnue-qsearch-max-ply
    # 0 --value-source nnue-static` reads like the arm's own negative control
    # and used to produce a plain default-configured static run.
    problems = unconsumable_flags(args)
    if problems:
        raise SystemExit("; ".join(problems))
    if cfg.value_source in NNUE_ARM_SOURCES and cfg.nnue_pack is None:
        raise SystemExit(f"--value-source {cfg.value_source} needs --nnue-pack")
    # `generate` owns the sidecar so a crashed run still leaves one behind.
    summary = generate(
        cfg, summary_json=None if args.summary_json is None else Path(args.summary_json),
    )

    res = summary["results"]
    plies = summary["plies"]
    shape = summary["policy_target_shape"]
    print(
        f"games={summary['games']} rows={summary['rows']} "
        f"shards={summary['shards_written']} "
        f"W/D/L={res['white_wins']}/{res['draws']}/{res['black_wins']}",
    )
    if plies:
        print(
            f"plies: min={plies['min']:.0f} p25={plies['p25']:.0f} "
            f"median={plies['median']:.0f} p75={plies['p75']:.0f} "
            f"max={plies['max']:.0f} mean={plies['mean']:.1f}",
        )
    print(f"terminations (games): {summary['terminations']}")
    print(f"terminations (rows):  {summary['terminations_by_rows']}")
    if shape:
        print(
            f"policy target vs uniform-over-legal: mean TV {shape['tv_to_uniform_mean']:.4f} "
            f"median {shape['tv_to_uniform_median']:.4f} "
            f"sharp(TV>{shape['sharp_row_tv_threshold']}) {shape['sharp_row_frac']:.4%} "
            f"uniform {shape['uniform_row_frac']:.4%}",
        )
    budget = summary["root_budget"]
    if budget:
        print(
            f"root budget: legal moves mean {budget['legal_moves_mean']:.1f} "
            f"max {budget['legal_moves_max']:.0f} | sims mean "
            f"{budget['sims_mean']:.1f} max {budget['sims_max']:.0f}",
        )
    arm = summary["nnue"]
    if arm:
        print(
            f"nnue arm: {arm['leaves']:.0f} leaves in {arm['batches']:.0f} batches "
            f"({arm['leaves_per_batch']:.1f}/batch) + {arm['roots']:.0f} roots "
            f"in-check(calls) {arm.get('in_check_call_frac', float('nan')):.4f} "
            f"expansion {arm.get('resolver_expansion_factor', float('nan')):.2f} "
            f"mate-band {arm['mate_band_frac']:.4%} "
            f"qnodes={arm.get('ctx_qnodes', 0):.0f} "
            f"depth_cutoffs={arm.get('ctx_depth_cutoffs', 0):.0f}",
        )
        if arm.get("ctx_depth_cutoffs", 0):
            # A depth cutoff means a line was neither resolved nor terminal and
            # was scored 0. Defensible as a backstop, indefensible as a silent
            # one -- so the run SAYS SO rather than leaving it in a column.
            print(
                f"⚑ {arm['ctx_depth_cutoffs']:.0f} resolver depth cutoffs: those "
                "leaf values are floors, not evaluations.",
            )
    print(
        f"throughput: {summary['games_per_hour']:.0f} games/h "
        f"{summary['rows_per_hour']:.0f} rows/h "
        f"in {summary['wall_seconds']:.1f}s",
    )
    if summary["partial"]:
        print(f"⚑ PARTIAL RUN: {summary['error']} orphans={summary['orphan_shards']}")
    print(f"summary: {summary['summary_json']}")
    return 1 if summary["partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
