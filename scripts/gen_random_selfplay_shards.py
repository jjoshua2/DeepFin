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
    ``sfnnue``   Stockfish's own evaluation of the leaf, mapped through
                 production's cp-logistic. See "The SF-NNUE value source".
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

The SF-NNUE value source
------------------------
⚑⚑ **DESIGN CONSTRAINT: STOCKFISH SUPPLIES VALUES ONLY.** No SF move ranking,
no MultiPV, no SF-chosen move, anywhere. SF is asked for a score and nothing
else; the returned PV and ``bestmove`` are never read, and the prior stays the
stub's uniform-over-legal. The stored ``policy_target`` therefore remains
**our own search's** improved policy -- ``softmax(log_prior + sigma*Qbar)`` over
SF-valued leaves -- and not a distillation of SF's move preference. This is not
a stylistic choice: the SF→policy-teacher lane is CLOSED (killed by dose
ladder2), and reading SF's move here would reopen it through the back door
while wearing a value source's name. ``test_no_code_path_reads_a_stockfish_move``
pins it by refusing the PV at the protocol level.

What it changes: under ``zero`` the search's ``Qbar`` is constant, so the
improved policy collapses to the prior and the stored target is uniform on
99.32 % of rows. With SF values ``Qbar`` separates the candidates, sequential
halving has something to rank, and the target carries search signal. That is
the whole point, and it is MEASURED per run rather than asserted -- read
``policy_target_shape`` in the sidecar and compare against the zero run's
``tv_to_uniform_median`` (~0.005).

⚑⚑ AND THE TRAP THAT COMES WITH IT -- ``--all-root-moves``, ON by default here.
Sequential halving picks its root candidates by ``gumbel + log_prior``. At
generation zero the prior is UNIFORM, so ``log_prior`` is constant and the
candidate set is decided by the GUMBEL DRAW ALONE: a uniformly random subset of
the legal moves. Under ``zero`` that costs nothing, because no move was going to
be ranked anyway. Under ``sfnnue`` it is the "sharp and wrong" failure built to
order -- the search would rank a random 16 of 40 moves confidently, with the best
move simply never evaluated, and the stored target is dense over the legal moves
either way so nothing downstream could tell.

MEASURED, from the start position (20 legal moves) at ``topk`` 16: exactly four
root moves -- ``a2a3``, ``b1c3``, ``b2b4``, ``c2c4`` -- were never evaluated, at
EVERY sims budget. And raising ``topk`` alone does not fix it: the candidate
count is also capped at ``sims / 2`` (measured at topk 218: sims 32 → 16
candidates, 64 → 32, 128 → 64, 256 → 128). Full coverage needs BOTH
``topk >= n_legal`` AND ``sims >= 2 * n_legal``, and no constant ``--sims`` can
promise the second for every branching factor. So the budget is per-position:
``max(--sims, 2 * n_legal)``, with ``topk`` at the 218-move ceiling. Measured
exact at 20/20, 44/44 and 49/49 legal moves, non-power-of-two included, and
``root_budget`` in the sidecar reports what it actually cost -- under this flag
``--sims`` is a floor, not the realized budget.

With every legal move a candidate and a flat prior, the stored target is exactly
what Josh's framing says it is: ``softmax(sigma * Qbar)`` over ALL legal moves,
with the prior contributing a constant that softmax removes. (Deepening --
prior-free sequential halving that revisits the survivors at a larger node
budget -- is NOT implemented here and is left as a follow-up flag rather than
smuggled in.)

Mechanics, and the three things that are not obvious:

* **The evaluator gets PLANES, not boards**, so the leaf position is rebuilt
  with the repo's own ``eval.audit.decode_board_from_planes`` -- the same
  decoder ``scripts/build_audit_set.py`` uses to put shard rows in front of
  Stockfish, so this tool is not a second opinion about what a row means. It is
  side-to-move canonical (white to move is always the side on move, the
  colour-mirrored twin when black was), which is exactly what an eval needs:
  chess is symmetric under that mirror. ⚑ It needs the LAYOUT to find the
  castling/EP/rule50 planes and reading the wrong one returns plausible
  booleans rather than raising, so ``--input-history-encoding`` is threaded to
  the decoder and pinned by a test.
* **The batch is PADDED, and the padding is stale buffer content.** The C tree
  rounds each batch up to a bucket size and the pad rows are whatever was in the
  buffer last (``gumbel_c._record_batch_dup`` says so in as many words), so they
  are usually *previous real positions* -- which decode fine, get evaluated, and
  are then sliced away by ``pol_all[:n_leaves]`` before the tree sees them.
  Their cost is absorbed by the cache, since a stale row is by construction one
  already evaluated. What CANNOT be evaluated is the uninitialised case: a NaN
  in the rule50 plane makes the decoder raise rather than return None, so
  non-finite rows are rejected before it is called. Such rows get q = 0 with no
  SF call -- the tree discards them anyway -- and are COUNTED (``undecodable``
  in the sidecar's ``sfnnue`` block).

  ⚑ That count is a PADDING measure and it is NOT small at every setting, so do
  not read a big number as breakage. It tracks how far the real leaf count falls
  short of the bucket the batch is rounded up to. MEASURED over 3 games at each
  budget, same seeds: **sims 8 → 71.5 %, 16 → 33.4 %, 32 → 0.02 %, 64 → 0.38 %,
  128 → 0.22 %** (bucket sizes 1/16/32/64/96). The default ``--sims`` 32 sits at
  the floor; below it the batches stop filling their bucket and most of what the
  evaluator is handed is pad. It costs nothing but the skipped work, because a
  real leaf ALWAYS decodes -- that is the load-bearing half, and it is tested
  directly (``test_every_reachable_position_decodes``) rather than inferred from
  this rate.
* **Terminal leaves keep the GAME RULE, never SF.** A mated side-to-move is
  q = -1 and a drawn position q = 0, decided from the board. Measured: the C
  tree resolves terminals itself and 0 of 51,140 leaves reached the evaluator
  game-over, so this is a guard rather than a hot path -- but a guard that
  matters, because ``analyse`` on a finished game has no score to give.

cp → value uses production's own cp-logistic, not a new scale:
``stockfish.wdl.cp_to_wdl(cp, mate, slope=0.006, draw_width_cp=120.0)`` at the
live yaml's ``selfplay.sf_wdl_cp_slope`` / ``selfplay.sf_wdl_cp_draw_width``
(that section is `selfplay:`, not the `stockfish:` the names suggest). Passing ``mate``
straight in is what matches the mate convention rather than transcribing it:
``cp_to_wdl`` gives mate precedence over cp and folds it through
``stockfish.wdl.mate_to_effective_cp``, THE single mate→score mapping that
``selfplay/finalize.py::_sf_move_score`` and ``train/target_builder.py`` both
delegate to. That split is a banked defect class -- the two mate bands once
disagreed and named different best moves on 1.34 % of live scored rows -- and
the way not to re-create it is to call the one home, which is what this does.

⚑ The cp-logistic's DRAW mass is computed and then dropped, deliberately. Both
search paths read a value back as ``softmax([w,d,l])[0] - [2]``
(``mcts/puct.py::_value_scalar_from_wdl_logits``, ``mcts/_mcts_tree.c:42``), so
only the W-L margin has a consumer; this tool writes no ``search_wdl``, so the
D channel has no reader at all. q = W - L is therefore fed through the SAME
``q_to_wdl_logits`` the other three sources use, keeping one logits constructor
and one round-trip test instead of a second mapping that could drift. The
alternative -- emitting the true triple -- would change what the search sees by
at most 3e-9 and would buy a D channel nobody reads.

⚑⚑ **``sfnnue`` IS NOT REPRODUCIBLE FROM THE SEED.** The other three sources are
pure functions of the position, so ``--seed`` fixes the games exactly. Stockfish
is not: one process serves the whole worker and its transposition table carries
across calls, so a fixed-node ``analyse`` depends on what was searched before
it. MEASURED on this box: re-evaluating a position after 10 intervening
searches changed the score on 7/40 positions at 32 nodes and **40/40 at 512**.
This is production's own property, not a defect introduced here (the ledger's
"production SF labels run on a dirty TT", set agreement 32.3 % warm), and it is
why the bounded position cache below is a consistency gain rather than a
shortcut. Consequences: ``test_same_seed_reproduces_the_first_shard_digest``
covers the deterministic sources only, and the same-seed re-run warning above
still applies for the opposite reason -- a re-run duplicates the POSITIONS even
though it will not reproduce the values.

The cache is bounded and FIFO, keyed on the decoded ``board.fen()``: a repeat
within a worker reuses the first answer instead of asking a differently-warmed
TT again. Measured dedup on real gen-0 trees: **2.38x** (58 % of leaves are
repeats), so it is also the difference between one and two hours of wall clock.
Its hit rate is reported per run rather than assumed.

Cost, measured, so nobody plans against a guess: SF answers ~1130 analyse/s at
32 nodes and ~680/s at 512 on this box, against ~5,900 unique leaves per game,
and the UCI round trip -- not the node budget -- dominates (32 → 512 nodes is
16x the search for 1.67x the wall clock). ``sfnnue`` is therefore ~50x slower
per game than ``zero``; see the PR's benchmark table for games/h.

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
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import torch
import yaml

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.encode import input_plane_count
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V2_THREATS
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.encoding.plane_decode import decode_step0_bitboards
from chess_anti_engine.eval.audit import decode_board_from_planes
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
from chess_anti_engine.stockfish.wdl import cp_to_wdl

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
VALUE_SOURCE_SFNNUE = "sfnnue"
VALUE_SOURCES: tuple[str, ...] = (
    VALUE_SOURCE_ZERO, VALUE_SOURCE_RANDOM, VALUE_SOURCE_MATERIAL,
    VALUE_SOURCE_SFNNUE,
)

# ── the SF-NNUE value source ─────────────────────────────────────────────────
# Where the binary comes from: the production config's own key, so this file
# carries no machine path and cannot drift from what production runs. Both the
# live yaml and `main` agree on it (checked 2026-08-24), which is not true of
# ~55 other keys -- see the defaults block above.
PRODUCTION_CONFIG = Path("configs/pbt2_small.yaml")
STOCKFISH_PATH_KEYS: tuple[str, str] = ("stockfish", "stockfish_path")
# ⚑ Production's cp-logistic, NOT a scale invented here: SELFPLAY.sf_wdl_cp_slope
# and selfplay.sf_wdl_cp_draw_width off the LIVE yaml (0.006 / 120.0, and
# selfplay.sf_wdl_use_cp_logistic is true, so this is the mapping the value
# target actually trains against). ⚑ The section is `selfplay:`, not the
# `stockfish:` the names suggest. Pinned by
# test_cp_logistic_constants_match_production.
# Deliberately NOT flags: `parametric_draw_from_q`'s docstring warns that
# passing anything other than the SF component's own knobs silently
# desynchronises the two halves of the blended value target, and a --slope flag
# here would be an invitation to do exactly that for no experiment anyone has
# prereg'd.
SFNNUE_CP_SLOPE = 0.006
SFNNUE_CP_DRAW_WIDTH = 120.0
# Shallow by design -- see the module docstring's cost note: the UCI round trip
# dominates, so this buys eval quality rather than costing wall clock.
DEFAULT_SFNNUE_NODES = 512
# Per-worker SF hash. Small on purpose: 4+ workers each hold one, and at a few
# hundred nodes a search cannot fill even this.
DEFAULT_SFNNUE_HASH_MB = 16
# Bounded FIFO position cache per worker. 200k FENs is ~30 MB and covers a
# whole game's tree several times over (~5,900 unique leaves per game).
DEFAULT_SFNNUE_CACHE = 200_000
# Mated side to move / drawn: the GAME RULE, never an SF score.
TERMINAL_Q_MATED = -1.0
TERMINAL_Q_DRAWN = 0.0

# ── all-root-moves coverage ──────────────────────────────────────────────────
# ⚑⚑ THE TRAP THIS EXISTS FOR. Sequential halving picks its root candidates by
# `gumbel + log_prior`, and at generation zero the prior is UNIFORM -- so
# `log_prior` is constant and the candidate set is chosen by the Gumbel draw
# ALONE: a uniformly random subset of the legal moves. With an uninformative
# value that is harmless (every move is equally unranked). With SF values it is
# not: the search would rank a random subset sharply and never look at the rest,
# so the stored target could be confident about 16 of 40 moves with the best move
# simply absent. That is the ledger's "sharp and wrong" failure manufactured on
# purpose, and it is invisible downstream -- the target is dense over the legal
# moves either way.
#
# MEASURED on the start position (20 legal moves) at topk 16: exactly 4 root
# moves -- a2a3, b1c3, b2b4, c2c4 -- were NEVER evaluated, at every sims budget.
#
# ⚑ And raising topk alone does NOT fix it: the candidate count is ALSO capped by
# the simulation budget at sims/2. MEASURED at topk 218: sims 32 -> 16 candidates,
# 64 -> 32, 128 -> 64, 256 -> 128. So full coverage needs BOTH topk >= n_legal AND
# sims >= 2*n_legal, and a constant `sims` cannot promise it for every branching
# factor. The budget is therefore made per-position: `2 * n_legal`, floored at
# `--sims`. MEASURED exact at 20/20, 44/44 and 49/49 legal moves, including the
# non-power-of-two ones.
MAX_LEGAL_MOVES = 218        # the most any legal chess position admits
ROOT_SIMS_PER_LEGAL_MOVE = 2  # the measured factor above, not a safety margin

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


def default_stockfish_path(config: Path = PRODUCTION_CONFIG) -> Path:
    """The binary production uses, read off the production config's own key.

    Not a constant in this file: the repo is public and a machine path does not
    belong in source, and a hardcoded guess would be a second opinion about
    which Stockfish is "the" one the moment the config moved.
    """
    section, key = STOCKFISH_PATH_KEYS
    with open(config, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    try:
        raw = loaded[section][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{config} has no {section}.{key}; pass --sf-binary explicitly",
        ) from exc
    return Path(str(raw))


def file_md5(path: Path) -> str:
    """Content hash of the binary actually opened -- provenance, not a name."""
    # md5, not a security claim: this labels which binary produced a corpus.
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q_from_cp_mate(cp: float | None, mate: int | None) -> float:
    """A Stockfish score → q in [-1, 1], through PRODUCTION's cp-logistic.

    THE cp/mate → value mapping for this tool, and a named function rather than
    three lines inside the engine wrapper so a test can drive it without an
    engine -- a mate-in-2 is otherwise awkward to obtain on demand, and a
    mapping only exercised through a live search is a mapping only tested where
    it happens to be reached.

    Both halves are delegated, neither transcribed:

    * ``cp_to_wdl`` gives ``mate`` precedence over ``cp`` (the UCI convention --
      SF emits at most one per info line) and folds it through
      ``stockfish.wdl.mate_to_effective_cp``, THE single mate→score mapping.
      ⚑ That single home exists because it was once two: ``finalize``'s mate
      band dominated cp while ``target_builder``'s sat INSIDE the cp range, and
      on 1.34 % of live scored rows the two named different best moves on the
      same position. Folding mate here with a local base would recreate exactly
      that, one file further out.
    * the slope and draw width are production's own
      ``selfplay.sf_wdl_cp_slope`` / ``sf_wdl_cp_draw_width``.

    Returns ``W - L``; see the module docstring for why the D channel is
    dropped rather than propagated.
    """
    wdl = cp_to_wdl(
        cp, mate, slope=SFNNUE_CP_SLOPE, draw_width_cp=SFNNUE_CP_DRAW_WIDTH,
    )
    return float(wdl[0]) - float(wdl[2])


def terminal_q(board: chess.Board) -> float | None:
    """The GAME RULE's value for a finished position, or None if it is not one.

    ⚑ Never SF. A mated side to move has lost, whatever an evaluation of a
    position with no moves would say -- and ``analyse`` on a finished game has
    no score to return in the first place.
    """
    if not board.is_game_over():
        return None
    return TERMINAL_Q_MATED if board.is_checkmate() else TERMINAL_Q_DRAWN


@dataclass
class SfnnueStats:
    """What the SF source did, so the sidecar reports it instead of implying it."""

    rows: int = 0
    analysed: int = 0
    cache_hits: int = 0
    terminal: int = 0
    undecodable: int = 0

    def merge(self, other: SfnnueStats) -> None:
        self.rows += other.rows
        self.analysed += other.analysed
        self.cache_hits += other.cache_hits
        self.terminal += other.terminal
        self.undecodable += other.undecodable

    def summary(self) -> dict[str, float]:
        if self.rows == 0:
            return {}
        looked_up = self.analysed + self.cache_hits
        return {
            "rows": float(self.rows),
            "analysed": float(self.analysed),
            "cache_hits": float(self.cache_hits),
            "cache_hit_frac": (
                self.cache_hits / looked_up if looked_up else 0.0
            ),
            "terminal_rows": float(self.terminal),
            # Expected ~0.02% (the C tree's all-zero batch padding). A rate that
            # left that band means real positions are being skipped.
            "undecodable_rows": float(self.undecodable),
            "undecodable_frac": self.undecodable / self.rows,
        }


class StockfishValueSource:
    """Fixed-node Stockfish evaluations of decoded leaves, as q in [-1, 1].

    ⚑ VALUES ONLY. ``analyse`` returns a PV and this class reads ``score`` and
    nothing else; ``_score_cp_mate`` is the only place a UCI reply is touched,
    and it takes the two score fields by name. There is no MultiPV, no
    ``play()``, and no move of SF's anywhere in this file -- see the module
    docstring's design constraint for why that is load-bearing rather than
    tidy.

    One engine per worker, opened lazily so constructing an evaluator in a test
    does not spawn a process, and reaped by ``close()`` in the worker's
    ``finally``.
    """

    def __init__(
        self,
        *,
        binary: Path,
        nodes: int,
        input_history_encoding: str,
        hash_mb: int = DEFAULT_SFNNUE_HASH_MB,
        cache_size: int = DEFAULT_SFNNUE_CACHE,
    ) -> None:
        if int(nodes) <= 0:
            raise ValueError(f"sfnnue nodes must be > 0, got {nodes}")
        self.binary = Path(binary)
        if not self.binary.is_file():
            raise FileNotFoundError(f"stockfish binary not found: {self.binary}")
        self.nodes = int(nodes)
        self.input_history_encoding = str(input_history_encoding)
        self.hash_mb = int(hash_mb)
        self.cache_size = max(0, int(cache_size))
        self.stats = SfnnueStats()
        self._engine: chess.engine.SimpleEngine | None = None
        self._cache: dict[str, float] = {}

    # -- provenance ---------------------------------------------------------
    @property
    def binary_md5(self) -> str:
        return file_md5(self.binary)

    # -- engine lifecycle ---------------------------------------------------
    def engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(str(self.binary))
            # Threads 1: every worker holds its own engine, and at a few hundred
            # nodes a second thread is pure overhead.
            engine.configure({"Threads": 1, "Hash": self.hash_mb})
            self._engine = engine
        return self._engine

    def close(self) -> None:
        """Reap the child. Idempotent, and it does not leave an orphan behind.

        ``quit()`` is the clean path and it can hang or raise on an engine that
        is already wedged, which is exactly when a leaked process matters --
        so the fallback is ``close()``, which drops the transport and takes the
        child with it.
        """
        engine, self._engine = self._engine, None
        if engine is None:
            return
        try:
            engine.quit()
        except BaseException as exc:
            _LOG.warning("stockfish quit() failed (%s); closing transport", exc)
            try:
                engine.close()
            except BaseException as exc2:
                _LOG.warning("stockfish close() failed: %s", exc2)

    # -- evaluation ---------------------------------------------------------
    @staticmethod
    def _score_cp_mate(info: chess.engine.InfoDict) -> tuple[int | None, int | None]:
        """``(cp, mate)`` from the side-to-move's POV. The ONLY reply field read.

        The decoded board is side-to-move canonical -- white to move IS the
        original mover -- so ``pov(chess.WHITE)`` is the side-to-move POV. Taking
        it from the board's own turn would read the same thing while implying
        the frame could be otherwise.
        """
        score = info.get("score")
        if score is None:
            raise RuntimeError("stockfish returned no score for an analysed position")
        pov = score.pov(chess.WHITE)
        return pov.score(), pov.mate()

    def q_for_board(self, board: chess.Board) -> float:
        """One decoded leaf → q, through production's cp-logistic."""
        cp, mate = self._score_cp_mate(
            self.engine().analyse(board, chess.engine.Limit(nodes=self.nodes)),
        )
        return q_from_cp_mate(None if cp is None else float(cp), mate)

    def _cached_q(self, board: chess.Board) -> float:
        key = board.fen()
        hit = self._cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit
        value = self.q_for_board(board)
        self.stats.analysed += 1
        if self.cache_size:
            if len(self._cache) >= self.cache_size:
                # FIFO, not LRU: dicts keep insertion order, and an exact
                # recency policy buys nothing over a window this much larger
                # than one game's tree.
                del self._cache[next(iter(self._cache))]
            self._cache[key] = value
        return value

    def q_for_planes(self, x: np.ndarray) -> np.ndarray:
        """(N, C, 8, 8) encoded leaves → (N,) q in [-1, 1], side-to-move POV."""
        arr = np.asarray(x, dtype=np.float32)
        out = np.zeros((arr.shape[0],), dtype=np.float64)
        for i in range(arr.shape[0]):
            self.stats.rows += 1
            # ⚑ FINITENESS FIRST, and it is not belt-and-braces. The C tree pads
            # each batch to a bucket size with STALE BUFFER CONTENT
            # (`gumbel_c._record_batch_dup`), and on the first batches that
            # content is uninitialised -- a NaN in the rule50 plane makes
            # `decode_board_from_planes` RAISE (`round(nan)`) rather than return
            # None, which killed the first run of this source. The decoder's own
            # None covers finite-but-invalid rows; this covers the rows it
            # cannot be asked about at all. Two disjoint conditions, not a
            # backstop over one.
            if not bool(np.isfinite(arr[i]).all()):
                self.stats.undecodable += 1
                continue
            board = decode_board_from_planes(
                arr[i], input_history_encoding=self.input_history_encoding,
            )
            if board is None:
                # The C tree's all-zero batch padding, whose value it discards.
                # Counted rather than passed to SF: an empty board has no
                # evaluation, and a rate outside ~0.02% means something else.
                self.stats.undecodable += 1
                continue
            forced = terminal_q(board)
            if forced is not None:
                self.stats.terminal += 1
                out[i] = forced
                continue
            out[i] = self._cached_q(board)
        return out


class UniformPriorEvaluator:
    """``BatchEvaluator`` stub: uniform prior over legal moves + a cheap value.

    ``eval_calls`` / ``eval_rows`` are not decoration: they are how a test
    proves the search actually consumed ``--sims`` rather than merely storing it
    (a bigger budget must ask this object for more positions).
    """

    def __init__(
        self, *, value_source: str, expected_planes: int, random_salt: int = 0,
        sf_source: StockfishValueSource | None = None,
    ) -> None:
        if value_source not in VALUE_SOURCES:
            raise ValueError(
                f"value_source must be one of {VALUE_SOURCES}, got {value_source!r}",
            )
        if (value_source == VALUE_SOURCE_SFNNUE) != (sf_source is not None):
            # Both directions. A missing source would silently fall through to
            # some other branch; a source handed to `zero` would open a
            # Stockfish nobody asked for and report a value_source that did not
            # produce the values.
            raise ValueError(
                f"value_source={value_source!r} and sf_source="
                f"{'set' if sf_source is not None else 'None'} disagree: "
                f"{VALUE_SOURCE_SFNNUE!r} requires an sf_source and no other "
                "source may have one",
            )
        self.value_source = str(value_source)
        self.expected_planes = int(expected_planes)
        self.random_salt = int(random_salt)
        self.sf_source = sf_source
        self.eval_calls = 0
        self.eval_rows = 0

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
        if self.value_source == VALUE_SOURCE_ZERO:
            q = np.zeros((n,), dtype=np.float64)
        elif self.value_source == VALUE_SOURCE_MATERIAL:
            q = material_q(arr)
        elif self.value_source == VALUE_SOURCE_SFNNUE:
            # Non-None is enforced in __init__; re-checked here so a future
            # edit that loosens the constructor fails loudly rather than
            # falling through to another source under this source's name.
            if self.sf_source is None:
                raise ValueError(
                    f"{VALUE_SOURCE_SFNNUE} evaluator has no sf_source",
                )
            q = self.sf_source.q_for_planes(arr)
        else:
            q = random_q(arr, salt=self.random_salt)
        return policy_logits, q_to_wdl_logits(q)

    def close(self) -> None:
        """Release the value source's process, if it has one. Idempotent."""
        if self.sf_source is not None:
            self.sf_source.close()


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
    sf_binary: Path | None = None
    sfnnue_nodes: int = DEFAULT_SFNNUE_NODES
    sfnnue_hash_mb: int = DEFAULT_SFNNUE_HASH_MB
    sfnnue_cache_size: int = DEFAULT_SFNNUE_CACHE
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
    """Simulations for THIS root, so every legal move is a candidate.

    ``--sims`` is a FLOOR, not the value: sequential halving caps the candidate
    set at ``sims / 2``, so a fixed budget silently truncates wide positions.
    Returns ``max(--sims, 2 * n_legal)`` when ``--all-root-moves`` is on, and
    ``--sims`` verbatim otherwise. See ``ROOT_SIMS_PER_LEGAL_MOVE`` for the
    measurement behind the factor.
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

    def add(self, *, legal_moves: int, sims: int) -> None:
        self.plies += 1
        self.legal_sum += int(legal_moves)
        self.legal_max = max(self.legal_max, int(legal_moves))
        self.sims_sum += int(sims)
        self.sims_max = max(self.sims_max, int(sims))

    def merge(self, other: RootBudgetStats) -> None:
        self.plies += other.plies
        self.legal_sum += other.legal_sum
        self.legal_max = max(self.legal_max, other.legal_max)
        self.sims_sum += other.sims_sum
        self.sims_max = max(self.sims_max, other.sims_max)

    def summary(self) -> dict[str, float]:
        if self.plies == 0:
            return {}
        return {
            "plies": float(self.plies),
            "legal_moves_mean": self.legal_sum / self.plies,
            "legal_moves_max": float(self.legal_max),
            "sims_mean": self.sims_sum / self.plies,
            "sims_max": float(self.sims_max),
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
    sfnnue: SfnnueStats = field(default_factory=SfnnueStats)
    root_budget: RootBudgetStats = field(default_factory=RootBudgetStats)
    eval_calls: int = 0
    eval_rows: int = 0
    seconds: float = 0.0


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


def build_sf_source(cfg: GenConfig) -> StockfishValueSource | None:
    """The SF value source for this config, or None for the pure sources.

    ⚑ ``input_history_encoding`` is threaded through because the plane decoder
    finds the castling/EP/rule50 block by LAYOUT, and reading the wrong offsets
    returns plausible booleans rather than raising -- so a mismatch here would
    hand Stockfish a position with the wrong castling rights and no error
    anywhere. Pinned by test_sfnnue_decodes_with_the_configured_layout.
    """
    if cfg.value_source != VALUE_SOURCE_SFNNUE:
        return None
    if cfg.sf_binary is None:
        raise ValueError(
            f"--value-source {VALUE_SOURCE_SFNNUE} needs a Stockfish binary; "
            f"none was resolved from {PRODUCTION_CONFIG} and --sf-binary was "
            "not given",
        )
    return StockfishValueSource(
        binary=Path(cfg.sf_binary),
        nodes=int(cfg.sfnnue_nodes),
        input_history_encoding=str(cfg.input_history_encoding),
        hash_mb=int(cfg.sfnnue_hash_mb),
        cache_size=int(cfg.sfnnue_cache_size),
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
        # The coverage guarantee and the two knobs that implement it. `topk`
        # above is read off the GumbelConfig the search receives, so these
        # three disagreeing would be visible rather than inferred.
        "all_root_moves": bool(cfg.all_root_moves),
        "root_sims_per_legal_move": (
            ROOT_SIMS_PER_LEGAL_MOVE if cfg.all_root_moves else 0
        ),
        "material_cp_scale": float(MATERIAL_CP_SCALE),
        "random_salt": int(evaluator.random_salt),
        # ── the SF value source, read off the source object the evaluator will
        # call -- not off cfg, so a flag dropped on the way in shows up here.
        # None on the three sources that have no engine, which is a reading and
        # not an omission.
        "sfnnue_nodes": (
            None if evaluator.sf_source is None else int(evaluator.sf_source.nodes)
        ),
        "sfnnue_hash_mb": (
            None if evaluator.sf_source is None else int(evaluator.sf_source.hash_mb)
        ),
        "sfnnue_cache_size": (
            None if evaluator.sf_source is None
            else int(evaluator.sf_source.cache_size)
        ),
        "sfnnue_history_encoding": (
            None if evaluator.sf_source is None
            else str(evaluator.sf_source.input_history_encoding)
        ),
        "sf_binary_path": (
            None if evaluator.sf_source is None else str(evaluator.sf_source.binary)
        ),
        # Provenance of the thing that actually produced the numbers: the path
        # is a name and names get repointed, the hash is the binary.
        "sf_binary_md5": (
            None if evaluator.sf_source is None else evaluator.sf_source.binary_md5
        ),
        # Constants, alongside `material_cp_scale` and for the same reason: a
        # corpus is only readable next to the value scale that built it.
        "sfnnue_cp_slope": float(SFNNUE_CP_SLOPE),
        "sfnnue_cp_draw_width": float(SFNNUE_CP_DRAW_WIDTH),
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


def play_game(
    *,
    cfg: GenConfig,
    gcfg: GumbelConfig,
    evaluator: UniformPriorEvaluator,
    rng: np.random.Generator,
    opening_cfg: OpeningConfig,
    budget: RootBudgetStats | None = None,
) -> GameOutcome:
    """Play one complete game with the production C Gumbel search."""
    start = sample_starting_board(rng=rng, cfg=opening_cfg)
    board = start.board
    cb = CBoard.from_board(board)
    start_fen = board.fen()
    records: list[PlyRecord] = []
    actions: list[int] = []

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
            budget.add(legal_moves=legal_moves, sims=sims_here)
        probs, acts, _values, masks, _tree, _root_ids = run_gumbel_root_many_c(
            None,
            [board],
            device="cpu",
            rng=rng,
            cfg=gcfg,
            evaluator=evaluator,
            cboards=[cb],
            per_game_simulations=[sims_here],
            # Explicit, never defaulted: these two are not GumbelConfig fields,
            # so no override surface can reach them and mcts/gumbel.py's standing
            # comment requires every call site to pass them by hand.
            vloss_weight=int(cfg.vloss_weight),
            target_batch=int(cfg.target_batch),
        )
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
        sf_source=build_sf_source(cfg),
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

    # ⚑ The `finally` reaps the value source's Stockfish child. A worker that
    # raises mid-game has one running, and `generate` deliberately keeps going
    # to build the partial sidecar -- so without this an orphaned engine
    # outlives the run that spawned it, once per worker, holding its hash table.
    try:
        for _ in range(int(spec.games)):
            outcome = play_game(
                cfg=cfg, gcfg=gcfg, evaluator=evaluator, rng=rng,
                opening_cfg=opening_cfg, budget=result.root_budget,
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
    finally:
        if evaluator.sf_source is not None:
            # Copied before close(), which does not clear the counters but is
            # the last moment the source is unambiguously this worker's.
            result.sfnnue = evaluator.sf_source.stats
        evaluator.close()

    result.eval_calls = evaluator.eval_calls
    result.eval_rows = evaluator.eval_rows
    result.seconds = time.perf_counter() - started
    return result


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


def summarize(
    *, cfg: GenConfig, results: list[WorkerResult], wall_seconds: float,
    shard_index_start: int, partial: bool = False, error: str | None = None,
) -> dict[str, Any]:
    plies: list[int] = []
    terminations: dict[str, int] = {}
    termination_rows: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    shape = PolicyShapeStats()
    sfnnue = SfnnueStats()
    root_budget = RootBudgetStats()
    games = rows = wins = draws = losses = eval_calls = eval_rows = 0
    for res in results:
        sfnnue.merge(res.sfnnue)
        root_budget.merge(res.root_budget)
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
    hours = max(wall_seconds, 1e-9) / 3600.0
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
                   "sf_binary": None if cfg.sf_binary is None else str(cfg.sf_binary)},
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
        # Empty for the three sources with no engine, which is the reading.
        "sfnnue": sfnnue.summary(),
        # ⚑ What the root budget REALLY was. Under --all-root-moves `--sims`
        # is only a floor, so `sims_mean`/`sims_max` are the realized cost
        # and `legal_moves_max` says how wide it had to stretch.
        "root_budget": root_budget.summary(),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
    started = time.perf_counter()
    results: list[WorkerResult] = []
    failure: str | None = None
    try:
        if len(specs) == 1:
            results.append(run_worker(specs[0]))
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
                    completed = future.result()
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
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
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
            "sfnnue (Stockfish VALUES ONLY through production's cp-logistic -- "
            "never SF moves; ~50x slower per game and not seed-reproducible) | "
            "random (⚑ STRUCTURED NOISE -- plumbing tests only, never a training "
            "corpus: it emits confident targets encoding a hash of the position)."
        ),
    )
    parser.add_argument(
        "--sf-binary", type=Path, default=None,
        help=(
            "Stockfish binary for --value-source sfnnue. Default: the "
            f"{'.'.join(STOCKFISH_PATH_KEYS)} key of {PRODUCTION_CONFIG}, i.e. "
            "the binary production itself uses."
        ),
    )
    parser.add_argument(
        "--sfnnue-nodes", type=int, default=DEFAULT_SFNNUE_NODES,
        help=(
            "Fixed node budget per leaf evaluation. ⚑ The UCI round trip "
            "dominates, not the search: 32 -> 512 nodes is 16x the nodes for "
            "~1.67x the wall clock, which is why the default is not smaller."
        ),
    )
    parser.add_argument(
        "--sfnnue-hash-mb", type=int, default=DEFAULT_SFNNUE_HASH_MB,
        help="Per-worker Stockfish hash (MB); each worker holds its own engine.",
    )
    parser.add_argument(
        "--sfnnue-cache-size", type=int, default=DEFAULT_SFNNUE_CACHE,
        help=(
            "Bounded FIFO of decoded-FEN -> q per worker; 0 disables it. "
            "Measured 2.38x dedup on real gen-0 trees."
        ),
    )
    parser.add_argument(
        "--all-root-moves", dest="all_root_moves", action="store_true",
        default=None,
        help=(
            "Make EVERY legal root move a search candidate: topk defaults to "
            "the 218-move ceiling and the per-root budget becomes "
            "max(--sims, 2*n_legal). ON by default for --value-source sfnnue "
            "and off otherwise. ⚑ Without it, a uniform prior makes "
            "sequential halving pick its candidates by the Gumbel draw "
            "alone, so the best move is absent from the target on wide "
            "positions -- measured 4 of 20 root moves never evaluated from "
            "the start position at topk 16."
        ),
    )
    parser.add_argument(
        "--no-all-root-moves", dest="all_root_moves", action="store_false",
        help="Keep production's topk-capped candidate set even under sfnnue.",
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


def config_from_args(args: argparse.Namespace) -> GenConfig:
    # ⚑ Resolved TOGETHER, because they are one decision. `--all-root-moves`
    # defaults ON for sfnnue (where an unranked candidate subset would make
    # the target sharp about a random half of the moves) and OFF elsewhere,
    # and topk follows it unless the operator named a value -- in which case
    # a contradiction is refused in `build_gumbel_config` rather than
    # resolved behind their back.
    all_root_moves = (
        str(args.value_source) == VALUE_SOURCE_SFNNUE
        if args.all_root_moves is None
        else bool(args.all_root_moves)
    )
    topk = (
        int(args.topk) if args.topk is not None
        else (MAX_LEGAL_MOVES if all_root_moves else DEFAULT_TOPK)
    )
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
        # Resolved HERE, not in the worker, so the path is settled once and every
        # worker reports the same binary -- and so a missing config fails at
        # startup rather than N times inside spawned children. Left None for the
        # sources that have no engine, rather than resolving a path nothing opens.
        sf_binary=(
            None if str(args.value_source) != VALUE_SOURCE_SFNNUE
            else Path(args.sf_binary) if args.sf_binary is not None
            else default_stockfish_path()
        ),
        sfnnue_nodes=int(args.sfnnue_nodes),
        sfnnue_hash_mb=int(args.sfnnue_hash_mb),
        sfnnue_cache_size=int(args.sfnnue_cache_size),
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
    sf = summary["sfnnue"]
    if sf:
        print(
            f"sfnnue: {sf['analysed']:.0f} analysed {sf['cache_hits']:.0f} cached "
            f"(hit {sf['cache_hit_frac']:.1%}) terminal={sf['terminal_rows']:.0f} "
            f"undecodable={sf['undecodable_rows']:.0f} ({sf['undecodable_frac']:.4%})",
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
