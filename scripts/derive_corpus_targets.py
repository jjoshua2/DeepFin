#!/usr/bin/env python3
"""Turn an NNUE-bootstrap corpus bank into training shards the control rig reads.

``scripts/gen_sf_rooted_corpus.py`` banks the LOWEST-LEVEL observations a
Stockfish staircase emitted -- every phase, every depth, every MultiPV rank,
with the cumulative node count on each line.  It does not bank a training
target, on purpose: the ledger's FREEZE-THE-OBSERVATIONS rule exists so that
choosing a target is a re-read of a corpus rather than a rerun of the search.

This is the re-read.  It maps each banked row's ``(move -> per-depth value)``
bank to ONE value per legal move under a named SCHEME, turns those values into a
policy and a value target, and writes ``shard_NNNNNN.zarr`` replay shards that
``scripts/lc0_control_train.py --shards`` consumes unchanged.

Usage::

    PYTHONPATH=. python3 scripts/derive_corpus_targets.py \\
        --corpus data/nnue_bootstrap/run01 \\
        --out data/nnue_rows/run01-uniform-d9 \\
        --scheme uniform-d9 --temp 1.0 [--floor 0.002]

THE SCHEMES
-----------
``uniform-d<D>``
    Every move's value at depth ``D``.  Each move is read from the DEEPEST PHASE
    that carries that move at that depth -- a narrowed phase's depth-``D`` block
    is the same iteration seen with a warmer transposition table, so it is the
    better-informed observation of the same cell.  ⚑ That means one row's values
    can come from several phases, and ``values_by_phase`` in the summary says
    how often; a ``uniform-d9`` corpus is therefore NOT "what a bare
    ``go depth 9`` would have printed".

``top<K>-d<D2>-rest-d<D1>``
    Rank every move by its ``D1`` value, read the top ``K`` at ``D2`` and leave
    the rest at ``D1``.  This is the OFFLINE HALVING SIMULATION: it prices a
    narrowing policy the generator did not run, against blocks it already
    banked.  ⚑ A move in the top ``K`` by ``D1`` that the generator's own
    narrowing did NOT carry to ``D2`` has no banked value there; that row is an
    envelope miss (see below) rather than a row quietly read at ``D1``.

``nodes-<N>``
    The deepest COMPLETE phase-0 depth whose banked cumulative node count is
    ``<= N``, per row.  ⚑ PHASE 0 ONLY, for both the depth choice and the
    values, and that is the one place this file deliberately differs from the
    two schemes above: the claim being reconstructed is "what a full-width
    ``go nodes N`` would have produced", and phase 1+ node counts are neither
    full width nor measured from a cold table.  Splicing a depth-13 narrowed
    reading into a budget arm would make its own headline false.
    A row whose SHALLOWEST complete banked depth already exceeds ``N`` is read
    at that shallowest depth and COUNTED as ``nodes_floor_hits`` -- the budget
    was not honoured for that row and the corpus says so, rather than the row
    silently pretending it was.

THE TARGETS
-----------
POLICY -- ``(1 - floor * n_legal) * softmax(q / temp) + floor`` over the
scheme's values, placed on the compact ``lc0_1858`` slots
``compact_index_for_move`` assigns and zero everywhere else.  ``n_legal`` is the
ROW's own legal-move count, so every row's policy sums to 1 and every legal move
carries at least ``--floor``.  At the default ``--floor 0`` the expression is
``softmax(q / temp)`` and the emitted bytes are byte-for-byte what this tool
wrote before the flag existed.  ``q`` comes from
``audit_label_candidates.q_from_effective_cp``, which reaches
``gen_random_selfplay_shards.cp_to_wdl_array`` as a module attribute AT CALL
TIME -- the same one function object the generator's own move selection and the
label gate's arms use.  ``tests/test_derive_corpus_targets.py`` proves it by
replacing that single object and watching this file's targets move.  Because the
generator selects with ``argmax(q/tau + Gumbel)``, ``--temp tau`` reproduces
exactly the distribution its own play was sampled from -- ``--floor`` is a
DEPARTURE from that distribution and is off unless asked for.

⚑ WHY A FLOOR AT ALL: T80's shape is a search-sharpened head PLUS an
exploration floor, and one temperature cannot fit both ends.  The ledger's
2026-08-29 re-analysis of the banked T80 ruler (``3e655b762``, bank
``scratchpad/t80_ruler/``) measured the miss at BOTH ends of a single-tau fit --
at tau 0.08 the T80 target puts 0.699 of its mass within 10cp of best against
the mapping's 0.560, and 0.039 beyond 150cp against 0.030, with the middle
over-weighted; the two ends carry 87% of the KL.  ``tau 0.04 + floor 0.002``
reads KL 0.336, **-31% against the best single temperature**, interior on both
axes of a 0.001-0.006 x 0.02-0.06 grid.  ⚑ THE LADDER ARM IS THE FLOOR ALONE:
the reorder (``66f29b703``) runs arm 5 as ``qtemp_0.067 + floor 0.002``, the
Gumbel-sigma head UNCHANGED and the floor as the single treatment -- sharpening
the head to 0.04 was declined on lever 1's ground that eval-space sharpening
amplifies label confidence on the ~7% of rows where d9 is wrong.  ``--floor`` is
orthogonal to ``--temp`` and says nothing about which head an arm uses.  ⚑ The
training-side consequence, and the reason this is a floor rather than a wider
temperature: a move the eval ranks last still keeps ``floor`` of the mass, so a
move MISCLASSIFIED by the eval is recoverable by the net instead of being zeroed
forever -- and it survives the shard's float16 cast, which does NOT survive the
alternative.  MEASURED on the test fixture's 9-move row at tau 0.04: the ninth
move's unfloored mass is 1.6e-09, float16's smallest subnormal is 6e-08, so the
shard stores that move as EXACTLY ZERO while the legal mask still names it
(``policy_support_lost_to_float16`` counts exactly that).  Floored it stores
0.00200.

⚑ THE FLOOR IS REFUSED AT STARTUP, NEVER PER ROW.  ``floor * n_legal >= 1``
would make the head's coefficient non-positive, and ``n_legal`` varies by row,
so the natural-looking implementation drops or refuses individual ROWS.  This
file already has a per-row drop idiom (``EnvelopeMiss`` /
``--max-envelope-misses``) and the floor deliberately does NOT use it: an
envelope miss is a property of the CORPUS -- the same rows for every reader --
whereas a floor-driven drop would be a property of the FLAG, so ``--floor``
would silently change WHICH POSITIONS the derived corpus contains.  Two arms of
a floor ladder would then differ on two axes at once and the paired comparison
the ladder exists for would be confounded.  So ``validate_floor`` refuses at
startup against the chess-theoretic bound ``MAX_LEGAL_MOVES`` (218), before the
first shard is opened, and every row of a run that starts is emitted.  The bound
is deliberately conservative -- it refuses ``--floor 0.005`` even though no real
position has 218 legal moves -- and the arm this was built for, ``floor 0.002``,
sits 2.3x below it (0.002 x 218 = 0.436).  ``apply_floor`` re-checks the same
inequality per row -- a COEFFICIENT check, ``floor * n_legal >= 1``, so at
floor 0.002 it fires from n_legal 500 up, which under the startup refusal
requires a position with more legal moves than the 218 bound admits: it makes
the bound falsifiable rather than assumed, and it is fatal rather than a drop,
so the emitted row set stays independent of the flag either way.

⚑⚑ THE TAKE-EFFECT STAMPS, AND WHY THERE ARE TWO.  With a floor the emitted
policy is NOT ``softmax(q / temp)``, so the closed-form ``recover_temp`` would
read a temperature that was never requested -- a stamp that lies.  A floored run
therefore switches estimator (the choice is stamped as
``temp_recovery_estimator``) to ``recover_floor_and_temp``, which recovers BOTH
knobs from the emitted row and NOTHING from the flags: floor cancels out of
DIFFERENCES of emitted probabilities, so the ratio
``(p_hi - p_mid) / (p_mid - p_lo)`` is a function of tau alone and inverts to
tau; the scale ``(p_hi - p_lo) / (s_hi - s_lo)`` then gives ``1 - floor *
n_legal`` and hence the floor.  A floor that was parsed and then not applied
reads back 0.  That estimator is EXACT but needs three distinct values and
enough numerical separation (see ``FLOOR_RECOVERY_MIN_SPREAD_PER_TAU`` for
what "enough" is and why), so it is joined by a coarse one that covers every
row: ``policy_min_legal_prob_stored``, the smallest mass any legal move
carries AFTER the shard's float32-then-float16 cast, which lands within one
float16 ULP of the floor on every floored row (the cast rounds to nearest in
BOTH directions -- 0.002 stores as 0.00200081, 0.0035 as 0.00349998) and
collapses to the softmax tail (1e-8 and below, or 0) without the floor.  One
instrument is exact on a subset, the other is approximate on all of it, and the
summary reports the coverage of the first rather than implying it is universal.
⚑ Neither stamp is merely published: ``enforce_take_effect`` compares the mean
recovered floor (and tau) against the flags before the summary is written and
kills the run on a mismatch, so a floored directory that carries a summary is
one whose rows were MEASURED to carry the floor.

⚑ ONE TRAINING-SIDE KNOB CAN STRIP THE FLOOR AFTER THE FACT:
``policy_target_temp`` (``retemper_main_policy_target`` in
``chess_anti_engine/train/losses.py``) applies ``p ** (1/T)`` plus a
renormalise, which compresses the floor's relative mass non-linearly.  It is
1.0 (identity) in every config and unset by ``lc0_control_train.py``, so the
floor ladder trains on what was derived -- but an arm that pairs a floor with
``policy_target_temp != 1.0`` is not training on the floor it stamped.

VALUE -- the construction is ``data/lc0_rows``'s, mirrored:

* ``wdl_target`` = the row's EXACT game result, already stored from that row's
  own side-to-move seat by ``result_from_pov``.  0=W / 1=D / 2=L.
* ``search_wdl`` = ``cp_to_wdl_array`` of the scheme's BEST-MOVE value, i.e. the
  searched root value of the position, side-to-move POV.
* ``sf_wdl`` is ABSENT.

⚑⚑ AND THE SEARCHED VALUE GOES IN ``search_wdl`` EVEN THOUGH IT IS A STOCKFISH
EVAL.  That reads backwards until you check what the consumer does with it:
``lc0_control_train.py``'s launch guard 1 calls
``assert_pid_cannot_reassert_sf_wdl``, which refuses ANY config with
``sf_wdl_frac > 0`` -- unconditionally, with no reference to whether the shards
carry an SF label.  A value written to ``sf_wdl`` therefore could not reach a
loss on this rig under any config it will start: accepted at write time, ignored
at train time, this repo's signature defect manufactured by its own tooling.
``search_wdl`` is the channel the rig can actually weight
(``search_wdl_frac``), and it is the channel ``lc0_data_to_rows`` already uses
for "the search's own root value".  ⚑ The honesty cost is real and is paid in
the manifest: a reader joining these shards with production ones must know that
here ``search_wdl`` is Stockfish's root value and not our MCTS's, so
``value_channels`` in ``<out>/derive_targets_summary.json`` says exactly that.

⚑ NO BLEND IS BAKED INTO A ROW.  The row carries the two components; the mixing
weights are the trainer's (``game_frac`` / ``search_wdl_frac``), exactly as on
the lc0 corpus.  ``required_training_overrides`` in the summary names the
combination that passes the rig's own guards, and a test asserts that it does by
running ``run_config_problems`` against the shards this tool actually wrote.

⚑ A ROW WITH NO RESULT IS SKIPPED AND COUNTED (``rows_dropped_no_result``).  The
generator's ply cap outside tablebase range banks ``result: null`` and its own
docstring refuses to call that a draw; ``wdl_target`` is a REQUIRED shard field
with no has-flag, so there is no way to emit such a row without inventing an
outcome for it.  The count is in the summary because dropping them shifts the
corpus's position mix and a consumer has to be able to see by how much.

⚑⚑ ZERO HISTORY, AND IT IS MEASURED RATHER THAN ASSERTED
--------------------------------------------------------
A corpus row is a FEN.  It carries ``game_id`` and ``ply``, but rows are banked
only above ``MIN_BANKED_PIECES`` and only on a dedup MISS, so the plies of one
game are not contiguous and the move stack cannot be rebuilt from the corpus
alone.  ``encode_position`` on a ``chess.Board(fen)`` therefore fills history
slot 0 and leaves slots 1..7 -- planes 13..103 -- ZERO, and every repetition
plane with them.  That is the same blindness the frozen rulers score under, and
it is a real difference from both production selfplay rows and the lc0 corpus,
whose 8 frames are all real.  The summary stamps
``history_slots_nonzero_max``, measured off the planes this run actually wrote,
so the claim is a reading and not a comment.

⚑⚑ TWO CORPUS RECORDS, AND A PARTIAL ONE IS NEVER SILENT
---------------------------------------------------------
``summary.json`` is written ONCE, at run END.  A corpus that is still running,
was killed, or is between ``--resume`` sessions has none -- possibly for days --
and every fact this tool needs was nonetheless banked at launch and as the run
went along.  So there are two records, and which one was read is stamped:

``summary``
    ``summary.json`` exists.  It is the complete, authoritative record: the
    shard inventory is checked against the disk BOTH ways (nothing missing,
    nothing unnamed), and ``config_realized_by_worker`` lets the cp map be
    cross-checked against what each worker's searcher actually converted with.

``manifest+progress``
    No ``summary.json``.  ``manifest.json`` (banked before the first game) gives
    ``config_sha256``, ``config_requested`` -- and with it the cp map -- and
    ``staircase_parsed``; the per-worker ``w<id>.progress.jsonl`` files give the
    shard inventory, one line per CLOSED shard.  ⚑ This record is genuinely
    WEAKER and the output says so rather than reading like a whole corpus:
    ``corpus.corpus_record`` names the mode, ``corpus_record_detail`` counts the
    shards and rows adopted, and it NAMES the facts a manifest cannot carry --
    ``config_realized_by_worker`` above all, so the cp cross-check that a summary
    run performs is reported as having covered ZERO workers here rather than
    quietly not happening.  A fact that lives only in a summary is reported
    missing, never guessed.

⚑ A LIVE CORPUS IS A MOVING TARGET, SO THE INVENTORY IS A SNAPSHOT.  The
progress files are read ONCE, before the first row, and the derivation runs
against exactly that list.  Shards that close while it runs are neither picked
up nor an error -- a run whose input grew halfway through would have a row count
that no later reading of the corpus could reproduce.  Shards on disk that the
snapshot does not name (the in-flight shard every live worker is holding open)
are COUNTED in the output and never read: a shard is listed only once it is
closed, and reading a file still being appended to is reading a truncated JSONL.
⚑ The reader is ``gen_sf_rooted_corpus.read_worker_progress``, imported rather
than reimplemented, so its torn-tail tolerance (a ``kill -9`` can cut the LAST
line short of its newline, and nothing else) is the same tolerance the resume
path applies -- a second decoder is how a format drifts from its writer.

WHAT IS SHARED RATHER THAN RESTATED
-----------------------------------
The corpus schema, the populated-directory refusal and the codec probe are
``gen_sf_rooted_corpus``'s, imported.  The cp->value mapping is the label gate's
(above).  The move -> compact-1858 map is ``moves.leela_index``'s, the same one
``lc0_data_to_rows`` uses.  The shard writer is ``replay.shard``'s, so these
files are byte-compatible with the rig's existing corpus by construction rather
than by imitation.  ⚑ No torch: the mapping needed here is move -> compact 1858,
which ``moves.leela_index`` provides off module-level tables, and the device
tensors in ``moves/torch_maps.py`` convert compact <-> AZ-4672, which this path
never does.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    local_shard_path,
    samples_to_arrays,
    save_local_shard_arrays,
)
from scripts import audit_label_candidates as gate
from scripts import gen_random_selfplay_shards as gen
from scripts import gen_sf_rooted_corpus as corpus

#: Derived-shard schema.  Bumped when the MEANING of an emitted column changes,
#: which is a different event from the corpus row schema changing -- a consumer
#: needs both numbers to know what it is holding, so both are stamped.
DERIVE_SCHEMA = 1

#: Pinned, not a flag.  The rig merges these shards into ONE ``DiskReplayBuffer``
#: alongside whatever else is named in ``--shards``, and the buffer hard-fails on
#: mixed encoding identity -- so a per-run knob here would turn a typo into an
#: unmergeable corpus, and a wrong-but-mergeable value into 146 planes fed to a
#: 175-plane net.  Changing these is a code edit, deliberately.
INPUT_HISTORY_ENCODING = "lc0_root_legacy_meta"
INPUT_EXTRA_FEATURES = "v2_threats"

#: The rows are encoded by TODAY's encoder, which applies the repetition-plane
#: fix unconditionally.  Vacuous on a zero-history row (nothing can repeat) and
#: stamped anyway, because it is replay IDENTITY: the same encoding name with a
#: different flag is a different plane set and the buffer refuses to mix them.
HISTORY_REP_FIX = True

#: 8192 rows x 175 planes x 64 squares of float16 is ~184 MB on disk, the same
#: rotation ``lc0_data_to_rows`` uses for the same reason.
DEFAULT_ROWS_PER_SHARD = 8192

DEFAULT_SEED = 20260827

#: ``--floor`` off.  The identity: ``(1 - 0 * n) * softmax + 0`` is the softmax,
#: and ``apply_floor`` short-circuits so a zero-floor run's bytes are the ones
#: this tool wrote before the flag existed rather than bytes that happen to
#: round the same way.
DEFAULT_FLOOR = 0.0

#: The most legal moves a chess position can have: 218, the constructed maximum
#: (R6R/2pbpppp/pppppppp/... family), and the bound python-chess's own move
#: generator cannot exceed on a legal position.  Used ONLY to turn the per-row
#: constraint ``floor * n_legal < 1`` into a startup refusal that no corpus can
#: sneak past -- see the module docstring on why the floor is never a per-row
#: drop.  ⚑ A bound, not a measurement: ``apply_floor`` re-checks the real
#: ``n_legal`` per row so the bound is falsifiable rather than assumed.
MAX_LEGAL_MOVES = 218

#: ``ShardMeta.run_id`` for every shard this tool writes.  The scheme is NOT
#: folded into it: the scheme, its parameters and this file's schema go into
#: their own zarr attrs (see ``_stamp_shard_attrs``) where a reader can parse
#: them, rather than into a string it would have to take apart.
SHARD_RUN_ID = "sf_rooted_corpus_targets"

SUMMARY_NAME = "derive_targets_summary.json"

#: Which of the corpus's two own records supplied its facts.  Stamped into every
#: derived manifest; see the module docstring.
CORPUS_RECORD_SUMMARY = "summary"
CORPUS_RECORD_PARTIAL = "manifest+progress"

#: The per-worker incremental inventory files, as a glob.  ⚑ Shaped to match
#: ``gen_sf_rooted_corpus.progress_name`` and nothing else: the shard files sit
#: in the same directory and a looser pattern would read one as an inventory.
PROGRESS_GLOB = "w*.progress.jsonl"

#: Below this q spread a row says NOTHING about the temperature, and the tau
#: ``recover_temp`` computes from it is float64 rounding rather than a reading.
#:
#: ⚑ MEASURED, not guessed at.  A totally won (or lost) position saturates EVERY
#: legal move's q at ±1.0, and the SMALLEST NONZERO spread the cp map can then
#: produce is **1.11e-16 -- exactly one float64 ULP below 1.0** (swept over
#: cp 1000..8000 at the production slope 0.006 / draw width 120).  ``recover_temp``
#: divides that gap by a log-probability difference computed at the same
#: precision, so the answer is quantisation: the same 1.11e-16 row reads tau
#: 1.0 at 2 legal moves, 0.5 at 5, ``None`` at 9 and 0.25 at 20 -- a function of
#: the MOVE COUNT, not of the temperature.  That is the mechanism behind the
#: dry-run REPORT this fix answers -- 7 saturated rows in 240k stamping
#: ``min=0.625 max=1.007`` on an otherwise healthy ``--temp 1.0`` run.  The
#: 1.11e-16 sweep and the per-move-count taus above are measurements taken here;
#: the 7-in-240k is that earlier report, quoted as its origin.
#:
#: 1e-9 is where tau stops being quantisation and starts being a reading: the
#: absolute error in ``log p_hi - log p_lo`` is ~4.4e-16, so the relative error
#: in ``tau = gap / log_gap`` is ~4.4e-16 / gap -- about 4e-7 at a spread of
#: 1e-9, and >100% at 1e-16.  It is ~4.5 million ULPs clear of the saturation
#: floor and orders of magnitude below the spread of any row whose moves differ
#: by even one centipawn, so it separates the two populations without touching a
#: real reading.  ⚑ A row whose moves are EXACTLY equal never reaches this test:
#: ``recover_temp`` already returns ``None`` on a zero gap, and such a row is
#: counted in neither the reading nor the skip.
#:
#: ⚑ DELIBERATELY CONSERVATIVE, and the cost is measured rather than assumed:
#: over the first 250,000 rows of the live ``run02`` corpus this holds out
#: **2 rows** (211,755 kept, 14,811 flat), whose spreads were 1.6e-12 and
#: 1.1e-11 -- rows whose tau was in fact still good to ~4 decimals.  Holding out
#: a handful of harmless rows costs a slightly smaller ``n``; admitting one
#: pathological row visibly corrupts a min/max that a reader uses to decide
#: whether ``--temp`` was applied at all.  The asymmetry is why the threshold
#: sits well above the failure rather than tight against it.
#: ⚑ It gates the STAMP only.  The row's targets are derived from its q exactly
#: as before -- a saturated position is a real position and its policy is a real
#: (flat) policy; what is refused is quoting a temperature read off it.
TEMP_RECOVERY_MIN_Q_SPREAD = 1e-9

#: The floored estimator needs THREE distinct values: two determine a ratio the
#: floor cancels out of only if there is a third point to form the second
#: difference with.  A row with fewer says nothing about (tau, floor) jointly --
#: with two distinct values the model has two unknowns and one independent
#: equation -- so it is skipped and COUNTED rather than fitted.
FLOOR_RECOVERY_MIN_DISTINCT_VALUES = 3

#: The conditioning gate on the floored estimator, and the reason it is a
#: RELATIVE one.  Both differences it divides are catastrophic cancellations
#: when the two probabilities are pinned at the floor: at ``--temp 0.0145`` a
#: move 0.6 of q behind the best contributes ~1e-18 on top of a floor of 2e-3,
#: and the float64 ULP of 2e-3 is 4.3e-19 -- so ``p_mid - p_lo`` comes out
#: strictly positive and is nonetheless two ULPs of rounding noise.  ⚑ MEASURED: over 20,000 random (n, q, temp, floor) rows
#: this gate holds out 77 and drops the worst recovered-tau error from
#: **1.9e-3 relative to 2.1e-9**, with the worst recovered floor at 1.1e-12
#: absolute; on 2,000 production-shaped rows (cp ramps, temp 0.04, floor 0.002)
#: it holds out NONE and the worst errors are 5.5e-11 (tau) and 9.5e-13
#: (floor).  1e-9 leaves ~7 significant digits in the smaller difference, and
#: the tau error is damped from there rather than amplified (``dtau/tau =
#: (tau/gap) * dR/R``).
FLOOR_RECOVERY_MIN_REL_GAP = 1e-9

#: The bracket the floored estimator bisects tau in, and its stopping width.
#: ``R(tau)`` is strictly decreasing from +inf to ``gap_hi / gap_lo``, so a
#: bracket check is a real test: a row whose observed ratio is not inside that
#: range is NOT fitted to the nearest end, it is skipped and counted.  The
#: bracket spans 12 decades around every temperature this tool will be asked
#: for; bisection is in LOG tau, so the width is relative.
FLOOR_RECOVERY_TEMP_BRACKET = (1e-8, 1e4)
FLOOR_RECOVERY_TEMP_RTOL = 1e-13

#: ⚑⚑ THE IDENTIFIABILITY GATE, in units of the RECOVERED temperature.  When
#: ``q_spread / tau`` is small the softmax is in its linear regime and the
#: emitted row carries only TWO numbers (a slope and an offset), so no
#: arithmetic can read (tau, floor) jointly off it -- the three-point ratio
#: sits on ``R(tau)``'s flat hot tail and the inverse returns whatever the
#: rounding noise picks (found independently by two reviewers of PR #486;
#: reproduced: spread 1e-9 at true (0.067, 0.002) recovered (0.0398, 0.035),
#: and a cp-3000-to-3200 all-moves-winning row, spread 6.5e-8, recovered
#: (0.087, -0.022)).  ⚑ The gate is on spread PER TAU, not raw spread, because
#: the linear regime is set by their ratio: MEASURED on 12-move linspace rows,
#: recovered-tau relative error scales as ``~1e-15 / (spread/tau)^2`` --
#: 4.3e-2 at spread/tau 1.5e-7, 7.4e-4 at 1.5e-6, 2.3e-7 at 1.5e-4 -- so at
#: this threshold the surviving readings are exact to ~1e-7 relative across
#: tau 0.02..1.0.  Rows under it are counted ill-conditioned, and their
#: targets are still derived and written exactly as before.
FLOOR_RECOVERY_MIN_SPREAD_PER_TAU = 1e-4

#: How far below zero a recovered floor may read before the reading is refused
#: as ill-conditioned rather than stamped.  A real emission cannot carry a
#: negative floor (``validate_floor`` refuses it at startup), so a
#: substantially negative reading is always the arithmetic failing -- but the
#: take-effect proof RELIES on an unapplied floor reading back ~0, and an
#: honest zero comes back as rounding residue of either sign (measured
#: |floor| <= 7e-13 over 20k unfloored rows).  The tolerance sits ~6 orders
#: above that residue and ~3 below any floor anyone would request.
FLOOR_RECOVERY_FLOOR_TOL = 1e-6

#: Which phases a scheme is allowed to read a move's value from.
VALUE_SOURCE_DEEPEST = "deepest_phase_covering"
VALUE_SOURCE_PHASE0 = "phase0_only"

#: ``lc0_root_legacy_meta`` history slots and the piece planes in each.  Used
#: only to MEASURE how many slots the encoder actually filled.
_HISTORY_SLOTS = 8
_PLANES_PER_SLOT = 13
_PIECE_PLANES_PER_SLOT = 12

_SCHEME_UNIFORM = re.compile(r"^uniform-d(\d+)$")
_SCHEME_TOPK = re.compile(r"^top(\d+)-d(\d+)-rest-d(\d+)$")
_SCHEME_NODES = re.compile(r"^nodes-(\d+)$")

_SCHEME_FORMS = (
    "uniform-d<D>",
    "top<K>-d<D2>-rest-d<D1>",
    "nodes-<N>",
)


class CorpusIntegrityError(RuntimeError):
    """The corpus is not what its own summary or row schema says it is."""


class EnvelopeMiss(RuntimeError):
    """ONE row's bank cannot answer the scheme's question.

    Distinct from :class:`CorpusIntegrityError`: the corpus is well formed, this
    row simply does not carry the block the scheme asks for (an aborted search,
    or a narrowing the generator did not run).  Whether that ends the run is the
    operator's call via ``--max-envelope-misses``; it is never silent.
    """


# -- the scheme ---------------------------------------------------------------


@dataclass(frozen=True)
class Scheme:
    """A parsed target scheme.  ``canonical`` is re-derived, never echoed."""

    kind: str
    #: uniform: the depth.  topk: D1, the depth the REST are read at.
    depth: int | None = None
    #: topk only: D2, the depth the top K are read at.
    deep_depth: int | None = None
    top_k: int | None = None
    nodes: int | None = None

    @property
    def canonical(self) -> str:
        """The scheme's spelling, rebuilt from the PARSED fields.

        Stamped instead of ``args.scheme`` so a stamp that agrees with the flag
        is evidence the parse agreed with it too, rather than a copy of the
        string that would match whatever the parser had decided.
        """
        if self.kind == "uniform":
            return f"uniform-d{self.depth}"
        if self.kind == "topk":
            return f"top{self.top_k}-d{self.deep_depth}-rest-d{self.depth}"
        return f"nodes-{self.nodes}"

    @property
    def value_source(self) -> str:
        return VALUE_SOURCE_PHASE0 if self.kind == "nodes" else VALUE_SOURCE_DEEPEST

    def params(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "deep_depth": self.deep_depth,
            "top_k": self.top_k,
            "nodes": self.nodes,
            "value_source": self.value_source,
        }


def parse_scheme(spec: str) -> Scheme:
    """``"top4-d13-rest-d9"`` -> a :class:`Scheme`, or ``ValueError``.

    The shape rules are the staircase's own, for the staircase's own reasons:

    * ``D2`` must strictly DEEPEN ``D1``.  ``top4-d9-rest-d9`` reads every move
      at one depth while claiming a two-tier read, and ``top4-d5-rest-d9`` would
      spend the "deep" tier on a shallower block than the rest -- a narrowing
      that makes the top moves the LEAST informed, which no consumer wants and
      no name here would disclose.
    * every depth, ``K`` and ``N`` is positive.  ``depth 0`` is the trap
      ``parse_staircase`` documents (Stockfish silently replaces ``go depth 0``
      with a real iteration), and here it would simply select no block.
    """
    text = spec.strip()
    match = _SCHEME_UNIFORM.match(text)
    if match:
        depth = int(match.group(1))
        _require_positive(text, depth=depth)
        return Scheme(kind="uniform", depth=depth)
    match = _SCHEME_TOPK.match(text)
    if match:
        top_k, deep, base = (int(match.group(i)) for i in (1, 2, 3))
        _require_positive(text, top_k=top_k, deep_depth=deep, depth=base)
        if deep <= base:
            raise ValueError(
                f"--scheme {text!r}: the top-K depth {deep} must strictly deepen "
                f"the rest depth {base}; a scheme whose 'deep' tier is not deeper "
                "is a uniform read wearing a two-tier name",
            )
        return Scheme(kind="topk", depth=base, deep_depth=deep, top_k=top_k)
    match = _SCHEME_NODES.match(text)
    if match:
        nodes = int(match.group(1))
        _require_positive(text, nodes=nodes)
        return Scheme(kind="nodes", nodes=nodes)
    raise ValueError(
        f"--scheme {spec!r} is not one of {', '.join(_SCHEME_FORMS)}",
    )


def _require_positive(spec: str, **values: int) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"--scheme {spec!r}: {name} must be positive, got {value}")


def scheme_vs_staircase_problems(
    scheme: Scheme, staircase: Sequence[dict[str, Any]],
) -> list[str]:
    """Why this corpus's STAIRCASE cannot answer this scheme, before reading rows.

    A run-level refusal on top of the per-row one: ``--scheme uniform-d20``
    against a corpus whose deepest rung is 13 is knowable from ``summary.json``,
    and finding it out one row at a time would spend a full read to say it.
    """
    if not staircase:
        return ["the corpus summary carries no staircase_parsed; nothing to check"]
    full_width_depth = int(staircase[0]["depth"])
    deepest = max(int(rung["depth"]) for rung in staircase)
    problems: list[str] = []
    # The BASE depth must be reachable at FULL WIDTH -- every move needs a value
    # there -- so it is bounded by phase 0's rung, not by the deepest one.
    if scheme.depth is not None and int(scheme.depth) > full_width_depth:
        problems.append(
            f"the scheme's base depth {scheme.depth} exceeds the corpus "
            f"envelope: the staircase's full-width phase reaches depth "
            f"{full_width_depth}",
        )
    # The DEEP depth is only ever asked of a narrowed subset, so the deepest
    # rung is its bound.
    if scheme.deep_depth is not None and int(scheme.deep_depth) > deepest:
        problems.append(
            f"the scheme's deep depth {scheme.deep_depth} exceeds the corpus "
            f"envelope: the staircase's deepest rung reaches depth {deepest}",
        )
    return problems


# -- one row's bank -----------------------------------------------------------


@dataclass(frozen=True)
class MoveValues:
    """One value per move, and the provenance of every one of them.

    ``depth_by_move``/``phase_by_move`` are not diagnostics: a scheme that reads
    two tiers is only doing so if the emitted rows say which tier each move came
    from, and those two arrays are what the summary's histograms are built from.
    """

    moves: tuple[str, ...]
    effective_cp: np.ndarray
    depth_by_move: tuple[int, ...]
    phase_by_move: tuple[int, ...]
    base_depth: int
    floor_hit: bool

    @property
    def best_index(self) -> int:
        """The scheme's best move.  Ties go to the earliest banked rank."""
        return int(np.argmax(self.effective_cp))


class RowBank:
    """Every ``(phase, depth)`` block of ONE corpus row, indexed for reading.

    Built per row and thrown away; the corpus is streamed, so nothing here is
    allowed to grow with the run.
    """

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        # phase index -> depth -> {"complete": bool, "nodes": int|None,
        #                          "values": {move: cp}, "order": (move, ...)}
        self._blocks: list[dict[int, dict[str, Any]]] = []
        for phase in row["phases"]:
            by_depth: dict[int, dict[str, Any]] = {}
            for block in phase["per_depth"]:
                lines = block["lines"]
                by_depth[int(block["depth"])] = {
                    "complete": bool(block["complete"]),
                    "nodes": (
                        None if block["nodes_at_depth"] is None
                        else int(block["nodes_at_depth"])
                    ),
                    "values": {str(line[1]): float(line[2]) for line in lines},
                    "order": tuple(str(line[1]) for line in lines),
                }
            self._blocks.append(by_depth)

    @property
    def phase_count(self) -> int:
        return len(self._blocks)

    def full_width_block(self, depth: int) -> dict[str, Any] | None:
        """Phase 0's COMPLETE block at ``depth``, or None.

        Phase 0 is the only rung searched at ``MultiPV = legal move count`` with
        no ``searchmoves``, so it is the only one whose block defines "every
        move".  ``complete`` is required: an aborted iteration lists a subset of
        the ranks, and reading a subset as the move set would silently drop
        moves out of the policy's support with a legal mask that still names
        them.
        """
        if not self._blocks:
            return None
        block = self._blocks[0].get(int(depth))
        if block is None or not block["complete"]:
            return None
        return block

    def full_width_depths(self) -> list[int]:
        """The depths phase 0 banked a COMPLETE block at, ascending."""
        if not self._blocks:
            return []
        return sorted(d for d, b in self._blocks[0].items() if b["complete"])

    def value_at(self, move: str, depth: int) -> tuple[float, int] | None:
        """``(effective_cp, phase_index)`` from the DEEPEST phase carrying it."""
        for index in range(len(self._blocks) - 1, -1, -1):
            block = self._blocks[index].get(int(depth))
            if block is None:
                continue
            value = block["values"].get(move)
            if value is not None:
                return float(value), index
        return None

    def node_ladder(self) -> list[tuple[int, int]]:
        """Phase 0's ``(depth, cumulative nodes)`` rungs, ascending by depth.

        Only COMPLETE blocks that reported a node count take part: a block with
        no ``nodes_at_depth`` cannot be priced, and an incomplete one is not the
        iteration a ``go nodes`` budget would have bought.
        """
        if not self._blocks:
            return []
        return [
            (depth, int(block["nodes"]))
            for depth, block in sorted(self._blocks[0].items())
            if block["complete"] and block["nodes"] is not None
        ]

    def nodes_depth(self, budget: int) -> tuple[int, bool]:
        """``(depth, floor_hit)`` for a ``go nodes budget`` reconstruction."""
        ladder = self.node_ladder()
        if not ladder:
            raise EnvelopeMiss(
                "no complete phase-0 block carries a node count, so no node "
                "budget can be priced against this row",
            )
        affordable = [depth for depth, nodes in ladder if nodes <= int(budget)]
        if affordable:
            return max(affordable), False
        # ⚑ The FLOOR, and it is an event rather than a default: the budget did
        # not buy even the shallowest banked iteration, so the row is read at
        # that iteration and the run counts how often it had to.
        return ladder[0][0], True


def _required(value: int | None, name: str) -> int:
    """A scheme field the parser guarantees for this kind.

    ``ValueError`` rather than ``assert``: an assert is removed under ``-O`` and
    the failure would then be an index into ``None``.
    """
    if value is None:  # pragma: no cover - parse_scheme fills these per kind
        raise ValueError(f"scheme is missing {name}")
    return int(value)


def apply_scheme(bank: RowBank, scheme: Scheme) -> MoveValues:
    """Collapse one row's bank to one value per move under ``scheme``."""
    if scheme.kind == "nodes":
        depth, floor_hit = bank.nodes_depth(_required(scheme.nodes, "nodes"))
        block = bank.full_width_block(depth)
        if block is None:  # pragma: no cover - node_ladder only lists complete blocks
            raise EnvelopeMiss(f"phase 0 has no complete block at depth {depth}")
        moves = block["order"]
        return MoveValues(
            moves=moves,
            effective_cp=np.array(
                [block["values"][move] for move in moves], dtype=np.float64,
            ),
            depth_by_move=(depth,) * len(moves),
            phase_by_move=(0,) * len(moves),
            base_depth=depth,
            floor_hit=floor_hit,
        )

    base = _required(scheme.depth, "depth")
    block = bank.full_width_block(base)
    if block is None:
        raise EnvelopeMiss(
            f"phase 0 has no complete block at depth {base}; the row's "
            f"full-width envelope is {bank.full_width_depths()}",
        )
    moves = block["order"]
    values: list[float] = []
    phases: list[int] = []
    for move in moves:
        read = bank.value_at(move, base)
        if read is None:  # pragma: no cover - phase 0 carries every move at base
            raise EnvelopeMiss(f"no banked value for {move} at depth {base}")
        values.append(read[0])
        phases.append(read[1])
    depths = [base] * len(moves)

    if scheme.kind == "topk":
        top_k = _required(scheme.top_k, "top_k")
        deep = _required(scheme.deep_depth, "deep_depth")
        # ⚑ The rank is by the D1 value with the uci as the tiebreak, so the
        # top-K set is a function of the BANK and not of iteration order. Using
        # Stockfish's own D1 rank as the tiebreak would tie this scheme to the
        # generator's narrowing, which is the very thing it exists to vary.
        order = sorted(range(len(moves)), key=lambda i: (-values[i], moves[i]))
        for index in order[:top_k]:
            read = bank.value_at(moves[index], deep)
            if read is None:
                raise EnvelopeMiss(
                    f"{moves[index]} is in the top {top_k} by depth "
                    f"{base} but no phase banked it at depth {deep}; the "
                    "generator's own narrowing did not carry it that far",
                )
            values[index], phases[index] = read[0], read[1]
            depths[index] = deep

    return MoveValues(
        moves=moves,
        effective_cp=np.asarray(values, dtype=np.float64),
        depth_by_move=tuple(depths),
        phase_by_move=tuple(phases),
        base_depth=base,
        floor_hit=False,
    )


# -- targets ------------------------------------------------------------------


def validate_temp(temp: float) -> float:
    """⚑ A non-positive or non-finite temperature is REFUSED, not taken to its limit.

    Exactly as ``gen_sf_rooted_corpus.gumbel_choice`` refuses it: the limit is a
    one-hot target, and silently emitting one would turn a decimal typo into a
    corpus whose policy carries no distributional information at all and no
    column that says so.  Called from ``main`` as well as from the softmax, so a
    bad ``--temp`` is refused before the first shard is read rather than on the
    first row that happens to reach the target builder.
    """
    tau = float(temp)
    if not tau > 0.0 or not math.isfinite(tau):
        raise ValueError(
            f"--temp must be finite and positive, got {temp!r}: the limit is a "
            "one-hot target, which is a different experiment",
        )
    return tau


def validate_floor(floor: float) -> float:
    """⚑ A floor is refused AT STARTUP or not at all -- never per row.

    Two refusals, both before the first shard is opened (``main`` calls this
    next to ``validate_temp``):

    * negative or non-finite -- a floor is a probability MASS per legal move,
      and a negative one would subtract mass from the tail it exists to
      protect while still summing to 1, so nothing downstream could see it;
    * ``floor * MAX_LEGAL_MOVES >= 1`` -- the head's coefficient
      ``1 - floor * n_legal`` must stay positive for the emitted row to be a
      distribution with the scheme's argmax on top, and 218 is the most legal
      moves any position can have;
    * a positive floor the SHARD CANNOT STORE -- the trainer reads float16
      (through float32, see ``_note_shapes``), and a floor below half of
      float16's smallest subnormal (2**-25 ~ 3e-8) serializes to exactly 0 on
      every cold tail, so the CLI would accept a flag, every stamp would echo
      it, and the trainer would see nothing.  Refused with the storage math in
      the message rather than documented as a caveat.

    ⚑ The second bound is CONSERVATIVE ON PURPOSE and the alternative was
    considered and rejected: refusing (or dropping) per row would make the
    emitted row set a function of ``--floor``, so two arms of a floor ladder
    would differ in which positions they contain as well as in their targets.
    See the module docstring.  The cost is that a floor between 1/218 and the
    row-wise limit is refused even though most rows could carry it; the arm
    this exists for is 0.002, which is 2.3x below the bound.
    """
    value = float(floor)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"--floor must be finite and >= 0, got {floor!r}: it is the mass "
            "every legal move keeps, and 0 is the off position",
        )
    if value * MAX_LEGAL_MOVES >= 1.0:
        raise ValueError(
            f"--floor {value!r} leaves no head: a position may have up to "
            f"{MAX_LEGAL_MOVES} legal moves, and {value!r} * {MAX_LEGAL_MOVES} "
            f"= {value * MAX_LEGAL_MOVES:.6g} >= 1 would drive the softmax's "
            f"coefficient (1 - floor * n_legal) to zero or below. The bound is "
            f"the chess-theoretic maximum rather than this corpus's, so which "
            f"floors are legal does not depend on which rows a corpus holds.",
        )
    stored = float(shard_stored(np.asarray([value], dtype=np.float64))[0])
    if value > 0.0 and stored <= 0.0:
        raise ValueError(
            f"--floor {value!r} vanishes in shard storage: the trainer reads "
            f"policy as float16 (via float32, see shard_stored), and this "
            f"value serializes to {stored!r} there, so every cold tail the "
            "floor exists to protect would reach the trainer as exactly zero "
            "while the flag and the stamps say otherwise. The smallest "
            "storable floor is float16's smallest subnormal, 2**-24 ~ 6e-8.",
        )
    return value


def softmax_at_temp(q: np.ndarray, *, temp: float) -> np.ndarray:
    """``softmax(q / temp)`` in float64, max-shifted."""
    tau = validate_temp(temp)
    scaled = np.asarray(q, dtype=np.float64) / tau
    shifted = np.exp(scaled - float(np.max(scaled)))
    return shifted / float(shifted.sum())


def shard_stored(values: np.ndarray) -> np.ndarray:
    """The values AS THE TRAINER WILL READ THEM: float32, then float16.

    ⚑ Two steps because the shard path is two steps -- ``sample_from_row``
    stores float32 and ``samples_to_arrays`` casts float16 -- and double
    rounding differs from a direct float64->float16 cast just above the
    half-way values: ``2**-25 * (1 + 2**-30)`` is above the tie, so the direct
    cast rounds UP to float16's smallest subnormal, but float32 first rounds
    it DOWN onto the tie exactly, and the tie then goes to even -- ZERO.
    Every stamp that claims to speak for the stored bytes goes through this
    function, so the claim and the storage cannot use different arithmetic.
    """
    return np.asarray(values).astype(np.float32).astype(np.float16)


def apply_floor(probs: np.ndarray, *, floor: float, n_legal: int) -> np.ndarray:
    """``(1 - floor * n_legal) * probs + floor``: the uniform exploration floor.

    Mass-preserving by construction -- the head is scaled by exactly what the
    floor adds -- and order-preserving, since it is an affine map with a
    positive coefficient, so the scheme's best move is still the argmax.  The
    smallest emitted probability is ``floor`` itself (IEEE addition of a
    non-negative to ``floor`` cannot round below it), which is what makes the
    coarse take-effect stamp a real reading.

    ⚑ ``floor == 0`` returns ``probs`` UNCHANGED rather than computing the
    identity, so a zero-floor run is the old behaviour by construction and not
    by a rounding argument.
    """
    value = validate_floor(floor)
    if value == 0.0:
        return probs
    if value * int(n_legal) >= 1.0:
        # Unreachable while `validate_floor`'s startup bound holds, and that is
        # the point: it is the bound's FALSIFIER, not a second gate. A position
        # with more than MAX_LEGAL_MOVES legal moves would make the emitted row
        # silently non-monotone, so this is fatal rather than a dropped row --
        # dropping would make the row set depend on the flag.
        raise CorpusIntegrityError(
            f"--floor {value!r} on a row with {int(n_legal)} legal moves leaves "
            f"the head coefficient {1.0 - value * int(n_legal):.6g}; "
            f"MAX_LEGAL_MOVES ({MAX_LEGAL_MOVES}) is supposed to bound this at "
            "startup, so this row falsifies that bound",
        )
    return (1.0 - value * int(n_legal)) * np.asarray(probs, dtype=np.float64) + value


def recover_temp(q: np.ndarray, probs: np.ndarray) -> float | None:
    """The temperature READ BACK off an emitted policy row, or None.

    ⚑ This is the realized stamp for ``--temp``, and it is realized because it
    is computed from the OUTPUT: for any two moves,
    ``log p_i - log p_j == (q_i - q_j) / tau``, so the emitted distribution and
    the values it came from determine tau with nothing echoed from the flag.
    A knob that was parsed, stored and then not applied cannot survive it.
    ⚑ It reads the DERIVED float64 distribution, before the shard's float16
    cast, so it certifies the computation rather than the storage; what the
    cast costs is reported separately as ``policy_support_lost_to_float16``.
    Returns None when every value is equal (the distribution is uniform at every
    temperature, so the row carries no information about tau).
    """
    values = np.asarray(q, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    usable = p > 0.0
    if int(usable.sum()) < 2:
        return None
    values, p = values[usable], p[usable]
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    gap = float(values[hi] - values[lo])
    if gap <= 0.0:
        return None
    log_gap = float(np.log(p[hi]) - np.log(p[lo]))
    if log_gap <= 0.0:
        return None
    return gap / log_gap


def distinct_values(q: np.ndarray) -> int:
    """How many DISTINCT values a row's move set carries.

    The floored estimator's own precondition, split out so the caller can count
    the rows it holds out for this reason separately from the ones it holds out
    for conditioning -- "the row could not say" and "the arithmetic could not
    say" are different facts about a corpus.
    """
    return int(np.unique(np.asarray(q, dtype=np.float64)).size)


def _log_emitted_ratio(gap_hi: float, gap_lo: float, temp: float) -> float:
    """``log R(tau)`` for ``R = (e^(a/t) - e^(b/t)) / (e^(b/t) - e^(c/t))``.

    Written in terms of the two GAPS ``a-b`` and ``b-c`` (the shared
    ``e^(b/t)`` and the softmax's normaliser both cancel), and evaluated
    through ``log1p``/``expm1`` so it neither overflows at a cold temperature
    nor cancels at a hot one: ``log(e^x - 1)`` is ``x + log1p(-e^-x)`` once
    ``x > 1``, which is exact for any x the bracket can reach.
    """
    x, y = gap_hi / temp, gap_lo / temp
    high = x + math.log1p(-math.exp(-x)) if x > 1.0 else math.log(math.expm1(x))
    return high - math.log(-math.expm1(-y))


def _solve_emitted_ratio_for_temp(
    gap_hi: float, gap_lo: float, ratio: float,
) -> float | None:
    """Invert :func:`_log_emitted_ratio` for tau, or None when out of bracket.

    ``R(tau)`` falls strictly from +inf (as tau -> 0, where the best move takes
    everything) to ``gap_hi / gap_lo`` (as tau -> inf, where the softmax is
    uniform and the differences go linear), so the inverse is unique and plain
    bisection in log tau is the whole solver.  ⚑ The bracket check is a
    MEASUREMENT, not a formality: a ratio outside the range no temperature can
    produce means the row's arithmetic has broken down, and returning None
    there is the difference between a skipped row and a row fitted to the
    nearest endpoint.
    """
    low, high = FLOOR_RECOVERY_TEMP_BRACKET
    target = math.log(ratio)
    if not _log_emitted_ratio(gap_hi, gap_lo, low) - target > 0.0:
        return None
    if not _log_emitted_ratio(gap_hi, gap_lo, high) - target < 0.0:
        return None
    for _ in range(200):
        mid = math.sqrt(low * high)
        if _log_emitted_ratio(gap_hi, gap_lo, mid) - target > 0.0:
            low = mid
        else:
            high = mid
        if high - low <= FLOOR_RECOVERY_TEMP_RTOL * high:
            break
    return math.sqrt(low * high)


def recover_floor_and_temp(
    q: np.ndarray, probs: np.ndarray, *, n_legal: int,
) -> tuple[float, float] | None:
    """(tau, floor) READ BACK off a floored policy row, or None.

    The realized stamp for ``--temp`` AND ``--floor`` when both are active, and
    like :func:`recover_temp` it echoes NEITHER flag -- it is a function of the
    emitted probabilities and the values they were built from:

    * a floor is an ADDITIVE constant, so it cancels out of differences:
      ``p_i - p_j == (1 - floor * n) * (s_i - s_j)``.  The ratio of two such
      differences drops the scale as well, leaving
      ``(p_hi - p_mid) / (p_mid - p_lo)`` a function of tau alone, which
      :func:`_solve_emitted_ratio_for_temp` inverts;
    * with tau in hand ``s`` is known, ``(p_hi - p_lo) / (s_hi - s_lo)`` is the
      scale ``1 - floor * n_legal``, and the floor follows.  A ``--floor`` that
      was parsed and never applied reads back 0 (measured: -5.7e-16 on an
      unfloored row), which is what makes this a take-effect proof rather than
      a restatement of the flag.

    The three points are the largest, second-largest and smallest DISTINCT
    values; ties share a probability, so any representative index does.
    Returns None when the row cannot support the reading -- fewer than
    ``FLOOR_RECOVERY_MIN_DISTINCT_VALUES`` distinct values, a difference at or
    below ``FLOOR_RECOVERY_MIN_REL_GAP`` of the probability it came out of, a
    ratio outside the bracket, a q spread below
    ``FLOOR_RECOVERY_MIN_SPREAD_PER_TAU`` of the recovered tau (the softmax's
    linear regime, where the row carries two numbers and cannot determine
    three -- fitting there stamps noise, see the constant), or a recovered
    floor outside ``[-FLOOR_RECOVERY_FLOOR_TOL, 1/n_legal)`` (no real emission
    can carry either, so such a reading is the arithmetic failing, not the
    row).  ⚑ Like ``recover_temp`` it reads the derived
    float64 distribution, so it certifies the computation; what the shard's
    float16 cast leaves is stamped separately as ``policy_min_legal_prob_stored``.
    """
    values = np.asarray(q, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if values.size != p.size or values.size < FLOOR_RECOVERY_MIN_DISTINCT_VALUES:
        return None
    unique = np.unique(values)
    if unique.size < FLOOR_RECOVERY_MIN_DISTINCT_VALUES:
        return None
    top, second, bottom = float(unique[-1]), float(unique[-2]), float(unique[0])
    hi = int(np.argmax(values == top))
    mid = int(np.argmax(values == second))
    lo = int(np.argmax(values == bottom))
    head, tail = float(p[hi] - p[mid]), float(p[mid] - p[lo])
    if head <= FLOOR_RECOVERY_MIN_REL_GAP * float(p[hi]):
        return None
    if tail <= FLOOR_RECOVERY_MIN_REL_GAP * float(p[mid]):
        return None
    tau = _solve_emitted_ratio_for_temp(top - second, second - bottom, head / tail)
    if tau is None:
        return None
    if float(top - bottom) < FLOOR_RECOVERY_MIN_SPREAD_PER_TAU * tau:
        return None
    reference = softmax_at_temp(values, temp=tau)
    span = float(reference[hi] - reference[lo])
    if span <= 0.0:
        return None
    scale = float(p[hi] - p[lo]) / span
    floor = (1.0 - scale) / float(n_legal)
    if floor < -FLOOR_RECOVERY_FLOOR_TOL or floor * float(n_legal) >= 1.0:
        return None
    return tau, floor


def q_spread(q: np.ndarray) -> float:
    """``max(q) - min(q)``: how much a row can say about the temperature at all.

    The companion to :func:`recover_temp`.  ``recover_temp`` answers "what tau
    produced this policy"; this answers "could this row have told anyone", and
    the two are separate because a saturated row still gets a real (flat) target
    -- only the tau read off it is meaningless.  See
    ``TEMP_RECOVERY_MIN_Q_SPREAD``.
    """
    values = np.asarray(q, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.max(values) - np.min(values))


#: ``result_from_pov``'s ``+1 / 0 / -1`` (the ROW's own side-to-move seat) as the
#: shard's ``wdl_target``.  ⚑ Both halves of this mapping are POV claims and
#: neither is checkable from the number alone, which is why the corpus stores the
#: result already rotated and this table is the only other place a sign appears.
_RESULT_TO_WDL: dict[float, int] = {1.0: 0, 0.0: 1, -1.0: 2}


def wdl_target_from_result(result: float) -> int:
    """0=W / 1=D / 2=L from the row's own-POV game result."""
    key = float(result)
    if key not in _RESULT_TO_WDL:
        raise CorpusIntegrityError(
            f"row result {result!r} is not one of +1.0/0.0/-1.0; "
            "result_from_pov emits nothing else and a fourth value would mean "
            "the corpus was written by something other than the generator",
        )
    return _RESULT_TO_WDL[key]


def history_slots_filled(planes: np.ndarray) -> int:
    """How many of the 8 history slots carry a piece.  MEASURED, per row.

    The zero-history claim in this module's docstring is a claim about the
    ENCODER's behaviour on a stackless board, and the cheapest way to keep it
    honest is to read it off the planes that were written.
    """
    filled = 0
    for slot in range(_HISTORY_SLOTS):
        start = slot * _PLANES_PER_SLOT
        if bool(np.any(planes[start : start + _PIECE_PLANES_PER_SLOT])):
            filled += 1
    return filled


# -- reading the corpus -------------------------------------------------------


def corpus_shard_paths(corpus_dir: Path) -> list[Path]:
    """The corpus's JSONL shards, found on DISK and sorted by name."""
    return [
        path
        for path in sorted(corpus_dir.iterdir())
        if path.name.endswith((".jsonl.zst", ".jsonl.gz"))
    ]


def check_shard_inventory(on_disk: Sequence[Path], summary: dict[str, Any]) -> None:
    """The shards present must be exactly the ones ``summary.json`` names.

    ⚑ BY BASENAME, because ``summary["shards"]`` stores the paths of the machine
    that produced the corpus and a corpus is routinely read from somewhere else.
    Comparing sets rather than counts is what catches the case this exists for:
    a partially copied corpus, whose missing shard would otherwise be a smaller
    training set that nothing named.
    """
    named = {Path(str(entry["path"])).name for entry in summary.get("shards", [])}
    if not named:
        # Refused, not skipped (review finding 2): the generator always writes
        # a real inventory, so an empty or missing `shards` list means a
        # damaged or foreign summary — and a lenient pass here is a gate that
        # silently does not fire, this codebase's signature defect.
        raise CorpusIntegrityError(
            "summary.json names no shards at all; the generator always records "
            "its inventory, so there is nothing to check the disk against and "
            "proceeding would train on whatever happens to be in the directory",
        )
    found = {path.name for path in on_disk}
    if named != found:
        missing = sorted(named - found)
        extra = sorted(found - named)
        raise CorpusIntegrityError(
            "the shards on disk are not the ones summary.json names: "
            f"missing {missing or 'none'}, unexpected {extra or 'none'}. A "
            "partially copied corpus would train on a subset nothing recorded.",
        )


def iter_corpus_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one corpus shard's rows, zstd or gzip."""
    if path.name.endswith(".jsonl.zst"):
        module = corpus.zstandard_module()
        if module is None:
            raise CorpusIntegrityError(
                f"{path.name} is zstd-compressed but the zstandard module is not "
                "importable in this environment",
            )
        with open(path, "rb") as binary:
            reader = module.ZstdDecompressor().stream_reader(binary)
            for line in _text_lines(reader):
                yield json.loads(line)
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _text_lines(reader: Any) -> Iterator[str]:
    with io.TextIOWrapper(reader, encoding="utf-8") as text:
        for line in text:
            if line.strip():
                yield line


# -- the run ------------------------------------------------------------------


@dataclass
class DeriveStats:
    """Every counter the summary reports.  All of them are events, not rates."""

    rows_read: int = 0
    rows_written: int = 0
    rows_dropped_no_result: int = 0
    rows_dropped_envelope: int = 0
    envelope_miss_examples: list[str] = field(default_factory=list)
    nodes_floor_hits: int = 0
    support_checks: int = 0
    depth_histogram: dict[int, int] = field(default_factory=dict)
    values_by_phase: dict[int, int] = field(default_factory=dict)
    deep_tier_moves: int = 0
    base_tier_moves: int = 0
    #: How many staircase rungs the ROWS carried, which is not necessarily how
    #: many ``staircase_parsed`` declares -- and ``values_by_phase`` is
    #: uninterpretable without it (a corpus of one-phase rows reads
    #: ``{"0": n}`` whether or not the scheme could ever have looked deeper).
    phases_per_row: dict[int, int] = field(default_factory=dict)
    history_slots_nonzero_max: int = 0
    repetition_planes_nonzero_rows: int = 0
    temp_recovered_n: int = 0
    temp_recovered_min: float = math.inf
    temp_recovered_max: float = -math.inf
    temp_recovered_sum: float = 0.0
    #: Rows whose q was SATURATED -- every legal move at ±1.0 to within a
    #: couple of ULPs, so the tau read back off them is rounding noise. Held
    #: out of the min/max/mean above and counted here instead; see
    #: ``TEMP_RECOVERY_MIN_Q_SPREAD``. ⚑ A count, not a suppression: saturated
    #: rows are a real feature of a corpus with won positions in it (2 in
    #: run02's first 250k), and a reader has to be able to see that the stamp
    #: above was computed WITHOUT them.
    temp_recovery_skipped_saturated: int = 0
    #: The floored estimator's readings (``--floor > 0`` runs only). ``n`` is
    #: its COVERAGE and is reported next to the two hold-out counters below, so
    #: a floor read off 3% of the rows can never be mistaken for one read off
    #: all of them.
    floor_recovered_n: int = 0
    floor_recovered_min: float = math.inf
    floor_recovered_max: float = -math.inf
    floor_recovered_sum: float = 0.0
    #: Rows with fewer than FLOOR_RECOVERY_MIN_DISTINCT_VALUES distinct values:
    #: the row itself cannot determine (tau, floor) jointly.
    floor_recovery_skipped_few_values: int = 0
    #: Rows whose emitted differences were below FLOOR_RECOVERY_MIN_REL_GAP,
    #: whose ratio fell outside the bracket, whose q spread was under
    #: FLOOR_RECOVERY_MIN_SPREAD_PER_TAU of the recovered tau (the softmax's
    #: linear regime -- two observable numbers cannot determine three), or
    #: whose recovered floor left [-FLOOR_RECOVERY_FLOOR_TOL, 1/n_legal): the
    #: arithmetic, not the row.  ⚑ Together with the saturation and
    #: few-values counters and `floor_recovered_n`, every written row is in
    #: exactly one bucket -- their sum reconstructs rows_written.
    floor_recovery_skipped_ill_conditioned: int = 0
    #: The COARSE floor stamp, and the only one that covers every written row:
    #: the smallest mass any legal move carries, measured AFTER the float16
    #: cast the shard stores (the same side of the cast as the support
    #: counters, and for the same reason -- a floor the trainer cannot read is
    #: not a floor).
    min_legal_prob_n: int = 0
    min_legal_prob_min: float = math.inf
    min_legal_prob_max: float = -math.inf
    min_legal_prob_sum: float = 0.0
    x_planes: int = 0
    policy_width: int = 0
    #: -1 until the first row is measured; a 0 sentinel would be a legal support.
    policy_support_min: int = -1
    policy_support_max: int = 0
    #: Legal moves whose probability survived the derivation but not the shard's
    #: float16 cast.  ⚑ Measured on the far side of the cast on purpose: a
    #: support counted in float64 is a number about a row nothing will ever read.
    policy_support_lost_to_float16: int = 0

    def note_temp(self, value: float) -> None:
        self.temp_recovered_n += 1
        self.temp_recovered_sum += value
        self.temp_recovered_min = min(self.temp_recovered_min, value)
        self.temp_recovered_max = max(self.temp_recovered_max, value)

    def note_floor(self, value: float) -> None:
        self.floor_recovered_n += 1
        self.floor_recovered_sum += value
        self.floor_recovered_min = min(self.floor_recovered_min, value)
        self.floor_recovered_max = max(self.floor_recovered_max, value)

    def note_min_legal_prob(self, value: float) -> None:
        self.min_legal_prob_n += 1
        self.min_legal_prob_sum += value
        self.min_legal_prob_min = min(self.min_legal_prob_min, value)
        self.min_legal_prob_max = max(self.min_legal_prob_max, value)

    def _reading(self, count: int, low: float, high: float, total: float) -> dict[str, Any]:
        """One recovered quantity as ``{n, min, max, mean}``, NaN when n == 0.

        ⚑ NaN rather than 0.0 for an empty reading: a floor stamped 0.0 by an
        estimator that never ran and a floor MEASURED at 0.0 are opposite
        facts, and 0.0 is a legal value of every quantity here.
        """
        return {
            "n": count,
            "min": low if count else math.nan,
            "max": high if count else math.nan,
            "mean": total / count if count else math.nan,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "rows_dropped_no_result": self.rows_dropped_no_result,
            "rows_dropped_envelope": self.rows_dropped_envelope,
            "envelope_miss_examples": list(self.envelope_miss_examples),
            "nodes_floor_hits": self.nodes_floor_hits,
            "realized_base_depth_histogram": {
                str(k): v for k, v in sorted(self.depth_histogram.items())
            },
            # ⚑ WHICH PHASE EACH VALUE CAME FROM. A `uniform-dD` corpus whose
            # values are all phase 0 and one whose top moves came from a
            # narrowed rung are different corpora with the same scheme name.
            "values_by_phase": {
                str(k): v for k, v in sorted(self.values_by_phase.items())
            },
            "deep_tier_moves": self.deep_tier_moves,
            "base_tier_moves": self.base_tier_moves,
            "phases_per_row": {
                str(k): v for k, v in sorted(self.phases_per_row.items())
            },
            "history_slots_nonzero_max": self.history_slots_nonzero_max,
            "repetition_planes_nonzero_rows": self.repetition_planes_nonzero_rows,
            "temp_recovered_from_emitted_policy": self._reading(
                self.temp_recovered_n,
                self.temp_recovered_min,
                self.temp_recovered_max,
                self.temp_recovered_sum,
            ),
            # ⚑ Reported SEPARATELY, at the same level as the reading it was
            # held out of: `n` above counts rows that could say something about
            # tau, this counts rows that could not, and the two together are
            # every row whose policy had two or more surviving moves.
            "temp_recovery_skipped_saturated": self.temp_recovery_skipped_saturated,
            "temp_recovery_saturation_q_spread_epsilon": TEMP_RECOVERY_MIN_Q_SPREAD,
            # ⚑ THE FLOOR, READ BACK OFF THE EMITTED ROWS. `n` here is the
            # floored estimator's coverage: it runs only under --floor > 0, and
            # only on rows that can carry it, so an unfloored run reads n=0 and
            # a floored one reads the floor it actually emitted -- 0 if the
            # flag was parsed and never applied.
            "floor_recovered_from_emitted_policy": self._reading(
                self.floor_recovered_n,
                self.floor_recovered_min,
                self.floor_recovered_max,
                self.floor_recovered_sum,
            ),
            "floor_recovery_skipped_few_values": self.floor_recovery_skipped_few_values,
            "floor_recovery_skipped_ill_conditioned": (
                self.floor_recovery_skipped_ill_conditioned
            ),
            "floor_recovery_min_distinct_values": FLOOR_RECOVERY_MIN_DISTINCT_VALUES,
            "floor_recovery_min_rel_gap": FLOOR_RECOVERY_MIN_REL_GAP,
            "floor_recovery_min_spread_per_tau": FLOOR_RECOVERY_MIN_SPREAD_PER_TAU,
            "floor_recovery_floor_tol": FLOOR_RECOVERY_FLOOR_TOL,
            # ⚑ EVERY WRITTEN ROW, unlike the estimator above: the smallest
            # mass any legal move carries after the shard's float32-then-
            # float16 cast. Within one float16 ULP of the floor on a floored
            # row -- the cast rounds to NEAREST, so roughly half of all floors
            # store just below the requested value (0.002 stores as
            # 0.00200081, but 0.0035 as 0.00349998) -- and the softmax's own
            # tail without one, which is orders of magnitude smaller or
            # exactly zero.
            "policy_min_legal_prob_stored": self._reading(
                self.min_legal_prob_n,
                self.min_legal_prob_min,
                self.min_legal_prob_max,
                self.min_legal_prob_sum,
            ),
            "x_planes": self.x_planes,
            "policy_width": self.policy_width,
            "policy_support_min": self.policy_support_min,
            "policy_support_max": self.policy_support_max,
            "policy_support_lost_to_float16": self.policy_support_lost_to_float16,
            "support_checks": self.support_checks,
        }


@dataclass(frozen=True)
class DeriveOptions:
    scheme: Scheme
    temp: float
    cp_slope: float
    cp_draw_width: float
    limit: int
    seed: int
    rows_per_shard: int
    max_envelope_misses: int
    #: ⚑ Defaulted, and it is the ONLY field here that is: 0.0 is the identity
    #: (``apply_floor`` short-circuits), so a caller that predates the flag
    #: keeps writing exactly the corpus it wrote before rather than failing to
    #: construct. Every OTHER field changes the targets when it changes, and
    #: none of them may be forgotten.
    floor: float = DEFAULT_FLOOR


class TargetDeriver:
    """Turns corpus rows into :class:`ReplaySample` rows under one scheme."""

    def __init__(self, options: DeriveOptions) -> None:
        self.options = options
        self.stats = DeriveStats()

    # -- the shared mapping ----------------------------------------------

    def q_of(self, effective_cp: np.ndarray) -> np.ndarray:
        """Effective cp -> q in [-1, 1], through the SHARED map.

        ``gate.q_from_effective_cp`` resolves ``gen.cp_to_wdl_array`` as a
        module attribute at call time, which is what makes this file, the
        generator's move selection and the label gate's arms ONE function
        object rather than three copies that agree today.
        """
        return gate.q_from_effective_cp(
            np.asarray(effective_cp, dtype=np.float64),
            slope=self.options.cp_slope,
            draw_width_cp=self.options.cp_draw_width,
        )

    def wdl_of(self, effective_cp: float) -> np.ndarray:
        """One effective cp -> (W, D, L), through the SAME object as ``q_of``."""
        wdl = gen.cp_to_wdl_array(
            np.asarray([float(effective_cp)], dtype=np.float64),
            slope=self.options.cp_slope,
            draw_width_cp=self.options.cp_draw_width,
        )
        return np.asarray(wdl, dtype=np.float32).reshape(-1)[:3]

    # -- one row ----------------------------------------------------------

    def sample_from_row(self, row: dict[str, Any]) -> ReplaySample | None:
        """One corpus row -> one replay row, or None when the row is dropped."""
        board = self._board_for(row)
        if row.get("result") is None:
            self.stats.rows_dropped_no_result += 1
            return None
        bank = RowBank(row)
        self.stats.phases_per_row[bank.phase_count] = (
            self.stats.phases_per_row.get(bank.phase_count, 0) + 1
        )
        values = apply_scheme(bank, self.options.scheme)
        if values.floor_hit:
            self.stats.nodes_floor_hits += 1
        self._check_support(board, values, row)

        q = self.q_of(values.effective_cp)
        probs = apply_floor(
            softmax_at_temp(q, temp=self.options.temp),
            floor=self.options.floor,
            n_legal=len(values.moves),
        )
        self._note_recovery(q, probs, n_legal=len(values.moves))

        policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float64)
        legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
        for move_uci, prob in zip(values.moves, probs):
            index = int(compact_index_for_move(board, chess.Move.from_uci(move_uci)))
            # ⚑ Range and COLLISION, both checked. A slot outside [0, 1858) is
            # the wrong index space (the AZ-4672 ids the search uses are the
            # near miss); a slot already claimed by another legal move is a
            # many-to-one map, which would fold two moves' mass onto one entry
            # and leave a legal mask that still names both. Neither shows up as
            # a shape error and neither changes the row's sum.
            if not 0 <= index < COMPACT_POLICY_SIZE:
                raise CorpusIntegrityError(
                    f"{_row_label(row)}: {move_uci} maps to policy slot {index}, "
                    f"outside the compact lc0_1858 space [0, {COMPACT_POLICY_SIZE})",
                )
            if legal_mask[index]:
                raise CorpusIntegrityError(
                    f"{_row_label(row)}: {move_uci} collides with an earlier "
                    f"legal move on policy slot {index}",
                )
            policy[index] = float(prob)
            legal_mask[index] = 1

        planes = self._encode(board)
        self._note_shapes(planes, policy, probs, values)

        return ReplaySample(
            x=planes,
            policy_target=policy.astype(np.float32),
            wdl_target=wdl_target_from_result(float(row["result"])),
            legal_mask=legal_mask,
            # ⚑ The SEARCHED value goes here and not in `sf_wdl`; see the module
            # docstring for the guard that makes `sf_wdl` unreachable on this rig.
            search_wdl=self.wdl_of(float(values.effective_cp[values.best_index])),
            has_policy=True,
            is_selfplay=True,
            is_network_turn=True,
            game_id=int(row["game_id"]),
            ply_index=int(row["ply"]),
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
        )

    def _note_recovery(
        self, q: np.ndarray, probs: np.ndarray, *, n_legal: int,
    ) -> None:
        """THE STAMPS ONLY -- ``probs`` is already this row's target either way.

        Which estimator runs is decided by ``--floor``, because the emitted
        policy is a different function under one: at floor 0 the closed-form
        ``recover_temp`` inverts it exactly, and above 0 it would return a
        temperature nobody asked for, so the joint
        :func:`recover_floor_and_temp` takes over and stamps both knobs.  ⚑ The
        saturation hold-out applies to BOTH: a position won outright pins every
        legal move's q at ±1.0 to within ULPs and neither estimator can read
        anything off it (see ``TEMP_RECOVERY_MIN_Q_SPREAD``).  The row is
        derived and written regardless -- what is refused is quoting a
        measurement taken off it.

        ⚑ ``temp_recovery_skipped_saturated`` COUNTS MORE ROWS UNDER A FLOOR,
        and that is an accounting difference rather than a corpus difference.
        The zero-floor path asks ``recover_temp`` first, and a row whose values
        are EXACTLY equal returns None there and is counted in neither the
        reading nor the skip -- the documented pre-floor behaviour.  The
        floored path tests the spread first, so those rows land in the skip.
        MEASURED on 20k rows of ``run02_snap_20260829`` at ``--temp 0.04``:
        1,333 skipped floored against 0 unfloored, and 1,333 is exactly the
        gap between the unfloored reading's n (17,270) and the rows written
        (18,603).  Under a floor every written row is in exactly one bucket;
        without one, flat rows are in none.

        ⚑ COST, since this runs per row: the joint estimator is ~113 us
        (bisection ~49 us of it, numpy call overhead most of the rest), and a
        floored 20k-row derivation of the production corpus measured 51.4 s
        against the same run unfloored at 45.2 s -- **+13.6% wall for the
        take-effect proof**. Stated rather than hidden; it is an offline
        derivation that runs once per arm.
        """
        stats = self.stats
        if self.options.floor <= 0.0:
            recovered = recover_temp(q, probs)
            if recovered is None:
                return
            if q_spread(q) < TEMP_RECOVERY_MIN_Q_SPREAD:
                stats.temp_recovery_skipped_saturated += 1
            else:
                stats.note_temp(recovered)
            return
        if q_spread(q) < TEMP_RECOVERY_MIN_Q_SPREAD:
            stats.temp_recovery_skipped_saturated += 1
            return
        if distinct_values(q) < FLOOR_RECOVERY_MIN_DISTINCT_VALUES:
            stats.floor_recovery_skipped_few_values += 1
            return
        reading = recover_floor_and_temp(q, probs, n_legal=n_legal)
        if reading is None:
            stats.floor_recovery_skipped_ill_conditioned += 1
            return
        tau, floor = reading
        stats.note_temp(tau)
        stats.note_floor(floor)

    def _board_for(self, row: dict[str, Any]) -> chess.Board:
        """The row's board, with the row's OWN metadata re-derived from it.

        ``stm`` and ``piece_count`` are banked alongside the FEN, so they are a
        free external check that the row is internally consistent -- and a
        disagreement means the fields a consumer would filter on describe a
        different position from the one it would encode.
        """
        board = chess.Board(str(row["fen"]))
        stm = "w" if board.turn == chess.WHITE else "b"
        if stm != str(row["stm"]):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: stm {row['stm']!r} "
                f"disagrees with the FEN's {stm!r}",
            )
        pieces = int(chess.popcount(board.occupied))
        if pieces != int(row["piece_count"]):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: piece_count "
                f"{row['piece_count']} disagrees with the FEN's {pieces}",
            )
        return board

    def _check_support(
        self, board: chess.Board, values: MoveValues, row: dict[str, Any],
    ) -> None:
        """The banked move set must be EXACTLY python-chess's legal moves.

        The external referee, and the same one ``lc0_data_to_rows`` uses:
        python-chess computes legality independently of anything in this repo,
        so a FEN that drifted from the search, a promotion spelled the other
        way, or a phase-0 block that is complete-but-narrow all show up here
        instead of as a policy whose support is quietly wrong.
        """
        legal = {move.uci() for move in board.legal_moves}
        banked = set(values.moves)
        if legal != banked:
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: the banked move set "
                f"is not the legal move set (only banked: {sorted(banked - legal)}; "
                f"only legal: {sorted(legal - banked)})",
            )
        if len(banked) != len(values.moves):
            raise CorpusIntegrityError(
                f"row {row.get('game_id')}/{row.get('ply')}: the banked block "
                "lists a move twice, which would fold two ranks onto one slot",
            )
        self.stats.support_checks += 1

    def _encode(self, board: chess.Board) -> np.ndarray:
        return np.asarray(
            encode_position(
                board,
                add_features=True,
                input_history_encoding=INPUT_HISTORY_ENCODING,
                input_extra_features=INPUT_EXTRA_FEATURES,
            ),
            dtype=np.float32,
        )

    def _note_shapes(
        self,
        planes: np.ndarray,
        policy: np.ndarray,
        probs: np.ndarray,
        values: MoveValues,
    ) -> None:
        stats = self.stats
        stats.x_planes = int(planes.shape[0])
        stats.policy_width = int(policy.shape[0])
        # ⚑ THE COARSE FLOOR READING, and it covers every written row. `probs`
        # is the row's legal moves and nothing else, so this is the smallest
        # mass a legal move carries -- taken on the far side of the cast the
        # shard stores, exactly like the support counters below, because a
        # floor the trainer reads as zero did not happen.  ⚑ THROUGH float32
        # FIRST: the shard path is float64 -> float32 (`sample_from_row`) ->
        # float16 (`samples_to_arrays`), and double rounding differs from the
        # direct cast exactly at the half-way values -- 2**-25 rounds directly
        # to float16's smallest subnormal but through float32 to ZERO, so a
        # one-step stamp could claim a tail the trainer reads as nothing.
        stats.note_min_legal_prob(float(shard_stored(probs).min()))
        # ⚑ AFTER the float16 cast the shard stores -- through float32 first,
        # same as the stamp above and for the same reason. A cold temperature
        # over a wide move list pushes the tail below float16's smallest
        # subnormal, and a support counted in float64 would report moves the
        # trainer will read as zero while the legal mask still names them.
        support = int((shard_stored(policy) > 0).sum())
        stats.policy_support_lost_to_float16 += int((policy > 0.0).sum()) - support
        stats.policy_support_min = (
            support if stats.policy_support_min < 0
            else min(stats.policy_support_min, support)
        )
        stats.policy_support_max = max(stats.policy_support_max, support)
        stats.history_slots_nonzero_max = max(
            stats.history_slots_nonzero_max, history_slots_filled(planes),
        )
        rep_planes = planes[
            [slot * _PLANES_PER_SLOT + _PIECE_PLANES_PER_SLOT
             for slot in range(_HISTORY_SLOTS)]
        ]
        if bool(np.any(rep_planes)):
            stats.repetition_planes_nonzero_rows += 1
        stats.depth_histogram[values.base_depth] = (
            stats.depth_histogram.get(values.base_depth, 0) + 1
        )
        for phase, depth in zip(values.phase_by_move, values.depth_by_move):
            stats.values_by_phase[phase] = stats.values_by_phase.get(phase, 0) + 1
            if depth == values.base_depth:
                stats.base_tier_moves += 1
            else:
                stats.deep_tier_moves += 1


# -- driving ------------------------------------------------------------------


def _row_label(row: dict[str, Any]) -> str:
    return f"game {row.get('game_id')} ply {row.get('ply')}"


def read_corpus_summary(corpus_dir: Path) -> dict[str, Any]:
    path = corpus_dir / corpus.SUMMARY_NAME
    if not path.exists():
        raise CorpusIntegrityError(
            f"{path} does not exist; a corpus without its summary carries no "
            "config_sha256, no staircase and no shard inventory, and every "
            "stamp this tool passes through would have to be invented",
        )
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    _check_row_schema(summary, source=path)
    return summary


def _check_row_schema(facts: Mapping[str, Any], *, source: Path) -> None:
    """Both records stamp ``row_schema``, and both are checked the same way."""
    row_schema = int(facts.get("row_schema", -1))
    if row_schema != corpus.ROW_SCHEMA:
        raise CorpusIntegrityError(
            f"corpus row schema {row_schema} != this build's "
            f"{corpus.ROW_SCHEMA} (read from {source.name}); the block keys "
            "this tool reads are not promised to mean the same thing across a "
            "schema bump",
        )


#: What ``manifest.json`` cannot carry, and what each absence costs.  ⚑ NAMED
#: rather than shrugged at: a partial derivation that reads like a complete one
#: is this repo's signature defect wearing a new hat, so the missing facts go
#: into the derived manifest by name instead of being quietly worked around.
FACTS_ONLY_IN_SUMMARY: dict[str, str] = {
    "config_realized_by_worker": (
        "the per-worker REALIZED cp map. summary mode cross-checks it against "
        "config_requested (cp_map.realized_workers_cross_checked says how many "
        "workers); a manifest is written before the first worker starts, so "
        "here that cross-check covers ZERO workers and the requested map is "
        "taken on trust"
    ),
    "search / dedup / terminations / opening_sources": (
        "per-run counters accumulated in memory and written once at the end. "
        "Not read by this tool, and not reconstructible from the shards"
    ),
    "rows / games": (
        "the corpus totals. The progress inventory's own per-shard row counts "
        "stand in (corpus_record_detail.rows_claimed_by_inventory), and they "
        "count CLOSED shards only"
    ),
}


@dataclass(frozen=True)
class CorpusRecord:
    """The corpus's own facts, and WHICH of its two records supplied them.

    ⚑ ``shards`` is a SNAPSHOT, resolved once.  In ``manifest+progress`` mode
    the corpus may be live, so the list is fixed here and never re-globbed; see
    the module docstring.
    """

    #: ``CORPUS_RECORD_SUMMARY`` or ``CORPUS_RECORD_PARTIAL``.
    mode: str
    #: The summary or the manifest.  Every stamp this tool passes through is
    #: read out of THIS dict, so a manifest that lacks one is a refusal or a
    #: named absence rather than a silently different value.
    facts: dict[str, Any]
    #: The shards to read, in name order.
    shards: tuple[Path, ...]
    #: Rows the inventory CLAIMS across those shards.  ⚑ Not the rows this run
    #: will emit: it is the inventory's own number, and comparing the two is
    #: how a reader sees what the schemes and the result filter dropped.
    rows_claimed: int
    #: The ``w*.progress.jsonl`` files read (partial mode only).
    progress_files: tuple[str, ...]
    #: Progress files whose torn final line was dropped -- the ONE damage
    #: ``read_worker_progress`` tolerates, and never silently.
    torn_tail_files: tuple[str, ...]
    #: Shard files on disk that the snapshot does not name: every live worker's
    #: in-flight shard, plus anything a kill left unlisted.  Counted, not read.
    unlisted_on_disk: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Whether the record itself claims the run FINISHED."""
        return self.mode == CORPUS_RECORD_SUMMARY

    def detail(self) -> dict[str, Any]:
        """The stamp that makes a partial derivation visibly partial."""
        return {
            "mode": self.mode,
            "document": (
                corpus.SUMMARY_NAME if self.complete
                else f"{corpus.MANIFEST_NAME} + {len(self.progress_files)} "
                     f"{PROGRESS_GLOB}"
            ),
            "run_finished": self.complete,
            "shards_adopted": len(self.shards),
            "rows_claimed_by_inventory": self.rows_claimed,
            "progress_files_read": list(self.progress_files),
            "progress_torn_tail_files": list(self.torn_tail_files),
            "shards_on_disk_not_in_inventory": list(self.unlisted_on_disk),
            "facts_only_in_summary": (
                {} if self.complete else dict(FACTS_ONLY_IN_SUMMARY)
            ),
        }


def read_corpus_record(corpus_dir: Path) -> CorpusRecord:
    """The corpus's facts and inventory, from whichever record it has.

    ``summary.json`` wins whenever it exists -- it is the complete record, and
    its inventory is checked against the disk in BOTH directions.  Without it,
    ``manifest.json`` plus the per-worker progress files carry every fact this
    tool reads except the ones :data:`FACTS_ONLY_IN_SUMMARY` names.
    """
    if (corpus_dir / corpus.SUMMARY_NAME).exists():
        summary = read_corpus_summary(corpus_dir)
        on_disk = corpus_shard_paths(corpus_dir)
        # ⚑ BEFORE the inventory check, which is where it was when this lived
        # in ``derive``. An empty directory and a directory missing one shard
        # are different problems and they kept different messages; moving the
        # inventory check earlier must not quietly re-route the first into the
        # second's wording.
        if not on_disk:
            raise CorpusIntegrityError(
                f"{corpus_dir} holds no .jsonl.zst/.jsonl.gz shards",
            )
        check_shard_inventory(on_disk, summary)
        return CorpusRecord(
            mode=CORPUS_RECORD_SUMMARY,
            facts=summary,
            shards=tuple(on_disk),
            rows_claimed=sum(
                int(entry.get("rows", 0)) for entry in summary.get("shards", [])
            ),
            progress_files=(),
            torn_tail_files=(),
            unlisted_on_disk=(),
        )
    return read_partial_corpus_record(corpus_dir)


def read_partial_corpus_record(corpus_dir: Path) -> CorpusRecord:
    """``manifest.json`` + ``w*.progress.jsonl``: a corpus that has not ended.

    ⚑ THE INTEGRITY CHECK IS THE GENERATOR'S OWN.  ``load_resume_manifest``
    already refuses a manifest whose ``config_sha256`` does not hash its own
    ``config_requested``, and that refusal is exactly the one this path needs:
    the cp map, the staircase and the row-identity join key all come out of that
    dict, so an edited manifest would have every derived target agreeing with a
    configuration nothing ran under.  Reimplementing the hash here would be a
    second copy free to drift from the writer's.

    ⚑ The progress files are read ONCE, here, and the returned list is the whole
    input.  A shard that closes while the derivation runs is not picked up, and
    the in-flight shard every live worker holds open is counted as unlisted
    rather than read -- it is being appended to, so its last line has no
    newline yet and reading it would either raise or silently truncate a row.
    """
    manifest_path = corpus_dir / corpus.MANIFEST_NAME
    if not manifest_path.exists():
        raise CorpusIntegrityError(
            f"{corpus_dir} holds neither {corpus.SUMMARY_NAME} nor "
            f"{corpus.MANIFEST_NAME}. The summary is written once at run END, "
            f"so a live or killed corpus legitimately has none -- but the "
            f"manifest is written BEFORE the first game, so a corpus with "
            f"neither carries no config_sha256, no cp map, no staircase and no "
            f"inventory, and every stamp this tool passes through would have to "
            f"be invented. If this directory was produced before "
            f"{corpus.MANIFEST_NAME} existed, it can only be derived from once "
            f"its run has written {corpus.SUMMARY_NAME}.",
        )
    try:
        # ⚑ IMPORTED, not mirrored: this is the generator's own refusal.
        manifest: dict[str, Any] = corpus.load_resume_manifest(corpus_dir)
    except ValueError as exc:
        raise CorpusIntegrityError(
            f"{manifest_path} cannot be trusted as this corpus's record: {exc}",
        ) from exc
    _check_row_schema(manifest, source=manifest_path)

    progress_paths = sorted(corpus_dir.glob(PROGRESS_GLOB))
    if not progress_paths:
        raise CorpusIntegrityError(
            f"{corpus_dir} has a {corpus.MANIFEST_NAME} but no "
            f"{PROGRESS_GLOB} files, so no shard inventory exists. A worker "
            "appends its first progress line when it closes its first shard; "
            "until then the corpus holds no CLOSED shard and there is nothing "
            "to derive from.",
        )

    listed: dict[str, Path] = {}
    rows_claimed = 0
    torn: list[str] = []
    for progress_path in progress_paths:
        try:
            records, was_torn = corpus.read_worker_progress(progress_path)
        except ValueError as exc:
            raise CorpusIntegrityError(
                f"{progress_path.name} is damaged somewhere other than its "
                f"torn final line, so the inventory it carries cannot be "
                f"trusted: {exc}",
            ) from exc
        if was_torn:
            torn.append(progress_path.name)
        for record in records:
            raw_path = record["path"]
            if raw_path is None:
                # ⚑ A GAME-COMPLETION RECORD, NOT A SHARD. `ShardWriter.close`
                # writes one when a worker ends on games that banked no rows;
                # it indexes games, and there is no file behind it. Treating it
                # as a shard would look for `None` on disk and refuse a corpus
                # whose only fault is that a dedup-served game ended a worker.
                continue
            # BY NAME, exactly as `resume_worker_state` resolves it: the stored
            # string is the absolute path of the machine that wrote the line,
            # and the file it means is the one beside the progress file.
            name = Path(str(raw_path)).name
            if name in listed:
                raise CorpusIntegrityError(
                    f"{name} is listed twice across {PROGRESS_GLOB}; shard "
                    "names are unique per worker and a resume continues the "
                    "index rather than reusing it, so a repeat means two runs' "
                    "inventories are mixed in this directory",
                )
            path = corpus_dir / name
            if not path.exists():
                raise CorpusIntegrityError(
                    f"{progress_path.name} lists {name} and it is not in "
                    f"{corpus_dir}; the inventory claims "
                    f"{int(record['rows'])} rows that are gone, and deriving "
                    "from what is left would train on a subset nothing recorded",
                )
            listed[name] = path
            rows_claimed += int(record["rows"])

    if not listed:
        raise CorpusIntegrityError(
            f"{corpus_dir}'s progress files name no shards at all (only "
            "game-completion records). No CLOSED shard exists yet, so there is "
            "nothing to derive from.",
        )
    unlisted = tuple(
        path.name for path in corpus_shard_paths(corpus_dir)
        if path.name not in listed
    )
    return CorpusRecord(
        mode=CORPUS_RECORD_PARTIAL,
        facts=manifest,
        shards=tuple(listed[name] for name in sorted(listed)),
        rows_claimed=rows_claimed,
        progress_files=tuple(path.name for path in progress_paths),
        torn_tail_files=tuple(torn),
        unlisted_on_disk=unlisted,
    )


def realized_cp_stamps(facts: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The per-worker REALIZED cp stamps that can be cross-checked, if any.

    ONE predicate, used by both the check and the stamp that reports how far it
    reached -- a second copy of "which stamps count" would let the manifest say
    a cross-check happened that the checker had skipped.  A dead worker's
    placeholder carries no cp keys and is not a claim anyone made.
    """
    by_worker = facts.get("config_realized_by_worker") or {}
    if not isinstance(by_worker, Mapping):
        return []
    return [
        (str(worker_id), dict(stamp))
        for worker_id, stamp in by_worker.items()
        if isinstance(stamp, Mapping) and "cp_slope" in stamp
    ]


def cp_map_params(facts: Mapping[str, Any]) -> tuple[float, float]:
    """The corpus's OWN cp->value parameters, not this tool's.

    ⚑ Deliberately not a flag.  The generator SELECTED its moves with these two
    numbers, so deriving targets under different ones would produce a policy
    whose ranking disagrees with the play that generated the positions, silently
    and only in the tail.  A corpus that does not record them is refused rather
    than defaulted.

    ``facts`` is the corpus's own record -- ``summary.json`` or, on a corpus
    that has not ended, ``manifest.json``.  BOTH carry ``config_requested``, so
    the map itself is read identically; only the realized cross-check below
    differs, and how far it reached is reported rather than assumed (see
    :data:`FACTS_ONLY_IN_SUMMARY`).
    """
    requested = facts.get("config_requested") or {}
    try:
        slope = float(requested["cp_slope"])
        draw_width = float(requested["cp_draw_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusIntegrityError(
            "the corpus record's config_requested carries no usable cp_slope / "
            "cp_draw_width; the cp->value mapping cannot be reconstructed and "
            "defaulting it would derive targets under a mapping the corpus was "
            "not generated with",
        ) from exc
    # ⚑ Cross-checked against every worker's REALIZED stamp (review finding 1):
    # `config_requested` is what the CLI said, the realized stamp is what the
    # searcher actually converted with. A generator defect that dropped the
    # knob on the way to the searcher would select moves under one map while
    # the request stamps another — and deriving under the requested map would
    # then disagree with the play that generated the positions. A dead
    # worker's placeholder carries no cp keys and is skipped.
    # ⚑ A MANIFEST CARRIES NO REALIZED STAMPS AT ALL, so on a partial corpus
    # this loop runs zero times and the requested map is taken on trust. That
    # is a real weakening, and it is REPORTED (`realized_workers_cross_checked`
    # in the derived manifest) rather than left to be inferred from silence.
    for worker_id, stamp in realized_cp_stamps(facts):
        realized = (
            float(stamp["cp_slope"]),
            float(stamp.get("cp_draw_width", draw_width)),
        )
        if realized != (slope, draw_width):
            raise CorpusIntegrityError(
                f"worker {worker_id}'s realized cp map {realized} disagrees "
                f"with config_requested ({slope}, {draw_width}); the corpus "
                "was selected under a mapping the request does not describe, "
                "and targets derived under either one would be wrong about "
                "the other",
            )
    return slope, draw_width


#: How far the mean recovered knob may sit from the requested one before the
#: run is refused: relative for both, with an absolute floor for tiny floors.
#: The estimator is exact to ~1e-7 relative on the rows it accepts, and the
#: failure this exists to catch is total (an unapplied floor reads ~0, i.e.
#: 100% off), so 5% is a wide-open corridor for honest runs and a wall for the
#: real failure.  Per-row scatter on real corpora stays inside it with margin
#: (tau in [0.066919, 0.067] on run02's first 20k).
TAKE_EFFECT_REL_TOL = 0.05
TAKE_EFFECT_FLOOR_ABS_TOL = 1e-6


def enforce_take_effect(options: DeriveOptions, stats: DeriveStats) -> None:
    """⚑⚑ The take-effect proof, CHECKED -- not just published for a human to diff.

    ``floor_requested`` and the recovered stamps land in the same summary, and
    a proof nobody compares is this codebase's signature defect one level up
    (a reviewer of PR #486 made exactly this point).  So the comparison runs
    HERE, before the summary is written: a floored run whose emitted rows do
    not carry the floor dies loudly, leaving shards and no summary -- the
    documented "this run DIED" state that ``refuse_populated_dir`` already
    fails closed on.

    Two asymmetries, both deliberate:

    * ``--floor > 0`` with ZERO estimator readings is refused outright.  The
      coarse stamp still covers every row, but the exact proof would be absent
      and silently absent is how gates rot.  A real corpus cannot trip this --
      run02's first 20k rows leave 17k+ readable -- only a pathological
      all-saturated input can, and such an input cannot prove a floor either.
    * ``--temp`` is checked only when readings exist.  An unfloored saturated
      corpus is a legal derivation whose rows genuinely cannot speak to tau,
      and refusing it would make the row set's readability a launch gate for a
      knob whose failure mode (`softmax_at_temp` ignoring ``temp``) has no
      plausible mechanism that also fakes ``temp_recovered_n == 0``.
    """
    if options.floor > 0.0:
        if stats.floor_recovered_n == 0:
            raise CorpusIntegrityError(
                f"--floor {options.floor} was requested but not one emitted row "
                "could be read back (floor_recovered_n == 0): the take-effect "
                "proof is absent, not passed. See the skip counters in the "
                "stats for where the rows went.",
            )
        mean_floor = stats.floor_recovered_sum / stats.floor_recovered_n
        tol = max(TAKE_EFFECT_REL_TOL * options.floor, TAKE_EFFECT_FLOOR_ABS_TOL)
        if abs(mean_floor - options.floor) > tol:
            raise CorpusIntegrityError(
                f"--floor {options.floor} was requested but the emitted rows "
                f"carry {mean_floor:.6g} (mean over {stats.floor_recovered_n} "
                f"readable rows, tolerance {tol:.3g}): the flag did not take "
                "effect as requested, and a corpus stamped with a floor it "
                "does not carry is worse than no corpus.",
            )
    if stats.temp_recovered_n > 0:
        mean_temp = stats.temp_recovered_sum / stats.temp_recovered_n
        if abs(mean_temp - options.temp) > TAKE_EFFECT_REL_TOL * options.temp:
            raise CorpusIntegrityError(
                f"--temp {options.temp} was requested but the emitted rows "
                f"carry {mean_temp:.6g} (mean over {stats.temp_recovered_n} "
                f"readable rows): the flag did not take effect as requested.",
            )


def derive(
    *,
    corpus_dir: Path,
    out_dir: Path,
    options: DeriveOptions,
    corpus_record: CorpusRecord | None = None,
) -> dict[str, Any]:
    """Read the corpus, write the shards, return the summary that describes them.

    ``corpus_record`` is threaded in by ``main`` (which already read it to
    resolve the cp mapping) so the corpus's record is read and validated ONCE
    per run; omitting it reads and validates it here, which is what a direct
    caller wants.

    ⚑⚑ THE SHARD LIST IS THE RECORD'S SNAPSHOT AND IS NEVER RE-READ.  On a live
    corpus (``manifest+progress`` mode) workers are closing shards and appending
    progress lines the whole time this function runs.  ``corpus_record.shards``
    was resolved once, before the first row, and that fixed list is the whole
    input: a shard that closes mid-derivation is neither picked up nor an error,
    and the in-flight shard a worker is holding open is counted in the output's
    ``shards_on_disk_not_in_inventory`` rather than read.  A derivation whose
    input grew while it ran would report a row count that no later reading of
    the same corpus could reproduce, which is a worse failure than deriving from
    slightly less than the corpus holds -- and the output says exactly how much
    it took.

    ⚑ ``derive_targets_summary.json`` is written LAST, so a directory holding
    shards and no summary is a run that DIED, not a corpus.  Nothing downstream
    has to detect that: the next attempt at the same ``--out`` hits
    ``refuse_populated_dir`` and stops, which is the fail-closed direction --
    the operator deletes a half-written directory deliberately rather than
    discovering later that half of it was described by nothing.
    """
    started = datetime.now(timezone.utc).isoformat()
    # ⚑ Here rather than in ``main`` so a caller cannot get past it: two runs'
    # shards in one directory is a corpus whose manifest describes half of it,
    # which is the same rule -- and the same reused function -- the generator
    # applies to its own out-dir.
    corpus.refuse_populated_dir(out_dir)
    record = (
        corpus_record if corpus_record is not None
        else read_corpus_record(corpus_dir)
    )
    summary = record.facts
    problems = scheme_vs_staircase_problems(
        options.scheme, summary.get("staircase_parsed", []),
    )
    if problems:
        raise CorpusIntegrityError(
            f"--scheme {options.scheme.canonical} cannot be answered by this "
            "corpus: " + "; ".join(problems),
        )
    shards = list(record.shards)
    if not shards:
        raise CorpusIntegrityError(f"{corpus_dir} holds no .jsonl.zst/.jsonl.gz shards")

    corpus_sha = str(summary.get("config_sha256", ""))
    deriver = TargetDeriver(options)
    rng = np.random.default_rng(options.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    pending: list[ReplaySample] = []
    shard_index = 0
    tt_carried: set[bool] = set()

    for path in shards:
        for row in iter_corpus_rows(path):
            if options.limit and deriver.stats.rows_read >= options.limit:
                break
            deriver.stats.rows_read += 1
            tt_carried.add(_check_row_identity(row, corpus_sha))
            try:
                sample = deriver.sample_from_row(row)
            except EnvelopeMiss as exc:
                deriver.stats.rows_dropped_envelope += 1
                if len(deriver.stats.envelope_miss_examples) < 8:
                    deriver.stats.envelope_miss_examples.append(
                        f"{_row_label(row)}: {exc}",
                    )
                if deriver.stats.rows_dropped_envelope > options.max_envelope_misses:
                    raise CorpusIntegrityError(
                        f"{_row_label(row)} cannot answer "
                        f"--scheme {options.scheme.canonical}: {exc}. That is "
                        f"{deriver.stats.rows_dropped_envelope} envelope miss(es) "
                        f"against --max-envelope-misses "
                        f"{options.max_envelope_misses}. Dropping rows changes "
                        "which positions the corpus contains, so the tolerance "
                        "is stated rather than assumed.",
                    ) from exc
                continue
            if sample is None:
                continue
            pending.append(sample)
            if len(pending) >= options.rows_per_shard:
                written.append(
                    _flush(out_dir, shard_index, pending, options, rng, corpus_sha),
                )
                deriver.stats.rows_written += len(pending)
                pending = []
                shard_index += 1
        if options.limit and deriver.stats.rows_read >= options.limit:
            break

    if pending:
        written.append(_flush(out_dir, shard_index, pending, options, rng, corpus_sha))
        deriver.stats.rows_written += len(pending)
    if not written:
        raise CorpusIntegrityError(
            "no rows survived; nothing was written. Read the drop counters "
            "before rerunning: an empty corpus and a corpus every row of which "
            "was dropped are different problems.",
        )

    enforce_take_effect(options, deriver.stats)
    out = build_summary(
        options=options,
        stats=deriver.stats,
        corpus_dir=corpus_dir,
        corpus_record=record,
        shards=written,
        started_utc=started,
        tt_carried=tt_carried,
    )
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(out, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return out


def _check_row_identity(row: dict[str, Any], corpus_sha: str) -> bool:
    """Every row belongs to the run ``summary.json`` describes.  Returns its TT flag.

    ⚑ The join key is the row's OWN ``config_sha256``, not the directory it was
    found in.  Two runs' rows in one directory would carry one summary's stamps
    on both halves, which is exactly what ``refuse_populated_dir`` exists to
    prevent upstream -- and this is the check that notices when it was defeated
    by hand.
    """
    schema = int(row.get("schema", -1))
    if schema != corpus.ROW_SCHEMA:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: row schema {schema} != {corpus.ROW_SCHEMA}",
        )
    run = row.get("run")
    if not isinstance(run, dict) or corpus.KEY_TT_CARRIED not in run:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: no run block carrying "
            f"{corpus.KEY_TT_CARRIED}; the derived shards could not disclose "
            "whether these searches shared a transposition table",
        )
    row_sha = str(run["config_sha256"])
    if corpus_sha and row_sha != corpus_sha:
        raise CorpusIntegrityError(
            f"{_row_label(row)}: config_sha256 {row_sha} does not match the "
            f"corpus summary's {corpus_sha}; these rows are from another run",
        )
    return bool(run[corpus.KEY_TT_CARRIED])


def _flush(
    out_dir: Path,
    index: int,
    samples: list[ReplaySample],
    options: DeriveOptions,
    rng: np.random.Generator,
    corpus_sha: str,
) -> dict[str, Any]:
    """Write one shard.  ``--seed`` permutes the rows inside it and nothing else.

    ⚑ Stated because the flag invites the other reading: the seed does NOT
    choose which rows are kept (``--limit`` takes a PREFIX of the corpus, which
    is a prefix of GAMES) and it does not touch a single target value.  It
    breaks up the game-contiguous order the corpus is written in, which is the
    order a mid-budget checkpoint would otherwise split by game.
    """
    order = rng.permutation(len(samples))
    ordered = [samples[int(i)] for i in order]
    path = local_shard_path(out_dir, index)
    save_local_shard_arrays(
        path,
        arrs=samples_to_arrays(ordered),
        meta=ShardMeta(
            run_id=SHARD_RUN_ID,
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            positions=len(ordered),
        ),
    )
    _stamp_shard_attrs(path, options, corpus_sha)
    return {"path": path.name, "rows": len(ordered)}


def _stamp_shard_attrs(path: Path, options: DeriveOptions, corpus_sha: str) -> None:
    """Put the scheme, its parameters and the code schema ON the shard.

    ⚑ A SECOND WRITE, and it has to be: ``save_local_shard_arrays`` funnels its
    meta through ``ShardMeta(**meta)``, which raises on any key that is not one
    of its declared fields -- so there is no way to carry a scheme name through
    it, and a scheme recorded only in the run's ``summary.json`` would be lost
    the moment one shard was copied somewhere else.  ``load_shard_arrays``
    returns the whole attrs dict, so these keys reach any reader that wants
    them and are ignored by every reader that does not.
    """
    group = zarr.open_group(str(path), mode="a")
    group.attrs.update({
        "derive_schema": DERIVE_SCHEMA,
        "derive_scheme": options.scheme.canonical,
        "derive_scheme_params": options.scheme.params(),
        "derive_temp": float(options.temp),
        # ⚑ ON THE SHARD, not only in the run's summary: a floored shard and an
        # unfloored one carry the same scheme name and the same temperature,
        # and a corpus is routinely read one shard at a time from somewhere
        # else. Without this a floor ladder's arms are indistinguishable once
        # their directories are apart.
        "derive_floor": float(options.floor),
        "derive_cp_slope": float(options.cp_slope),
        "derive_cp_draw_width": float(options.cp_draw_width),
        "derive_corpus_config_sha256": corpus_sha,
        "derive_corpus_row_schema": corpus.ROW_SCHEMA,
    })


def build_summary(
    *,
    options: DeriveOptions,
    stats: DeriveStats,
    corpus_dir: Path,
    corpus_record: CorpusRecord,
    shards: Sequence[dict[str, Any]],
    started_utc: str,
    tt_carried: set[bool],
) -> dict[str, Any]:
    """The output manifest.  Every knob appears as a REALIZED reading."""
    facts = corpus_record.facts
    return {
        "schema": DERIVE_SCHEMA,
        "started_utc": started_utc,
        "tool": "scripts/derive_corpus_targets.py",
        "corpus": {
            "dir": corpus_dir.name,
            # ⚑ WHICH RECORD THESE FACTS CAME FROM. "summary" is the complete
            # run; "manifest+progress" is a corpus that has not written its
            # summary yet (live, killed, or between --resume sessions) and is
            # therefore a PARTIAL read of it. A derived corpus that could not
            # say which one it was would be indistinguishable from a whole one.
            "corpus_record": corpus_record.mode,
            "corpus_record_detail": corpus_record.detail(),
            "config_sha256": facts.get("config_sha256"),
            # ⚑ A manifest has no top-level run_id; both records carry it inside
            # config_requested, and the summary's own top-level copy is read
            # FROM there, so the fallback is the same value and not a guess.
            "run_id": facts.get(
                "run_id", (facts.get("config_requested") or {}).get("run_id"),
            ),
            "row_schema": corpus.ROW_SCHEMA,
            "staircase_parsed": facts.get("staircase_parsed"),
            # Passed THROUGH, from the rows themselves rather than the summary:
            # a consumer must not mistake these for independent searches.
            corpus.KEY_TT_CARRIED: sorted(tt_carried),
            "banked_rows_min_piece_count": facts.get(
                "banked_rows_min_piece_count",
            ),
        },
        # ⚑ Rebuilt from the PARSED scheme object, not echoed from the flag.
        "scheme": {"canonical": options.scheme.canonical, **options.scheme.params()},
        "temp_requested": options.temp,
        "floor_requested": options.floor,
        "cp_map": {
            "q_function": (
                f"{gate.q_from_effective_cp.__module__}."
                f"{gate.q_from_effective_cp.__qualname__}"
            ),
            # ⚑ Resolved AT RUN TIME off the module attribute the mapping
            # actually goes through, so a monkeypatched or swapped object shows
            # up in the corpus that was written under it.
            "wdl_function": (
                f"{gen.cp_to_wdl_array.__module__}."
                f"{gen.cp_to_wdl_array.__qualname__}"
            ),
            "cp_slope": options.cp_slope,
            "cp_draw_width": options.cp_draw_width,
            "source": "the corpus's own config_requested",
            # ⚑ HOW FAR THE CROSS-CHECK REACHED, as a count rather than a
            # claim. `config_requested` is what the CLI asked for; the realized
            # stamps are what each worker's searcher actually converted with,
            # and only summary.json carries them. On a partial corpus this
            # reads 0 -- the check did not fire, and the number says so instead
            # of the manifest looking exactly like a cross-checked one.
            "realized_workers_cross_checked": len(realized_cp_stamps(facts)),
            "realized_cross_check_available": corpus_record.complete,
        },
        "input": {
            "input_history_encoding": INPUT_HISTORY_ENCODING,
            "input_extra_features": INPUT_EXTRA_FEATURES,
            "history_rep_fix": HISTORY_REP_FIX,
            "history_frames_total": _HISTORY_SLOTS,
            "zero_history": stats.history_slots_nonzero_max <= 1,
            "why_zero_history": (
                "a corpus row is a FEN; banked plies are non-contiguous "
                "(dedup misses above MIN_BANKED_PIECES only) so the move stack "
                "cannot be rebuilt, and encode_position fills slot 0 only"
            ),
        },
        "policy": {
            "encoding": "lc0_1858",
            "width": COMPACT_POLICY_SIZE,
            "construction": (
                "softmax(q / temp) over the scheme's values" if options.floor <= 0.0
                else (
                    "(1 - floor * n_legal) * softmax(q / temp) + floor over the "
                    "scheme's values, n_legal = the ROW's legal-move count"
                )
            ),
            "floor": options.floor,
            "floor_max_legal_moves_bound": MAX_LEGAL_MOVES,
            # ⚑ WHICH ESTIMATOR PRODUCED `temp_recovered_from_emitted_policy`.
            # A floored policy is not softmax(q/temp), so the closed form would
            # report a temperature that was never requested; the joint
            # estimator recovers tau and the floor together. Naming it is what
            # keeps two runs' temp stamps comparable.
            "temp_recovery_estimator": (
                "closed_form_two_move" if options.floor <= 0.0
                else "floored_three_move_bisection"
            ),
        },
        # ⚑ Both flags are COMPATIBILITY stamps, and both are lies of the same
        # shape `lc0_data_to_rows` already tells: no network played these moves
        # (Stockfish did) and the "selfplay" was engine-vs-itself. They are set
        # so the rows survive the trainer's network-turn filter, and they are
        # named here so a reader does not take them for provenance.
        "row_flags": {
            "is_network_turn": (
                "true — no network played these plies; set so the rows are not "
                "dropped by the train-on-network-turns filter"
            ),
            "is_selfplay": "true — Stockfish against itself, no curriculum arm",
            "priority": "1.0 — no surprise weighting exists for a corpus row",
        },
        "value_channels": {
            "wdl_target": (
                "the corpus row's exact game result, already stored from that "
                "row's own side-to-move seat (result_from_pov). 0=W/1=D/2=L"
            ),
            "search_wdl": (
                "cp_to_wdl_array of the SCHEME's best-move value: Stockfish's "
                "searched root value, side-to-move POV. ⚑ NOT our MCTS's "
                "value, which is what this column means on production shards"
            ),
            "sf_wdl": (
                "ABSENT — deliberately. lc0_control_train.py's launch guard 1 "
                "(assert_pid_cannot_reassert_sf_wdl) refuses any config with "
                "sf_wdl_frac > 0 regardless of what the shards carry, so a "
                "value written here could never reach a loss on this rig"
            ),
            "categorical_target": "ABSENT — outcome-derived, and wdl_target carries the outcome",
            "moves_left": "ABSENT — a corpus row does not know its game's length",
        },
        "value_blend": {
            "baked_into_rows": False,
            "note": (
                "the row carries the two components; the mixing weights are the "
                "trainer's, exactly as on data/lc0_rows"
            ),
        },
        "required_training_overrides": {
            "sf_wdl_frac": 0.0,
            "sf_wdl_frac_floor": 0.0,
            "search_wdl_frac": (
                "the whole non-outcome share. These shards carry no sf_wdl, so "
                "losses.py would redirect any SF share onto the raw game "
                "outcome; the searched value is in search_wdl"
            ),
        },
        "limit_requested": options.limit,
        "seed": options.seed,
        "seed_effect": "permutes rows WITHIN each shard; changes no target value",
        "rows_per_shard": options.rows_per_shard,
        "max_envelope_misses": options.max_envelope_misses,
        "realized": stats.summary(),
        "shards": list(shards),
        "python": sys.version.split()[0],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def format_summary(out: dict[str, Any]) -> str:
    realized = out["realized"]
    recovered = realized["temp_recovered_from_emitted_policy"]
    floor_read = realized["floor_recovered_from_emitted_policy"]
    min_mass = realized["policy_min_legal_prob_stored"]
    record = out["corpus"]["corpus_record_detail"]
    lines = [
        f"scheme={out['scheme']['canonical']} temp={out['temp_requested']} "
        f"floor={out['floor_requested']} "
        f"value_source={out['scheme']['value_source']}",
        # ⚑ FIRST-CLASS, not buried in the json: a partial derivation printed
        # exactly like a whole one is how it gets quoted as a whole one.
        f"corpus_record={out['corpus']['corpus_record']} "
        f"(run_finished={record['run_finished']}) "
        f"shards adopted={record['shards_adopted']} "
        f"rows claimed by inventory={record['rows_claimed_by_inventory']} "
        f"unlisted on disk={len(record['shards_on_disk_not_in_inventory'])}",
        f"rows read={realized['rows_read']} written={realized['rows_written']} "
        f"dropped(no result)={realized['rows_dropped_no_result']} "
        f"dropped(envelope)={realized['rows_dropped_envelope']}",
        f"nodes_floor_hits={realized['nodes_floor_hits']} "
        f"base depths={realized['realized_base_depth_histogram']} "
        f"values by phase={realized['values_by_phase']}",
        f"temp recovered from the emitted policy: n={recovered['n']} "
        f"min={recovered['min']:.6f} max={recovered['max']:.6f} "
        f"skipped(saturated)={realized['temp_recovery_skipped_saturated']} "
        f"estimator={out['policy']['temp_recovery_estimator']}",
        # ⚑ The floor's two readings on one line: the algebraic one (exact, on
        # the rows that can carry it) and the stored min mass (approximate, on
        # every row). Printed even at --floor 0, where they read n=0 and the
        # softmax's own tail -- which is what an unfloored run looks like.
        f"floor recovered from the emitted policy: n={floor_read['n']} "
        f"min={floor_read['min']:.6g} max={floor_read['max']:.6g} "
        f"skipped(few values)={realized['floor_recovery_skipped_few_values']} "
        f"skipped(ill-conditioned)="
        f"{realized['floor_recovery_skipped_ill_conditioned']}",
        f"min stored mass on a legal move: min={min_mass['min']:.6g} "
        f"max={min_mass['max']:.6g} mean={min_mass['mean']:.6g} "
        f"n={min_mass['n']}",
        f"x planes={realized['x_planes']} policy width={realized['policy_width']} "
        f"support {realized['policy_support_min']}..{realized['policy_support_max']} "
        f"history slots filled<={realized['history_slots_nonzero_max']}",
        f"shards={len(out['shards'])}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scheme", required=True, help=" | ".join(_SCHEME_FORMS))
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument(
        "--floor", type=float, default=DEFAULT_FLOOR,
        help="uniform exploration floor, as a probability PER LEGAL MOVE: the "
             "emitted policy is (1 - floor * n_legal) * softmax(q / temp) + "
             "floor. 0 (default) is off and emits the plain softmax. Refused "
             f"at startup at or above 1 / {MAX_LEGAL_MOVES} (the most legal "
             "moves a position can have), so which floors are legal never "
             "depends on which rows a corpus holds.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="stop after this many CORPUS ROWS READ (0 = the whole corpus). "
             "Rows dropped by a scheme or a missing result still count as read.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument(
        "--max-envelope-misses", type=int, default=0,
        help="how many rows may be dropped for lacking the block the scheme "
             "asks for before the run refuses. 0 (default) refuses on the first.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scheme = parse_scheme(str(args.scheme))
    temp = validate_temp(float(args.temp))
    # ⚑ Next to --temp, and for the same reason: before the corpus is opened,
    # so a bad floor is refused rather than discovered on the first row that
    # happens to have enough legal moves to break the head coefficient.
    floor = validate_floor(float(args.floor))
    if int(args.rows_per_shard) <= 0:
        raise ValueError(
            f"--rows-per-shard must be positive, got {args.rows_per_shard!r}",
        )
    if int(args.max_envelope_misses) < 0:
        raise ValueError(
            f"--max-envelope-misses must be >= 0, got {args.max_envelope_misses!r}",
        )
    corpus_dir = Path(args.corpus)
    # ⚑ ONCE, and the SNAPSHOT of a live corpus's inventory is taken here.
    corpus_record = read_corpus_record(corpus_dir)
    slope, draw_width = cp_map_params(corpus_record.facts)
    out = derive(
        corpus_dir=corpus_dir,
        out_dir=Path(args.out),
        corpus_record=corpus_record,
        options=DeriveOptions(
            scheme=scheme,
            temp=temp,
            floor=floor,
            cp_slope=slope,
            cp_draw_width=draw_width,
            limit=max(0, int(args.limit)),
            seed=int(args.seed),
            rows_per_shard=int(args.rows_per_shard),
            max_envelope_misses=int(args.max_envelope_misses),
        ),
    )
    print(format_summary(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
