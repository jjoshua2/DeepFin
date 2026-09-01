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

⚑ AT ``--value-scheme search`` (the default) NO BLEND IS BAKED INTO A ROW.  The
row carries the two components; the mixing weights are the trainer's
(``game_frac`` / ``search_wdl_frac``), exactly as on the lc0 corpus.
``required_training_overrides`` in the summary names the combination that passes
the rig's own guards, and a test asserts that it does by running
``run_config_problems`` against the shards this tool actually wrote.

THE VALUE SCHEMES (``--value-scheme``)
--------------------------------------
The ledger's 2026-08-31 VALUE-TARGET ROUND asks whether the game's OUTCOME
carries long-horizon information the depth-9 teacher misses -- and it cannot be
asked with the trainer's ``game_frac`` knob, because the generator NEVER plays
deterministic best (Gumbel temp 1.0 for plies 0-19, 0.3 after), so a flat share
of the raw outcome is a flat share of the sampled blunders that produced it.
The four arms differ ONLY in what goes in ``search_wdl``; ``--scheme``,
``--temp``, ``--floor``, the policy target and the row set are identical across
them, which is what makes the round paired rather than four experiments.

``search``  (V0, the default and the control)
    ``Q_t``: the scheme's searched root value.  ⚑ Byte-for-byte the corpus this
    tool wrote before the flag existed -- rows still enter the pending shard one
    at a time on an ungrouped path, so the control is the OLD CODE rather than
    new code that agrees with it.

``qz50``  (A)  ``0.5 * Q_t + 0.5 * Z_t``, per row.

``qzphase``  (B)  ``(1-w) * Q_t + w * Z_t``, ``w = ply_t / terminal_ply``.  The
    prereg's stated complexity bar for C.

``qzsegment``  (C)  THE CLEAN-SEGMENT RETROSPECTIVE TARGET.  From each row, scan
    FORWARD over the plies actually banked; the scan ends at the first move
    whose played regret exceeds ``--qz-r-boundary``, or that cannot be priced at
    all, and otherwise runs to the game's last banked row.  Reaching the end
    means the outcome is attributable to this position, so ``F_t`` is the game
    result; stopping early means it is not, so ``F_t`` is the searched value
    just before the blunder.  ``T_t = (1-w_t) * Q_t + w_t * F_t`` with ``w_t``
    from the teacher's own instability.

    ⚑⚑ THE SHAPE THAT MATTERS IS THAT ``F_t`` HAS NO LENGTH TERM.  This arm
    replaced a per-step lambda-return (``G_t = (1-lam) Q_t + lam flip(G_{t+1})``),
    which attenuates the terminal outcome by ``lam`` per PLY -- so on an 80-ply
    fortress, exactly the position class the arm exists for, the outcome arrives
    at the middlegame multiplied by ``lam**80`` and the arm is V0 with extra
    steps.  Here a clean segment of ANY length delivers the outcome undiminished
    and one blunder hard-stops it.  ``Z`` and any ``F`` taken from the other
    seat are W/L-swapped into the row's own seat first.

    Two DIAGNOSTIC ablations share the code path (second amendment):
    ``--qz-w-const c`` (C-no-u: constant weight, segment logic intact) and
    ``--qz-no-boundary`` (C-no-segment: blunder boundaries suppressed, u map
    intact).  Both at once is refused -- it is arm A with a different constant.
    ⚑ Under both, a row that cannot price its OWN transition (no played move,
    or a gap in the banked plies ahead of it) still falls back to pure ``Q``:
    that is the absence of a verdict rather than one, and the amendment carves
    it out explicitly.  Under ``--qz-no-boundary`` the carve-out is per-row --
    rows BEHIND such a row scan past it to the outcome, since suppressing what
    lies ahead is exactly what that cell ablates.

⚑⚑ A NON-``search`` SCHEME BAKES THE BLEND IN, AND THE MANIFEST SAYS SO LOUDLY.
``wdl_target`` still carries the RAW outcome (it is a required shard field with
no has-flag), so a trainer run at ``game_frac > 0`` against these shards mixes
the outcome in a SECOND time on top of the share the scheme already chose --
producing a target that is no arm of the round, including the one it is named
after.  ``value_blend.baked_into_rows`` flips to true and
``required_training_overrides`` gains ``game_frac: 0.0`` for exactly that
reader.

⚑⚑ AND THE SCHEME IS PROVED TO HAVE REACHED THE BYTES, NOT MERELY STAMPED.
``enforce_value_scheme_take_effect`` compares the emitted ``search_wdl`` against
the vector V0 would have written for the same row, on every arm: the control
must read a delta of exactly zero (so the instrument is known to be able to read
zero) and every other arm must have moved some row off it.  ``qzsegment``
additionally must have priced at least one played move, and each ablation flag
must have CHANGED a row -- a positive reading, because "no row stopped at a
boundary" is also what a blunder-free corpus looks like.  A value scheme that
parsed, stamped a summary and turned out to be V0 all along would read as a
clean experimental null, because it WOULD BE the control.

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
import multiprocessing as mp
import re
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
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
    arrays_to_samples,
    load_shard_arrays,
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
#:
#: ⚑ 1 IS THE UNCHANGED CONTRACT: ``search_wdl`` holds the bare searched root
#: value.  Every shard this tool wrote before the value round holds that, and
#: so does arm V0 (``--value-scheme search``) today.
DERIVE_SCHEMA = 1

#: ⚑⚑ THE SAME COLUMN, A DIFFERENT MEANING.  Under any ``--value-scheme`` other
#: than ``search``, ``search_wdl`` is no longer the searched root value: some
#: share of the GAME OUTCOME has already been blended into it.  The column's
#: name, dtype and shape are identical, so nothing about the file announces the
#: change -- which is precisely why the schema number moves.
#:
#: ⚑ ``derive_value_scheme`` also records this and is strictly more informative,
#: but it is a NEW key: a consumer written against schema 1 does not know to
#: look for it, and an unknown attr is silently ignored by every zarr reader.
#: The version number is the field such a consumer ALREADY reads, so it is the
#: only one that can reach it.  Two stamps, and the coarse one exists because
#: the precise one cannot be seen by code that predates it (Codex review of
#: PR #491).
#:
#: ⚑ V0 STAYS AT 1, deliberately: the control arm must remain byte-identical to
#: what this tool produced before the value round existed, attrs included.
DERIVE_SCHEMA_BAKED_VALUE = 2

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

#: Where the workers' intermediate rows live, under ``--out``.  Removed on
#: success; left behind on failure, where ``refuse_populated_dir`` then makes the
#: next attempt at the same ``--out`` refuse rather than half-overwrite.
SPILL_DIR_NAME = "spill"

#: Surviving rows per spill file.  ⚑ NOT tied to ``--rows-per-shard``: it bounds
#: a worker's RESIDENT sample list (a row's planes are 175*8*8 float32, ~45 KB
#: unpacked), and the output shard boundaries are decided by the coordinator
#: from the survivor counts regardless of how the rows were chunked on the way
#: there.  A repack that needs rows spanning two spill files simply reads two.
#:
#: ⚑ IT SETS THE PEAK, and the round-trip verification roughly triples it: at a
#: cut a lane holds the samples, the arrays built from them, the arrays read
#: back, and the samples rebuilt from those.  2,048 rows is ~90 MB of planes,
#: so a lane peaks near 300 MB and `--workers 7` near 2 GB -- against the
#: repack's ~600 MB per process at the default `--rows-per-shard 8192`, which
#: is the sequential path's own footprint and cannot be lowered without moving
#: the shard boundaries.
SPILL_CHUNK_ROWS = 2048

_SPILL_RUN_ID = "derive_parallel_spill"

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

# -- the value round's four arms ----------------------------------------------
#
# ⚑ ``--scheme`` and ``--value-scheme`` are ORTHOGONAL and neither implies the
# other: ``--scheme`` chooses which banked (phase, depth) cells become the POLICY
# target and the row's own searched value ``Q``, and ``--value-scheme`` chooses
# how much of the GAME's retrospect is mixed into that ``Q``.  The ledger's value
# round holds ``--scheme`` fixed at the reigning policy arm and moves only this.

#: V0, the control: ``search_wdl`` is the scheme's searched root value and
#: nothing else.  ⚑ THE DEFAULT, and byte-for-byte the corpus this tool wrote
#: before the flag existed -- the emission path for it is untouched (rows still
#: enter the pending shard one at a time, ungrouped), so the control arm is the
#: old code rather than new code that agrees with it.
VALUE_SCHEME_SEARCH = "search"
#: A: flat ``0.5 * Q + 0.5 * Z``.  Per-row; the game is never assembled.
VALUE_SCHEME_QZ50 = "qz50"
#: B: the phase ramp ``(1 - ply/terminal_ply) * Q + (ply/terminal_ply) * Z``.
#: The prereg's stated complexity bar for C.
VALUE_SCHEME_QZPHASE = "qzphase"
#: C: the clean-segment retrospective target (ledger amendment 2026-08-31).
#: ⚑ NAMED FOR WHAT IT COMPUTES.  The pre-amendment arm was a per-step
#: lambda-return and was going to be spelled ``qzlambda``; there is no lambda
#: left in it, and a flag value that names a mechanism the code does not run is
#: the same defect as a knob that never reaches its consumer.  ``qzlambda`` is
#: REFUSED with a message naming the amendment rather than quietly accepted.
VALUE_SCHEME_QZSEGMENT = "qzsegment"

VALUE_SCHEMES = (
    VALUE_SCHEME_SEARCH,
    VALUE_SCHEME_QZ50,
    VALUE_SCHEME_QZPHASE,
    VALUE_SCHEME_QZSEGMENT,
)

#: The value schemes that need the whole GAME assembled before any of its rows
#: can be emitted.  ``qz50`` is not one of them: its blend is a function of the
#: row alone, so it stays on the ungrouped path with V0.
VALUE_SCHEMES_NEEDING_GAME = (VALUE_SCHEME_QZPHASE, VALUE_SCHEME_QZSEGMENT)


def derive_schema_for(value_scheme: str) -> int:
    """The shard schema this value scheme emits.  See :data:`DERIVE_SCHEMA`.

    ⚑ A FUNCTION OF THE SCHEME, not of the run: the schema records what the
    ``search_wdl`` column MEANS, and the only thing that changes its meaning is
    which arm wrote it.  ``search`` is the pre-value-round contract and keeps
    the pre-value-round number.
    """
    if value_scheme == VALUE_SCHEME_SEARCH:
        return DERIVE_SCHEMA
    return DERIVE_SCHEMA_BAKED_VALUE

#: ⚑⚑ THE FROZEN PARAMETERS, from the ledger's 2026-08-31 prereg and its
#: same-day amendment.  Every one of them is a DEFAULT with a flag, so a
#: sensitivity run is possible -- and every one is pinned by a test against the
#: literal below, so a drift is a failing test rather than a quietly different
#: corpus.  They were calibrated on 98,304 rows / 12 shards of
#: ``run02_snap_20260829`` BEFORE any arm was trained; re-deriving them from a
#: later corpus is a new prereg, not a tweak.

#: The blunder boundary, in q units, for arm C's forward scan.  A transition
#: whose played regret exceeds it ends the clean segment.
#:
#: ⚑ ITS PROVENANCE IS ARITHMETIC AND IS CHECKED, NOT ASSERTED.  The prereg
#: froze a soft gate ``c = exp(-[max(0, r - r_free) / tau_r]^2)`` with
#: ``r_free = 0.06`` (the temp-low played-regret p75, so ordinary sampling noise
#: is exempt) and ``tau_r = 0.25`` (p95 = 0.2918 lands c ~ 0.37).  The amendment
#: replaced that soft per-step gate with a hard segment boundary and placed it at
#: the gate's HALF-CREDIT point, ``r_free + tau_r * sqrt(ln 2)`` = 0.268140...,
#: which the ledger froze rounded to 0.27.  ``QZ_R_FREE_CALIBRATED`` and
#: ``QZ_TAU_R_CALIBRATED`` below are kept for exactly that derivation and for the
#: summary's provenance stamp -- they are NOT flags, because the amended arm C
#: does not consume them and a flag that is parsed and then ignored is this
#: repo's signature defect.  ``test_the_boundary_is_the_frozen_gates_half_credit_point``
#: recomputes the derivation.
QZ_R_BOUNDARY = 0.27
QZ_R_FREE_CALIBRATED = 0.06
QZ_TAU_R_CALIBRATED = 0.25

#: The instability map ``w = 0.5 + 0.45 * min(u / 0.05, 1)``.  ``u``'s measured
#: quantiles on the calibration sample are p50 0.0058 / p90 0.0469, so 0.05 is
#: about the p90 -- the top decile of instability saturates the map and
#: everything below it interpolates.
QZ_U_SCALE = 0.05
QZ_W_MIN = 0.5
QZ_W_SPAN = 0.45

#: The three full-width rungs arm C reads its regret and instability off.  ⚑ NOT
#: flags and NOT tied to ``--scheme``: the prereg names d9/d8/d7 explicitly, and
#: a depth ladder that moved these with the policy scheme would change what the
#: gate MEANS between two arms that were supposed to differ only in their policy
#: teacher.  ``value_scheme_vs_staircase_problems`` refuses a corpus that cannot
#: answer them, at startup, before the first row.
QZ_GATE_DEPTH_TOP = 9
QZ_GATE_DEPTH_MID = 8
QZ_GATE_DEPTH_LOW = 7


@dataclass(frozen=True)
class QzParams:
    """Arm C's knobs: the two frozen ones, and the two ablation switches.

    ⚑ THE ABLATIONS ARE THE SECOND AMENDMENT'S ATTRIBUTION MACHINERY, on this
    one code path rather than as separate schemes.  ``C-no-u`` (``w_const``) and
    ``C-no-segment`` (``no_boundary``) each remove exactly ONE of arm C's two
    mechanisms, so a C win is attributable: C beating ``C-no-u`` says the
    depth-instability feature improves the net rather than merely correlating
    with teacher error, and ``C-no-segment`` losing on long-horizon positions is
    the direct evidence for time-as-effective-search-depth.  They are DIAGNOSTIC
    arms, not adoption candidates -- the ledger says so and so does the summary.

    ⚑ BOTH AT ONCE IS REFUSED.  Constant ``w`` AND no boundaries leaves nothing
    of arm C: it is ``(1-c) * Q + c * Z`` for a constant ``c``, i.e. arm A with a
    different constant, and the prereg names no such cell.  Running it would
    produce a directory whose ``value_scheme`` says ``qzsegment`` and whose rows
    are a fourth arm nobody preregistered.
    """

    r_boundary: float = QZ_R_BOUNDARY
    u_scale: float = QZ_U_SCALE
    #: ``C-no-u``: the blend weight, as a constant, instead of the frozen u map.
    #: ⚑ NO DEFAULT VALUE IN CODE.  The prereg's ablation uses 0.725 (the frozen
    #: map's midpoint) and that number arrives from the driver's command line;
    #: pinning it here would make the ablation's setting a property of this file
    #: rather than of the experiment that chose it.
    w_const: float | None = None
    #: ``C-no-segment``: ``F_t`` is the terminal outcome for every row whose own
    #: move can be priced, blunder boundaries ignored.
    no_boundary: bool = False

    def __post_init__(self) -> None:
        if self.w_const is not None and self.no_boundary:
            raise ValueError(
                "--qz-w-const and --qz-no-boundary together remove BOTH of arm "
                "C's mechanisms, leaving (1-c)*Q + c*Z for a constant c -- arm A "
                "with a different constant. The ledger's 2026-08-31 second "
                "amendment defines two ablations, C-no-u and C-no-segment, and "
                "no cell that is both; a run of it would stamp value_scheme "
                f"{VALUE_SCHEME_QZSEGMENT!r} on rows no prereg describes.",
            )
        if self.w_const is not None and not 0.0 <= float(self.w_const) <= 1.0:
            raise ValueError(
                f"--qz-w-const must lie in [0, 1], got {self.w_const!r}: it is a "
                "blend weight between the row's own searched value and its "
                "retrospective one, and outside that range the target is an "
                "extrapolation past both endpoints rather than a mixture.",
            )
        # ⚑ FINITE, not merely positive. `nan <= 0.0` is False, so a bare
        # `<= 0` check ACCEPTS `--qz-u-scale nan`: `segment_blend_weight` then
        # propagates NaN into every weight, every target, and every emitted
        # shard, and the take-effect gate only notices at the END -- after a
        # full corpus has been written to the output directory. Refused at
        # startup, like every other bad knob here (Codex review of PR #491).
        if not math.isfinite(float(self.u_scale)) or float(self.u_scale) <= 0.0:
            raise ValueError(
                f"--qz-u-scale must be positive and finite, got {self.u_scale!r}",
            )
        if not math.isfinite(float(self.r_boundary)):
            raise ValueError(f"--qz-r-boundary must be finite, got {self.r_boundary!r}")

    @property
    def is_ablation(self) -> bool:
        """Whether this is one of the two diagnostic cells rather than C-full."""
        return self.w_const is not None or self.no_boundary

    def without_ablation(self) -> QzParams:
        """The C-full parameters this cell ablates, for the direct comparison."""
        return QzParams(r_boundary=self.r_boundary, u_scale=self.u_scale)

    @property
    def variant(self) -> str:
        """Which of the three C cells this is, as one name for the stamps."""
        if self.w_const is not None:
            return "C-no-u"
        if self.no_boundary:
            return "C-no-segment"
        return "C-full"

    def params(self) -> dict[str, Any]:
        """The realized reading, plus the provenance of the frozen boundary."""
        derived = QZ_R_FREE_CALIBRATED + QZ_TAU_R_CALIBRATED * math.sqrt(math.log(2.0))
        return {
            "variant": self.variant,
            "adoption_candidate": self.variant == "C-full",
            "r_boundary": float(self.r_boundary),
            "u_scale": float(self.u_scale),
            "w_const": None if self.w_const is None else float(self.w_const),
            "no_boundary": bool(self.no_boundary),
            "w_min": QZ_W_MIN,
            "w_span": QZ_W_SPAN,
            "gate_depths": {
                "top": QZ_GATE_DEPTH_TOP,
                "mid": QZ_GATE_DEPTH_MID,
                "low": QZ_GATE_DEPTH_LOW,
            },
            "r_boundary_provenance": {
                "r_free_calibrated": QZ_R_FREE_CALIBRATED,
                "tau_r_calibrated": QZ_TAU_R_CALIBRATED,
                "half_credit_point": derived,
                "note": (
                    "r_free + tau_r * sqrt(ln 2) is the c=0.5 point of the "
                    "prereg's soft gate; the ledger froze it rounded to 0.27"
                ),
            },
        }


def parse_value_scheme(spec: str) -> str:
    """``--value-scheme`` -> one of :data:`VALUE_SCHEMES`.

    ⚑ ``qzlambda`` is named and REFUSED rather than left to argparse's generic
    "invalid choice".  It is the spelling the pre-amendment arm C would have had,
    so a driver script written against the first prereg would otherwise be told
    only that its scheme is unknown -- and the operator's next move would be to
    guess.  The amendment is named in the message instead.
    """
    name = str(spec)
    if name == "qzlambda":
        raise ValueError(
            "--value-scheme qzlambda no longer exists: the ledger's 2026-08-31 "
            "amendment replaced the per-step lambda-return (which geometrically "
            "attenuated the terminal outcome, so a long clean segment never "
            f"reached the middlegame) with {VALUE_SCHEME_QZSEGMENT!r}, the "
            "clean-segment retrospective target. The two are different targets; "
            "renaming the flag silently would have made them look like one.",
        )
    if name not in VALUE_SCHEMES:
        raise ValueError(
            f"--value-scheme {name!r} is not one of {', '.join(VALUE_SCHEMES)}",
        )
    return name


def value_scheme_vs_staircase_problems(
    value_scheme: str, staircase: Sequence[dict[str, Any]],
) -> list[str]:
    """Why this corpus's staircase cannot answer this VALUE scheme, before rows.

    ⚑ THIS GATE IS THE DIFFERENCE BETWEEN AN ARM AND A NULL.  Arm C reads its
    regret and instability off d9/d8/d7, and a corpus whose full-width phase
    stops at d5 banks none of them -- every row would be a missing-data row, so
    every scan would stop immediately, every ``w`` would be 0, and the emitted
    corpus would be arm V0 wearing arm C's name.  It would be COUNTED (the
    missing-depth counters would read 100%) and it would still be a full
    derivation that trained, which is exactly the shape of failure the ledger
    calls this repo's signature defect.  Refused at startup instead.
    """
    if value_scheme != VALUE_SCHEME_QZSEGMENT:
        return []
    if not staircase:
        return ["the corpus summary carries no staircase_parsed; nothing to check"]
    full_width_depth = int(staircase[0]["depth"])
    needed = (QZ_GATE_DEPTH_TOP, QZ_GATE_DEPTH_MID, QZ_GATE_DEPTH_LOW)
    if full_width_depth < max(needed):
        return [
            f"--value-scheme {value_scheme} reads its blunder gate and its "
            f"instability off full-width depths {needed}, and this corpus's "
            f"full-width phase reaches only depth {full_width_depth}; every row "
            "would be a missing-data row and the arm would silently be V0",
        ]
    return []

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

    ⚑ Two steps because the shard path is two steps -- ``derive_row``
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


# -- the value schemes --------------------------------------------------------


def flip_wdl(wdl: np.ndarray) -> np.ndarray:
    """One WDL 3-vector seen from the OTHER seat: W and L swap, D is invariant.

    ⚑ The only place a POV rotation happens in this file, and it exists because
    the value schemes below are the only construction here that reads one row's
    number into ANOTHER row's target.  ``wdl_target``/``result`` need no rotation
    -- the generator's ``result_from_pov`` already stored each row's outcome from
    that row's own seat -- so a second rotation of those would be a sign bug that
    no shape check could see.
    """
    values = np.asarray(wdl, dtype=np.float64).reshape(-1)[:3]
    return np.array([values[2], values[1], values[0]], dtype=np.float64)


def onehot_wdl(index: int) -> np.ndarray:
    """``0=W / 1=D / 2=L`` as a 3-vector, in the seat the index was taken from."""
    vector = np.zeros((3,), dtype=np.float64)
    vector[int(index)] = 1.0
    return vector


def segment_blend_weight(u: float | None, *, u_scale: float) -> float:
    """``w = 0.5 + 0.45 * min(u / u_scale, 1)``, or 0.0 when ``u`` is unknown.

    THE FROZEN MAP, and it is the same one the pre-amendment arm C used as a
    per-step lambda -- the amendment changed WHERE it is applied (a whole-segment
    blend weight instead of a per-ply decay), not the map.  ``u`` is the
    iterative-deepening instability ``|q(d9)-q(d8)| + |q(d8)-q(d7)|``: a teacher
    whose value is still moving between its last three iterations is one to
    listen to the FUTURE about, and a settled one keeps more of its own ``Q``.

    ⚑ ``u is None`` -> 0.0, which is PURE ``Q``, and that is the fail direction
    the ledger's amendment names ("fail toward Q, counted").  A row that did not
    bank d7/d8/d9 cannot say how stable its teacher was, and the alternative
    reading -- treating an unmeasurable ``u`` as 0 and taking the map's FLOOR of
    0.5 -- would hand half the target to a retrospective value on exactly the
    rows whose reliability is unknown.  Counted as
    ``qz_rows_missing_depths`` rather than absorbed.
    """
    if u is None:
        return 0.0
    scale = float(u_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"--qz-u-scale must be positive and finite, got {scale!r}")
    return QZ_W_MIN + QZ_W_SPAN * min(float(u) / scale, 1.0)


@dataclass(frozen=True)
class RowValueFacts:
    """Everything ONE row contributes to its game's value targets.

    Held instead of the row (a corpus row is a few KB of JSON and a game is
    hundreds of plies) and instead of the :class:`ReplaySample` (whose ``x`` is
    11,200 floats).  Every field is already in the row's OWN side-to-move seat;
    the rotations are :func:`game_value_targets`'s and are applied there.
    """

    #: The scheme's searched root value as WDL -- byte-for-byte the vector arm
    #: V0 writes, so V0/A/B/C differ ONLY in how much retrospect is mixed in.
    q_wdl: np.ndarray
    #: ``0=W / 1=D / 2=L``, the row's own-seat game result.
    z_index: int
    ply: int
    #: ``"w"``/``"b"`` -- the POV key, read off the row rather than inferred
    #: from ply parity, so a non-contiguous banked game rotates correctly.
    stm: str
    #: ``q(best_d9) - q(played_d9)``, or None when the row did not bank a
    #: full-width d9 block or its played move is absent from it.
    played_regret: float | None
    #: ``|q(d9)-q(d8)| + |q(d8)-q(d7)|``, or None when a rung is missing.
    instability: float | None
    #: Which of the two above was unavailable, for the counters.
    missing_played_move: bool
    missing_depths: bool


#: Why a forward scan stopped, stamped per row so the summary reports the split
#: rather than only the blend.  ⚑ THE TWO BOUNDARY CASES ARE SEPARATE because
#: they mean opposite things: ``boundary_self`` is a row whose OWN move was a
#: blunder or could not be priced, so its target is pure ``Q`` and it learned
#: nothing retrospective; ``boundary_ahead`` is a row that DID look forward and
#: found a clean stretch ending in someone else's blunder.  Only the second is
#: the mechanism arm C is testing, and a summary that added them would report a
#: corpus of dead rows as a working segment scheme.
SEGMENT_TERMINAL = "terminal"
SEGMENT_BOUNDARY_SELF = "boundary_self"
SEGMENT_BOUNDARY_AHEAD = "boundary_ahead"


@dataclass(frozen=True)
class SegmentReading:
    """One row's arm-C intermediate values, kept for the summary's counters."""

    stop: str
    #: Distance in BANKED ROWS from the row to where its scan stopped.
    span: int
    weight: float
    #: What the frozen u map WOULD have said, computed even under
    #: ``--qz-w-const``.  ⚑ The take-effect instrument for that flag: the run
    #: refuses unless the constant actually moved some row off the map.
    weight_from_map: float
    #: True when ``--qz-no-boundary`` changed THIS row's future -- the default
    #: scheme would have stopped it at a blunder ahead and the flag ran it to
    #: the terminal outcome instead.  ⚑ The take-effect instrument for that
    #: flag, and a positive one: "no row stopped at a boundary" is also what a
    #: blunder-free corpus looks like, so the count that proves the flag fired
    #: is the count of rows it CHANGED.
    boundary_suppressed: bool


def game_value_targets(
    facts: Sequence[RowValueFacts],
    *,
    value_scheme: str,
    params: QzParams,
    banked_tail_ply: int | None = None,
) -> tuple[list[np.ndarray], list[SegmentReading | None]]:
    """One game's rows -> one WDL 3-vector per row, under ``value_scheme``.

    PURE: no I/O, no stats, no corpus.  Every arm of the ledger's value round is
    a different line here and nothing else, which is what makes the four arms
    paired by construction rather than by four scripts agreeing.

    ``banked_tail_ply`` is the ply of the last row the CORPUS banked for this
    game when that row did not survive derivation -- see the section on the
    game's last banked row below, and :meth:`GameGrouper.note_dropped`.

    ``search`` (V0)
        ``Q_t``.  The vector this tool has always written; returned through the
        same path as the others so the take-effect stamp measures it too.

    ``qz50`` (A)
        ``0.5 * Q_t + 0.5 * Z_t``.  Per-row; the game is not consulted.

    ``qzphase`` (B)
        ``(1 - w) * Q_t + w * Z_t`` with ``w = ply_t / terminal_ply``, where
        ``terminal_ply`` is the ply of the game's LAST BANKED row -- see the
        note below on what that is and is not.
        ⚑ ``terminal_ply == 0`` -- a game with a single banked row at ply 0 --
        is ``w = 1.0``, not ``0/0``: that row IS the terminal one, and arm C
        puts pure ``Z`` there for the same reason.  Counted as
        ``qz_single_row_games``.

    ``qzsegment`` (C)
        THE CLEAN-SEGMENT RETROSPECTIVE TARGET (ledger amendment 2026-08-31).
        Scan FORWARD from row ``t`` over the transitions ``t, t+1, ...``; the
        scan stops at the first row whose played move is a blunder
        (``played_regret > r_boundary``) or whose regret cannot be computed at
        all, and otherwise runs off the end of the game.

        * ran to the end  -> ``F_t`` = the one-hot game result, in row ``t``'s
          own seat (``Z_t``: the corpus already stored it rotated).
        * stopped at ``j`` -> ``F_t`` = ``Q_j`` rotated into row ``t``'s seat.
          For ``j == t`` -- the row's OWN move is the blunder -- that is
          ``Q_t``, so the target collapses to pure ``Q`` and the row learns
          nothing retrospective.  Fail toward Q, by construction rather than by
          a branch.

        ``T_t = (1 - w_t) * Q_t + w_t * F_t`` with ``w_t`` from
        :func:`segment_blend_weight`, or the constant ``--qz-w-const``.

        ⚑⚑ NO GEOMETRIC DECAY, AND THAT IS THE WHOLE AMENDMENT.  The arm this
        replaced applied ``lambda`` per PLY, so a clean 80-ply proof reached the
        middlegame multiplied by ``lambda**80`` -- i.e. not at all, on exactly
        the fortress/50-move positions the arm exists for.  Here a clean segment
        of ANY length delivers the terminal outcome undiminished, and one
        blunder hard-stops it.  ``tests/test_derive_corpus_targets.py``'s
        ten-ply clean-game test pins that: row 0's target is a blend of ``Q_0``
        and the TERMINAL result, and reintroducing any per-step factor fails it.

        ⚑ ``--qz-no-boundary`` (``C-no-segment``) drops the blunder stop: every
        row whose own move can be priced runs to the terminal outcome.  A row
        that CANNOT price its own move still stops at itself -- the ledger's
        fail-toward-Q direction is about what a row can certify, not about the
        boundaries, and the ablation removes the second without weakening the
        first.  The LAST banked row is exempt from that, because C-full exempts
        it (see the comment at the branch).

    ⚑⚑ WHAT "THE GAME'S LAST BANKED ROW" IS, AND WHAT IT IS NOT
    -----------------------------------------------------------
    Both B and C anchor on it, and it is NOT reliably the game's terminal
    position.  ⚑ IT IS ALSO NOT RELIABLY ``facts[-1]``: a row the CORPUS banked
    can fail to survive derivation (a tolerated ``EnvelopeMiss`` under
    ``--max-envelope-misses``, or a row with a null ``result``), and a drop at
    the TAIL of a game leaves no later same-game row from which a ply gap could
    be observed -- so the last SURVIVING row would silently inherit the
    terminal treatment.  That is what ``banked_tail_ply`` closes: when it is
    set, the corpus banked a row at that ply, this derivation did not price the
    move out of ``facts[-1]``, and both arms are told so.  B anchors its ramp on
    the ply the corpus banked rather than the ply that survived, so its weights
    do not depend on the drop tolerance; C treats ``facts[-1]`` as unable to
    reach the outcome and stops it at itself.

    ⚑⚑ WHICH DROP PATH CAN ACTUALLY REACH IT — MEASURED, because the obvious
    answer is wrong.  ``result is None`` is the numerically dominant drop (1,897
    of 24,576 rows over ``run02_snap_20260829``'s first three shards, 7.7%) and
    it can NEVER produce a truncated tail on a corpus shaped like that one:
    ``result`` is a whole-GAME property there, so a game either has it on every
    row or on none.  Measured over the same 124 games: **9 games null on every
    row, 115 on none, and ZERO games mixed** -- an unfinished game disappears
    entirely rather than losing its tail, which is both correct and harmless.
    So the live trigger is the OTHER path: a tolerated ``EnvelopeMiss`` under
    ``--max-envelope-misses > 0`` drops INDIVIDUAL rows and can land on a game's
    last one.  Like the ply-gap boundary beside it, this is latent on run02
    (``games_with_dropped_tail = 0`` over 115 assembled games) and is a property
    of the next corpus, or of the next run that sets a drop tolerance.

    Three separate things end a game's banked rows, and only the
    third is the game ending:

    * **The banking floor.** ``gen_sf_rooted_corpus`` banks a row only at
      ``piece_count >= MIN_BANKED_PIECES`` (7) and adjudicates the game from
      tablebases at 6 or fewer, so most games stop banking ONE UNBANKED
      TRANSITION before they end -- the capture into TB range.  MEASURED here on
      ``run02_snap_20260829``: the last banked row has ``piece_count == 7`` in
      **101 of 124 games (81.5%)** over the first three shards and **95 of 133
      (71.4%)** over ``w00-00041..43``.  ⚑ An earlier revision of this comment
      offered PLY CONTIGUITY as the evidence that the last banked row is the
      terminal ply.  It is not evidence of that at all: contiguity is a
      statement about INTERNAL gaps (measured 0 in both samples, which is why
      ``GameGrouper``'s re-open refusal is well founded) and says nothing about
      a truncated TAIL.  Two different claims, and the wrong one was cited.
    * **``--limit``.** It takes a PREFIX of the corpus, so the last game read is
      cut wherever the count ran out.  Counted as ``qz_games_cut_by_limit``.
    * The game actually ending inside banking range.

    In ALL THREE the arms behave identically and deliberately: the last banked
    row is treated as terminal, so under B it gets ``w = 1`` and under C it gets
    ``F = Z``, the game's own recorded ``result``.  ⚑ That is CORRECT for the
    first and third -- ``result`` is the real outcome of the whole game however
    it was reached, adjudication included (measured: the last banked row carries
    an ``adjudication`` block in 91 of those 124 games), so a row one unbanked
    capture from the end is being told the truth.  Arm C simply never
    regret-checks that final transition: rows reaching it take ``Z``
    unconditionally, which is the one place the blunder gate does not cover.
    For ``--limit`` it is a row handed an outcome the derivation did not read
    far enough to witness, which is why that case alone carries a counter.

    ⚑ ``--limit`` is NOT refused for the grouped schemes, deliberately.  The
    alternative -- dropping the cut game's rows -- would make ``--limit N`` emit
    fewer than ``N`` rows as a FUNCTION OF THE VALUE SCHEME, so two arms of a
    paired round would hold different POSITIONS and not merely different
    targets.  That is the same argument ``--floor`` is refused at startup for
    rather than dropped per row.  A production arm passes no ``--limit`` and its
    summary proves it with ``games_cut_by_limit: 0``.
    """
    count = len(facts)
    if count == 0:
        return [], []
    if value_scheme == VALUE_SCHEME_SEARCH:
        return [np.asarray(f.q_wdl, dtype=np.float64) for f in facts], [None] * count
    if value_scheme == VALUE_SCHEME_QZ50:
        return (
            [
                0.5 * np.asarray(f.q_wdl, dtype=np.float64)
                + 0.5 * onehot_wdl(f.z_index)
                for f in facts
            ],
            [None] * count,
        )
    if value_scheme == VALUE_SCHEME_QZPHASE:
        # ⚑ THE PLY THE CORPUS BANKED, not the ply that survived derivation.
        # B's `w` is `ply / terminal_ply` and `terminal_ply` is defined as the
        # game's last BANKED row; a dropped tail row was banked. Reading
        # `facts[-1]` instead would hand the last surviving row `w = 1` -- full
        # terminal weight on a position the game demonstrably continued past --
        # and would make every earlier row's weight a function of
        # `--max-envelope-misses`, so two runs of the same arm over the same
        # corpus could differ in their targets because one tolerated a drop.
        terminal_ply = int(facts[-1].ply if banked_tail_ply is None else banked_tail_ply)
        targets: list[np.ndarray] = []
        for fact in facts:
            weight = 1.0 if terminal_ply <= 0 else float(fact.ply) / float(terminal_ply)
            targets.append(
                (1.0 - weight) * np.asarray(fact.q_wdl, dtype=np.float64)
                + weight * onehot_wdl(fact.z_index),
            )
        return targets, [None] * count
    if value_scheme != VALUE_SCHEME_QZSEGMENT:  # pragma: no cover - parse refuses first
        raise ValueError(f"unknown value scheme {value_scheme!r}")

    def uncertifiable(index: int) -> bool:
        """The transition out of this row cannot be priced AT ALL.

        Two ways, and neither is a blunder verdict -- they are the ABSENCE of
        one.  ⚑ Precisely what ``--qz-no-boundary`` does with them: a row that
        cannot certify its OWN transition still stops at itself (pure Q), which
        is the ledger amendment's stated carve-out; rows BEHIND it scan past,
        because that ablation's whole definition is that nothing ahead of a row
        stops it.  So the carve-out is per-row, not a boundary -- stated here
        because "the ablation preserves fail-toward-Q" is true of the row's own
        target and false of its neighbours', and those are different claims.

        * the row's played move is missing from its own d9 block (or absent);
        * ⚑ the row is the LAST surviving one and ``banked_tail_ply`` says the
          corpus banked more of this game than survived derivation.  Same hole
          as the next bullet, at the one index where the arithmetic there
          cannot see it;
        * ⚑ the next BANKED row is more than one ply later.  A corpus row is
          banked on a dedup MISS only, so a game's plies need not be
          contiguous, and the moves inside a gap were never banked -- they
          cannot be priced even in principle.  Counting the gap and then
          scanning straight across it (what this did before) lets an earlier
          row inherit the terminal outcome through an UNOBSERVED blunder,
          inside a stretch the summary is calling clean.  Latent on run02,
          whose ``ply_gaps_nonunit`` is 0, and a property of the next corpus
          rather than of this code.
        """
        fact = facts[index]
        if fact.played_regret is None:
            return True
        if index + 1 < count:
            return facts[index + 1].ply - fact.ply != 1
        # ⚑ THE SAME HOLE AT THE TAIL, where no next row exists to reveal it.
        # A dropped final row means the corpus banked at least one more move
        # that this derivation never priced, so running off the end here is
        # running ACROSS an unchecked move to claim its outcome -- the exact
        # attribution error arm C exists to prevent, and invisible to the gap
        # clause above because there is no `facts[index + 1]` to subtract.
        return banked_tail_ply is not None

    def blocked(index: int) -> bool:
        regret = facts[index].played_regret
        return uncertifiable(index) or (
            regret is not None and regret > params.r_boundary
        )

    # ⚑ ONE BACKWARD PASS computes every row's forward scan: the first boundary
    # at or after row i is row i itself when i is a boundary, and otherwise
    # whatever row i+1 already resolved.  O(game length) rather than the O(n^2)
    # the "scan forward from each row" reading invites.
    #
    # ⚑⚑ IT STARTS AT THE LAST BANKED ROW, WHOSE MOVE IS A REAL PLAYED MOVE.
    # An earlier cut started at ``count - 2`` on the reasoning that the final
    # banked row "has no outgoing transition". It has one: the generator writes
    # ``played_move`` and then PUSHES it (``gen_sf_rooted_corpus.play_game``),
    # and because the loop adjudicates at the TOP, that push is what ends the
    # game -- the last banked move is usually the game's LAST move, and it is
    # banked, priced and full-width like every other. Skipping it meant a
    # game-losing final blunder stopped nothing: that row and every clean row
    # behind it blended toward an outcome the blunder had just produced, which
    # is the exact attribution error arm C exists to prevent. Found by the
    # Codex review of PR #491.
    next_boundary: list[int | None] = [None] * count
    following: int | None = None
    for index in range(count - 1, -1, -1):
        next_boundary[index] = index if blocked(index) else following
        following = next_boundary[index]

    targets = []
    readings: list[SegmentReading | None] = []
    for index, fact in enumerate(facts):
        from_map = segment_blend_weight(fact.instability, u_scale=params.u_scale)
        weight = from_map if params.w_const is None else float(params.w_const)
        default_stop = next_boundary[index]
        if params.no_boundary:
            # ⚑ Self-only: a row that cannot price its own move still fails
            # toward Q, exactly as under the full scheme. What the flag removes
            # is the effect of OTHER rows' blunders on this row's future.
            #
            # ⚑ SELF-ONLY, and that is the ablation's definition rather than a
            # weaker version of C-full's rule: a row that cannot certify its own
            # transition keeps the fail-toward-Q carve-out, and a row BEHIND one
            # scans past it, because "nothing ahead of a row stops it" is
            # precisely what C-no-segment removes.
            #
            # ⚑ The LAST banked row takes the same rule as every other, because
            # C-full now gives it the same rule: its move is a real played move
            # (see the backward pass above). An earlier cut exempted it here, to
            # match a C-full that wrongly treated it as terminal-by-construction;
            # with C-full corrected, the exemption would REINTRODUCE the
            # divergence it was added to remove.
            stop = index if uncertifiable(index) else None
        else:
            stop = default_stop
        suppressed = bool(
            params.no_boundary
            and default_stop is not None
            and default_stop != stop,
        )
        q_own = np.asarray(fact.q_wdl, dtype=np.float64)
        if stop is None:
            future = onehot_wdl(fact.z_index)
            stop_kind, span = SEGMENT_TERMINAL, count - 1 - index
        else:
            boundary = facts[stop]
            future = np.asarray(boundary.q_wdl, dtype=np.float64)
            if boundary.stm != fact.stm:
                future = flip_wdl(future)
            stop_kind = (
                SEGMENT_BOUNDARY_SELF if stop == index else SEGMENT_BOUNDARY_AHEAD
            )
            span = stop - index
        targets.append((1.0 - weight) * q_own + weight * future)
        readings.append(
            SegmentReading(
                stop=stop_kind,
                span=span,
                weight=weight,
                weight_from_map=from_map,
                boundary_suppressed=suppressed,
            ),
        )
    return targets, readings


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

    # -- the value round's counters -------------------------------------
    #: Games assembled, and the banked-row length of each.  ⚑ ``qz_games`` is 0
    #: on the ungrouped path (V0 and A never assemble a game), which is what
    #: distinguishes "no games" from "games of length 0" in the summary.
    qz_games: int = 0
    qz_game_rows_min: int = 0
    qz_game_rows_max: int = 0
    #: Games whose only banked row is at ply 0, so arm B's ``ply/terminal_ply``
    #: is 0/0.  See :func:`game_value_targets` for why that reads as ``w = 1``.
    qz_single_row_games: int = 0
    #: Games flushed because ``--limit`` stopped the read, not because the next
    #: game began.  ⚑ Their last banked row is treated as terminal, so its arm-B
    #: weight is 1 and its arm-C future is the game RESULT -- a result the
    #: truncated game did reach, but at a ply this derivation never read. A
    #: limited run is a smoke test, and this counter is why nobody mistakes one
    #: for an arm.
    qz_games_cut_by_limit: int = 0
    #: Rows whose banked plies jumped by more than one. A dedup miss makes a
    #: game's plies non-contiguous, and arm C's POV rotation is read off ``stm``
    #: rather than ply parity precisely so that such a step still rotates
    #: correctly -- this counts how often it happened. MEASURED 0 on
    #: run02_snap_20260829's first three shards.
    qz_ply_gaps_nonunit: int = 0
    #: ⚑ Games whose LAST banked row did not survive derivation -- a tolerated
    #: ``EnvelopeMiss`` or a null ``result`` at the tail. The last SURVIVING row
    #: of such a game is not the game's last banked row, so arm C stops it at
    #: itself instead of blending toward an outcome across the unpriced dropped
    #: move, and arm B anchors its ramp on the banked ply rather than the
    #: surviving one. The take-effect reading for that fix: it is 0 on a corpus
    #: with no trailing drops, so a nonzero value is the only proof the path
    #: ran. MEASURED 0 over run02_snap_20260829's first three shards (115
    #: assembled games, 1,897 null-result drops, none of them game-final) --
    #: see :func:`game_value_targets` for why that path cannot truncate a tail
    #: there and which one can.
    qz_games_with_dropped_tail: int = 0
    #: Rows that could not price their own played move, and rows missing one of
    #: the d9/d8/d7 full-width rungs. Disjoint; see ``_value_facts``.
    qz_rows_missing_played_move: int = 0
    qz_rows_missing_depths: int = 0
    #: Where each row's forward scan stopped, split three ways -- see the
    #: SEGMENT_* constants for why ``self`` and ``ahead`` are never summed.
    qz_stop_terminal: int = 0
    qz_stop_boundary_self: int = 0
    qz_stop_boundary_ahead: int = 0
    #: Rows whose future ``--qz-no-boundary`` actually CHANGED. The take-effect
    #: proof for that flag; see :class:`SegmentReading`.
    qz_boundary_suppressed: int = 0
    #: Rows where ``--qz-w-const`` differs from what the frozen map would have
    #: said. The take-effect proof for that flag.
    #:
    #: ⚑ CROSS-CHECK IT AGAINST ``qz_rows_missing_depths`` BEFORE READING A
    #: C-full vs C-no-u CONTRAST. On a row with no d7/d8/d9 the frozen map gives
    #: ``w = 0`` (pure Q, the fail-toward-Q rule) while a constant gives ``w =
    #: c``, so those rows differ between the two arms by TWO changes -- the
    #: constant AND the loss of the missing-data fallback -- not one. That is
    #: accepted (the ledger defines C-no-u as "constant blend w", full stop) and
    #: it is bounded exactly by the counter two fields up, which reads 0 on
    #: run02_snap_20260829. A future corpus with missing rungs would need this
    #: subtracted before the contrast is one-dimensional again.
    qz_w_const_differs_from_map: int = 0
    #: ⚑⚑ THE ABLATIONS' ONLY TAKE-EFFECT GATE: rows whose EMITTED, QUANTIZED
    #: target differs from the one C-full would have emitted for the same row.
    #: The two counters above are diagnostics about intermediate choices and
    #: cannot carry this claim -- see `apply_value_scheme` for why each of them
    #: can increment on a row whose emitted vector does not move at all.
    qz_rows_differ_from_c_full: int = 0
    qz_regret_n: int = 0
    qz_regret_min: float = math.inf
    qz_regret_max: float = -math.inf
    qz_regret_sum: float = 0.0
    qz_instability_n: int = 0
    qz_instability_min: float = math.inf
    qz_instability_max: float = -math.inf
    qz_instability_sum: float = 0.0
    qz_weight_n: int = 0
    qz_weight_min: float = math.inf
    qz_weight_max: float = -math.inf
    qz_weight_sum: float = 0.0
    #: ⚑⚑ THE TAKE-EFFECT INSTRUMENT FOR ``--value-scheme`` ITSELF: the L1
    #: distance between the vector READ BACK OFF ``sample.search_wdl`` after the
    #: write (through the shard's own float32-then-float16 cast, see
    #: ``write_value_target``) and the vector V0 would have written for the same
    #: row, through that same cast. ⚑ "Read back", not "computed": measuring the
    #: computed target would make a dropped WRITE invisible, which a reviewer
    #: demonstrated on the first cut of this file by replacing the assignment
    #: with the baseline and watching the gate stay silent.
    #: It is measured on EVERY row of EVERY scheme -- V0 included, where it must
    #: read exactly 0 -- so
    #: the proof is symmetric: a value scheme that was parsed and never applied
    #: reads 0 and the run dies, and a `search` run that somehow blended reads
    #: nonzero and dies too. `enforce_take_effect` compares it before the
    #: summary is written; publishing it alone would be a stamp nobody diffs.
    value_delta_n: int = 0
    value_delta_min: float = math.inf
    value_delta_max: float = 0.0
    value_delta_sum: float = 0.0
    value_delta_rows_nonzero: int = 0

    def note_qz_regret(self, value: float) -> None:
        self.qz_regret_n += 1
        self.qz_regret_sum += value
        self.qz_regret_min = min(self.qz_regret_min, value)
        self.qz_regret_max = max(self.qz_regret_max, value)

    def note_qz_instability(self, value: float) -> None:
        self.qz_instability_n += 1
        self.qz_instability_sum += value
        self.qz_instability_min = min(self.qz_instability_min, value)
        self.qz_instability_max = max(self.qz_instability_max, value)

    def note_qz_weight(self, value: float) -> None:
        self.qz_weight_n += 1
        self.qz_weight_sum += value
        self.qz_weight_min = min(self.qz_weight_min, value)
        self.qz_weight_max = max(self.qz_weight_max, value)

    def note_value_delta(self, value: float) -> None:
        self.value_delta_n += 1
        self.value_delta_sum += value
        self.value_delta_min = min(self.value_delta_min, value)
        self.value_delta_max = max(self.value_delta_max, value)
        if value > 0.0:
            self.value_delta_rows_nonzero += 1

    def note_game(self, rows: int, *, cut_by_limit: bool) -> None:
        self.qz_games += 1
        self.qz_game_rows_min = (
            rows if self.qz_games == 1 else min(self.qz_game_rows_min, rows)
        )
        self.qz_game_rows_max = max(self.qz_game_rows_max, rows)
        if cut_by_limit:
            self.qz_games_cut_by_limit += 1

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
            # ⚑ ONE BLOCK, always present, even on a `search` run where every
            # counter in it reads 0. A key that appears only under some schemes
            # would make "this run wrote no games" and "this summary predates
            # the value round" the same observation.
            "value_scheme_realized": {
                "games": self.qz_games,
                "game_rows_min": self.qz_game_rows_min,
                "game_rows_max": self.qz_game_rows_max,
                "single_row_games": self.qz_single_row_games,
                "games_cut_by_limit": self.qz_games_cut_by_limit,
                "ply_gaps_nonunit": self.qz_ply_gaps_nonunit,
                "games_with_dropped_tail": self.qz_games_with_dropped_tail,
                "rows_missing_played_move": self.qz_rows_missing_played_move,
                "rows_missing_depths": self.qz_rows_missing_depths,
                "stop_terminal": self.qz_stop_terminal,
                "stop_boundary_self": self.qz_stop_boundary_self,
                "stop_boundary_ahead": self.qz_stop_boundary_ahead,
                "boundary_suppressed_by_flag": self.qz_boundary_suppressed,
                "w_const_differs_from_map": self.qz_w_const_differs_from_map,
                "rows_differing_from_c_full": self.qz_rows_differ_from_c_full,
                "played_regret": self._reading(
                    self.qz_regret_n,
                    self.qz_regret_min,
                    self.qz_regret_max,
                    self.qz_regret_sum,
                ),
                "instability_u": self._reading(
                    self.qz_instability_n,
                    self.qz_instability_min,
                    self.qz_instability_max,
                    self.qz_instability_sum,
                ),
                "blend_weight_w": self._reading(
                    self.qz_weight_n,
                    self.qz_weight_min,
                    self.qz_weight_max,
                    self.qz_weight_sum,
                ),
                # ⚑ THE TAKE-EFFECT READING, and it is CHECKED by
                # `enforce_take_effect` before this summary is written.
                "l1_delta_vs_search_value": self._reading(
                    self.value_delta_n,
                    self.value_delta_min,
                    self.value_delta_max,
                    self.value_delta_sum,
                ),
                "rows_differing_from_search_value": self.value_delta_rows_nonzero,
            },
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
    #: ⚑ Defaulted for the SAME reason as ``floor`` and no other: ``search`` is
    #: the identity -- the vector this tool has always written -- so a caller
    #: that predates the value round keeps deriving V0's corpus.
    value_scheme: str = VALUE_SCHEME_SEARCH
    qz: QzParams = field(default_factory=QzParams)
    #: Surviving rows per ``--workers`` spill file.  ⚑⚑ A FIELD, NOT A MODULE
    #: CONSTANT, AND THAT IS THE WHOLE POINT.  It changes no output -- the
    #: coordinator cuts output shards from the survivor counts however the rows
    #: were chunked on the way there -- so the only thing it is good for is
    #: reaching the multi-chunk spill and repack path from a test.  As a
    #: constant it was UNREACHABLE: lanes are spawned children that re-import
    #: this module, so a monkeypatch cannot follow them, and every fixture
    #: corpus is small enough that each lane wrote exactly ONE chunk.  MEASURED
    #: (independent review of PR #493): with the constant in place, deleting the
    #: final ``drain`` broke nothing and a truncating ``drain`` passed all twelve
    #: end-to-end identity tests.  Production runs thousands of chunks per lane
    #: through that path.  Travels to the lanes inside ``_WorkerTask``, so a test
    #: that lowers it lowers it for the code that actually writes the spill.
    spill_chunk_rows: int = SPILL_CHUNK_ROWS

    @property
    def needs_game(self) -> bool:
        """Whether rows must be assembled into games before any can be emitted."""
        return self.value_scheme in VALUE_SCHEMES_NEEDING_GAME


@dataclass(frozen=True)
class DerivedRow:
    """One corpus row's two products: the replay row, and its value facts.

    ⚑ Returned as a pair rather than the sample alone because the value schemes
    need numbers the sample does not carry (the played regret, the d9/d8/d7
    instability, the row's seat) and rebuilding them would mean walking the
    row's JSON a second time -- and, worse, a second reader of the corpus
    schema.  ``facts`` is ~100 bytes against ``sample.x``'s 11,200 floats, so a
    game buffer of either is dominated by the samples regardless.
    """

    sample: ReplaySample
    facts: RowValueFacts


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

    def derive_row(self, row: dict[str, Any]) -> DerivedRow | None:
        """One corpus row -> one replay row plus its value facts, or None when dropped.

        ⚑ ONE PASS.  The value schemes need this row's searched value, its game
        result, its played regret and its d9/d8/d7 instability -- all of them
        read off the SAME ``RowBank`` the policy target came from.  A second
        entry point that rebuilt the bank would double the JSON walk that
        dominates this tool's wall time, and would be a second reader of the
        corpus schema (this file's own docstring on why there is one decoder).
        """
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

        q_wdl = self.wdl_of(float(values.effective_cp[values.best_index]))
        z_index = wdl_target_from_result(float(row["result"]))
        sample = ReplaySample(
            x=planes,
            policy_target=policy.astype(np.float32),
            wdl_target=z_index,
            legal_mask=legal_mask,
            # ⚑⚑ LEFT UNSET ON PURPOSE. The searched value belongs in this
            # column and not in `sf_wdl` (see the module docstring for the guard
            # that makes `sf_wdl` unreachable on this rig) -- but it is written
            # by `write_value_target`, once, for EVERY arm including V0, and
            # `facts.q_wdl` below is where the V0 vector and the take-effect
            # baseline both come from.
            #
            # Seeding it with `q_wdl` here is what hid the original defect: a
            # write that never happened still read back as a plausible vector.
            # Unset, a missing write is a missing COLUMN, which nothing
            # downstream can mistake for a value.
            search_wdl=None,
            has_policy=True,
            is_selfplay=True,
            is_network_turn=True,
            game_id=int(row["game_id"]),
            ply_index=int(row["ply"]),
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
        )
        return DerivedRow(
            sample=sample,
            facts=self._value_facts(row, bank, q_wdl=q_wdl, z_index=z_index),
        )

    def _value_facts(
        self,
        row: dict[str, Any],
        bank: RowBank,
        *,
        q_wdl: np.ndarray,
        z_index: int,
    ) -> RowValueFacts:
        """This row's contribution to its game's value targets.

        ⚑ THE GATE READS PHASE 0's FULL-WIDTH BLOCKS, not ``value_at``'s
        deepest-phase-covering rule that the policy target uses.  ``u`` is a
        DIFFERENCE between three iterations and only means "how far has this
        search still moving" if all three are the same kind of reading: a d9
        spliced out of a narrowed rung searched with a warmer table against a d8
        from the cold full-width scout is a difference between two search
        SHAPES, not between two depths.  The regret is read the same way for the
        same reason, so ``r`` and ``u`` are commensurable.

        ⚑ Both readings are OPTIONAL and their absence is a value, not an
        exception: the ledger's fail direction is toward ``Q``, and a row that
        cannot be certified clean must stop a scan rather than be assumed clean.
        """
        blocks = {
            depth: bank.full_width_block(depth)
            for depth in (QZ_GATE_DEPTH_TOP, QZ_GATE_DEPTH_MID, QZ_GATE_DEPTH_LOW)
        }
        missing_depths = any(block is None for block in blocks.values())
        top = blocks[QZ_GATE_DEPTH_TOP]
        played = row.get("played_move")
        played_cp: float | None = (
            None if top is None or played is None
            else top["values"].get(str(played))
        )
        # ⚑ THE TWO COUNTERS ARE DISJOINT, deliberately. A row missing the d9
        # block cannot price its played move either, and counting it in both
        # buckets would make the summary's own totals unreadable -- a reader
        # comparing `qz_rows_missing_played_move` against `rows_written` would
        # be reading the depth failures a second time. A row is in the
        # played-move bucket only when the block it needed was THERE and the
        # move was not in it (or the row banked no played move at all).
        missing_played_move = played is None or (
            top is not None and played_cp is None
        )

        # ⚑ ONE numpy call for every q this row's gate needs. `q_from_effective_cp`
        # is a logistic evaluated through the shared object; four separate calls
        # per row cost ~30 us x 5.5M rows for arithmetic that vectorises for free.
        wanted: list[float] = []
        for depth in (QZ_GATE_DEPTH_TOP, QZ_GATE_DEPTH_MID, QZ_GATE_DEPTH_LOW):
            block = blocks[depth]
            wanted.append(
                math.nan if block is None else max(block["values"].values()),
            )
        wanted.append(math.nan if played_cp is None else float(played_cp))
        q_values = self.q_of(np.asarray(wanted, dtype=np.float64))

        instability: float | None = None
        if not missing_depths:
            instability = float(
                abs(q_values[0] - q_values[1]) + abs(q_values[1] - q_values[2]),
            )
        played_regret: float | None = None
        if top is not None and not missing_played_move:
            played_regret = float(q_values[0] - q_values[3])

        if missing_depths:
            self.stats.qz_rows_missing_depths += 1
        if missing_played_move:
            self.stats.qz_rows_missing_played_move += 1
        if played_regret is not None:
            self.stats.note_qz_regret(played_regret)
        if instability is not None:
            self.stats.note_qz_instability(instability)

        return RowValueFacts(
            q_wdl=np.asarray(q_wdl, dtype=np.float64),
            z_index=int(z_index),
            ply=int(row["ply"]),
            stm=str(row["stm"]),
            played_regret=played_regret,
            instability=instability,
            missing_played_move=missing_played_move,
            missing_depths=missing_depths,
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
        # FIRST: the shard path is float64 -> float32 (`derive_row`) ->
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
    #: The SAME claim, per shard and aligned with ``shards``.  ⚑ Kept here
    #: rather than re-derived by the caller that wants it: both records already
    #: read these numbers to produce ``rows_claimed``, and a second reader of
    #: the inventory is a second thing to keep in step with the writer.
    #: ``--workers`` prefix-sums this to place ``--limit`` at a global row
    #: index, and then checks every claim it used against the rows the shard
    #: actually held.
    shard_rows: tuple[int, ...]
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
        # ⚑ BY BASENAME, the same join ``check_shard_inventory`` just proved is
        # total in both directions -- the entries store the producing machine's
        # absolute paths and a corpus is routinely read from somewhere else.
        claimed_by_name = {
            Path(str(entry["path"])).name: int(entry.get("rows", 0))
            for entry in summary.get("shards", [])
        }
        return CorpusRecord(
            mode=CORPUS_RECORD_SUMMARY,
            facts=summary,
            shards=tuple(on_disk),
            rows_claimed=sum(
                int(entry.get("rows", 0)) for entry in summary.get("shards", [])
            ),
            shard_rows=tuple(claimed_by_name[path.name] for path in on_disk),
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
    claimed_rows: dict[str, int] = {}
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
            claimed_rows[name] = int(record["rows"])
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
        shard_rows=tuple(claimed_rows[name] for name in sorted(listed)),
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
    enforce_value_scheme_take_effect(options, stats)


def enforce_value_scheme_take_effect(
    options: DeriveOptions, stats: DeriveStats,
) -> None:
    """⚑⚑ The value round's take-effect proof, on the EMITTED vectors.

    Every check here is a comparison against ``search_wdl`` as it was actually
    written -- ``write_value_target`` reads the column back off the row through
    the shard's own cast, and ``_flush`` reads the whole column back off the
    written FILE -- not against the flag that asked for it.  The failure this exists
    for is the one the ledger calls this repo's signature defect: a
    ``--value-scheme`` that parses, stamps a summary, writes 5.5M rows, trains
    for 9,680 steps, and turns out to have been V0 all along -- an arm that
    reads as a clean null because it WAS the control.

    Four claims, each with its own instrument:

    * ``search`` must write the searched value UNCHANGED.  The control's zero is
      measured, not assumed, so the instrument is known to be able to read zero.
    * any other scheme must have MOVED some row off it.
    * ``qzsegment`` must have priced at least one row's own move, or its gate
      never ran and its ``w`` never came from a real ``u``.
    * an ablation must have moved a row OFF C-FULL'S OWN EMITTED TARGET,
      compared as stored.  ⚑ Not off its intermediate counters: a weight that
      differs from the map still emits C-full's vector on a self-boundary,
      where it multiplies a difference of zero, and a suppressed boundary whose
      replacement future equals the one it replaced emits C-full's vector too.
      Both counters can be positive on a corpus that IS C-full.
    """
    delta_zero = stats.value_delta_max <= 0.0
    if options.value_scheme == VALUE_SCHEME_SEARCH:
        if not delta_zero:
            raise CorpusIntegrityError(
                f"--value-scheme {VALUE_SCHEME_SEARCH} is the control arm and "
                "must write the scheme's searched root value unchanged, but "
                f"{stats.value_delta_rows_nonzero} of {stats.value_delta_n} "
                f"emitted rows differ from it (max L1 {stats.value_delta_max:.6g}). "
                "A control that is not the control invalidates every arm "
                "measured against it.",
            )
        return
    if stats.value_delta_n == 0 or delta_zero:
        raise CorpusIntegrityError(
            f"--value-scheme {options.value_scheme} was requested but every one "
            f"of the {stats.value_delta_n} emitted rows carries exactly the "
            "searched root value V0 writes: the flag was accepted and did not "
            "reach the bytes. Read value_scheme_realized's counters for where "
            "it stopped.",
        )
    if options.value_scheme != VALUE_SCHEME_QZSEGMENT:
        return
    if stats.qz_weight_n == 0:
        raise CorpusIntegrityError(
            f"--value-scheme {VALUE_SCHEME_QZSEGMENT} emitted "
            f"{stats.value_delta_n} rows and computed not one blend weight: the "
            "segment scheme did not run.",
        )
    if stats.qz_regret_n == 0:
        raise CorpusIntegrityError(
            f"--value-scheme {VALUE_SCHEME_QZSEGMENT} could not price the played "
            f"move of a single row (missing d{QZ_GATE_DEPTH_TOP} blocks: "
            f"{stats.qz_rows_missing_depths}, missing played moves: "
            f"{stats.qz_rows_missing_played_move}), so every scan stopped where "
            "it started and the arm is V0 wearing another name.",
        )
    if options.qz.is_ablation and stats.qz_rows_differ_from_c_full == 0:
        raise CorpusIntegrityError(
            f"the {options.qz.variant} ablation was requested but not one of "
            f"{stats.value_delta_n} emitted rows differs from the target C-full "
            "would have written for it, as stored. Its intermediate counters "
            f"(w_const_differs_from_map={stats.qz_w_const_differs_from_map}, "
            f"boundary_suppressed={stats.qz_boundary_suppressed}) can be "
            "positive on rows whose emitted vector does not move -- a weight "
            "multiplied into a zero difference, or a replacement future equal "
            "to the one it replaced -- so they are diagnostics, not evidence. "
            "This corpus IS C-full's, and the comparison the ablation exists "
            "for would be empty.",
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
    value_problems = value_scheme_vs_staircase_problems(
        options.value_scheme, summary.get("staircase_parsed", []),
    )
    if value_problems:
        raise CorpusIntegrityError(
            f"--value-scheme {options.value_scheme} cannot be answered by this "
            "corpus: " + "; ".join(value_problems),
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
    grouper = GameGrouper(deriver) if options.needs_game else None

    def emit(batch: GameBatch | None) -> None:
        """Take a batch's rows into ``pending`` and cut full shards off it.

        ⚑ THE SHARD BOUNDARIES ARE UNCHANGED BY BATCHING.  The ungrouped path
        calls this one row at a time, so the ``while`` runs at most once and cuts
        at exactly ``rows_per_shard`` -- the same boundary the pre-flag code cut
        at.  A grouped path pushing a 241-row game could otherwise flush a
        241-row-oversized shard, and two arms of the value round would then
        differ in their shard layout as well as in their targets.
        """
        nonlocal pending, shard_index
        if batch is None or not batch.rows:
            return
        pending.extend(apply_value_scheme(
            batch.rows,
            options=options,
            stats=deriver.stats,
            banked_tail_ply=batch.banked_tail_ply,
        ))
        while len(pending) >= options.rows_per_shard:
            chunk = pending[: options.rows_per_shard]
            written.append(_flush(out_dir, shard_index, chunk, options, rng, corpus_sha))
            deriver.stats.rows_written += len(chunk)
            pending = pending[options.rows_per_shard :]
            shard_index += 1

    for path in shards:
        for row in iter_corpus_rows(path):
            if options.limit and deriver.stats.rows_read >= options.limit:
                break
            deriver.stats.rows_read += 1
            tt_carried.add(_check_row_identity(row, corpus_sha))
            try:
                derived = deriver.derive_row(row)
            except EnvelopeMiss as exc:
                deriver.stats.rows_dropped_envelope += 1
                # ⚑⚑ TELL THE GROUPER BEFORE `continue`. Both drop paths used
                # to skip it entirely, so a dropped row was invisible to the
                # game -- and a drop of a game's LAST row left the previous row
                # looking like the game's end. See `note_dropped`.
                if grouper is not None:
                    grouper.note_dropped(row)
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
            if derived is None:
                # ⚑ THE SAME NOTIFICATION ON THE OTHER DROP PATH. `derive_row`
                # returns None for a null `result`. ⚑ MEASURED that this path
                # cannot truncate a tail on a run02-shaped corpus -- `result` is
                # a whole-game property there, so such a game loses every row
                # rather than its last (9 games all-null, 115 none, 0 mixed of
                # 124). Notified anyway: the invariant is the corpus's, not this
                # code's, and a branch that is correct only while an upstream
                # habit holds is the kind that breaks silently.
                if grouper is not None:
                    grouper.note_dropped(row)
                continue
            if grouper is None:
                emit(GameBatch(rows=[derived]))
            else:
                emit(grouper.add(row, derived))
        if options.limit and deriver.stats.rows_read >= options.limit:
            break

    if grouper is not None:
        # ⚑ ``cut_by_limit`` is a claim about WHY this game ended, and it is only
        # true when the read actually stopped short. A whole-corpus run's last
        # game ended because the corpus did.
        emit(grouper.flush(cut_by_limit=bool(
            options.limit and deriver.stats.rows_read >= options.limit,
        )))
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


def write_value_target(sample: ReplaySample, target: np.ndarray) -> np.ndarray:
    """Put the value target ON the row, and hand back WHAT THE SHARD WILL HOLD.

    ⚑⚑ THE READ-BACK IS OFF THE ATTRIBUTE, NEVER OFF ``target``, and that is the
    entire reason this is a function instead of one assignment inline.  The
    first cut of this file measured its take-effect delta from the vector it had
    just COMPUTED and then assigned on the next line, so the two numbers could
    not disagree by construction: an independent reviewer replaced the
    assignment with ``sample.search_wdl = baseline`` -- the flag parses, the
    scheme computes, the WRITE drops it, which is this repo's signature defect
    in its purest form -- and the gate stayed silent.  Reading the attribute
    back closes that: the delta is now a statement about the column.

    ⚑ THROUGH ``shard_stored``, i.e. the shard's own float32-then-float16 cast,
    for the same reason the floor stamp goes through it: ``search_wdl`` is a
    float16 column (``replay/shard.py``'s ``_OptFieldSpec``), so a float64
    reading is two casts away from the bytes and cannot see a storage-level
    loss.  A blend the trainer would read as the bare searched value is not a
    blend, however exactly it was computed.

    ⚑ THIS IS THE SOLE WRITER OF ``search_wdl`` ON THIS PATH, and ``derive_row``
    deliberately leaves the column ``None`` rather than pre-seeding it with
    ``Q``.  A pre-seed is what made the original defect invisible: with ``Q``
    already on the row, an assignment that never happened still read back as a
    plausible vector, and only the arms whose target differs from ``Q`` could
    ever notice.  Unset, a missing write is a missing COLUMN -- ``has_search_wdl``
    goes to 0 and every downstream gate, here and in the trainer's launch
    preflight, is looking straight at it.

    ⚑ HONEST LIMIT, since the alternative reads better than it is: returning
    ``shard_stored(target)`` instead of reading the attribute back is an
    EQUIVALENT expression for as long as the write succeeds -- both sides go
    through the same cast of the same numbers.  Reading the attribute is not a
    better arithmetic, it is the only form that still says something TRUE when
    the write does not happen, which is the case the reviewer built.  What
    neither form can catch is a serializer that stores one thing and reports
    another; that needs the bytes, and ``_flush`` reads them back off the shard.
    """
    sample.search_wdl = np.asarray(target, dtype=np.float64).astype(np.float32)
    return np.asarray(shard_stored(stored_value_target(sample)), dtype=np.float64)


def stored_value_target(sample: ReplaySample) -> np.ndarray:
    """The value column AS IT NOW STANDS ON THE ROW, or a refusal.

    ⚑ A SEPARATE FUNCTION so the ``None`` branch is REACHABLE.  Inline, right
    after the assignment, a type checker narrows ``search_wdl`` to non-optional
    and the guard reads as dead code -- which it is, until the assignment above
    it is edited away, and that is precisely the edit it exists to catch
    (basedpyright flagged the inline form as an unnecessary comparison, which
    was a true statement about the narrowing and a false one about the risk).
    Read through the field's own declared ``ndarray | None`` and the branch is
    live again.
    """
    written = sample.search_wdl
    if written is None:
        raise CorpusIntegrityError(
            "the value target for this row was never written to search_wdl. "
            "Every reading taken of it downstream would be a reading of "
            "nothing, and the shard would carry has_search_wdl=0 on a row that "
            "was handed a target.",
        )
    return np.asarray(written, dtype=np.float64)


def apply_value_scheme(
    rows: Sequence[DerivedRow],
    *,
    options: DeriveOptions,
    stats: DeriveStats,
    banked_tail_ply: int | None = None,
) -> list[ReplaySample]:
    """Write the value scheme's target into each row's ``search_wdl``, in place.

    ⚑ THE ONE PLACE ``search_wdl`` IS DECIDED, and it runs for EVERY scheme
    including ``search``.  V0 could have skipped it -- the samples already carry
    ``Q`` -- and then the take-effect delta below would be measured on three
    arms out of four, with the control's zero being an assumption instead of a
    reading.  A gate that cannot fire on the control is a gate whose passing
    says nothing.

    ⚑ MUTATES the samples rather than rebuilding them: ``ReplaySample`` carries
    an 11,200-float ``x`` and ``dataclasses.replace`` would copy every field of
    every row of a 5.5M-row corpus to change one 3-vector.
    """
    facts = [row.facts for row in rows]
    targets, readings = game_value_targets(
        facts,
        value_scheme=options.value_scheme,
        params=options.qz,
        banked_tail_ply=banked_tail_ply,
    )
    # ⚑⚑ THE ABLATION'S TAKE-EFFECT EVIDENCE IS THE EMITTED TARGET ITSELF.
    # The weight and stop-kind counters below prove that an INTERMEDIATE choice
    # changed, which is not the same claim: `w_const_differs_from_map`
    # increments on a self-boundary where `future == q_own`, and there the
    # weight multiplies a difference of zero and the emitted vector is
    # identical; `boundary_suppressed` increments even when the replacement
    # future happens to equal the one it replaced. A corpus could satisfy both
    # counters, satisfy the separate nonzero-vs-V0 gate on other rows, and
    # still be byte-for-byte C-full. So when an ablation is asked for, C-full is
    # computed alongside it and the two are compared AS STORED (Codex review of
    # PR #491).
    reference: list[np.ndarray] | None = None
    if options.value_scheme == VALUE_SCHEME_QZSEGMENT and options.qz.is_ablation:
        reference, _ = game_value_targets(
            facts,
            value_scheme=options.value_scheme,
            params=options.qz.without_ablation(),
            # ⚑ THE SAME TAIL FACT. C-full and the ablation must differ ONLY in
            # the mechanism being ablated; handing C-full a different view of
            # where the game's banked rows end would put a second difference
            # into the comparison the gate reads.
            banked_tail_ply=banked_tail_ply,
        )
    samples: list[ReplaySample] = []
    for index, (row, target, reading) in enumerate(zip(rows, targets, readings)):
        # ⚑ The baseline goes through the SAME cast as the read-back, so the
        # control's delta is exactly 0 rather than 0-to-within-float16 -- an
        # exact zero is a much stronger reading for the gate to enforce.
        baseline = np.asarray(
            shard_stored(np.asarray(row.facts.q_wdl, dtype=np.float64)),
            dtype=np.float64,
        )
        # ⚑ WRITE FIRST, THEN MEASURE, and measure what came back off the row.
        stored = write_value_target(row.sample, np.asarray(target, dtype=np.float64))
        stats.note_value_delta(float(np.abs(stored - baseline).sum()))
        if reference is not None:
            c_full = np.asarray(
                shard_stored(np.asarray(reference[index], dtype=np.float64)),
                dtype=np.float64,
            )
            if not np.array_equal(stored, c_full):
                stats.qz_rows_differ_from_c_full += 1
        if reading is not None:
            stats.note_qz_weight(reading.weight)
            if reading.stop == SEGMENT_TERMINAL:
                stats.qz_stop_terminal += 1
            elif reading.stop == SEGMENT_BOUNDARY_SELF:
                stats.qz_stop_boundary_self += 1
            else:
                stats.qz_stop_boundary_ahead += 1
            if reading.boundary_suppressed:
                stats.qz_boundary_suppressed += 1
            if (
                options.qz.w_const is not None
                and reading.weight != reading.weight_from_map
            ):
                stats.qz_w_const_differs_from_map += 1
        samples.append(row.sample)
    return samples


@dataclass(frozen=True)
class GameBatch:
    """The rows handed to :func:`apply_value_scheme` at once, plus their context.

    ⚑ THE TAIL FACT TRAVELS WITH THE ROWS.  ``banked_tail_ply`` is a property of
    the GAME, is known only to :class:`GameGrouper`, and is needed only by
    :func:`game_value_targets` -- and there is a shard-cutting loop in between.
    Carrying it as a field beside the rows keeps it attached to the game it
    describes; a parallel variable would be one refactor away from being read
    for the wrong game.
    """

    rows: list[DerivedRow]
    #: The ply of the last row the CORPUS banked for this game, when that row
    #: did NOT survive derivation.  ``None`` -- the overwhelmingly common case,
    #: and the whole ungrouped path -- means the last surviving row IS the last
    #: banked one and the arms behave exactly as they did before this field
    #: existed.
    banked_tail_ply: int | None = None


class GameGrouper:
    """Buffers derived rows into whole games for the schemes that need one.

    ⚑ ONLY the schemes that need a game are grouped (``VALUE_SCHEMES_NEEDING_GAME``).
    V0 and A stay on the ungrouped path, so the CONTROL ARM's bytes are produced
    by the code that produced them before this flag existed rather than by new
    code that agrees with it.

    ⚑⚑ A RE-OPENED GAME IS REFUSED, NOT REORDERED.  This closes a game the
    moment a row of a different ``(worker_id, game_id)`` arrives, which is
    correct exactly while each game's rows are contiguous in the corpus -- they
    are (MEASURED on run02_snap_20260829: 124 game-runs over the first three
    shards, zero re-openings), because one worker writes one game at a time and
    the shards are read in worker-then-index order.  If that ever stopped
    holding, the silent outcome would be a game split into fragments, each
    fragment's LAST row treated as terminal and handed the game's outcome -- a
    corpus that looks entirely normal and whose arm-C targets are wrong in the
    middle of every game.  So the invariant is checked rather than assumed.
    """

    def __init__(self, deriver: TargetDeriver) -> None:
        self._deriver = deriver
        self._key: tuple[int, int] | None = None
        self._rows: list[DerivedRow] = []
        self._last_ply: int | None = None
        # ⚑⚑ TWO PLY CURSORS, AND THEY MUST NOT BE MERGED. `_last_ply` is the
        # last SURVIVING ply and drives the internal-gap detection: a row
        # dropped at ply 40 with ply 41 surviving must read as a gap
        # (41 - 39 = 2), so a drop must NOT advance it. `_last_seen_ply` is the
        # last ply the CORPUS offered, drop or not, and exists only to notice
        # that a game's banked rows ran past its surviving ones. Advancing
        # `_last_ply` on a drop would silence the gap; ignoring drops in
        # `_last_seen_ply` would silence the tail.
        self._last_seen_ply: int | None = None
        self._closed: set[tuple[int, int]] = set()

    @property
    def closed_keys(self) -> frozenset[tuple[int, int]]:
        """Every game this grouper assembled and closed.

        ⚑ The re-open refusal in :meth:`add` is a WITHIN-grouper check, and a
        partitioned read has one grouper per lane -- so a game split across two
        lanes reaches neither grouper's ``_closed``.  Published here so
        ``derive_parallel`` can check the lanes' sets are pairwise disjoint,
        which is the same invariant one lane up.
        """
        return frozenset(self._closed)

    def note_dropped(self, row: dict[str, Any]) -> None:
        """Record that the corpus banked this row and derivation did not keep it.

        ⚑⚑ THE ONLY WAY A TRAILING DROP IS EVER OBSERVABLE.  An INTERNAL drop
        announces itself: the next surviving row of the same game arrives with a
        non-unit ply step and :meth:`add` sees the gap.  A drop of a game's LAST
        row announces nothing at all -- no later same-game row ever arrives, so
        the previous surviving row looks exactly like a game's natural end and
        is handed the terminal treatment (arm C blends toward ``Z`` across the
        dropped row's unpriced move; arm B gives it ``w = 1``).  This is the
        notification that makes the two cases symmetrical.

        ⚑ ATTRIBUTED BY KEY, not to whatever game is open.  A drop of the FIRST
        row of the NEXT game arrives while this game is still open, and charging
        it here would truncate a game that was never truncated.  A leading drop
        is then simply not recorded, which is correct: it shortens no scan and
        moves no ramp anchor.
        """
        key = self._key_of(row)
        if key != self._key:
            return
        ply = int(row["ply"])
        if self._last_seen_ply is None or ply > self._last_seen_ply:
            self._last_seen_ply = ply

    def _key_of(self, row: dict[str, Any]) -> tuple[int, int]:
        """``(worker_id, game_id)``, or a refusal naming what is missing.

        ⚑ Read by SUBSCRIPT and raised on, rather than ``.get``-ed to a default:
        a row with no game identity cannot be grouped, and a scheme that groups
        by game must say so instead of quietly filing it under a placeholder.
        """
        try:
            return (int(row["worker_id"]), int(row["game_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusIntegrityError(
                f"{_row_label(row)} carries no usable (worker_id, game_id), so "
                f"--value-scheme {self._deriver.options.value_scheme} cannot "
                "tell which game it belongs to. The grouped schemes assemble "
                "whole games; a row that cannot be attributed to one is not a "
                "row they can derive.",
            ) from exc

    def add(self, row: dict[str, Any], derived: DerivedRow) -> GameBatch | None:
        """Buffer one row; return the game it displaced, if any."""
        key = self._key_of(row)
        finished: GameBatch | None = None
        if key != self._key:
            finished = self.flush(cut_by_limit=False)
            if key in self._closed:
                raise CorpusIntegrityError(
                    f"{_row_label(row)} re-opens worker {key[0]} game {key[1]}, "
                    "which was already closed. --value-scheme "
                    f"{self._deriver.options.value_scheme} assembles each game "
                    "from a CONTIGUOUS run of corpus rows; interleaved games "
                    "would be split into fragments whose last row is silently "
                    "treated as the game's terminal position.",
                )
            self._key = key
            self._last_ply = None
            self._last_seen_ply = None
        ply = int(row["ply"])
        if self._last_ply is not None and ply - self._last_ply != 1:
            self._deriver.stats.qz_ply_gaps_nonunit += 1
        self._last_ply = ply
        if self._last_seen_ply is None or ply > self._last_seen_ply:
            self._last_seen_ply = ply
        self._rows.append(derived)
        return finished

    def flush(self, *, cut_by_limit: bool) -> GameBatch | None:
        """Close the open game and hand back its rows plus its tail fact."""
        if self._key is None or not self._rows:
            return None
        rows, key = self._rows, self._key
        # ⚑ STRICTLY GREATER, not merely different: only a ply BEYOND the last
        # survivor proves the game continued past it. A drop at or below it is
        # an internal drop, which the gap clause in `game_value_targets` already
        # covers.
        #
        # ⚑ HONEST NOTE, so nobody later "restores" a difference that is not
        # there: because `_last_seen_ply` keeps the MAXIMUM ply seen (in both
        # `add` and `note_dropped`), it can never fall below `rows[-1]`'s ply,
        # so `>` and `!=` are EQUIVALENT here today. Mutating one into the other
        # is an equivalent mutant and no test can distinguish them. `>` is kept
        # because it states the condition the code means; the max-keeping is
        # what actually enforces it.
        seen = self._last_seen_ply
        banked_tail_ply = (
            seen if seen is not None and seen > rows[-1].facts.ply else None
        )
        if banked_tail_ply is not None:
            self._deriver.stats.qz_games_with_dropped_tail += 1
        self._deriver.stats.note_game(len(rows), cut_by_limit=cut_by_limit)
        if len(rows) == 1 and rows[0].facts.ply == 0:
            self._deriver.stats.qz_single_row_games += 1
        self._closed.add(key)
        self._rows = []
        self._key = None
        self._last_ply = None
        self._last_seen_ply = None
        return GameBatch(rows=rows, banked_tail_ply=banked_tail_ply)


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
    return _flush_ordered(
        out_dir, index, [samples[int(i)] for i in order], options, corpus_sha,
    )


def _flush_ordered(
    out_dir: Path,
    index: int,
    ordered: list[ReplaySample],
    options: DeriveOptions,
    corpus_sha: str,
) -> dict[str, Any]:
    """Write one shard whose row ORDER has already been decided.

    ⚑ SPLIT OUT OF :func:`_flush` AND NOTHING ELSE CHANGED.  The sequential path
    still draws its permutation from the run's one generator and then calls
    this, so its statements are the statements it always ran.  The split exists
    because ``derive_parallel`` replays the whole permutation chain up front --
    it cannot hand a generator that is about to produce the i-th draw -- and a
    second copy of the writer is the one thing that could make a parallel shard
    differ from a sequential one without any check noticing.
    """
    path = local_shard_path(out_dir, index)
    arrs = samples_to_arrays(ordered)
    save_local_shard_arrays(
        path,
        arrs=arrs,
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
    _verify_value_column_on_disk(path, arrs)
    return {"path": path.name, "rows": len(ordered)}


def _verify_value_column_on_disk(path: Path, arrs: Mapping[str, np.ndarray]) -> None:
    """⚑⚑ READ THE VALUE COLUMN BACK OFF THE WRITTEN SHARD.

    The row-level stamp in :func:`write_value_target` proves the target reached
    the ``ReplaySample``; this proves it reached the FILE.  Between the two sits
    ``samples_to_arrays`` -> ``save_local_shard_arrays`` -> zarr, and the shard
    schema's ``search_wdl`` is an OPTIONAL column gated by a ``has_search_wdl``
    flag -- exactly the shape of thing that can be dropped without any error:
    the shard would be well formed, every row would carry ``has_search_wdl=0``,
    the trainer's ``compute_loss`` would fall the search component back to the
    raw one-hot outcome, and the arm would train on something no summary
    describes.  ``lc0_control_train``'s own launch guard measures that coverage
    for exactly this reason; measuring it HERE means a broken shard never leaves
    this process.

    Cheap enough to be unconditional: the column is ``(rows, 3)`` float16, ~48 KB
    on a full 8,192-row shard against the ~184 MB of planes beside it.  Lazy, so
    only that column is read.
    """
    stored, _ = load_shard_arrays(path, lazy=True)
    for key in ("search_wdl", "has_search_wdl"):
        want = np.asarray(arrs[key])
        if key not in stored:
            raise CorpusIntegrityError(
                f"{path.name}: the shard was written without a {key!r} column, "
                "so its value target is gone. Every row was handed one; a "
                "consumer would silently fall back to the raw game outcome.",
            )
        got = np.asarray(stored[key])
        if got.shape != want.shape or not np.array_equal(got, want):
            raise CorpusIntegrityError(
                f"{path.name}: the {key!r} column read back off the shard is "
                f"not the one that was written (shape {got.shape} vs "
                f"{want.shape}). The value target this run computed is not the "
                "value target the file holds.",
            )


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
        "derive_schema": derive_schema_for(options.value_scheme),
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
        # ⚑ ON THE SHARD, for the same reason `derive_floor` is: four arms of
        # the value round share a --scheme, a --temp and a --floor and differ
        # ONLY in what search_wdl holds. Without these two keys their shards are
        # indistinguishable the moment one is copied out of its directory, and
        # the round's whole point is that they are compared against each other.
        "derive_value_scheme": options.value_scheme,
        "derive_value_scheme_params": (
            options.qz.params()
            if options.value_scheme == VALUE_SCHEME_QZSEGMENT else {}
        ),
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
        "schema": derive_schema_for(options.value_scheme),
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
        "value_scheme": value_scheme_manifest(options),
        "value_channels": {
            "wdl_target": (
                "the corpus row's exact game result, already stored from that "
                "row's own side-to-move seat (result_from_pov). 0=W/1=D/2=L"
            ),
            "search_wdl": (
                "cp_to_wdl_array of the SCHEME's best-move value: Stockfish's "
                "searched root value, side-to-move POV. ⚑ NOT our MCTS's "
                "value, which is what this column means on production shards"
            ) if options.value_scheme == VALUE_SCHEME_SEARCH else (
                f"the --value-scheme {options.value_scheme} TARGET: a blend of "
                "the searched root value and the game's retrospect, already "
                "computed, side-to-move POV. ⚑ NOT the bare searched value -- "
                "see value_scheme and value_blend"
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
        "value_blend": (
            {
                "baked_into_rows": False,
                "note": (
                    "the row carries the two components; the mixing weights are "
                    "the trainer's, exactly as on data/lc0_rows"
                ),
            }
            if options.value_scheme == VALUE_SCHEME_SEARCH
            else {
                "baked_into_rows": True,
                "note": (
                    f"⚑ --value-scheme {options.value_scheme} BAKED the Q/Z "
                    "blend into search_wdl. wdl_target still carries the raw "
                    "outcome, so a trainer run with game_frac > 0 would mix the "
                    "outcome in a SECOND time on top of the share this scheme "
                    "already put there. See required_training_overrides."
                ),
            }
        ),
        "required_training_overrides": {
            "sf_wdl_frac": 0.0,
            "sf_wdl_frac_floor": 0.0,
            "search_wdl_frac": (
                "the whole non-outcome share. These shards carry no sf_wdl, so "
                "losses.py would redirect any SF share onto the raw game "
                "outcome; the searched value is in search_wdl"
            ) if options.value_scheme == VALUE_SCHEME_SEARCH else (
                "1.0 — the whole value target. This scheme already blended the "
                "outcome into search_wdl at the weight it chose"
            ),
            # ⚑ A FLOAT, next to its float siblings. It was prose, which read
            # fine and broke the one thing a consumer does with this dict:
            # `float(overrides["sf_wdl_frac"])` is exactly how the rig's own
            # guard test consumes the siblings, and `float("0.0 — REQUIRED…")`
            # raises. The reason lives in its own `_note` key instead of inside
            # the value (review finding 3).
            **(
                {}
                if options.value_scheme == VALUE_SCHEME_SEARCH
                else {
                    "game_frac": 0.0,
                    "game_frac_note": (
                        "REQUIRED. wdl_target still holds the raw outcome and "
                        "any positive share would double-count it against the "
                        "share this scheme baked in, which is a different "
                        "target from every arm of the round including this "
                        "one. lc0_control_train.py's launch preflight enforces "
                        "it from the shards' own derive_value_scheme stamp"
                    ),
                }
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


#: One line per arm, keyed by ``--value-scheme``: what the emitted ``search_wdl``
#: IS, spelled as arithmetic.  ⚑ In the manifest rather than only in this file's
#: docstring, because a derived directory is routinely read months later by
#: someone who has only the shards and the summary.
_VALUE_SCHEME_CONSTRUCTION = {
    VALUE_SCHEME_SEARCH: "Q_t — the scheme's searched root value, unchanged (V0)",
    VALUE_SCHEME_QZ50: "0.5 * Q_t + 0.5 * Z_t (A)",
    VALUE_SCHEME_QZPHASE: (
        "(1 - w) * Q_t + w * Z_t, w = ply_t / terminal_ply of the game's last "
        "banked row (B)"
    ),
    VALUE_SCHEME_QZSEGMENT: (
        "(1 - w_t) * Q_t + w_t * F_t (C): F_t is the game result in row t's "
        "seat when a forward scan over played regret reaches the game's last "
        "banked row, else the searched value at the first row whose played "
        "regret exceeds r_boundary (or cannot be priced), rotated into row t's "
        "seat. w_t = 0.5 + 0.45 * min(u_t / u_scale, 1), u_t = "
        "|q(d9)-q(d8)| + |q(d8)-q(d7)|. NO per-ply decay"
    ),
}


def _qzsegment_construction(qz: QzParams) -> str:
    """Arm C's arithmetic AS REALIZED, which for an ablation is not C-full's.

    ⚑⚑ THE ABLATIONS EMIT A DIFFERENT TARGET AND MUST SAY SO.  The dict above
    describes C-full, and a manifest that printed it for ``--qz-w-const`` would
    claim ``w_t`` came from the instability map while the corpus was written
    with a constant, and for ``--qz-no-boundary`` would claim the scan stopped
    at the first blunder while nothing stopped it.  ``params`` beside it does
    expose the flags, but ``construction`` is the field this manifest presents
    as THE arithmetic a later consumer should read, so a contradiction between
    the two is resolved by whichever the reader happens to trust.  Built from
    ``QzParams.variant`` and the realized numbers instead (Codex review of
    PR #491).

    ⚑ The variants are spelled out rather than assembled from fragments: the
    point of the field is that someone holding only the shards can reconstruct
    the target, and a sentence stitched from three conditionals is exactly the
    kind of prose that reads fluently and means something slightly wrong.
    """
    future_full = (
        "F_t is the game result in row t's seat when a forward scan over "
        "played regret reaches the game's last banked row, else the searched "
        "value at the first row whose played regret exceeds r_boundary "
        f"({float(qz.r_boundary):g}) or cannot be priced, rotated into row t's "
        "seat"
    )
    if qz.no_boundary:
        return (
            "(1 - w_t) * Q_t + w_t * F_t (C-no-segment ABLATION, diagnostic — "
            "NOT an adoption candidate): F_t is the game result in row t's "
            "seat for every row whose OWN played move can be priced; the "
            f"r_boundary ({float(qz.r_boundary):g}) blunder stop is REMOVED, so "
            "no row ahead of t stops t's scan. A row that cannot price its own "
            "move, or whose own banked tail is missing, still stops at itself "
            "and takes F_t = Q_t. w_t = 0.5 + 0.45 * min(u_t / u_scale, 1), "
            "u_t = |q(d9)-q(d8)| + |q(d8)-q(d7)|. NO per-ply decay"
        )
    if qz.w_const is not None:
        return (
            "(1 - w) * Q_t + w * F_t (C-no-u ABLATION, diagnostic — NOT an "
            f"adoption candidate): w is the CONSTANT {float(qz.w_const):g} for "
            "every row; the frozen instability map 0.5 + 0.45 * min(u_t / "
            "u_scale, 1) is REMOVED and u_t is not consulted. "
            f"{future_full}. NO per-ply decay"
        )
    return (
        f"(1 - w_t) * Q_t + w_t * F_t (C): {future_full}. "
        "w_t = 0.5 + 0.45 * min(u_t / u_scale, 1), u_t = "
        "|q(d9)-q(d8)| + |q(d8)-q(d7)|. NO per-ply decay"
    )


def value_scheme_manifest(options: DeriveOptions) -> dict[str, Any]:
    """The value arm, as a REALIZED reading rather than an echo of the flag."""
    construction = (
        _qzsegment_construction(options.qz)
        if options.value_scheme == VALUE_SCHEME_QZSEGMENT
        else _VALUE_SCHEME_CONSTRUCTION[options.value_scheme]
    )
    manifest: dict[str, Any] = {
        "name": options.value_scheme,
        "construction": construction,
        "grouped_by_game": options.needs_game,
        "frozen_by": (
            "docs/experiment_ledger.md — 2026-08-31 PREREG VALUE-TARGET ROUND "
            "and its two same-day amendments"
        ),
        "q_definition": (
            "cp_to_wdl_array of the --scheme's best-move value: the SAME vector "
            "the search arm writes, so V0/A/B/C differ only in the retrospect"
        ),
        "z_definition": (
            "one-hot of the row's own `result`, which result_from_pov already "
            "stored in that row's seat — never rotated again"
        ),
    }
    if options.value_scheme == VALUE_SCHEME_QZSEGMENT:
        manifest["params"] = options.qz.params()
    return manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _value_scheme_line(out: dict[str, Any]) -> str:
    """The value arm and its take-effect reading, on one console line."""
    scheme = out["value_scheme"]
    read = out["realized"]["value_scheme_realized"]
    delta = read["l1_delta_vs_search_value"]
    variant = (scheme.get("params") or {}).get("variant")
    head = f"value_scheme={scheme['name']}" + (f" ({variant})" if variant else "")
    return (
        f"{head} games={read['games']} "
        f"rows differing from V0={read['rows_differing_from_search_value']} "
        f"L1 vs V0 mean={delta['mean']:.6g} max={delta['max']:.6g} | "
        f"stop terminal/self/ahead="
        f"{read['stop_terminal']}/{read['stop_boundary_self']}/"
        f"{read['stop_boundary_ahead']} "
        f"w mean={read['blend_weight_w']['mean']:.4g} "
        f"u mean={read['instability_u']['mean']:.4g} "
        f"regret mean={read['played_regret']['mean']:.4g} | "
        f"missing played_move={read['rows_missing_played_move']} "
        f"depths={read['rows_missing_depths']} "
        f"nonunit ply gaps={read['ply_gaps_nonunit']} "
        f"dropped tails={read['games_with_dropped_tail']}"
    )


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
        # ⚑ ON THE CONSOLE, not only in the json: an operator launching four
        # arms in a row reads this line, and "which arm did I just derive" is
        # the question a ladder gets wrong most cheaply.
        _value_scheme_line(out),
        f"x planes={realized['x_planes']} policy width={realized['policy_width']} "
        f"support {realized['policy_support_min']}..{realized['policy_support_max']} "
        f"history slots filled<={realized['history_slots_nonzero_max']}",
        f"shards={len(out['shards'])}",
    ]
    return "\n".join(lines)


# ── --workers: the same corpus, derived in parallel ────────────────────────────
#
# ``--workers N > 1``, and the contract is not "equivalent" but IDENTICAL
# CONTENT:
#
#     every array of every ``shard_*.zarr`` decompresses to the same dtype,
#     shape and bytes as ``--workers 1`` writes; the shards are the same shards
#     in the same order holding the same rows in the same permuted positions;
#     every ``.zattrs`` stamp matches; and ``derive_targets_summary.json``
#     matches field for field but for ``started_utc``.
#
# ⚑⚑ AND NOT THE COMPRESSED FILES, WHICH CAN NEVER MATCH, FOR TWO SEPARATE
# REASONS.  ``save_local_shard_arrays`` writes through ``numcodecs`` Blosc.
#
#   1. ITS MULTI-THREADED ENCODER IS NON-DETERMINISTIC.  MEASURED 2026-08-31 on
#      ``numcodecs 0.13.1``: one process compressing one float16 array three
#      times produced three different digests, and two back-to-back SEQUENTIAL
#      derivations of one fixture corpus at one ``--seed`` differed in 11 of 324
#      files -- every one an ``x`` or ``policy_target`` chunk, i.e. exactly the
#      arrays large enough for Blosc to split across threads.  So ``--workers 1``
#      does not reproduce its OWN bytes.
#   2. THE TWO PATHS DO NOT EVEN USE THE SAME ENCODER.  ``numcodecs`` disables
#      Blosc's thread pool outside the main process, so ``_get_use_threads()`` is
#      True where ``--workers 1`` writes (the main process) and False in every
#      spawned lane, where the repack writes.  Threaded and context encoders
#      emit different bytes for identical input.  ⇒ the parallel output is
#      byte-STABLE run to run, the sequential output is not, and the two are
#      different byte streams by construction.
#
# The single-threaded encoder IS deterministic, so the tool COULD be made
# byte-reproducible by pinning ``numcodecs.blosc.use_threads = False`` -- at a
# wall cost, and by changing what ``--workers 1`` writes, which is the one thing
# this flag must not do.  So the identity claim is stated over the DATA, which is
# what a shard means and all any consumer reads.  ⚑ Do not "restore" a byte-level
# assertion here or in the tests; ``codec_is_deterministic`` in
# ``tests/test_derive_parallel.py`` measures the regime instead, and tightens the
# comparison automatically if it ever changes.
#
# :func:`derive` above is a single pass over the corpus's shard snapshot doing
# three separable things -- read and derive each row, fold every row into an
# order-dependent running statistic, and cut the surviving rows into
# seed-permuted output shards.  Only the FIRST is parallelisable; the other two
# are DEFINED by the global row order and have to be reconstructed exactly.
#
# ⚑⚑ ``--workers 1`` DOES NOT COME THROUGH HERE AT ALL.  ``main`` calls
# :func:`derive_parallel` only above 1, so the default path is not "the parallel
# driver with one lane" -- it is the same :func:`derive` call, statement for
# statement, that this tool ran before the flag existed.  A refactor that routed
# the default through a one-lane coordinator would make every future bug here a
# production bug, and the identity claim would then have nothing to compare
# against.
#
# ⚑ IT LIVES IN THIS FILE, next to the loop it mirrors, rather than in a module
# of its own.  Every part of it -- the deriver, the grouper, ``apply_value_scheme``,
# the shard writer -- is :func:`derive`'s own, because reimplementing any of them
# is the single thing that could make a parallel shard differ from a sequential
# one; and a change to that loop then shows up in the same diff as the mirror it
# has to stay in step with.
#
# How the four hazards are closed
# -------------------------------
#
# **1. The row order.**  Workers take CONTIGUOUS ranges of the record's shard
# snapshot, so the union of their processed rows, sorted by global input-row
# index, is exactly the sequential read order.  ``--limit`` is a cut at a global
# input-row INDEX, not a per-worker budget: the worker owning that index stops
# there and the ones after it read nothing.
#
# **2. Whole games.**  ``qzphase``/``qzsegment`` assemble a game before any of its
# rows can be emitted, and ⚑ MEASURED on ``run02_snap_20260829``: a game DOES span
# input shards there (``w00-00000`` ends on game 1176 and ``w00-00001`` opens on
# it), so a partition that assumed otherwise would split a game and hand each
# fragment's last row the terminal treatment.  The handoff is a strict mirror:
# worker ``k`` skips its range's leading run of rows carrying the raw
# ``(worker_id, game_id)`` of the last row of the shard BEFORE its range, and
# worker ``k-1`` overflows past its own range end while rows carry that SAME key,
# read off that SAME row.  Both sides name one row, so the skip and the overflow
# cannot disagree.
#
# ⚑ THE OVERFLOW KEY IS THE RAW ONE, NOT "THE LAST ROW I PROCESSED".  Those differ
# whenever a shard's final rows were DROPPED (a null ``result``, a tolerated
# envelope miss) and belong to a different game from the last surviving row: the
# skipping side reads the raw key regardless of drops, so an overflow anchored on
# the last SURVIVING row would stop one game early and the rows in between would
# be processed by nobody.  See ``test_a_dropped_tail_at_the_boundary_is_not_lost``.
#
# ⚑ AND IT IS CONDITIONAL: a worker overflows only when its range-end key DIFFERS
# from its carry-in key.  When they are equal the whole range lies inside one
# game's run, the worker skipped every row of it, and the game belongs to an
# earlier worker that is already overflowing through it -- overflowing again would
# process those rows twice.
#
# **3. The order-dependent floats.**  Seven of the summary's readings are running
# IEEE sums (``temp_recovered_sum``, ``floor_recovered_sum``,
# ``min_legal_prob_sum``, ``qz_regret_sum``, ``qz_instability_sum``,
# ``qz_weight_sum``, ``value_delta_sum``), and float addition is not associative,
# so per-worker subtotals added together are NOT the sequential sum.  Every worker
# therefore BANKS the individual float64 values in the order it produced them, and
# the coordinator replays them through :class:`DeriveStats`'s own ``note_*``
# methods in worker order.  ⚑ REPLAYED THROUGH THE METHODS rather than merged
# field by field: ``note_value_delta`` also maintains a count and a nonzero
# tally, ``note_game`` maintains a first-game sentinel, and a hand-written merge
# of the fields they touch is a second implementation free to drift.  The same
# rule is why the game stream is banked and replayed rather than merged.
#
# **4. The shard permutation.**  ``_flush`` draws ``rng.permutation(n)`` from ONE
# generator seeded with ``--seed``, once per output shard, in shard order.  The
# coordinator can only replay that chain once every surviving row count is known,
# so it does: workers spill, the coordinator prefix-sums their survivor counts,
# draws the whole chain in shard order (the partial last shard included), and
# hands each repack task its own already-drawn order.
#
# Runtime guards, all unconditional
# ---------------------------------
#
# Byte-identity is a claim about a run that SUCCEEDED, so every way this could
# quietly emit a different corpus is checked before the summary is written:
#
# * the inventory's claimed rows against the rows actually streamed, for every
#   shard a worker read to EOF -- the partition's global indices are computed from
#   the claimed counts, so a claim that is wrong puts ``--limit`` in the wrong
#   place;
# * ``sum(rows_read)`` against the rows the partition said would be read;
# * ``sum(rows_written)`` against the rows the output shards actually hold;
# * the workers' closed game keys, pairwise disjoint -- a game assembled by two
#   workers is the split-game failure the handoff exists to prevent, and it is the
#   one failure that produces a plausible-looking corpus;
# * the MERGED envelope-miss count against ``--max-envelope-misses``.  ⚑ A worker
#   can only ever see its own share, so the per-worker refusal that mirrors the
#   sequential one is fail-OPEN across the partition; this is the check that is
#   not.


# ── the ordered streams ──────────────────────────────────────────────────────
#
# ⚑ NAME -> the DeriveStats method that consumes one value.  The coordinator
# replays each stream through its own method, so these are the only places the
# summary's order-dependent readings are ever computed -- here and in the
# sequential path, which is the SAME code.

_ORDERED_STREAMS: tuple[tuple[str, str], ...] = (
    ("temp_recovered", "note_temp"),
    ("floor_recovered", "note_floor"),
    ("min_legal_prob", "note_min_legal_prob"),
    ("qz_regret", "note_qz_regret"),
    ("qz_instability", "note_qz_instability"),
    ("qz_weight", "note_qz_weight"),
    ("value_delta", "note_value_delta"),
)

#: Which ``DeriveStats`` fields each replayed stream OWNS.  A field listed here
#: is produced entirely by the replay and must never also be summed off the
#: partials; :data:`_MERGE_COVERAGE` is what stops the two lists drifting apart.
_STREAM_OWNED_FIELDS: dict[str, tuple[str, ...]] = {
    "temp_recovered": (
        "temp_recovered_n", "temp_recovered_sum",
        "temp_recovered_min", "temp_recovered_max",
    ),
    "floor_recovered": (
        "floor_recovered_n", "floor_recovered_sum",
        "floor_recovered_min", "floor_recovered_max",
    ),
    "min_legal_prob": (
        "min_legal_prob_n", "min_legal_prob_sum",
        "min_legal_prob_min", "min_legal_prob_max",
    ),
    "qz_regret": (
        "qz_regret_n", "qz_regret_sum", "qz_regret_min", "qz_regret_max",
    ),
    "qz_instability": (
        "qz_instability_n", "qz_instability_sum",
        "qz_instability_min", "qz_instability_max",
    ),
    "qz_weight": (
        "qz_weight_n", "qz_weight_sum", "qz_weight_min", "qz_weight_max",
    ),
    # ⚑ ``value_delta_rows_nonzero`` is in here, not in the summed fields:
    # ``note_value_delta`` increments it, so summing it too would double it.
    "value_delta": (
        "value_delta_n", "value_delta_sum", "value_delta_min",
        "value_delta_max", "value_delta_rows_nonzero",
    ),
}

#: Produced by replaying the banked ``note_game`` calls, for the same reason.
_GAME_OWNED_FIELDS: tuple[str, ...] = (
    "qz_games", "qz_game_rows_min", "qz_game_rows_max", "qz_games_cut_by_limit",
)

#: Event counters: every one of them is a count of rows or games, so the
#: partition's disjointness makes the sum exact.
_SUM_FIELDS: tuple[str, ...] = (
    "rows_read",
    "rows_dropped_no_result",
    "rows_dropped_envelope",
    "nodes_floor_hits",
    "support_checks",
    "deep_tier_moves",
    "base_tier_moves",
    "repetition_planes_nonzero_rows",
    "temp_recovery_skipped_saturated",
    "floor_recovery_skipped_few_values",
    "floor_recovery_skipped_ill_conditioned",
    "policy_support_lost_to_float16",
    "qz_single_row_games",
    "qz_ply_gaps_nonunit",
    "qz_games_with_dropped_tail",
    "qz_rows_missing_played_move",
    "qz_rows_missing_depths",
    "qz_stop_terminal",
    "qz_stop_boundary_self",
    "qz_stop_boundary_ahead",
    "qz_boundary_suppressed",
    "qz_w_const_differs_from_map",
    "qz_rows_differ_from_c_full",
)

#: Extrema over rows, which no ordering can move.
_MAX_FIELDS: tuple[str, ...] = (
    "history_slots_nonzero_max",
    "policy_support_max",
)

#: ``{bucket: count}`` histograms, merged key by key.
_DICT_SUM_FIELDS: tuple[str, ...] = (
    "depth_histogram", "values_by_phase", "phases_per_row",
)

#: Assigned (not accumulated) per row, so every row writes the same value and a
#: disagreement between workers is a real corpus fault rather than a merge one.
_CONSTANT_FIELDS: tuple[str, ...] = ("x_planes", "policy_width")

#: ``-1`` until the first row is measured, because ``0`` is a legal support --
#: so a plain ``min`` would adopt an untouched worker's sentinel.
_SENTINEL_MIN_FIELDS: tuple[str, ...] = ("policy_support_min",)

#: Merged from the per-worker examples' GLOBAL row indices; see
#: :func:`_merge_stats`.
_ORDERED_EXAMPLE_FIELDS: tuple[str, ...] = ("envelope_miss_examples",)

#: ⚑ COUNTED WHERE THE SHARDS ARE WRITTEN, which on this path is the repack and
#: not the lanes: a lane spills surviving rows and writes no shard, so summing
#: its ``rows_written`` would report 0 for every run.  The coordinator sets it
#: from the shards that exist and cross-checks it against the survivor counts,
#: which is a stronger statement than the sequential path's running total.
_REPACK_OWNED_FIELDS: tuple[str, ...] = ("rows_written",)

#: ⚑⚑ THE ANTI-DRIFT GATE, and the reason the six lists above are data instead
#: of code.  ``tests/test_derive_parallel.py`` asserts this covers
#: ``dataclasses.fields(DeriveStats)`` exactly, so a counter added to the
#: sequential summary without a merge rule fails a test instead of silently
#: reading 0 in every parallel run -- which is this repo's signature defect
#: aimed squarely at the instrument that would have caught it.
_MERGE_COVERAGE: frozenset[str] = frozenset(
    _SUM_FIELDS
    + _MAX_FIELDS
    + _DICT_SUM_FIELDS
    + _CONSTANT_FIELDS
    + _SENTINEL_MIN_FIELDS
    + _ORDERED_EXAMPLE_FIELDS
    + _REPACK_OWNED_FIELDS
    + _GAME_OWNED_FIELDS
    + tuple(name for owned in _STREAM_OWNED_FIELDS.values() for name in owned),
)


class ParallelDeriveError(CorpusIntegrityError):
    """A guard on the parallel path refused.  A ``CorpusIntegrityError``, so the
    caller's existing handling of a refusal is unchanged."""


# ── the banking stats ────────────────────────────────────────────────────────


class _BankingStats(DeriveStats):
    """``DeriveStats`` that also keeps every ordered value it was handed.

    ⚑ A SUBCLASS, so the counters the summary reads are still maintained by the
    sequential code.  The bank is a pure ADDITION alongside each ``note_*``; if
    one of them ever stops being a running fold the parallel path keeps agreeing
    with the sequential one by construction rather than by a second opinion.
    """

    def __init__(self, spill_dir: Path | None = None) -> None:
        super().__init__()
        #: ⚑ A BUFFER, NOT AN ARCHIVE. `min_legal_prob` and `value_delta` take a
        #: value from EVERY surviving row and a grouped run adds four more, so
        #: holding a lane's whole range would be ~6 lists of ~790k Python floats
        #: on a 5.5M-row / 7-lane derivation -- around a gigabyte of small
        #: objects that exists only to be summed once. `drain` appends them to
        #: their files at every spill cut, so what is resident is one cut's
        #: worth. ⚑ RAW float64 BYTES, appended: the fold has to see the exact
        #: doubles the sequential path folded, and `tofile`/`fromfile` is the
        #: form with no header to keep consistent across appends.
        self.spill_dir = spill_dir
        self.banked: dict[str, list[float]] = {
            name: [] for name, _ in _ORDERED_STREAMS
        }
        #: ``(rows, cut_by_limit)`` per assembled game, in assembly order.
        self.banked_games: list[tuple[int, bool]] = []
        #: ``(global_row_index, text)`` for the first envelope misses this
        #: worker saw.  The GLOBAL index is what lets the coordinator take the
        #: first eight in the sequential order rather than the first eight of
        #: whichever worker happened to finish first.
        self.banked_envelope: list[tuple[int, str]] = []

    def note_temp(self, value: float) -> None:
        self.banked["temp_recovered"].append(value)
        super().note_temp(value)

    def note_floor(self, value: float) -> None:
        self.banked["floor_recovered"].append(value)
        super().note_floor(value)

    def note_min_legal_prob(self, value: float) -> None:
        self.banked["min_legal_prob"].append(value)
        super().note_min_legal_prob(value)

    def note_qz_regret(self, value: float) -> None:
        self.banked["qz_regret"].append(value)
        super().note_qz_regret(value)

    def note_qz_instability(self, value: float) -> None:
        self.banked["qz_instability"].append(value)
        super().note_qz_instability(value)

    def note_qz_weight(self, value: float) -> None:
        self.banked["qz_weight"].append(value)
        super().note_qz_weight(value)

    def note_value_delta(self, value: float) -> None:
        self.banked["value_delta"].append(value)
        super().note_value_delta(value)

    def note_game(self, rows: int, *, cut_by_limit: bool) -> None:
        self.banked_games.append((int(rows), bool(cut_by_limit)))
        super().note_game(rows, cut_by_limit=cut_by_limit)

    def stream_path(self, name: str) -> Path:
        if self.spill_dir is None:
            raise ParallelDeriveError(
                "this DeriveStats was built without a spill directory, so it "
                "cannot bank its ordered streams",
            )
        return self.spill_dir / f"stream_{name}.f64"

    def drain(self) -> None:
        """Append what has accumulated since the last call, in order."""
        for name, values in self.banked.items():
            if not values:
                continue
            with open(self.stream_path(name), "ab") as handle:
                np.asarray(values, dtype=np.float64).tofile(handle)
            values.clear()

    def plain(self) -> DeriveStats:
        """A picklable ``DeriveStats`` holding exactly this worker's counters."""
        return DeriveStats(**{
            spec.name: getattr(self, spec.name) for spec in fields(DeriveStats)
        })


# ── the partition ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShardRange:
    """One worker's contiguous slice of the record's shard snapshot."""

    lo: int
    hi: int

    @property
    def empty(self) -> bool:
        return self.hi <= self.lo


def shard_row_counts(record: CorpusRecord) -> tuple[int, ...]:
    """Rows the corpus's INVENTORY claims for each shard, in snapshot order.

    ⚑ CLAIMED, not measured: measuring would mean decompressing the whole corpus
    before the first row is derived.  The claims decide where ``--limit`` falls,
    so every shard a worker reads to EOF is checked against its claim while it
    streams and the run is refused on a mismatch -- before any output shard is
    finalised.  See :func:`_check_claimed_rows`.

    ⚑ READ OFF THE RECORD, not re-read off the inventory: both of
    ``read_corpus_record``'s branches already parse these numbers, and a second
    reader here would be free to drift from the one the sequential path's
    ``rows_claimed`` comes from.
    """
    counts = record.shard_rows
    if len(counts) != len(record.shards):
        raise ParallelDeriveError(
            f"the corpus record carries {len(counts)} per-shard row claims for "
            f"{len(record.shards)} shards. --workers partitions by claimed "
            "rows and cuts --limit at a global row index, so a shard with no "
            "claim cannot be placed; derive this corpus with --workers 1.",
        )
    negative = [
        record.shards[index].name
        for index, value in enumerate(counts) if int(value) < 0
    ]
    if negative:
        raise ParallelDeriveError(
            f"the corpus inventory claims a negative row count for {negative[:4]} "
            f"({len(negative)} shard(s)); the partition cannot be placed.",
        )
    return tuple(int(value) for value in counts)


def plan_ranges(
    counts: Sequence[int], *, workers: int, limit: int,
) -> tuple[list[ShardRange], int, int]:
    """Split the shards ``--limit`` will actually reach into balanced ranges.

    Returns the ranges, the number of shards in play, and the number of input
    rows the run will read.

    ⚑ THE SHARDS PAST ``--limit`` ARE NOT PARTITIONED.  A 5.5M-row limit over a
    10M-row corpus touches the first ~671 of 1,232 shards; splitting all 1,232
    evenly would leave three of seven workers with nothing to do and cost most
    of the speedup, while changing no output.  The ranges still cover a
    contiguous prefix, so the global row indices are unchanged.

    ⚑⚑ AND ONLY WHEN THE LIMIT STRICTLY BINDS, because otherwise the guard that
    checks the claims is scoped BY the claims.  A shard is dropped from play on
    the strength of the inventory's numbers, and ``_check_claimed_rows`` can only
    check a shard some lane opened -- so a shard whose claim UNDERSTATES what it
    holds could be excluded on a claim nothing ever tests.  MEASURED: three
    8-row shards, the inventory claiming 0 for the third, ``--limit 30``:
    ``--workers 1`` reads 24 rows and above 1 read 16, no guard firing, a smaller
    corpus written and stamped.  ⚑ That is only reachable when ``limit >= the
    claimed total`` -- when the limit falls INSIDE the claims, the shards it
    excludes start past the row the sequential read stops on, and every shard
    before it is opened and its claim checked.  So a non-binding limit puts every
    shard back in play, where a lying claim is caught rather than believed.
    """
    if workers < 1:
        raise ValueError(f"--workers must be >= 1, got {workers}")
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + int(count))
    total = prefix[-1]
    rows_to_read = min(limit, total) if limit else total
    shards_in_play = len(counts)
    if limit and limit < total:
        # The last shard holding a row below the limit; the ones after it are
        # never opened by any worker, exactly as the sequential loop's `break`
        # never opens them.
        shards_in_play = 0
        for index in range(len(counts)):
            if prefix[index] < rows_to_read:
                shards_in_play = index + 1
    if shards_in_play == 0:
        raise ParallelDeriveError(
            "--limit leaves no corpus shard to read; the sequential path would "
            "write nothing and refuse.",
        )
    lanes = min(workers, shards_in_play)
    edges = [0]
    for lane in range(1, lanes):
        want = rows_to_read * lane / lanes
        index = edges[-1] + 1
        while index < shards_in_play and prefix[index] < want:
            index += 1
        # Every remaining lane keeps at least one shard, and every lane keeps
        # the one it already has: an empty range would give a lane no range-end
        # key, and the handoff is defined by one.
        edges.append(max(
            edges[-1] + 1, min(index, shards_in_play - (lanes - lane)),
        ))
    edges.append(shards_in_play)
    ranges = [ShardRange(edges[i], edges[i + 1]) for i in range(lanes)]
    if any(item.empty for item in ranges):
        raise ParallelDeriveError(
            f"the partition produced an empty range ({ranges}); this is a bug "
            "in plan_ranges, not a corpus fault",
        )
    return ranges, shards_in_play, rows_to_read


# ── the worker ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _WorkerTask:
    index: int
    shards: tuple[Path, ...]
    prefix: tuple[int, ...]
    span: ShardRange
    options: DeriveOptions
    corpus_sha: str
    spill_dir: Path
    shards_in_play: int


@dataclass
class _WorkerResult:
    index: int
    stats: DeriveStats
    #: ``(rows, cut_by_limit)`` per game, in assembly order.
    games: list[tuple[int, bool]]
    #: ``(global_row_index, text)`` for this worker's first envelope misses.
    envelope: list[tuple[int, str]]
    #: The raw-float64 file this lane banked each ordered stream into.
    stream_paths: dict[str, str]
    #: One entry per spill file, in this worker's emission order.
    chunk_rows: list[int]
    #: ``{shard_index: rows}`` for every shard this worker read to EOF.
    actual_rows: dict[int, int]
    closed_keys: list[tuple[int, int]]
    tt_carried: list[bool]
    survivors: int


def _raw_game_key(row: dict[str, Any]) -> tuple[int, int]:
    """The row's ``(worker_id, game_id)`` READ OFF THE JSON, drops included.

    ⚑ The handoff's key must not depend on whether a row survives derivation:
    the skipping worker sees the raw row and the overflowing worker must apply
    the identical predicate to the identical row.  ``GameGrouper._key_of``
    answers the same question for rows that DID survive and raises a
    scheme-specific refusal for ones that cannot; this one is deliberately the
    plain read.

    ⚑ IT RUNS FIRST ON A GROUPED ARM, so a row with no game identity is refused
    HERE rather than by ``_key_of``, and the two messages name different flags --
    this one ``--workers``, that one ``--value-scheme``. Both are true and the
    corpus fault is the same; the wording differs between ``--workers 1`` and
    above it, which is stated so nobody reads the difference as a difference in
    what the two paths accept. They accept the same rows.

    ⚑ Read by SUBSCRIPT and raised on, for the same reason ``_key_of`` is: the
    partition is defined by these keys, and a row filed under a placeholder
    would put a game boundary where there is none.  ⚑ ``_key_of`` reaches every
    row of a grouped run too -- both drop paths call ``note_dropped`` -- so this
    refuses exactly the rows the sequential path refuses, only sooner, and it
    says the same thing.
    """
    try:
        return (int(row["worker_id"]), int(row["game_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusIntegrityError(
            f"{_row_label(row)} carries no usable (worker_id, game_id), so "
            "--workers cannot tell which lane's game it belongs to. The "
            "grouped schemes assemble whole games from contiguous runs of "
            "rows; a row that cannot be attributed to one is not a row they "
            "can derive.",
        ) from exc


def _carry_in_key(
    shards: Sequence[Path], before: int,
) -> tuple[int, int] | None:
    """The raw key of the last row BEFORE ``shards[before]``, or ``None``.

    ⚑ SCANS BACK OVER EMPTY SHARDS rather than reading only ``before - 1``.  A
    zero-row shard has no last row, and stopping there would hand the lane a
    ``None`` carry-in -- it would skip nothing, while the lane before it is
    overflowing through the same rows on the key it read past the empty shard.
    Every row derived twice, and the only thing that would notice is the
    ``rows_read`` guard, which is a crash rather than a correct read.  The
    mirror holds because the overflowing lane's range-end key is likewise the
    last key it SAW, not the last shard it opened.
    """
    for index in range(before - 1, -1, -1):
        key: tuple[int, int] | None = None
        for row in iter_corpus_rows(shards[index]):
            key = _raw_game_key(row)
        if key is not None:
            return key
    return None


def _spill_path(spill_dir: Path, worker: int, chunk: int) -> Path:
    return spill_dir / f"w{worker:03d}" / f"chunk_{chunk:06d}.zarr"


def _write_spill_chunk(path: Path, samples: list[ReplaySample]) -> None:
    """Bank one run of surviving rows, and PROVE the banking is lossless.

    ⚑⚑ THE ROUND TRIP IS CHECKED HERE, ON EVERY CHUNK, and it is the whole
    correctness argument for the spill.  The repack rebuilds ``ReplaySample``\\ s
    out of these files and hands them to the sequential writer, so anything the
    columnar form cannot carry -- a field ``derive_row`` starts setting
    tomorrow, an optional column the pruner drops -- would come back absent and
    the output would differ from the sequential corpus in a way no other check
    looks at.  Comparing ``samples_to_arrays`` BEFORE against
    ``samples_to_arrays(arrays_to_samples(<read back off disk>))`` AFTER states
    exactly the property the repack depends on, over the same reader and writer
    the repack uses, and costs a few percent of a derivation that is dominated
    by JSON parsing and position encoding.

    ⚑ Read back off DISK rather than compared in memory: the serializer sits
    between the two, and ``_verify_value_column_on_disk`` exists in the
    sequential path for exactly that reason.
    """
    want = samples_to_arrays(samples)
    save_local_shard_arrays(
        path,
        arrs=want,
        meta=ShardMeta(
            run_id=_SPILL_RUN_ID,
            input_history_encoding=INPUT_HISTORY_ENCODING,
            history_rep_fix=HISTORY_REP_FIX,
            policy_encoding="lc0_1858",
            policy_size=COMPACT_POLICY_SIZE,
            positions=len(samples),
        ),
    )
    stored, _ = load_shard_arrays(path, lazy=False)
    got = samples_to_arrays(arrays_to_samples(dict(stored)))
    if set(got) != set(want):
        raise ParallelDeriveError(
            f"{path.name}: the spill round trip changed which columns the rows "
            f"carry (lost {sorted(set(want) - set(got))}, gained "
            f"{sorted(set(got) - set(want))}). The repacked shard would not be "
            "the shard the sequential path writes.",
        )
    for key in sorted(want):
        if not np.array_equal(np.asarray(got[key]), np.asarray(want[key])):
            raise ParallelDeriveError(
                f"{path.name}: column {key!r} did not survive the spill round "
                "trip unchanged, so the repacked shard would differ from the "
                "sequential one in that column.",
            )


def _run_worker(task: _WorkerTask) -> _WorkerResult:
    """Derive one contiguous range of shards, plus its game handoff, to spill."""
    options = task.options
    limit = int(options.limit)
    deriver = TargetDeriver(options)
    spill_dir = task.spill_dir / f"w{task.index:03d}"
    spill_dir.mkdir(parents=True, exist_ok=True)
    bank = _BankingStats(spill_dir)
    deriver.stats = bank
    grouper = GameGrouper(deriver) if options.needs_game else None

    span = task.span
    carry_in: tuple[int, int] | None = None
    if grouper is not None and span.lo > 0:
        carry_in = _carry_in_key(task.shards, span.lo)

    buffered: list[ReplaySample] = []
    chunk_rows: list[int] = []
    survivors = 0
    tt_carried: set[bool] = set()
    actual_rows: dict[int, int] = {}
    max_gidx: int = -1
    skipping = carry_in is not None
    # The raw key of the last row seen inside the worker's OWN range; the
    # overflow predicate and the next worker's skip predicate both name it.
    range_end_key: tuple[int, int] | None = None

    def cut(rows: list[ReplaySample]) -> None:
        _write_spill_chunk(
            _spill_path(task.spill_dir, task.index, len(chunk_rows)), rows,
        )
        chunk_rows.append(len(rows))
        bank.drain()

    def emit(batch: GameBatch | None) -> None:
        nonlocal survivors
        if batch is None or not batch.rows:
            return
        produced = apply_value_scheme(
            batch.rows,
            options=options,
            stats=deriver.stats,
            banked_tail_ply=batch.banked_tail_ply,
        )
        buffered.extend(produced)
        survivors += len(produced)
        while len(buffered) >= options.spill_chunk_rows:
            cut(buffered[:options.spill_chunk_rows])
            del buffered[:options.spill_chunk_rows]

    stop = False
    for shard_index in range(span.lo, task.shards_in_play):
        overflow = shard_index >= span.hi
        if overflow and (
            grouper is None or range_end_key is None or range_end_key == carry_in
        ):
            # Nothing to carry: the ungrouped schemes assemble no game, or this
            # worker's whole range lay inside the carried-in game -- which an
            # EARLIER worker owns and is already overflowing through, so
            # carrying it again would derive those rows twice.
            break
        base = task.prefix[shard_index]
        seen = 0
        complete = True
        for row in iter_corpus_rows(task.shards[shard_index]):
            gidx = base + seen
            seen += 1
            if limit and gidx >= limit:
                complete = False
                stop = True
                break
            if overflow:
                if _raw_game_key(row) != range_end_key:
                    complete = False
                    stop = True
                    break
            elif grouper is not None:
                # ⚑ ONLY THE GROUPED SCHEMES KEY THE ROW.  V0 and A never
                # assemble a game, so nothing here needs one -- and the
                # sequential path does not read `worker_id` on those arms at
                # all.  Keying every row unconditionally would make a corpus
                # that derives fine at `--workers 1` refuse above it, which is
                # a difference in what the flag ACCEPTS rather than in what it
                # writes, and no output comparison would ever show it.
                range_end_key = _raw_game_key(row)
                if skipping:
                    if range_end_key == carry_in:
                        continue
                    skipping = False
            max_gidx = gidx
            deriver.stats.rows_read += 1
            tt_carried.add(_check_row_identity(row, task.corpus_sha))
            try:
                derived = deriver.derive_row(row)
            except EnvelopeMiss as exc:
                deriver.stats.rows_dropped_envelope += 1
                if grouper is not None:
                    grouper.note_dropped(row)
                if len(bank.banked_envelope) < 8:
                    bank.banked_envelope.append(
                        (gidx, f"{_row_label(row)}: {exc}"),
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
            if derived is None:
                if grouper is not None:
                    grouper.note_dropped(row)
                continue
            if grouper is None:
                emit(GameBatch(rows=[derived]))
            else:
                emit(grouper.add(row, derived))
        if complete and not overflow:
            actual_rows[shard_index] = seen
        if stop:
            break

    if grouper is not None:
        # ⚑ THE SAME PREDICATE THE SEQUENTIAL FLUSH USES, expressed in global
        # indices: ``rows_read >= limit`` there is true exactly when the read
        # consumed the row at index ``limit - 1``, and exactly one lane does.
        # A lane that stopped because its range ended, or because the carried
        # game did, flushes ``False`` -- the game ended, the budget did not.
        emit(grouper.flush(cut_by_limit=bool(limit and max_gidx + 1 >= limit)))
    if buffered:
        cut(list(buffered))
        buffered.clear()

    bank.drain()
    stream_paths = {
        name: str(bank.stream_path(name))
        for name, _ in _ORDERED_STREAMS
        if bank.stream_path(name).exists()
    }

    return _WorkerResult(
        index=task.index,
        stats=bank.plain(),
        games=list(bank.banked_games),
        envelope=list(bank.banked_envelope),
        stream_paths=stream_paths,
        chunk_rows=chunk_rows,
        actual_rows=actual_rows,
        closed_keys=sorted(grouper.closed_keys) if grouper is not None else [],
        tt_carried=sorted(tt_carried),
        survivors=survivors,
    )


# ── the repack ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SpillSlice:
    worker: int
    chunk: int
    lo: int
    hi: int


@dataclass(frozen=True)
class _RepackTask:
    index: int
    slices: tuple[_SpillSlice, ...]
    order: np.ndarray
    options: DeriveOptions
    corpus_sha: str
    out_dir: Path
    spill_dir: Path


def _slice_arrays(arrs: dict[str, Any], lo: int, hi: int) -> dict[str, np.ndarray]:
    """Rows ``[lo, hi)`` of a spill chunk's columns.

    ⚑ The two encoding-identity markers (``input_history_encoding``,
    ``history_rep_fix``) are 0-d scalars describing the whole file and are
    carried through unsliced; every other column is per row.
    """
    out: dict[str, np.ndarray] = {}
    for key, value in arrs.items():
        array = np.asarray(value)
        out[key] = array if array.ndim == 0 else array[lo:hi]
    return out


def _repack_shard(task: _RepackTask) -> dict[str, Any]:
    """Rebuild one output shard's rows and write it through the sequential writer."""
    samples: list[ReplaySample] = []
    for part in task.slices:
        arrs, _ = load_shard_arrays(
            _spill_path(task.spill_dir, part.worker, part.chunk), lazy=False,
        )
        samples.extend(arrays_to_samples(_slice_arrays(dict(arrs), part.lo, part.hi)))
    order = np.asarray(task.order)
    if order.shape != (len(samples),):
        raise ParallelDeriveError(
            f"shard {task.index}: the replayed permutation is for "
            f"{order.shape} rows and the repack assembled {len(samples)}; the "
            "seed chain and the shard layout disagree.",
        )
    ordered = [samples[int(i)] for i in order]
    return _flush_ordered(
        task.out_dir, task.index, ordered, task.options, task.corpus_sha,
    )


def _measure_spill(
    spill_dir: Path, results: Sequence[_WorkerResult],
) -> None:
    """Every spill file holds the rows its lane said it holds.

    ⚑ READ OFF DISK, and that is the point.  The obvious version of this check
    compares the lanes' ``chunk_rows`` with their ``survivors`` -- and both are
    incremented from the same ``len()`` inside the same function, so it is true
    by construction and cannot fire (an independent reviewer of this PR proved
    it).  A gate that cannot fail is this repo's signature defect, so the
    comparison is made against the ``x`` column's own first dimension instead:
    a different source, answering the same question, before the repack reads a
    single row -- and it fires on a real fault (a lane that writes ``rows[:-1]``
    while reporting ``len(rows)`` is invisible to the per-chunk round trip,
    which compares the file with what it was handed and is self-consistent).
    """
    for item in results:
        for chunk, rows in enumerate(item.chunk_rows):
            path = _spill_path(spill_dir, item.index, chunk)
            stored = int(np.asarray(
                zarr.open_group(str(path), mode="r")["x"].shape,
            )[0])
            if stored != rows:
                raise ParallelDeriveError(
                    f"{path.name}: lane {item.index} banked {rows} rows and the "
                    f"file holds {stored}. The repack cuts output shards by "
                    "counting those rows, so it would take the wrong slice of "
                    "every shard after this one.",
                )


def _plan_repack(
    survivors: Sequence[int], chunk_rows: Sequence[Sequence[int]], rows_per_shard: int,
) -> list[tuple[_SpillSlice, ...]]:
    """Map each output shard onto the spill slices that hold its rows, in order."""
    flat: list[tuple[int, int, int, int]] = []
    cursor = 0
    for worker, chunks in enumerate(chunk_rows):
        for chunk, rows in enumerate(chunks):
            flat.append((cursor, cursor + rows, worker, chunk))
            cursor += rows
    if cursor != sum(survivors):
        # ⚑ NOT A GATE ON THE SPILL -- `_measure_spill` already read those rows
        # off disk. Both numbers here come from `chunk_rows`, so this is an
        # invariant of the loop above it and cannot fire on a corpus; it is kept
        # because a future caller could pass the two apart, and it is labelled
        # so nobody counts it as coverage.
        raise ParallelDeriveError(
            f"the spill files hold {cursor} rows and the workers reported "
            f"{sum(survivors)} survivors; the two must be the same rows.",
        )
    plans: list[tuple[_SpillSlice, ...]] = []
    start = 0
    while start < cursor:
        end = min(start + rows_per_shard, cursor)
        parts: list[_SpillSlice] = []
        for lo, hi, worker, chunk in flat:
            if hi <= start or lo >= end:
                continue
            parts.append(_SpillSlice(
                worker=worker, chunk=chunk,
                lo=max(start, lo) - lo, hi=min(end, hi) - lo,
            ))
        plans.append(tuple(parts))
        start = end
    return plans


# ── merging ──────────────────────────────────────────────────────────────────


def _stream_files(
    results: Sequence[_WorkerResult],
) -> dict[str, list[str]]:
    """Each banked stream's files, IN LANE ORDER.

    ⚑ LANE ORDER IS READ ORDER.  Lane ``k``'s rows are all globally after lane
    ``k-1``'s -- the ranges are contiguous and the handoff hands each boundary
    game to exactly one of them -- so reading the lanes in index order
    reproduces the sequence the sequential read produced these values in.  A
    ``dict``-iteration or completion order here would be a different sequence
    and therefore a different IEEE sum.

    ⚑ PATHS, NOT VALUES, and that is the point.  Returning the concatenated
    floats would undo what the lanes' incremental ``drain`` bought: ~6 per-row
    streams over a 5.5M-row corpus is ~1 GB of Python floats, and materialising
    all of it in the COORDINATOR is worse than in the lanes, because it is one
    process and adding lanes cannot reduce it.  :func:`_stream_values` walks the
    files one at a time, so the peak is one lane's file.

    ⚑ A FUNCTION so the ordering is testable without deriving a corpus: a small
    corpus's subtotals are usually bit-identical to its sequential sum anyway,
    so an end-to-end test cannot be relied on to notice.
    """
    streams: dict[str, list[str]] = {}
    for name, _ in _ORDERED_STREAMS:
        paths = [
            path
            for item in sorted(results, key=lambda entry: entry.index)
            if (path := item.stream_paths.get(name)) is not None
        ]
        if paths:
            streams[name] = paths
    return streams


def _stream_values(paths: Sequence[str]) -> Iterator[float]:
    """The banked doubles, one file at a time, in the order they were banked.

    ⚑ ``fromfile`` per path rather than one concatenation: the fold only ever
    needs the next value, and holding the whole sequence is the memory this
    design exists to avoid.  ``tolist`` on ONE file still materialises that
    file, which is bounded by a lane's share and is the point at which the cost
    stops scaling with the corpus.
    """
    for path in paths:
        yield from np.fromfile(path, dtype=np.float64).tolist()


def _merge_stats(
    results: Sequence[_WorkerResult], *, streams: Mapping[str, Sequence[str]],
) -> DeriveStats:
    """One ``DeriveStats`` describing the whole run, in the sequential order.

    ``streams`` maps each banked stream to its lanes' files in lane order; the
    values are replayed through the very methods the sequential path calls,
    which is what makes the IEEE sums identical rather than merely close.
    """
    merged = DeriveStats()
    # ⚑ LANE ORDER, like `_concat_streams`: `note_game`'s first-game sentinel
    # makes the game replay order-sensitive too, and two functions that disagree
    # about who owns the ordering is one refactor away from a bug.
    results = sorted(results, key=lambda item: item.index)
    for name in _SUM_FIELDS:
        setattr(merged, name, sum(getattr(item.stats, name) for item in results))
    for name in _MAX_FIELDS:
        setattr(merged, name, max(
            (getattr(item.stats, name) for item in results), default=0,
        ))
    for name in _DICT_SUM_FIELDS:
        totals: dict[int, int] = {}
        for item in results:
            for key, value in getattr(item.stats, name).items():
                totals[key] = totals.get(key, 0) + value
        setattr(merged, name, totals)
    for name in _CONSTANT_FIELDS:
        seen = {getattr(item.stats, name) for item in results} - {0}
        if len(seen) > 1:
            raise ParallelDeriveError(
                f"the workers disagree about {name}: {sorted(seen)}. Every row "
                "of a corpus carries the same shape, so this is the corpus "
                "changing shape mid-read, not a merge fault.",
            )
        setattr(merged, name, seen.pop() if seen else 0)
    for name in _SENTINEL_MIN_FIELDS:
        measured = [
            getattr(item.stats, name) for item in results
            if getattr(item.stats, name) >= 0
        ]
        setattr(merged, name, min(measured) if measured else -1)
    # ⚑ Off the rule rather than beside it: a field named in one place as data
    # and in another as code is two rules that can disagree.
    (example_field,) = _ORDERED_EXAMPLE_FIELDS
    setattr(merged, example_field, [
        text for _, text in sorted(
            (entry for item in results for entry in item.envelope),
            key=lambda entry: entry[0],
        )[:8]
    ])
    for rows, cut in [game for item in results for game in item.games]:
        merged.note_game(rows, cut_by_limit=cut)
    for name, method in _ORDERED_STREAMS:
        note = getattr(merged, method)
        for value in _stream_values(streams.get(name, ())):
            note(value)
    return merged


def _check_claimed_rows(
    results: Sequence[_WorkerResult], counts: Sequence[int], shards: Sequence[Path],
) -> None:
    """Every shard a worker read to EOF held the rows the inventory claimed."""
    problems: list[str] = []
    for item in results:
        for shard_index, actual in sorted(item.actual_rows.items()):
            claimed = int(counts[shard_index])
            if actual != claimed:
                problems.append(
                    f"{shards[shard_index].name}: inventory claims {claimed}, "
                    f"holds {actual}",
                )
    if problems:
        raise ParallelDeriveError(
            "the corpus inventory's row counts are not the rows on disk: "
            + "; ".join(problems[:6])
            + f" ({len(problems)} shard(s)). --workers places --limit at a "
            "global row index computed from those claims, so the partition "
            "would cut somewhere the sequential read does not. Nothing was "
            "finalised; derive with --workers 1.",
        )


def _check_rows_read(
    results: Sequence[_WorkerResult], rows_to_read: int,
) -> None:
    """The lanes between them read the corpus prefix the partition covers.

    ⚑ THE ONLY THING THAT NOTICES OVERLAPPING RANGES ON AN UNGROUPED ARM.  V0 and
    A assemble no game, so `_check_closed_keys` has nothing to compare; a handoff
    that let two lanes derive the same rows shows up here and nowhere else.  (On
    a grouped arm the reverse holds, which is why both are kept.)
    """
    rows_read = sum(item.stats.rows_read for item in results)
    if rows_read != rows_to_read:
        raise ParallelDeriveError(
            f"the lanes read {rows_read} corpus rows and the partition covers "
            f"{rows_to_read}; the ranges do not tile the corpus prefix the "
            "sequential read walks.",
        )


def _check_envelope_budget(
    results: Sequence[_WorkerResult], max_envelope_misses: int,
) -> None:
    """``--max-envelope-misses`` is a budget over the WHOLE read.

    ⚑⚑ THE FAIL-OPEN A PARTITION CREATES.  Each lane carries its own copy of the
    sequential refusal and each lane sees only its own share, so three misses
    split three ways pass every lane's budget of 1 and the run succeeds where
    ``--workers 1`` refuses.  Nothing else looks at the total.
    """
    misses = sum(item.stats.rows_dropped_envelope for item in results)
    if misses > max_envelope_misses:
        raise CorpusIntegrityError(
            f"{misses} envelope miss(es) across the whole read against "
            f"--max-envelope-misses {max_envelope_misses}. Each lane sees only "
            "its own share, so this is the count the sequential run would have "
            "refused on. Dropping rows changes which positions the corpus "
            "contains, so the tolerance is stated rather than assumed.",
        )


def _check_rows_written(
    written: Sequence[dict[str, Any]], survivors: Sequence[int],
) -> int:
    """The shards the repack planned hold every surviving row exactly once.

    ⚑ AN ASSERTION ABOUT THE REPACK'S TILING, not about the writer: it compares
    what `_flush_ordered` was handed with what the lanes produced, and both are
    upstream of the bytes.  The on-disk statement is
    `_verify_value_column_on_disk`, per shard, inside the writer.  Stated because
    the two are easy to conflate and only one of them has read a file.
    """
    rows_written = sum(int(entry["rows"]) for entry in written)
    if rows_written != sum(survivors):
        raise ParallelDeriveError(
            f"the repacked shards were handed {rows_written} rows and the lanes "
            f"produced {sum(survivors)} surviving rows; the repack lost or "
            "duplicated rows.",
        )
    return rows_written


def _check_closed_keys(results: Sequence[_WorkerResult]) -> None:
    """No game was assembled by two workers."""
    owner: dict[tuple[int, int], int] = {}
    clashes: list[str] = []
    for item in results:
        for key in item.closed_keys:
            previous = owner.get(key)
            if previous is not None:
                clashes.append(f"worker {key[0]} game {key[1]} closed by "
                               f"lanes {previous} and {item.index}")
            else:
                owner[key] = item.index
    if clashes:
        raise ParallelDeriveError(
            "the same game was assembled by two lanes: "
            + "; ".join(clashes[:6])
            + f" ({len(clashes)} game(s)). Each fragment's last row would be "
            "treated as terminal and handed the game's outcome, which is a "
            "corpus that looks entirely normal and whose grouped-scheme targets "
            "are wrong in the middle of every split game.",
        )


# ── the driver ───────────────────────────────────────────────────────────────


def derive_parallel(
    *,
    corpus_dir: Path,
    out_dir: Path,
    options: DeriveOptions,
    corpus_record: CorpusRecord | None = None,
    workers: int,
    context: str = "spawn",
) -> dict[str, Any]:
    """The ``--workers N > 1`` path.  Writes what ``--workers 1`` writes."""
    if workers < 2:
        raise ValueError(
            f"derive_parallel is the N > 1 path and was asked for {workers}; "
            "the sequential derivation is derive_corpus_targets.derive, and "
            "routing --workers 1 through here would give the default path no "
            "independent implementation to be compared against.",
        )
    started = datetime.now(timezone.utc).isoformat()
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
    value_problems = value_scheme_vs_staircase_problems(
        options.value_scheme, summary.get("staircase_parsed", []),
    )
    if value_problems:
        raise CorpusIntegrityError(
            f"--value-scheme {options.value_scheme} cannot be answered by this "
            "corpus: " + "; ".join(value_problems),
        )
    shards = list(record.shards)
    if not shards:
        raise CorpusIntegrityError(
            f"{corpus_dir} holds no .jsonl.zst/.jsonl.gz shards",
        )
    corpus_sha = str(summary.get("config_sha256", ""))

    counts = shard_row_counts(record)
    ranges, shards_in_play, rows_to_read = plan_ranges(
        counts, workers=workers, limit=int(options.limit),
    )
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + int(count))

    out_dir.mkdir(parents=True, exist_ok=True)
    spill_dir = out_dir / SPILL_DIR_NAME
    spill_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context(context)

    tasks = [
        _WorkerTask(
            index=index,
            shards=tuple(shards),
            prefix=tuple(prefix),
            span=span,
            options=options,
            corpus_sha=corpus_sha,
            spill_dir=spill_dir,
            shards_in_play=shards_in_play,
        )
        for index, span in enumerate(ranges)
    ]
    # ⚑ ONE pool for BOTH phases.  Under the "spawn" start method every lane
    # re-imports this module and its torch-backed dependencies, which is several
    # seconds each; a second pool for the repack would pay that twice for no
    # isolation the first pool does not already give.
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_run_worker, tasks, chunksize=1)
        results.sort(key=lambda item: item.index)

        _check_claimed_rows(results, counts, shards)
        _check_closed_keys(results)
        _check_rows_read(results, rows_to_read)
        _check_envelope_budget(results, options.max_envelope_misses)

        survivors = [item.survivors for item in results]
        if sum(survivors) == 0:
            raise CorpusIntegrityError(
                "no rows survived; nothing was written. Read the drop counters "
                "before rerunning: an empty corpus and a corpus every row of which "
                "was dropped are different problems.",
            )
        _measure_spill(spill_dir, results)
        plans = _plan_repack(
            survivors, [item.chunk_rows for item in results], options.rows_per_shard,
        )
        # ⚑ ONE generator, drawn in shard order, exactly as `_flush` draws it.
        rng = np.random.default_rng(options.seed)
        sizes = [sum(part.hi - part.lo for part in plan) for plan in plans]
        orders = [rng.permutation(size) for size in sizes]

        repack = [
            _RepackTask(
                index=index,
                slices=plan,
                order=orders[index],
                options=options,
                corpus_sha=corpus_sha,
                out_dir=out_dir,
                spill_dir=spill_dir,
            )
            for index, plan in enumerate(plans)
        ]
        written = pool.map(_repack_shard, repack, chunksize=1)

    stats = _merge_stats(results, streams=_stream_files(results))
    stats.rows_written = _check_rows_written(written, survivors)

    enforce_take_effect(options, stats)
    out = build_summary(
        options=options,
        stats=stats,
        corpus_dir=corpus_dir,
        corpus_record=record,
        shards=written,
        started_utc=started,
        tt_carried={flag for item in results for flag in item.tt_carried},
    )
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(out, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _remove_spill(spill_dir)
    return out


def _remove_spill(spill_dir: Path) -> None:
    """Drop the intermediate rows once the summary describes the shards.

    ⚑ AFTER the summary, never before: the summary is written last on purpose
    (a directory of shards with no summary is a run that DIED), and deleting the
    only other copy of the rows before that stamp exists would turn a crash in
    the last few lines into a full re-derivation.
    """
    shutil.rmtree(spill_dir, ignore_errors=True)


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
        "--value-scheme", default=VALUE_SCHEME_SEARCH,
        help="which of the value round's four arms writes search_wdl. "
             f"{VALUE_SCHEME_SEARCH} (default, V0) is the searched root value "
             f"unchanged; {VALUE_SCHEME_QZ50} (A) is 0.5*Q + 0.5*Z; "
             f"{VALUE_SCHEME_QZPHASE} (B) ramps Z in by ply/terminal_ply; "
             f"{VALUE_SCHEME_QZSEGMENT} (C) blends in the outcome of the row's "
             "CLEAN SEGMENT — the forward run of plies before the first blunder "
             "— at a weight set by the teacher's d9/d8/d7 instability. Only the "
             "last two assemble whole games.",
    )
    parser.add_argument(
        "--qz-r-boundary", type=float, default=None,
        help=f"--value-scheme {VALUE_SCHEME_QZSEGMENT} only: the played-move "
             "regret, in q units, above which a transition ends the clean "
             f"segment. Default {QZ_R_BOUNDARY} — the half-credit point of the "
             f"prereg's frozen soft gate (r_free {QZ_R_FREE_CALIBRATED} + tau_r "
             f"{QZ_TAU_R_CALIBRATED} * sqrt(ln 2)), rounded as the ledger froze it.",
    )
    parser.add_argument(
        # ⚑ TWO SPELLINGS, ONE dest, and both live. The first prereg called this
        # the lambda's u scale; the amendment removed the lambda and the map
        # became a segment blend weight, so the honest name lost the word. The
        # old spelling is kept because drivers were written against it — as an
        # ALIAS onto the same value, never as a second knob that is parsed and
        # dropped.
        "--qz-u-scale", "--qz-lambda-u-scale", type=float, default=None,
        dest="qz_u_scale",
        help=f"--value-scheme {VALUE_SCHEME_QZSEGMENT} only: the instability u "
             f"at which the blend weight saturates. w = {QZ_W_MIN} + {QZ_W_SPAN} "
             f"* min(u / scale, 1). Default {QZ_U_SCALE}, about u's p90 on the "
             "prereg's calibration sample.",
    )
    parser.add_argument(
        "--qz-w-const", type=float, default=None,
        help=f"--value-scheme {VALUE_SCHEME_QZSEGMENT} ablation C-no-u: use this "
             "CONSTANT blend weight instead of the u map, leaving the segment "
             "logic untouched. Absent (default) uses the map. Diagnostic arm: "
             "it attributes a C result to the instability feature rather than "
             "to the segment.",
    )
    parser.add_argument(
        "--qz-no-boundary", action="store_true", default=None,
        help=f"--value-scheme {VALUE_SCHEME_QZSEGMENT} ablation C-no-segment: "
             "every row whose own played move can be priced takes the game's "
             "outcome as its future, blunder boundaries ignored; the u map still "
             "sets the weight. Diagnostic arm. Refused together with "
             "--qz-w-const, which would leave neither of C's mechanisms.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="derive with this many processes over contiguous ranges of the "
             "corpus's shard snapshot. 1 (default) is the sequential read, and "
             "is not the parallel driver with one lane -- it is the same code "
             "path it has always been. Above 1 the emitted CONTENT is "
             "identical to it: the same shards holding the same rows in the "
             "same permuted order, every array equal, the same summary but for "
             "started_utc. Not the compressed bytes -- Blosc's threaded encoder "
             "is non-deterministic and --workers 1 does not reproduce its own "
             "either; see this file's --workers section. Intermediate rows are "
             "spilled under <out>/spill and removed on success: one output "
             "corpus of scratch disk in total (every surviving row is spilled "
             "once, whatever the lane count) and roughly 1 GB of RAM per lane. "
             "The lanes are CPU-bound next to whatever else the box is running.",
    )
    parser.add_argument(
        "--spill-chunk-rows", type=int, default=SPILL_CHUNK_ROWS,
        help="surviving rows per --workers spill file. Changes no output -- the "
             "output shard boundaries come from the survivor counts however the "
             "rows were chunked -- and exists so the multi-chunk spill and "
             "repack path is reachable at a size a test can drive. Lower it to "
             "trade memory for spill files.",
    )
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
    # ⚑ Next to --temp and --floor, and for the same reason: an unknown value
    # scheme, an out-of-range constant weight or the refused both-ablations cell
    # is rejected before the corpus is opened rather than after a partial write.
    value_scheme = parse_value_scheme(str(args.value_scheme))
    # ⚑ A qz knob on a scheme that cannot consume it is REFUSED, not ignored.
    # `--value-scheme qz50 --qz-w-const 0.725` looks exactly like a configured
    # ablation and would derive arm A, and a driver that lost its --value-scheme
    # line is precisely how that typo gets made.
    #
    # ⚑⚑ ON PRESENCE, NOT ON VALUE, and the difference is the whole doctrine.
    # Every qz option defaults to the None SENTINEL and the frozen default is
    # applied below, so "was it passed" is a real question. Testing
    # `args.qz_r_boundary != QZ_R_BOUNDARY` instead -- which is what the first
    # cut did -- waves `--value-scheme search --qz-r-boundary 0.27` straight
    # through: a knob EXPLICITLY passed and then ignored, inside the one check
    # written to forbid exactly that (review finding 7).
    if value_scheme != VALUE_SCHEME_QZSEGMENT:
        unused = [
            name for name, given in (
                ("--qz-r-boundary", args.qz_r_boundary),
                ("--qz-u-scale/--qz-lambda-u-scale", args.qz_u_scale),
                ("--qz-w-const", args.qz_w_const),
                ("--qz-no-boundary", args.qz_no_boundary),
            ) if given is not None
        ]
        if unused:
            raise ValueError(
                f"{', '.join(unused)} only affect --value-scheme "
                f"{VALUE_SCHEME_QZSEGMENT}, and this run asked for "
                f"{value_scheme}. Accepting them here would stamp a summary "
                "naming knobs that touched none of the emitted rows.",
            )
    qz = QzParams(
        r_boundary=(
            QZ_R_BOUNDARY if args.qz_r_boundary is None
            else float(args.qz_r_boundary)
        ),
        u_scale=QZ_U_SCALE if args.qz_u_scale is None else float(args.qz_u_scale),
        w_const=None if args.qz_w_const is None else float(args.qz_w_const),
        no_boundary=bool(args.qz_no_boundary),
    )
    if int(args.rows_per_shard) <= 0:
        raise ValueError(
            f"--rows-per-shard must be positive, got {args.rows_per_shard!r}",
        )
    if int(args.spill_chunk_rows) <= 0:
        raise ValueError(
            f"--spill-chunk-rows must be positive, got {args.spill_chunk_rows!r}; "
            "a non-positive chunk would spill one file per row",
        )
    if int(args.max_envelope_misses) < 0:
        raise ValueError(
            f"--max-envelope-misses must be >= 0, got {args.max_envelope_misses!r}",
        )
    workers = int(args.workers)
    if workers < 1:
        raise ValueError(
            f"--workers must be >= 1, got {args.workers!r}",
        )
    corpus_dir = Path(args.corpus)
    # ⚑ ONCE, and the SNAPSHOT of a live corpus's inventory is taken here.
    corpus_record = read_corpus_record(corpus_dir)
    slope, draw_width = cp_map_params(corpus_record.facts)
    options = DeriveOptions(
        scheme=scheme,
        temp=temp,
        floor=floor,
        cp_slope=slope,
        cp_draw_width=draw_width,
        limit=max(0, int(args.limit)),
        seed=int(args.seed),
        rows_per_shard=int(args.rows_per_shard),
        max_envelope_misses=int(args.max_envelope_misses),
        value_scheme=value_scheme,
        spill_chunk_rows=int(args.spill_chunk_rows),
        qz=qz,
    )
    if workers > 1:
        # ⚑ A DIFFERENT FUNCTION, NOT A PARAMETER ON THE SAME ONE.  `--workers
        # 1` runs `derive` -- the code this tool has always run, statement for
        # statement -- and the identity claim is worth exactly as much as
        # the independence of the two sides, so the default path is not "the
        # parallel driver with one lane".
        out = derive_parallel(
            corpus_dir=corpus_dir,
            out_dir=Path(args.out),
            corpus_record=corpus_record,
            options=options,
            workers=workers,
        )
    else:
        out = derive(
            corpus_dir=corpus_dir,
            out_dir=Path(args.out),
            corpus_record=corpus_record,
            options=options,
        )
    print(format_summary(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
