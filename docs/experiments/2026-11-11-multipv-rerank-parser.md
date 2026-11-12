# MultiPV rerank parser fix and bound clamping

**Status: not adopted.** Merging this to main does not deploy it. The live run
uses the code it was started on; this change reaches generated data only when
a run is restarted onto a commit that contains it, and that restart must
follow the adoption condition below.

"Merging does not deploy" holds for the live run only. Node-limited MultiPV
tools run from main change behaviour on merge, because each calls
`StockfishUCI.search(..., nodes=...)` and so goes through the changed
`_SearchInfoAccumulator`: `scripts/rvg_label_pass.py`,
`scripts/build_audit_set.py`, `scripts/audit_targets.py` (its SF soft-target
labelling) and `scripts/mpv6_tail_panel.py`.

## What changes

`_SearchInfoAccumulator` in `chess_anti_engine/stockfish/uci.py` builds the
MultiPV list that feeds the SF policy target, the stored `sf_multipv_raw`
block, the max-based `sf_p0_regret` vector and the curriculum opponent's
candidate set. Two changes, in two separate commits so either can be reverted
on its own:

1. **Rerank identity (cherry-picked from PR #1201, commit `2c272ed72`).** When
   Stockfish stops on its node limit, it can print the line under search with
   an upper or lower bound, while the ranks below it carry scores from the
   previous depth. Main skipped the bound line but kept the *previous* move at
   that rank, so one move could appear twice and another disappear. The
   duplicated move then took policy mass twice and double weight in the
   curriculum opponent's uniform draw. The fix files each move under its own
   latest exact evidence and emits each move once.
2. **Bound clamping (this branch).** A move's kept exact score can contradict
   the newer bound for that move: for example, an earlier exact -391 when the
   current depth says the move is at most -465. The kept score is now replaced
   by the bound line (its value and WDL). That is the closest score consistent
   with both pieces of evidence. Dropping the move instead was rejected:
   a fail-high (lowerbound) move is often SF's new best, and dropping it
   would give it zero policy mass and remove it from the regret vector's max.
   **Exception:** the row the scalar eval came from (the last exact
   `multipv 1` line) is never clamped. Clamping it would separate row 0 from
   `res.cp` (the value label is untouched). The eval/PV orphan detector would
   then fire on roughly 3–5% of labelled rows (16 of 300 searches at 75k nodes,
   33 of 1000 at 150k), above its 1% warn threshold. Those contradictions on
   the scalar's own move stay as they are on main.

## Measured rates (real Stockfish, banked by the PR #1201 review)

Single process, Threads 1, Hash 16, **MultiPV 6**, `go nodes N` on
random-playout positions. A search is "changed" when its emitted
`(move, cp, mate)` list differs from main's.

| | 75k nodes (300 searches) | 150k nodes (1000 searches) |
| --- | --- | --- |
| searches ending on a bound line | 109 (36.3%) | 309 (30.9%) |
| main emits a duplicated move | 23 (7.7%) | 64 (6.4%) |
| changed by the rerank fix alone | 24 (8.0%) | 67 (6.7%) |
| kept score contradicts the final bound (rerank fix alone) | 61 | 169 |
| of those, on the scalar's own row (left unclamped) | 16 | 33 |
| clamped by change 2 | 45 (15.0%) | 136 (13.6%) |
| changed by both changes together | 47 (15.7%) | 151 (15.1%) |
| clamp gap, cp (min / median / max) | 1 / 14 / 341 | 1 / 13 / 192 |
| eval/PV orphan flag: main / this branch | 0 / 0 | 0 / 2 (0.2%) |
| row 0 equals SF's `bestmove` on bound-ending searches (this branch) | 109 / 109 | 309 / 309 |

The two orphan hits are a benign route that the parser fix opens: rank 1 ends
on a bound for `bestmove`, while the scalar stays on the previous exact
rank-1 line of another move. They are documented in
`replay/shard.py::sf_eval_pv_orphan_flags` and pinned by
`tests/test_stockfish_multipv_bound_real.py`. Reordering rows so the
scalar's move comes first was evaluated and rejected, for two reasons.
First, `_choose_curriculum_opponent_move` plays `cand_indices[0]` as SF's
best on full-strength (SF-refute) plies, so reordering would make it play
a stale move instead of `bestmove`. Second, in one of the two cases no
emitted row carries the scalar at all, because that move's newer exact
line has a different score.

**CAST compatibility caveat.** `scripts/cast_probe.py` drops a pair when
either row is orphaned. So the benign route also removes about 0.2% of rows
from that probe's sample, and up to about twice that in pairs. The trainer
only reports the orphan rate; it does not filter on it. PR #1201's second
commit added a per-row `sf_eval_witness` shard field to remove this. That
commit is not taken here: it is a reader/writer schema change of about 900
lines, aimed at a rate below the 1% warn threshold.

## Independent review measurements (wider MultiPV, fresh banks)

The table above covers MultiPV 6 only. The independent reviewer of PR #1212
replayed real Stockfish transcripts through three parsers: main, the rerank
fix alone ("parser"), and this branch ("head"). Two banks are the author's
MultiPV 6 banks above (bound-ending subset only). Four are new. The two
150k-node fresh banks send `ucinewgame` before every search.

| bank | searches | main dup | main lost | PR dup/lost | changed vs main (parser / head) |
| --- | --- | --- | --- | --- | --- |
| author MPV6 75k (bound subset of 300) | 109 | 23 | 24 | 0/0 | 24 / 47 |
| author MPV6 150k (bound subset of 1000) | 309 | 64 | 65 | 0/0 | 67 / 151 |
| new MPV20 75k | 150 | 27 (18%) | 27 | 0/0 | 27 / 42 |
| new MPV40 75k | 150 | 39 (26%) | 39 | 0/0 | 39 / 47 |
| new MPV6 150k, ucinewgame per search | 300 | 27 | 28 | 0/0 | 28 / 57 |
| new MPV40 150k, ucinewgame per search | 200 | 40 (20%) | 40 | 0/0 | 40 / 55 (27.5%) |

"dup" is a move emitted twice. "lost" is a move SF searched that main drops.
Under this branch, row 0 equals Stockfish's `bestmove` on every search in every
bank. Main misses it twice in the author's 150k MultiPV 6 bank. So at
production width 40, main corrupts about 20–26% of searches (a duplicate plus a
lost move), roughly three times the MultiPV 6 rate. The eval/PV orphan flag
read 0 of 200 on the MPV40 150k bank under this branch.

**Clamp vs re-searched truth.** All 101 clamped rows from the fresh banks were
re-searched to get a reference value for the clamped move:

- `T_full`: `ucinewgame; go depth d+1` at the same MultiPV.
- `T_sm`: `go depth d+1 searchmoves <move>` at MultiPV 1.

| truth | clamped closer / ties / stale closer | median abs error, stale → clamped (cp) | mean win-prob error, stale → clamped |
| --- | --- | --- | --- |
| `T_full` | 78 / 2 / 21 | 26 → 12 | 0.039 → 0.018 |
| `T_sm` | 68 / 1 / 32 | 30 → 24 | 0.042 → 0.028 |

All 5 lower-bound clamps were closer under `T_full`. Limits: the positions are
random playouts, not production game positions; node budgets are at most 150k;
no real mate clamp occurred, so the mate branch of the clamp is untested
against truth.

**Bank (the independent reviewer's, not the author's):**
`~/chess-artifacts/experiments/2026-11-11-multipv-rerank-review/`.
It holds the generators (`gen.py`, `gen_fresh.py`), the census, clamp and
truth scripts (`census.py`, `clamped.py`, `truth.py`), the transcript banks
(`bank_*.json`, `fbank_*.json`), the clamped rows and the re-searched truth
(`clamped_rows.json`, `clamped_fresh.json`, `truth_rows.json`), and the
parser variants and mutation outputs used in the review.

## What this does NOT establish

- **No strength claim.** Nothing here shows the net gets stronger from cleaner
  targets. The measurement is parser output on a frozen transcript bank.
- **Rates may not transfer to production.** Production labels run with
  MultiPV 40 and far larger node budgets. Wider MultiPV means more ranks
  carry previous-depth scores, so these rates are illustrative, not a forecast.
  The review's MultiPV 40 banks (20–26% of searches corrupted on main) narrow
  the width gap, but not the position or node-budget gap.
- **The value label is unchanged.** `res.cp`/`res.mate` still come from the
  last exact rank-1 line, even when a later bound contradicts it. That is a
  separate value-target question, not addressed here.
- **Non-pvIdx "exact" lines can themselves be bounds.** Stockfish prints a
  bound flag only on the line under search. A root move resolved earlier in
  the same aspiration window may carry a fail-low value printed as exact
  (bank case `benign_orphan_rank1_fail_high`: d6b4 printed -583, depth 11).
  This change does not detect that.

## Confounds for a readout window

The change also alters the **curriculum opponent's candidate set**: main's
duplicate double-weighted one move in the uniform within-regret draw, and the
full-strength path could play a non-bestmove. So it moves `sf_played_regret`
and the PID's difficulty, not only the training targets. Per CLAUDE.md,
`wdl_regret` measures the agent, so a regret step after adoption cannot be
read as net improvement.

## Adoption condition

- One data-affecting change per readout window. Adopt only in a window with
  no other data-affecting change, or name the overlap on a Confounds line in
  that window's record.
- Before adoption, write the window's hypothesis, control, one deciding
  yardstick with an exact command, and kill/success thresholds in a dated
  record. That record should check the realized change rate on production
  labels. A suitable check: count rows whose stored `sf_multipv_raw` differs
  under re-parse, from a banked raw-transcript sample at production MultiPV
  and node settings.
- Record the verdict at adoption, in the same session as the readout, judged
  by the pre-committed rule.
