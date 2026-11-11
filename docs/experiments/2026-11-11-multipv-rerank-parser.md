# MultiPV rerank parser fix and bound clamping

**Status: not adopted.** Merging this to main does not deploy it. The live run
uses the code it was started on; this change reaches generated data only when
a run is restarted onto a commit that contains it, and that restart must
follow the adoption condition below.

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

## What this does NOT establish

- **No strength claim.** Nothing here shows the net gets stronger from cleaner
  targets. The measurement is parser output on a frozen transcript bank.
- **Rates may not transfer to production.** Production labels run with
  MultiPV 40 and far larger node budgets. Wider MultiPV means more ranks
  carry previous-depth scores, so these rates are illustrative, not a forecast.
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
