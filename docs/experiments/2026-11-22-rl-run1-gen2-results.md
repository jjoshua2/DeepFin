# 2026-11-22: RL run 1, generation 2 — results

Results of the pre-registered [generation-2 record](2026-11-22-rl-run1-generation2-zq-recal.md) (merged as
`4ddf03b1e`, revision 5). Every reading below is judged only by that record's pre-committed rules. Stage 0 is
complete. The generation-2 stages are still to run; their sections are named blanks.

**Bank.** `~/chess-artifacts/operations/rl-run1-gen2/` (`BANK_SHA256SUMS` [`2d6e1b67`], the sealed launch bank). It
is read-only for this record. File stamps are sha256 prefixes of the file as cited; in the paths below, `G2/` means
that directory.

## Stage 0: replication of match 8's player under a new seed

### Verdict

**Stage 0 reads "adoptable": consistent with match 8, but it does not replicate it.**
- The pooled Elo of the baked generation-1 net vs the SL base is **+4.94 [−1.62, +11.50]**.
- By the record's Stage 0 rules (`pool.json` → `result.pooled.elo_ci95`; the arm is the baked seat):
  - LB95 −1.62 is not > 0, so the reading is not "replicates";
  - LB95 > −10, so the reading is "adoptable";
  - UB95 +11.50 is not < 0, so the reading is not "damage".
- The launcher wrote that verdict: `G2/stage0/verdict.json` [`dbf8b51f`], which reads
  `{"elo_ci95": [-1.6157, 11.5001], "damage": false, "adoptable": true, "replicates": false, "reading": "adoptable"}`.
- The net it played is `played_sha256` `8dd6ac8c…`, which is `G2/gen1/gen1_baked.pt` [`8dd6ac8c`].
- Exit: `G2/stage0/exit` [`46b147b4`], `0 2026-11-22T16:06:23-05:00`.
- **Nothing is adopted on Stage 0 alone** (record, Stage 0 rules).

**What follows from the record:**
- Generation 2 proceeds, because Stage 0 is not damage. The generation-2 launcher refuses unless the gen-1 file's
  sha256 equals `played_sha256`.
- Stage 0 is not "replicates", so **the "two measured steps" row of the licensing table is now unreachable** for this
  run, whatever generation 2 reads.
- The rows still open are:
  - "one measured step (step 2)", if gen 2 vs gen 1 improves;
  - "one measured step and a cumulative improvement over the base", if gen 2 vs the base improves as well;
  - a cumulative improvement alone;
  - no second measured step.

### Estimate (the record's estimator, on the banked games)

The estimator is `pool.json`, unchanged: clustered by opening key, normal approximation on the score scale, ends
mapped to Elo. Source: `G2/stage0/arenas/zq_recal_rep/pool.json` [`842537eb`].
- Inputs: `results.jsonl` [`a3cfec67`] and the seat game logs `zq_recal_rep_1_A.games.jsonl` [`61fb85c4`] and
  `zq_recal_rep_1_B.games.jsonl` [`1eb3c374`].
- The arm is the baked generation-1 net (Zq + recalibration (0.7036, −0.2271) baked into the WDL head). The base is
  `rl-run1-pilot/sl_base.pt` (`a39f900c…`).
- Settings: 64 v 64 sims, matched_sims, training search shape, move temperature 0.1, no Gumbel noise, production
  Syzygy, EVAL book (16 plies), seed 2026112281.

| | Games (pairs) | Arm W / D / L | Draw rate | Arm score | Elo [95% CI] |
|---|---|---|---|---|---|
| Seat A (arm = candidate) | 3,200 (1,600) | 963 / 1,308 / 929 | 0.409 | 0.5053 | +3.69 [−5.26, +12.65] |
| Seat B (arm = reference) | 3,200 (1,600) | 966 / 1,325 / 909 | 0.414 | 0.5089 | +6.19 [−2.70, +15.09] |
| **Pooled (the reading)** | **6,400 (1,600 clusters)** | **1,929 / 2,633 / 1,838** | **0.411** | **0.5071** | **+4.94 [−1.62, +11.50]**, SE 0.0048 on the score scale |

- **The seats agree.** The seat offset is −1.25 [−7.30, +4.80], so its interval contains 0. The correlation between
  seats A and B is 0.080.
- Pair outcomes (pentanomial, arm points per pair 0 / 0.5 / 1 / 1.5 / 2):
  - seat A: 125 / 368 / 580 / 402 / 125;
  - seat B: 111 / 377 / 590 / 388 / 134.
- Every game ended by the rules or by Syzygy: 0 unresolved, 0 at the 600-ply cap. The longest games were 314 plies
  (A) and 381 (B).
- No opening repeated within the run. Coverage was 0.950.

### Against match 8's nomination (descriptive only)

- Match 8 read **+9.61 [+3.27, +15.96]** on seed 2026112191 (6,400 games, SE 3.24).
- Stage 0 reads +4.94 [−1.62, +11.50] on seed 2026112281: the same player, the same opponent, a new seed.
- The point estimates differ by 4.7 Elo. Each interval contains the other's point estimate.
- **No pooled match 8 + Stage 0 estimate is computed or stated.** The record excludes one in advance: "Match 8 was
  selected as a winner, so pooling would carry its selection bias; Stage 0 alone is the unbiased estimate of that
  player." A combined number would be exactly that selection-biased quantity, so this record does not give one,
  not even as a descriptive figure.
- Read as an estimate of the player, the unbiased number is Stage 0's. Match 8 was the best of a set of readings, and
  its replication came out smaller, which is what selection would produce. Stage 0 does not tell selection apart from
  noise.

### What "adoptable but not replicated" means for "RL improved on the SL base"

**It is not established that RL improved on the SL base.**
- The one registered reading that nominated an improvement (match 8, +9.6) did not replicate under the record's
  rule. The unbiased estimate of that player is +4.9, with an interval that includes 0.
- Stage 0 also shows **no damage**: the baked net is not measurably worse than the SL base (UB95 +11.5, LB95 −1.6),
  so it is safe to collect from.
- That is all "adoptable" means here. Under this record:
  - "RL improved on the SL base" can be claimed only via generation 2's cumulative reading (gen 2 vs base improves).
  - "two measured steps" cannot be claimed at all.

### Operator facts

These are as reported by the operator and read from the bank files cited.

**Attempt 1.** Refused by the guard's device-holder scan before any game.
- An agent's pytest held `/dev/dxg`. This is the first line of `G2/stage0/run.out` [`d1c7253e`]: "REFUSED: a
  process holds the GPU device (/dev/dxg …) … pytest …".
- Nothing was played.

**Attempt 2.**
- The release line is `stage0 idle_mib=2398` in `G2/GPU_RELEASED_BY_OPERATOR` [`efee002a`]. Its first line per
  stage wins, and the operator replaced it for this attempt.
- `G2/stage0/started` [`a247a7c1`].
- The operator reports the start at 14:33:07.
- Seat A's first game is stamped 14:38:35 in its header.
- Exit 0 at 16:06:23.
- The owner's standing go is quoted in `G2/stage0/OWNER_GO` [`7d859aea`].

**Re-guard waits at the two seat boundaries.** These are in `G2/stage0/guard.log` [`08088d84`]. Each wait ended in a
pass:
- 14:35:58–14:36:09: an agent's pytest held `/dev/dxg`; passed after 5 s of waits.
- 14:37:06–14:38:29: pytest, then GPU utilisation 16 % / 11 %; passed after 40 s.
- 15:21:52–15:25:40: utilisation 13–17 %, pytest on `/dev/dxg`, then memory 1,884 MiB outside [1,886, 2,910]; passed
  after 115 s.
- The operator raised the hold file at the boundaries. No game was played while a refusal stood.

**`gpu_scan.sh: line 171: desc: bad array subscript`, printed twice in `run.out`.** It does not change any scan's
verdict. Source: `G2/gpu_scan.sh` [`949408e7`].
- Line 171 is the descendant-closure loop: `[ -n "${desc[${pp[$p]}]:-}" ]`. The message appears when a pid's parent
  was recorded as empty.
- That happens when a process exits between the scan opening its `/proc/<pid>/status` and reading it. The read
  fails inside `while read`, the loop ends with status 0, the `|| continue` does not fire, and the pid is recorded
  with an empty uid and parent.
- Bash 5.1.16 (the box's version) treats the bad subscript as an expansion error in that one `[ … ]` test. The test
  fails, the `&&` chain skips marking the pid, and the closure, the loop and the function all continue to their
  normal return. This was reproduced on a synthetic array with an empty parent value.
- The pid is never scanned in any case: an empty uid keeps it out of the scanned set `mine`.
- The only possible side effect is that a live child of that pid is treated as a non-descendant, which means it gets
  scanned, not exempted. So the error can make the scan stricter, never looser.
- **Not a blocker for the generation-2 stages.** A one-line guard, `[ -n "${pp[$p]}" ]` before the lookup, goes into
  the next bank revision.

### Known limits (Stage 0)

- One player, two matches (match 8 and Stage 0). No multiplicity correction is registered; Stage 0 is the tenth
  registered reading.
- Inference noise: Stage 0 plays the baked file, which is equal to match 8's player within the measured noise
  (record, part A).
- From the #1491 statistics review (MINOR; none affects this reading; they go into the next bank revision):
  - the lock-holder test cases should signal acquisition instead of sleeping;
  - `run_smoke.sh` should refuse `TRAIN_MEM_SMOKE_CPU=1` when `GEN2_BANK` is unset;
  - the Commands section should wait on `stage0/exit`.
- The mutant sweep of the launch scripts is in `~/chess-artifacts/operations/rl-run1-gen2.mutants_v7/`
  (`SHA256SUMS` [`3b6ce759`]): 103 / 103 killed; the audit-rule alone runs passed; the control passed 247 / 247.

## Generation 2

The blanks below are filled from the bank when each stage is read. Each is judged by its own rule in the record.

- **Memory smoke (pre-launch, 3 updates):** _blank: per-update peaks, verdict, bank file + stamp._
- **B2 collect from `gen1_baked.pt`:** _blank: games, rows, wall, checks, bank file + stamp._
- **B3 equivalence, collect checks and the take-effect proof:** _blank._
- **B4 training of generation 2:** _blank: updates, losses, trunk2 determinism check, bank file + stamp._
- **B5 per-generation recalibration (T, b):** _blank: fitted values, bounds check, bake verification._
- **B6 match: gen 2 vs gen 1 (the eleventh reading; PRIMARY):** _blank: Elo [95% CI], seats, reading._
- **B6 match: gen 2 vs the SL base (the twelfth reading):** _blank: Elo [95% CI], seats, reading._
- **Licensed claim** (from `verdicts.json` under the record's table, with Stage 0 = adoptable): _blank._
