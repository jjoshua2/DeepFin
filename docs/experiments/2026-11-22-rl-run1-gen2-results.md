# 2026-11-22: RL run 1, generation 2 — results

Results of the pre-registered [generation-2 record](2026-11-22-rl-run1-generation2-zq-recal.md)
(merged as `4ddf03b1e`, revision 5, with the exact optimizer check and launcher recovery in
[#1517](https://github.com/jjoshua2/DeepFin/pull/1517), revision 6). Stage 0 and generation 2 both completed.
The registered verdict is **no second measured step**.

**Bank.** `~/chess-artifacts/operations/rl-run1-gen2/`. The original launch seal is
`G2/BANK_SHA256SUMS.v5` [`2d6e1b67`] and the recovered launch seal is `G2/BANK_SHA256SUMS` [`d0542e31`].
File stamps are SHA-256 prefixes. `G2/` means the bank directory.

## Stage 0: replication of match 8's player under a new seed

### Verdict

**Stage 0 reads "adoptable": consistent with match 8; does not replicate it** (the record's Stage 0 table, verbatim).
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
- Generation 2 proceeds: the record says "Generation 2 proceeds on any **non-damage** Stage 0". The generation-2 launcher refuses unless the gen-1 file's
  sha256 equals `played_sha256`.
- Stage 0 is not "replicates", so **the "two measured steps" row of the licensing table is now unreachable** for this
  run, whatever generation 2 reads.
- The possible rows after Stage 0, with the record's licensed-claim text quoted (record, "What this record can license"):
  - gen 2 vs gen 1 improves and gen 2 vs base improves: "**one measured step (step 2) and a cumulative improvement
    over the base; attribution between generations unread**";
  - gen 2 vs gen 1 improves and gen 2 vs base does not: "one measured step (step 2)";
  - gen 2 vs gen 1 does not improve and gen 2 vs base improves: "**no second measured step; a cumulative improvement
    over the base (gen 2 vs base improves); attribution between generations unread**";
  - neither improves: "no second measured step".
- **With Stage 0 not "replicates", attribution between generations is unread in every reachable outcome.** B6's
  escape, "attribution between generations unread unless Stage 0 replicated and gen 2 vs gen 1 improved", is closed.
- **Option (ii), the fresh generation-1 replicate (record part C3).** The record's advice against using it
  before generation 2 applied to Stage 0's result. For the later decision after B6, see the registered
  verdict below.

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
- Interpretation: read as an estimate of the player, the unbiased number is Stage 0's. Match 8 was the best of a set
  of readings, and its replication came out smaller, which is what selection would produce. Stage 0 does not tell
  selection apart from noise.

### What "adoptable but not replicated" means for "RL improved on the SL base"

**It is not established that RL improved on the SL base.**
- The one registered reading that nominated an improvement (match 8, +9.6) did not replicate under the record's
  rule. The unbiased estimate of that player is +4.9, with an interval that includes 0.
- Stage 0 also shows **no damage**: the baked net is not measurably worse than the SL base (UB95 +11.5, LB95 −1.6).
  By the record, "generation 2 proceeds on any non-damage Stage 0".
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
- The release line is `stage0 idle_mib=2398 2026-11-22T14:31:36-05:00`, the first line of
  `G2/GPU_RELEASED_BY_OPERATOR.before_v6` [`66473b70`] (its first line per stage wins; the operator replaced it for this attempt).
  The file's second line, `gen2 …`, was appended later for the generation-2 stage (below); the stage0 line is
  untouched. A later append (for example a relaunch line) moves the whole-file stamp, so the stage0 line can be
  re-checked on its own: the file's first 47 bytes (that line and its newline) hash to `efee002a`, the stamp the
  file had before the gen2 line was appended.
- Start 14:31:43: `G2/stage0/started` [`a247a7c1`] reads `2026-11-22T14:31:43-05:00`, and the first line of
  `G2/resume.log` (append-only; in progress as of 16:47: the running generation-2 stage is still appending to it, so it is not stamped) reads
  `2026-11-22T14:31:43-05:00 stage0 launch pid 2832905`.
- Seat A's game log header is `created` 14:38:35; its first game's `ts` is 14:39:09.
- Exit 0 at 16:06:23.
- The owner's standing go is quoted in `G2/stage0/OWNER_GO` [`7d859aea`].

**Three re-guard waits.** These are in `G2/stage0/guard.log` [`08088d84`]. Each wait ended in a pass:
- before the smoke seat, 14:35:58–14:36:09: an agent's pytest held `/dev/dxg`; passed after 5 s of waits;
- between the smoke seat and real seat A, 14:37:06–14:38:29: pytest, then GPU utilisation 16 % / 11 %; passed after
  40 s;
- between seats A and B, 15:21:52–15:25:40: utilisation 13–17 %, pytest on `/dev/dxg`, then memory 1,884 MiB outside
  [1,886, 2,910]; passed after 115 s.
- The operator raised the hold file at the boundaries. No game was played while a refusal stood.

**The gap between Stage 0's exit (16:06:23) and the memory smoke (16:40:08).** The operator ran two diagnostics
unrelated to this record on the idle GPU: an engine warm-start test (16:06–16:10) and the forward-chunking
measurement for PR #1511 (16:14–16:40). Each went through the operator's GPU gate, and neither wrote anything inside
this bank; their times are the operator's report. The memory smoke's own gate then measured a fresh idle of 2,181 MiB
(16:40:02, per the operator): the first lines of `G2/gen2/smoke_mem.log` [`efb7ea1e`] read "OK: memory within
[1669, 2693] MiB, utilisation max 1 % over 5 samples, no GPU job of this user", and the smoke started at
2026-11-22T16:40:08-05:00.

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

The operator completed the run after recovering with the revision 6 launcher. The first training
attempt remains in the bank. `G2/gen2/exit` [`47acd1b1`] records exit 0 at 23:11:01 ET on November 22.

- **Memory smoke (3 updates, after Stage 0's verdict):** **PASS**, exit 0 at 16:46:16 (per the operator).
  `G2/gen2/smoke_mem.json` [`b2d0b1a1`]: `cpu_logic_smoke` false (a real GPU run), 722,742 rows, 1,412 batches in
  the plan, peak reserved 16.957 GiB and peak allocated 15.377 GiB of 31.816 GiB, update 2 → 3 deltas 0.0 GiB
  (reserved now and peak reserved), wall 364 s. Per-update rows: `G2/gen2/smoke_mem.jsonl` [`55e32161`]. Log:
  `G2/gen2/smoke_mem.log` [`efb7ea1e`].

### B2 and B3: collection and take effect

- The original generation-2 run used the baked generation-1 net from Stage 0. From 16:50:00 to 18:32:15 ET,
  it collected **8,000 games and 796,341 policy rows in 42 shards** (102 min 15 s). The release for that
  attempt is retained in `G2/GPU_RELEASED_BY_OPERATOR.before_v6` [`66473b70`].
  Sources: `G2/gen2/collect.started` [`d4d77bb4`], `G2/gen2/collect.exit` [`9c2cf3a6`],
  `G2/gen2/collect/gen_0001/collect/manifest.json` [`778b7ecd`].
- The gate found 8,000 distinct game IDs, zero unresolved games, zero duplicates, no resumed workers, and
  matching manifest and game row counts. It recorded a 49.74% draw rate and zero buffer-full flushes.
  Source: `G2/gen2/collect_gate.json` [`983188fa`].
- Against the pilot collection, rows per game rose from 95.25 to 99.54, draw rate from 49.00% to 49.74%,
  mean game length from 111.57 to 116.02 plies, and draw share of outcome-target rows from 49.99% to 51.22%.
  Both had zero buffer-full flushes. The collections used different seeds and position mixes.
- The baked generation-1 model matched the external recalibration seat under autocast on the 1,218
  hold-out rows available in the pilot's first shard (the check caps the sample at 2,048).
  Mean absolute error was 0.000349 against seat rounding noise of 0.000993, with zero WDL argmax
  disagreements. On the 39,135 generation-2 hold-out rows that stored the actor's root WDL
  (`root_net_wdl`), mean absolute error was 0.00111 against the baked net and 0.03900 against the raw net.
  The respective limits were at most 0.004 and at least 0.02. Sources:
  `G2/gen2/equiv.json` [`16e5f1be`], `G2/gen2/take_effect_collect.json` [`eea29d8f`],
  `G2/gen2/checks.exit` [`2e48032e`].

### B4: Zq2 training and recovery

- The pretraining trunk value/policy ratio was 1.209, within the registered [0.59, 1.35] sanity range.
  The eight batch ratios had a mean of 1.253 and SE of 0.137. Source:
  `G2/gen2/trunk2_verdict.json` [`bab8e9ae`].
- The first training attempt finished, then the launcher refused the optimizer readback because the inherited
  learning-rate ratio and the flat scheduler gave a slightly different matrix-group rate. Revision 6
  compared the exact closed form, and accepted the earlier completed steps with their inputs unchanged.
  The operator relaunched with the 20:11:44 ET release. The first model (SHA-256 prefix `1db00eca`) was
  kept aside. The rerun model (prefix `5d019e5e`) differed byte for byte and was the source for the bake.
  This was a descriptive comparison. Sources: `G2/gen2/arm_Zq2.first_attempt.json` [`5565cf92`],
  `G2/GPU_RELEASED_BY_OPERATOR` [`1d815e88`], `G2/gen2/arm_Zq2.check.json` [`f11f1c76`],
  `G2/gen2/gen2_baked.pt.json` [`68a005b5`].
- The rerun took 467.65 s and passed every optimizer check. It completed **1,478 updates**, drawing
  756,711 rows once each, with `w_wdl = 1.0` and search-WDL fraction 0.5. Recorded total loss was
  1.58382 (policy 0.91928, blended WDL 0.66454). On 39,630 held-out rows, policy CE went from
  0.91432 to 0.90298 and value NLL from 0.72084 to 0.68833. These are losses on the stored targets.
  Sources: `G2/gen2/arm_Zq2.check.json` [`f11f1c76`],
  `G2/gen2/arm_Zq2/gen_0001/train/metrics.json` [`859bc31a`].

### B5: recalibration and bake

- The fit toward baked generation 1 returned **T = 0.69965142, b = −0.24673345** on 39,630 held-out rows.
  Both are inside the registered ranges (T 0.4951–1.4212, b −0.4542 to +0.2271). Cross-entropy to the
  reference WDL went from 0.575319 to 0.558539. Source: `G2/gen2/fit2.log` [`6be395e0`].
- Only the final WDL weights and bias changed in the bake. On 1,218 verification rows, the maximum logit
  and probability errors were 5.87e−6 and 9.95e−8, within their limits of 1e−4 and 1e−5, and the policy
  was bit identical. Both arena pools identify baked generation 2 as model SHA-256 prefix `5881d2af`.
  Sources: `G2/gen2/gen2_baked.pt.json` [`68a005b5`], `G2/gen2/gen2_baked.pt.verify.json` [`eba7abbd`].

### B6: registered strength readings

Both matches used the pilot code pin `0c795fb7a`, the same 64-sim search on each side, T0.1, no Gumbel
noise, production Syzygy and the 16-ply EVAL book. Each played 3,200 games in each seat, for 6,400 games
over 1,600 opening clusters. All estimates below are in generation 2's direction, including seat B.
The estimator clusters by opening key and maps the score interval to Elo.

| Reading | Seed | Seat A Elo [95% CI] | Seat B Elo [95% CI] | Pooled Elo [95% CI] | Registered result |
|---|---:|---|---|---|---|
| 11. Generation 2 vs baked generation 1 (primary) | 2026112283 | −4.45 [−12.86, +3.95] | +6.73 [−1.76, +15.23] | **+1.14 [−5.04, +7.32]** | Adoptable |
| 12. Generation 2 vs the SL base | 2026112284 | +2.61 [−6.46, +11.67] | +5.10 [−3.31, +13.52] | **+3.85 [−2.46, +10.17]** | Adoptable |

- The seat-offset intervals both include zero: −5.59 [−11.36, +0.17] vs generation 1 and
  −1.25 [−7.30, +4.80] vs the SL base. Neither match had an unresolved or capped game.
- Sources: `G2/gen2/arenas/g2_vs_g1/pool.json` [`cce191d3`] and
  `G2/gen2/arenas/g2_vs_g1/results.jsonl` [`70dca585`],
  `G2/gen2/arenas/g2_vs_base/pool.json` [`e85d60f9`] and
  `G2/gen2/arenas/g2_vs_base/results.jsonl` [`245aa537`].

### Licensed claim

**No second measured step.** Both lower bounds clear −10 Elo and fall below zero. Both upper bounds are
positive. This yields adoptable on both matches, with neither improvement nor damage under the registered
rules. There is no measured cumulative improvement over the SL base, and Stage 0 did not replicate the
first step. The terminal verdict is `G2/gen2/verdicts.json` [`f02646f7`]. B6 leaves the next
experiment choice with the owner.
