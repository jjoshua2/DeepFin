# Downside300 policy with the selected BT4/SF value recipe

Draft prepared before reading the active V100-versus-V50 arena. The value choice
and exact execution bindings remain open; no materialization, training or match
is launched by this record. Recommendation: use the existing matched **35,314,577
rows / 21 sources**, provided a single-copy implementation fits the disk budget.

## Question and comparison

Does the moderate SF tactical correction add playing strength when combined with
our selected value supervision? The original 18.91M Downside300 comparison was
**+17.66 Elo [−17.97, +53.66]** at 400 simulations: unresolved under its historical
rule, but a plausible orthogonal intervention. The matched 35M V50 comparison was
**+31.30 [8.53, 54.34]** against SF-only values. Neither result establishes their
combination or the best dose. See the [policy record](2026-09-12-sf-allmove-downside.md)
and [value comparison](2026-09-13-native-bt4-value-endpoint.md).

After the current arena completes, record a provisional value choice using its
estimate, uncertainty, scientific implications and cost. An interval excluding zero
is not required for advancing a useful recipe. Preserve the arena's original
classification separately; do not rewrite its precommitted rule. The choice is
between the already trained V50 and V100 packages, not an untested intermediate dose.

The new candidate differs from that selected same-corpus checkpoint only in policy
supervision. Start with normalized stored B100 policy at teacher temperature 0.5.
In ordinary positions, multiply moves whose raw phase-zero d9 score deficit is
**strictly greater than 300 cp** by **0.5**, retain weight 1 for other moves, and
renormalize. Leave mate-domain positions unchanged. This is the existing
`stored-b100-allmove-sf-gapgt300-weight0.5-ordinary-v1` recipe, not a move deletion,
SF one-hot substitution, new sharpening sweep or deeper labeling pass.

Retain the selected V50 or V100 `search_wdl` bytes, masks and every other non-policy
array. V50 means normalized 50% stored SF / 50% native BT4 WDL; V100 means native
BT4 WDL alone. All other training settings stay matched: seed 101, one game epoch,
batch 512, two planning/two loading workers, compiler cap two, frozen architecture,
optimizer and objectives. Target 68,974 updates and canonical schedule `ca3922c4…`;
actual staging, columns and realized schedule must establish that match.

## Feasibility and why prefer 35M

| Route | CPU preparation evidence and estimate | Incremental disk and scientific tradeoff |
| --- | --- | --- |
| Matched 35M, one final copy | Original Downside rebuild took 13,298.722 s. The new G10 pilot took 6.435 s for 8,192 rows, decoding 8,243 raw rows. Naive extrapolation over the remaining 16.404M G10 rows adds about 3.58 h: roughly 7.3 h total. This is a planning estimate, not measured full-G10 throughput. | Prior exact 21-root B100 accounting was 26.245 GiB allocated. Propose 32 GiB aggregate new output plus 8 GiB other-writer allowance above the 150-GiB floor, requiring 190 GiB free at launch. Chosen-value inputs must be accounted for before launch; this is not a fresh census. Reuses the selected trained 35M control and tests nearer intended bootstrap scale. |
| Matched original 18.91M, commonly called 20M | Restore the verified archived Downside corpus, then adapt the value writer. The prior original value rewrite took about 4,824 s; restore cost is unmeasured. Rebuilding Downside instead costs the observed 3.69 h before value integration. | Restored Downside shard allocation was 13.553 GiB, plus a fresh combined output and restore staging where needed. Thus restoring and separately copying need not save much peak disk. Training is about 2.2 h rather than 4 h, but a matching chosen-value control at this corpus/seed must be identified or trained; the 35M checkpoint is not a matched control. |

The G10 pilot proves the selected-row join works for one shard, not a global
throughput rate: raw shards spanning output boundaries can be decoded repeatedly,
and source density/cache state differ. The original production timing is stronger
evidence than scaling the faster pilot over all 35M rows. Saved full-run timing only separates producer (13,278.11 s) from the enclosing
13,298.72 s operator; it does not distinguish raw decoding, copying and policy
arithmetic inside the producer. No dominant internal phase or speedup is established.
CPU preparation can overlap useful Ceres labeling, so this is not seven hours of
required idle GPU time. A 12-hour CPU ceiling is
proposed for the single-copy 35M route, including qualification and cleanup; no
promise that it will finish inside the estimate. If the concrete route needs two
full new copies, or cannot retain the reserve, reconsider the 20M alternative
before production rather than silently doubling disk or shortening the corpus.

## Required narrow integration

The historical writer routes cannot simply be chained. `sf_policy_rewrite.tactical_source`
requires the B100 parent's non-policy summary to equal original SF; it rejects a
V50/V100 parent. `bt4_value_rewrite` requires the unchanged global B100 policy
recipe, and combined-corpus admission also fixes that policy identity. Giving any
of these a relabeled summary would invalidate the experiment.

This change implements an explicit combined-policy materializer path that authenticates original SF scores,
original B100 policy and the already qualified selected value corpus, then copies
that value corpus once and applies the unchanged Downside arithmetic to its policy.
The original B100 policy must match the selected value input before mutation;
all non-policy output content must match that selected value input afterward.
Retain exact selected-G10 phase-zero rosters/physical-row joins, original exclusions,
and mate handling. Do not alter the existing single-intervention routes or recompute
mixed values from already mixed values. The materializer reuses existing V50/V100 metadata admission and authenticates the
actual per-shard writer file manifest before copying. It preserves the chosen value
summary and records its exact parent binding in the policy recipe. Explicit combined
training admission, completion mapping and match-role support remain required;
materializer support does not make those later paths launch-ready.

Qualification must disclose float16 storage error and any positive move-entry
underflow. The previous full Downside corpus lost three positive move entries,
not three rows; the G10 pilot lost none. Preserve all selected rows and non-policy
bytes. No raw teacher inference, model reads or corpus scans were performed for
this feasibility draft.

## Bounded execution and decision

Proposed limits: CPU materialization/qualification 43,200 seconds inclusive, two
CPU threads at low priority, GPU hidden, 48-GiB startup/32-GiB running RAM headroom,
150-GiB retained disk floor, and a periodic 32-GiB allocated-output threshold rather
than a filesystem quota. Exact affinity and final available disk are scheduling
inputs; do not collide with current operators. Use the existing owned-process
cleanup and STOP behavior; keep failed partial outputs and receipts, with no
automatic retry or scope expansion.

One new 35M training stage has a 21,600-second ceiling within the existing
27,030-second outer allocation. Budget 600 seconds for actual CPU pair preparation,
then one 512-game / 256-pair comparison at 400 simulations, priors 1, seed 20260913,
existing development panel `14470ee9…`, maximum 300 plies, no tablebases, rolling
128 and batch 4096. Arena stage ceiling 7,200 seconds; outer ceiling 12,030 seconds.
No extra search depth or repeated match is scheduled automatically.

Read the complete paired bank, report score/Elo and the existing nominal paired
95% interval, and decide whether to retain the candidate using effect size,
uncertainty, complexity and its contribution to the bootstrap objective. A positive
point estimate is not automatic promotion; an interval crossing zero is not an
automatic rejection or demand for more games. An estimated material loss makes
this fixed correction less attractive; a promising result can justify provisional
use or a distinct follow-up without claiming replication. Extra games earn compute
only if likely to change a practical decision.

This is a policy contrast conditional on one chosen value recipe, not a factorial
estimate of policy-by-value interaction. Reused development openings, one training
seed and the three historical control caveats remain. It establishes neither 100M
transfer nor readiness to resume RL.
