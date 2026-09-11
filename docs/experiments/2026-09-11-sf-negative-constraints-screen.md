# Saved SF constraints for neural bootstrap targets

Registered September 11, 2026. Status: all 262,079 rows completed and the disjoint two-bank readout passed independent review. Tactical300 training remains on hold behind the registered Ceres anchors; the negative-constraint family remains plausible. No new target rewrite, teacher inference, training or playing result.

## Decision

Use the existing complete run06 G10 increment to determine whether SF has a useful role identifying bad neural choices while BT4 ranks several good alternatives. Keep Ceres collection and the [registered policy/value strength anchors](2026-09-11-ceres-weighted-bootstrap.md) first in the GPU queue. This CPU screen informs subsequent candidates; it does not replace those anchors or expand bulk SF search depth.

The reviewed PRs contribute different pieces:

| PR and reviewed head | Contribution and limit |
| --- | --- |
| [640](https://github.com/jjoshua2/DeepFin/pull/640), `98122708af351f1cfb732f333f444473bb5b9c04` | Tactical300 transfers half the inferior-move mass to SF maxima when their gap to the next distinct score exceeds 300 cp; separate mate handling. A stronger sparse correction, not a demonstrated improvement. |
| [643](https://github.com/jjoshua2/DeepFin/pull/643), `cb672a02f3030afaa24480e61b706eaab8fff790` | Retrospective checks using saved narrowed d10/d12 rosters. Its gap matches Tactical300, despite the broader best-versus-BT4 wording in the PR description. |
| [644](https://github.com/jjoshua2/DeepFin/pull/644), `99af252cf9d6846d49f840e5cbe2e7011248a4e9` | Ranking, covered-policy regret, teacher disagreement and routing diagnostics. Reporting corrections are needed before a substantive run. |
| [645](https://github.com/jjoshua2/DeepFin/pull/645), `1e76b7b484d451e8dadd1fc1492faef265a64af6` | Shorter shared instructions and task-specific navigation; no scientific result. |

All four were open at review; issue comments, review bodies and inline threads were inspected and empty. Source review included an independent reviewer of PR644. These are the reviewed versions, not claims about future revisions.

## Why this differs from Tactical300

For SF scores **+50, +45, -400**, with BT4 preferring the third move, the best-to-runner-up gap is **5 cp**, but the SF regret of BT4's choice is **450 cp**. Ordinary Tactical300 does nothing. Removing some probability from the third move while preserving BT4's ranking of the first two is a different hypothesis from reinforcing an isolated SF best move.

The earlier [Tactical100 comparison](2026-09-10-bt4-sf-tactical-training.md) was unresolved at -13.6 Elo [-52.4,+24.9]. Neither that result nor the proposed stronger Tactical300 establishes whether the all-move negative-constraint family is useful.

## First screen and resource limit

Use original complete run06 G10: **262,079 positions in 32 derived shards**, original BT4 policy and already saved SF observations. Adapter manifest:

`/home/josh/projects/chess/scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_common_increment_v1/run06_g10/adapter_manifest.json`

SHA256: `8b81f9633050f8e2f286259671e3cefb39bfb30dffa29c7d76608723d36505c8`. Original derivation summary: `ab214d52665ee4ad6ccf756c0515344dc9653c3b51de15965fc74323198b7fd9`.

Qualify the corrected consumer on the first shard with a **10-minute inclusive limit**. Record actual runtime, peak RSS, source identities, audit revision and output digests. Then decide whether to scan the remaining 31 shards within a **90-minute inclusive limit**, using the first shard once in the combined result. Stop on failed identity, malformed required input, resource limit or STOP marker; retain failure evidence and never reinterpret it as a negative recipe result.

Use two CPU threads on cores 2,3, GPU hidden, 8 GiB address-space cap, 4 GiB fresh-output cap, at least 150 GiB SSD reserve and the existing shared preparation lock/owned-process termination helpers. No new engines, teacher inference or GPU arena. Do not extend a failed budget automatically.

## Measurements and corrections

Bank compact per-row measurements with source/shard/game/position identity so later intervals and alternative summaries reuse observations:

- Both SF gaps: best versus next distinct score, and best versus BT4's preferred move/set; explicitly define tie handling.
- BT4 probability on d9 moves more than 100/300/500/1000 cp below the best. These existing thresholds describe scale; they are not a tuning grid that chooses an Elo optimum.
- Saved deeper confirmation, contradiction, tie and unavailable counts **and affected BT4 probability mass**. Distinguish an existential best-set comparison from constraints involving every tied winner.
- Covered probability mass including zero-coverage rows. Conditional regret has its own positive-coverage denominator; compare recipes on a common eligible population.
- Unknown deeper reversals remain unknown, rather than entering position strata as false. Preserve final depth/selector and mate eligibility.
- Unchanged BT4 T=0.5 versus the exact ordinary Tactical300 preview, including actual probability moved. A new soft-veto target is a subsequent candidate, not silently substituted for Tactical300.

Report absolute SF-score and check/legal-count/ply slices where banked data support them. Categorical mates remain separate from ordinary centipawn gaps. For uncertainty, cluster by source-qualified game identity rather than treating adjacent positions as independent; any formal threshold comparison will use the same frozen rows.

## How the result changes the next action

The screen asks whether the negative-constraint mechanism deserves a small target candidate, not whether SF agrees with itself significantly. Report the joint tradeoff between how much neural mass is affected, how much is actually adjudicable, and contradiction versus confirmation on that adjudicable mass. If effects are negligible or mostly unscored, do not spend a training epoch merely because the code exists. If a substantive correction is supported, define one soft suppression candidate and a matched strength comparison before training; do not optimize twenty thresholds against the SF ruler.

Narrowed d10/d12 rosters are selected by SF. Missing moves have no invented deeper value, and low conditional regret or few observed reversals cannot prove reliability on unscored moves. The exploratory screen has no promotion threshold and no automatic escalation to a full training run.

## Value and Ceres follow-ups

PR644 currently requires Ceres before computing any value metrics; today's G10 policy adapter also lacks native WDL. Reuse the separately authenticated native BT4 WDL and [matched SF/BT4 products](2026-09-11-g10-value-pilot-and-coverage.md) through an explicit join if the next value diagnostic earns its cost. Deeper-SF WDL agreement measures consistency with that teacher, not objective accuracy or Elo.

The original-18.91M Ceres collection is not G10 row coverage. A later small G10 disagreement/control sample could compare Ceres, BT4 and arithmetic mixtures without training a student for every weight. Stratified selections need population weights and independent controls. No new Ceres collection is launched by this record.

Retrospective routing is lower priority: capture of already observed reversals does not establish prospective search savings or outcomes on unsearched moves. Existing d8/d9/d10/d12 data remain the affordable starting point.

## Real-data qualification and remaining cohort launch

The corrected runtime in [PR648](https://github.com/jjoshua2/DeepFin/pull/648), commit `bf668d7c5dc35558bb3b0351dc7edb48409330a6`, passed independent review and final whole-repository static checks. Twenty focused tests passed before the final position-category banking change; its targeted row-bank regression passed afterward.

The first 8,192-row shard completed in **46.90 seconds** (producer 46.81 seconds; peak RSS 1,250,120 KiB). Independent readback verified all 8,192 unique row identities, finite JSON, policy/ranking aggregate counts and mass, paired eligibility, and unknown-outcome accounting without rereading source arrays. The shard contains 6,103 ordinary d9 rows and 2,089 mate-domain exclusions. This qualifies the consumer and cost; it is not a representative full-cohort result or evidence of strength.

Based on that measured cost, the remaining **253,887 rows / 31 shards** launched under the registered 90-minute limit, with unchanged runtime and resource bounds. Session 17956 owned this pass; shard zero was excluded and its completed bank was reused once. The completed readout below verified the disjoint union against the original 32-shard inventory.

[Compact qualification and launch evidence](evidence/sf-constraints-pilot-20260911.json) pins the receipts, metric bank, independent review and remaining-shard plan. Bulk per-row evidence stays at the named host paths.


## Completed cohort and next decision

The remaining 31-shard pass completed in **23m23s**, with peak RSS **1,993,580 KiB**.
The one-pass metric-bank readout verified **262,079 unique rows / 32 original shards**
and reconstructed each input summary's counts and mass. It retained sufficient
statistics for **1,331 source-qualified game clusters**. No raw source, teacher
inference or bootstrap was repeated for this publication.

Of all rows, **187,628 are ordinary d9 positions** and **74,451 (28.41%) are
mate-domain exclusions**. Malformed required baselines remain fatal; the successful
run did not silently omit invalid identities. BT4 and Tactical300 share 182,856
valid conditional-regret rows, with 4,772 ordinary rows lacking eligible final
policy metrics and zero observed zero-coverage rows. Those denominators remain
separate from ranking constraints, which can include mate-domain final scores.

| Strict d9 inferior-move threshold | Mean BT4 mass on flagged moves, across ordinary rows | Flagged mass without a complete deeper comparison | Contradicted fraction of adjudicable flagged mass |
| --- | ---: | ---: | ---: |
| >100 cp |18.929% |52.44% |13.17% |
| >300 cp |6.100% |59.03% |11.66% |
| >500 cp |3.948% |59.77% |12.32% |
| >1000 cp |3.076% |60.41% |13.62% |

Here a contradiction means **at least one tied d9 winner scores below the
inferior move** under the conservative all-winner test. The 11.66% is
probability-mass-weighted and conditional on available comparisons; it is **not an
SF error rate**, a row error rate, or evidence that BT4 beats the surviving best
SF move. At >300 cp, confirmed/contradicted/tied/unscored mass averages
2.0545% / 0.2914% / 0.1532% / 3.6007% of each ordinary policy. Missing counterfactual
scores dominate the flagged mass and cannot be treated as agreement.

The exact ordinary Tactical300 preview moves mean probability mass **0.9817%**
across ordinary rows and triggers on **13,495 rows**. On the 182,856 common valid
rows, its conditional-regret change versus unchanged BT4 is **−48.23cp**,
with stored 95% game-cluster interval **[−52.70,−43.63]cp**. Coverage rises
**0.2755 percentage points**. Each target is independently normalized on the
scored roster; common rows remove population mismatch, not this normalization
or the missing-move bias. Scientific triggers use the exact >300 next-lower gap
and positive inferior mass; 31,669 additional bit-level differences caused only
normalization drift and are excluded from the changed-row population.

| Phase0 d9 best-score bin | Ordinary rows | Common valid pairs | Mean >300cp inferior BT4 mass | Mean Tactical300 conditional-regret change |
| --- | ---: | ---: | ---: | ---: |
| Absolute score <=100cp |17,031 |17,012 |0.45266% |−0.971cp |
| Score >100cp |97,360 |94,548 |6.92705% |−59.984cp |
| Score <−100cp |73,237 |71,296 |6.31341% |−43.911cp |

**99.81% of the summed paired-cp reduction lies outside absolute d9-best score
<=100cp.** These are score bins, not established decisive positions. The balanced
bin is only 9.1% of ordinary rows and has little mean affected mass; neither
that observation nor the other bins establishes irrelevance to Elo. Rare
important tactics remain possible. No finer concentration analysis is claimed. Separately,
4,180 ordinary rows have BT4-top regret>300cp while the best-to-next-lower gap
is<=300cp; only 40 of those lie in the balanced bin. This remains a different
axis from Tactical300, not an optimized alternative recipe.

The stored intervals use 1,000 cluster resamples, seed 20260911, grouped by
(source_namespace, raw_shard, worker_id, game_id). They describe frozen-bank
sampling uncertainty conditional on these teachers/checkpoints and observed
rosters, not training-seed uncertainty, unbiased teacher truth, search savings
or Elo. Score and position strata are descriptive.

**Hold Tactical300 training behind the registered Ceres policy/value anchors.**
Retain Tactical300 and SF negative constraints as plausible later candidates; this
SF-agreement result alone neither promotes nor rejects the family. Ceres may
change the strongest neural baseline or correct some of the same errors. A bounded,
authenticated same-row G10 Ceres conflict/calibration sample with agreement
controls could offer new discrimination; it is an option, not a mandatory new calibration
gate. Existing stratified selections are not
automatically collector-qualified or representative; preserve inclusion
probabilities and held-out games before drawing population conclusions. This
record launches no new teacher collection, target rewrite or training.

[Compact completed evidence](evidence/sf-constraints-completed-20260911.json)
pins the full readout, both terminal receipts and metric-bank hashes, game-cluster
sufficient statistics, completed-input plan and independent reviews. The [complete machine-readable readout](evidence/sf-constraints-full-readout-20260911.json)
publishes all reported strata and intervals. The original bulk row banks remain at
their pinned host locations.
