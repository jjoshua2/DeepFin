# Bootstrap evidence and scale-readiness audit — September 16

This audit supersedes interpretations in the local autonomous-loop recap where they
conflict with the evidence below. It does not erase results or change completed
experiments' stopping rules. UTC September 16 corresponds to local September 15
for the inspected evening jobs.

## Decision

Prepare larger-scale training now. Do not wait for every small target variant to
be resolved. A full 100M launch is not ready: corpus eligibility, teacher coverage,
storage staging and training admission still need verification. Advance the largest
complete common corpus, likely 40–60M first, while generation and preparation grow
toward 100M. This is an intermediate transfer, not a redefinition of the goal.

The next high-value science is one matched combined-teacher training comparison
at an adequate horizon, followed by scale transfer and independent training seeds.
Do not schedule more legacy-checkpoint blowouts or require one last match to grant
Ceres permission to participate. Estimates, uncertainty, mechanism and cost guide
provisional recipe choice; confidence intervals crossing zero are not vetoes.

## What the games actually establish

The independent audit checked 55 completed queued arena banks, totaling 15,104
games. Every inspected bank had its requested count and complete swapped pairs.
That is useful preservation of results. It is not a requalification of every
checkpoint byte, runtime or launch. The operator's newer launch records are less
comprehensive than the earlier pinned package contracts.

| Question | Evidence | Appropriate interpretation |
| --- | --- | --- |
| More training | Two-epoch versus one-epoch B100 pools to about +145 Elo over 512 games | Large practical improvement for these checkpoints; prioritize adequate training |
| Ceres value contribution | CeresV25 versus B100: 1,792 games, +21.55 Elo, nominal paired interval +7.75 to +35.41 | Promising; adaptive selection/stopping and fixed models limit confirmation claims |
| Teacher temperature | Direct T1 versus T0.5 two-epoch: 768 games, −5.43 Elo, nominal interval −26.48 to +15.58 | T0.5 remains usable, but superiority is not established |
| V100 versus V50 | Two 512-game comparisons give opposite modest estimates and a near-zero pool | Both remain plausible value choices |
| SF/Ceres policy variants | SoftSF10 strongly negative; several other estimates near zero or modestly positive | Tested recipes have limited support, not proof entire teacher families are useless |

The repeated “seeds” in the recap are **arena seeds on the same trained
checkpoints**, not independent training replications. More such games narrow
conditional match uncertainty, not variation across training runs. Ceres' nominal
pooled interval is positive; calling it “not a 95% lock” was numerically misleading,
but adaptive accumulation still prevents treating that interval as a clean
prospective confirmation. The 896 pairs contain 894 distinct opening FENs.

The proposed 80% training / 20% unique-data decomposition is unsupported. The
20M one-epoch model has 18,910,484 presentations and 36,935 updates; its two-epoch
model has 37,820,968 and 73,870. The 35M model has 35,314,577 and 68,974, uses seed
101 instead of 0, and V50/V100 also change the value target. Source distribution
and schedule differ. Elo differences are not additive causal shares. A clean
unique-data test needs the same recipe, training seed and exposure budget, with
repeat-data versus additional-data arms.

## Correct the teacher inventory before planning compute

- B100 means **100% BT4 policy weight**, using a one-evaluation teacher policy;
  it does not mean 100-node BT4 search.
- All **35,314,577 rows / 21 cohorts** have the original SF targets, B100 policy
  and native BT4 value alternatives. All 42 policy/value recipe references were
  reread and hash-verified. The G10 contribution is 16,404,093 rows.
- V100 already is the pure-BT4 **main WDL** endpoint. Original SF columns remain
  in the corpus; retention does not establish active SF supervision. The inspected
  training summary reports zero auxiliary loss terms.
- CeresV25 already is **50% SF + 25% BT4 + 25% Ceres WDL**, with B100 policy.
  It is a completed three-way blend, not another teacher to blend a second time.
- Supported Ceres coverage is **25,306,004 / 35,314,577 rows**, leaving 10,008,573.
  This is indexed evidence, not an exhaustive search for unindexed newer banks.
- Run06/run07 closed progress indices total **61,837,781 raw rows / 7,451 shards**.
  Those generation manifests bank SF `all:9,8:10,4:12` observations. They are not
  unlabeled selfplay. Deriving targets from saved observations is not new SF search.
- A prior inventory already covered **59,264,839 raw rows with BT4 policy**, of
  which **23,827,971** also had native BT4 WDL. This is receipt coverage, not a
  newly verified eligibility claim for every payload.
- Beyond 35M, another 1,517,925 B100 rows are materialized and 4,219,426 SF baseline
  rows are derived. The latter still needs adapter/materialization work, and the
  combined 41,051,928 candidate rows need union admission. Do not call 41M ready.

Closed raw rows are not necessarily eligible, nonduplicated training examples.
The original 18.91M plus the 61.84M G10 pool is about 80.75M physical candidate rows,
not 80.75M ready training rows. Later narrowed d10/d12 observations also must not be
mistaken for complete legal-move root scores.

## Mechanism claims that should remain hypotheses

Eventual game length is affected by the engines and outcome. Stratifying on it
can induce selection bias: fast wins and slow losses can produce a long-game
“weakness” without a distinct stamina defect. Replication of the association does
not establish a phase-conditioned teacher recipe. Opening pawn structure is a
pre-game feature, but the small, adaptively selected locked-position subgroups
still need a prospective comparison. The worst64/best64 retest failed its extreme
predictions; that does not validate a different mechanism by elimination.

`termination=rules` means a natural chess-rule ending, including checkmate. It
is not an adjudication flag. The current game records omit moves/final FEN, so
mate/endgame breakdowns cannot be recovered from those fields. A 300-ply cap does
not certify a fortress. Bank moves or final board state and precise termination
reasons in future diagnostic games.

The actual running CPU audit uses **4,000 FEN-only positions** and several search
profiles, not the advertised 512-row raw-policy map. Its live script has no
`--max-positions` argument. A curated 64-row agreement slice also cannot show that
a network has already learned all useful SF tactics or that quiet disagreements
favor another teacher. Full-history, cheap forward-only measurements with a
prespecified question would be more relevant to the next target decision.

## Current work and next allocation

At inspection the event-driven supervisor was alive. The T1-versus-35M-V100 match
finished and the final CeresV25/B100 512-game match started. Two direct Stockfish
checks remain queued. No queue/process was changed by this audit. Those checks
are deployment-profile observations, not proof of a unique-data mechanism; verify
UCI settings, warmup, move logs and achieved time/nodes before interpreting them.
The arena compiler spawned a 32-worker pool despite earlier two-thread budgets;
future operators should restore explicit resource limits.

No corpus generator was visible in the process inspection. Investigate the saved
stop/recovery state and resume a bounded supervised fleet rather than blindly
restarting every old worker. Preserve checkpoint/data provenance and ongoing jobs.

The SSD had about **181 GiB free (94% used)**; the external drive about **7.3 TiB**.
With a 150-GiB reserve, full 100M duplication is not feasible. Reclaim only verified
cold archives, stage bounded working blocks, and qualify external-drive I/O before
making it the training store.

Measured Ceres collection was 264,629 rows in 1,907.48 seconds. At that throughput,
the missing 10M costs about **20 GPU-hours**; all 100M about **200 GPU-hours**.
These are linear planning estimates for the current pipeline, not optimized limits.
In contrast, the measured 35M V100 training charge was about 3.98 hours, implying
about **11.3 hours per 100M pass** if throughput scales. Both costs need actual
larger-run qualification; do not misprice Ceres or invent BT4-100 search costs.

Recommended sequence:

1. Publish this corrected inventory and results; stop adding redundant arenas.
2. Recover supervised generation, admit already jointly labeled rows, and prepare
   storage and missing-value labeling toward the largest complete common corpus.
3. Test one literal combined recipe against a matched reference. A useful starting
   candidate is 50/50 BT4/Ceres policy averaging with each teacher at T=0.5,
   plus 50% SF / 25% BT4 / 25% Ceres main WDL using the existing CeresV25
   value transformations. This is a proposal, not a claimed optimum. Use matching rows, seed, schedule and horizon.
   A joint positive result establishes the package; individual attribution would
   need ablations. The existing CeresV25 value evidence already earns consideration.
4. Transfer two plausible packages to the larger corpus, allowing uncertainty in
   the choice. If the mixed policy underperforms, keep the useful value blend and
   B100 policy. Confirm with a fresh training seed when it can change selection.
5. Advance to 100M as eligibility and I/O permit. Minor SF thresholds and mined
   subgroups should not delay scale; add a conditional rule only after a cheap,
   held-out diagnostic supplies a concrete reason to test it.

This is an allocation recommendation, not a launch manifest. No new training,
labeling, fleet restart or queue alteration occurred during this audit. GitHub main
was still at PR #753 when inspected: the recent autonomous results had not been
published there.

[Compact audit evidence](evidence/bootstrap-status-audit-20260916.json) retains
bank summaries and hashes of the local reports/inventory sources. Bulk game logs
remain at the listed local paths. The original results are preserved.

## Overnight execution follow-up

After the audit, the user authorized taking over the queue and then explicitly
requested 20–24 hours of automatic work. The replacement supervisor from PR #757
preserved the active CeresV25 match through completion. Its next small labeling
attempt failed: one fixed-32 inference ran, then a sandbox-versus-host mount
identity mismatch stopped validation. No completed sidecar was produced. The
failed attempt remains preserved; the retry checks host mount identity before
imports as well as during mapped-library verification.

A reviewed sequential batch is now running: **13 cohorts, 10,008,573 rows and 1,228
shards**, covering the remaining Ceres gap in the matched 35,314,577-row corpus.
Expected duration is **19.72 hours** from measured template throughput; individual
block caps sum to **23.18 hours**, with a **24-hour outer termination bound**.
The first 8,192-row shard completed successfully in 30.4 seconds after the host
identity correction. This is useful teacher labeling, not another arena sweep or a training launch.
Success retains legal policy and both raw value heads; it does not itself establish
training admission or playing strength.

The batch stops at the first failed command or qualification. It checks 48 GiB
available RAM and 170 GiB free disk at startup, retains running floors of 32 GiB
RAM and 150 GiB disk, and samples a 12 GiB aggregate output cap. Expected teacher
output is about 9 GiB. These are sampled guards, not a hard reservation against
other writers. Existing CPU adapter and generator jobs retain their original
3-hour and 8-hour bounds; no unreviewed training was added to fill time.

The batch wrapper is [PR #759](https://github.com/jjoshua2/DeepFin/pull/759): five
focused tests passed and separate wrapper/plan reviews passed. Frozen code and
input identities, source cohorts, limits and queue adoption are recorded in the
[overnight evidence](evidence/bootstrap-overnight-20260916.json). The supervisor
advances from saved completion receipts without conversational polling. On failure
it preserves completed blocks and leaves a terminal record for diagnosis.

The next combined-teacher materializer is separately proposed in
[PR #758](https://github.com/jjoshua2/DeepFin/pull/758), with independent review
still pending; it is not part of this overnight labeling queue.

The user's eventual billion-position objective strengthens the priority on teacher
throughput, unique positions and adequate training. Cheap broad SF labels plus
selective later deepening remain a hypothesis to compare with the existing
staircase. Neither a d8 optimum nor a billion-position growth breakpoint has been
established; source positions and versioned teacher labels should remain reusable.

## Reboot continuation — September 16, 13:00 UTC window

The original overnight batch stopped before the reboot, after about 44.7 minutes.
Its output-size scan tried to stat a temporary chunk after an atomic rename. This
was an operator failure, not a teacher-quality result. Block00 qualified all
264,297 rows; block01 retained 14 completed shards (114,688 rows) and an unfinished
next shard. All original receipts and payloads remain preserved.

PR #759 now fixes that race by tolerating FileNotFoundError during sampled scans;
permission and I/O errors still fail closed. Nine focused tests pass, including
an actual rename between directory enumeration and stat, with independent review.

After the user's explicit request for at least another 12 hours, a fresh
boot-bound queue was activated for the 11 untouched cohorts: **9,216,672 rows /
1,130 shards**, expected **18.14 hours**, with a **22-hour inclusive outer limit**.
The 21.12 hours of summed block allowances fit that bound. Previous block00 and
partial block01 are excluded to avoid repeating completed inference. Complete
coverage of the partially processed cohort remains follow-up work.

All 30 library file stamps were unchanged across the reboot; host boot and mount
metadata were rebound and verified, as were source/command pins. The previously
reviewed RAM, disk, output and sequential failure-stop controls remain in force.
The CPU adapter completed successfully before reboot and is not repeated.

[Recovery evidence](evidence/bootstrap-reboot-20260916.json) records the exact
queue and launch identities. Activation does not claim completion of the batch.

CPU generation also resumed for 12 hours at the existing 2+1 concurrency, with
32 GiB available-RAM and 150 GiB disk guards. Initial automatic approval review
rejected the resume cleanup risk. A separate copy-only operation then preserved
and independently verified all 34 files, including every one of the 16 incomplete
tails. The revised wrapper rehashes both copies and originals and requires its
exact all-codec deletion roster to equal those backed-up tails. Renewed review
accepted this concrete recovery; both generator processes and their monitor
reported running. Backups remain outside the corpus, and completed shards are
preserved. This changes neither label depth nor logical worker partitioning.
