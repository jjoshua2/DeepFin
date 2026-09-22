# 500M bootstrap continuation and partial factorial readout — September 21

## Decision

Continue the frozen 58,090,688-row factorial through D and its two registered
matches before selecting the policy/value package. Retain BT4 batch128 and keep
prefetch off by default until its queued complete GPU pipeline screen passes.
The larger 1,963,948-row packed-storage stream now passes exact tensor parity;
actual GPU training throughput remains to be measured. SF-free E preparation
continues separately. No live training inputs, targets, engine binary, or search
settings were changed by this continuation.

This is a partial readout, not a final factorial winner or a 500M throughput claim.
The main checkout predates several open research PRs; current evidence was recovered
from their pinned runtimes and actual completion artifacts, not inferred from main.
The preceding records are in [factorial PR #786](https://github.com/jjoshua2/DeepFin/pull/786),
[storage PR #791](https://github.com/jjoshua2/DeepFin/pull/791),
[packed sampler PR #795](https://github.com/jjoshua2/DeepFin/pull/795), and
[generation PR #789](https://github.com/jjoshua2/DeepFin/pull/789).

## Factorial health and target interpretation

| Arm | Policy | Main WDL | Verified state |
| --- | --- | --- | --- |
| A | BT4 T=0.5 | 50% SF / 50% BT4 | Complete, 58,090,688 rows / 113,459 updates |
| B | Equal BT4 T=0.5 / Ceres T=0.5 | Same as A | Complete, same rows/updates |
| C | Same as A | Equal thirds SF / BT4 / Ceres | Complete, same rows/updates |
| D | Same as B | Same as C | Active; initialization verified, epoch completion pending |

Ceres values retain the registered 60% primary T=0.55 / 40% secondary T=1.5
calibration. The equal-third implementation is 2/3 normalized stored V50 plus
1/3 Ceres, inheriting the stored V50 float16 rounding. Sharpen each policy teacher
before mixing; do not sharpen BT4 twice or sharpen the mixture again.

All four actual initial tensor fingerprints agree. A/B/C completed one exact epoch
with zero same-game repeats inside batches, and each realized schedule hash equals
its own plan hash. Independent review mapped all 7,108 staged shards to identical
ordered base inputs and source partitions. The physical schedule hashes differ
because they include different target content/paths. A canonical realized batch-stream
hash was not banked: equivalence is supported by canonical input identity and the
reviewed deterministic scheduler, rather than a full epoch replay.

The realized loss guard reports 100% prepared search_wdl supervision and zero
outcome leakage for A/B/C. The SF share is already inside that prepared target;
zero direct sf_wdl fraction does not mean SF supervision was removed. The legacy
control-validity flag is false in all arms because no purity receipt/live config was
supplied and exact game epochs differ from the historical replacement sampler.
These are matched development experiments, not independently held-out or live-RL
control claims. One training seed limits generalization.

| Completed direct contrast | Games / pairs | Elo | Paired 95% interval |
| --- | ---: | ---: | ---: |
| B minus A: policy addition at V50 | 256 / 128 | +6.79 | [-24.95, +38.63] |
| C minus A: Ceres substitution / lower SF with BT4 policy | 256 / 128 | +21.74 | [-12.49, +56.40] |

The working hypothesis remains that BT4 and Ceres are useful for both policy and
value, as reiterated by the user during this readout. Neither completed contrast
disproves that hypothesis: both point estimates are positive. This factorial tests
the incremental effects of these particular mixtures and horizon; it cannot establish
that either teacher is generally unnecessary. Keep D as a serious scale candidate;
an interval crossing zero is not a reason to discard either teacher.

Both use 400 simulations and prior temperature 1.0, with the frozen opening/seed
protocol. The Ceres addition has the larger observed estimate, but neither result settles
superiority. C-A replaces one-third of the existing equal SF/BT4 mixture with
Ceres: both SF and BT4 fall from one-half to one-third. It does not isolate Ceres
benefit from reducing either incumbent teacher. The historical online-RL
SF-removal result does not establish that SF values are useful in this bootstrap.
Treat SF value supervision as a candidate, not a required dependency. A matched
SF-free BT4+Ceres value mixture is the relevant follow-up, with policy held fixed.
Do not derive pure BT4 values by subtracting rounded V50 probabilities; use their
original authenticated teacher bank. Existing factorial runs remain unchanged. D-C and D-B remain the registered conditional contrasts needed
to evaluate interactions and the combined recipe. Do not select extra games based
on whether an interval narrowly crosses zero.

Training wall times inclusive of startup were A 7.48h, B 11.53h, C 8.30h. These
are descriptive, with cache/overlap/storage confounds; they do not measure a causal
cost of either target. D was alive and reading corpus data at the September 21
15:12 UTC check. The supervisor was alive, disk had about 490GiB free, and available
RAM was about 74GiB. Low GPU use during startup alone was not treated as a hang.
D-C and D-B remain serialized behind D under the existing GPU lease and September
26 deadline. Their descriptor pins and completion destinations passed preflight;
D's final artifacts must still bind successfully at dispatch.

## Recovered arena bookkeeping

Both completed arenas were falsely marked failed: after the A-overlap plan moved,
the runner wrote `complete.json` beside the new plan, but the registered completion
path still pointed to the old scratchpad directory. Both arena processes exited
zero. Independent review revalidated actual donor bytes, plan/runtime pins, bank
hashes, all 256 games, swapped opening pairs, and result equality.

Under the existing operator lock, the two queue items were reconciled to `logged`
with their actual receipts and explicit repair provenance. Original descriptors and
failed-run records remain untouched. Queue/state snapshots and intended updates
were saved before mutation; readback passed. No game was rerun, and active D and
both queued matches were preserved. The remaining thirteen historical failed items
are not thereby current incidents.

Evidence: [independent review](evidence/500m-continuation-20260921/factorial-review.json),
[repair receipt](evidence/500m-continuation-20260921/reconciliation.json), and
[compact results](evidence/500m-continuation-20260921/factorial-results.json).
Full repair artifacts are under
`~/chess-artifacts/operations/factorial58-reconcile-20260921/`.

## Scale optimization readout

| Measurement | Completed observation | Decision |
| --- | --- | --- |
| Generation at concurrency4 | G10 17.37 eligible rows/s; all-d8 64.23/s | 3.70x short-screen gain; changed play/labels, no strength equivalence |
| Packed exact sampler, batch512 | External ZIP 25.88s vs NVMe directories 24.43s over 262,144 rows | Ordered tensors identical; favor packing, full GPU/cold-scale test outstanding |
| BT4 batches128/256/512 | About 800/786/753 producer rows/s | Retain128; 256 WDL difference 0.002916 exceeds 0.002 threshold |
| Scalar root d8/d10 SF labels | About 36.6/35.7 labels/s per engine on 64 saved histories | Separate labeling from new-position generation |
| SF reset profile, root d8 | 1.646s reset / 0.116s search over 64 rows | Test retaining tablebase mappings while clearing TT/history |

Generation banked 10,431 G10 and 38,615 d8 rows in approximately ten minutes per
arm, with no result/support failures among listed closed rows. These are short,
startup-affected observations, not stable fleet estimates or deduplicated counts.
500M in 30 days requires 192.9 rows/s; another 441.91M beyond the known eligible
58.09M requires 170.5/s. The previous 418M planning remainder assumes an 82M starting
inventory, which is not interchangeable with 58.09M fully qualified training rows.
Do not treat new-position generation, scalar SF labels, and GPU teacher throughput
as the same rate or extrapolate four-worker scaling without measurement.

Selected storage samples imply roughly 318–354GB for 500M derived rows and about
1.28TB for raw rows before additional teachers/variants/checkpoints. Packed storage
preserves bytes and lazy metadata reads; a cold large-working-set and short GPU
training confirmation remain necessary. No active inputs were migrated.

Empty-Syzygy profiling reduces reset time but changes ranked all-d8 outputs on
5/64 rows, so removing tablebases is not an equivalent optimization. The prepared
retention option leaves TT and history clears unconditional and retains only immutable
tablebase state. The next bounded qualification compares deployed-original,
candidate-disabled, and candidate-enabled outputs on the existing 72-case bank,
including real tablebase probes and path/option lifecycle. Timing compares the same
candidate binary disabled/enabled to avoid compiler/build confounding.

Build cap: two affinity CPUs, nice19, 600s, 40GiB available RAM, 100GiB free disk,
2GiB new output. Qualification cap: 15min wall / 30 CPU core-minutes, CPU only,
owned-process cleanup, raw per-case banking. Any move/score/rank mismatch prevents
an equivalence pass; missing meaningful tablebase probes is inconclusive. Passing
finite fixtures does not automatically authorize production adoption.

[Scale evidence and hashes](evidence/500m-continuation-20260921/scale-evidence.json)
identify all completed artifacts, including prepared markers that are stale.

## Retention qualification completed

The isolated build completed in 48.11s. Its first loader attempt failed because the
system libstdc++ lacked GLIBCXX_3.4.32; that attempt is preserved. The successful
qualification explicitly supplied `~/.local/gcc-15.3/lib64`, recorded in
the manifest. The deployed engine remains unchanged.

All 648 normal searches (72 cases x 3 settings x 3 engine variants) and 48 option
variants completed: 232 complete triples. Independent reparsing of all 700 raw UCI
searches, including four lifecycle probes, found exact moves, ranks, scores, depths,
bestmoves and WDLs. Nodes and tablebase-hit counts also matched within triples.
Real 6-piece and 7-to-6-piece probes hit tables; mapped files survived retention-on
resets/Clear Hash and cleared on retention-off resets and explicit path reloads.
ProbeLimit0 suppressed hits and changed the eight option fixtures. Toggling the
50-move rule changed none of those fixtures, so this bank does not demonstrate a
behaviorally active 50-move-rule contrast.

The harness exit status alone did not gate every requirement. A separate saved
validator and independent bank review checked complete coverage, requested depths,
real probes and mapping lifecycle before the finite-fixture PASS.

| Same candidate binary | Disabled total seconds | Enabled total seconds | Ratio |
| --- | ---: | ---: | ---: |
| root d8, 72 cases | 2.401 | 0.880 | 2.73x |
| root d10, 72 cases | 2.466 | 0.983 | 2.51x |
| all-move d8, 72 cases | 7.068 | 5.599 | 1.26x |

These are reset+search timings from one fixed-order, cache-affected short screen
with UCI banking overhead. The full qualification used 52.00s wall and 36.76 child
CPU-seconds. It establishes a promising optional labeling optimization with finite
fixture equivalence, not an end-to-end generation rate or automatic deployment.
Its value for the eventual training pipeline is conditional on whether SF labeling
is retained; faster SF labels are not evidence that the labels improve training.

[Completion receipt](evidence/500m-continuation-20260921/retention-complete.json),
[independent review](evidence/500m-continuation-20260921/retention-review.json), and
[exact runtime manifest](evidence/500m-continuation-20260921/retention-manifest.json)
identify the binaries, patch, inputs and harness. All raw observations, the failed
loader attempt, build logs and exact harness/validator are under
`~/chess-artifacts/operations/sf-retain-qualification-20260921/`.


## Does SF value supervision earn its cost?

[Banked endpoint evidence and source hashes](evidence/500m-continuation-20260921/sf-value-evidence.json)
support the following comparisons. The current bootstrap evidence does not establish that a nonzero SF value share is
necessary. Historical online-RL removal is a different intervention and must not be
used as proof here. In the matched 35M series, reducing SF100 to SF50/BT4-50 gained
+31.30 Elo [8.53,54.34]. Removing the residual SF50 in favor of native BT4 values
then gave +8.82 [-16.84,34.58] and -18.34 [-43.19,6.33] in two 512-game matches.
Those are arena samples of the same seed101 checkpoints, not independent training
replications. They neither establish monotonic benefit from reducing SF nor show
that the residual SF share improves training.

The current factorial's C-A and D-B add Ceres while lowering both SF and BT4
from50% to one-third each. Their outcome cannot attribute improvement separately
to adding Ceres and reducing either incumbent teacher. The earlier CeresV25 recipe also retained50% SF, so it is not an
SF-free BT4+Ceres control.

### Next value contrast, specified before its outcomes

Keep the combined policy target fixed at half BT4 T=0.5 / half Ceres T=0.5.
Compare D's equal-third SF/BT4/Ceres values against E's **half native BT4 / half
Ceres, zero SF**. This isolates whether SF earns weight relative to the same fixed
1:1 neural-teacher mixture. Ceres retains the same dual-head calibration; use
original authenticated native BT4 WDL, not inversion of rounded V50 data.

E must use the same58,090,688 canonical rows, initialization, seed121, batch512,
exact one-epoch schedule, optimizer, losses and masks as D. Admit full original
BT4/Ceres value coverage and actual value-loss wiring before training. Preserve
D's completed checkpoint and results; E follows existing D matches, not a restart
or rewrite of the four-arm factorial. Missing coverage is a preparation blocker,
not permission to compare a subset against all of D.

Deciding measurement: one fixed400-simulation E-versus-D arena,256 games/128 swapped
opening pairs, prior temperature1.0, frozen opening bank and seed2026092101.
Bank games and report paired intervals. This is a fixed-budget decision sample;
do not extend to chase significance. Prefer E provisionally if its estimated
strength is at least as good while removing SF label cost; a negative estimate
favors retaining D for this screen. In either case, an interval spanning meaningful
wins and losses remains unresolved, and one training seed limits transfer. The
user's neural-teacher hypothesis remains viable even if this exact mixture loses.

Budget ceiling: CPU preparation16h, one E training epoch20h plus2h disk-pause
allowance, one arena1h, existing resource floors and serialized GPU lease. No
500M rewrite, arbitrary coefficient sweep or extra matches is implied. These are
caps, not runtime estimates. Do not launch until actual coverage, immutable plans,
recovery bindings and independent review pass. At this readout the contrast is
specified only; no E targets, training or matches have been launched.

## Additional startup optimization identified

D was confirmed alive on the host at2h09m with about135% CPU,305.91GB logical
reads and6.75GB physical reads. Its last log message places startup after model
construction and before the completed exact-epoch plan. Source tracing finds
repeated schema2 overlay validation: qualification, storage hashes, lazy proxy
creation and objective census reopen manifests, decode replacement probabilities,
check normalization/legal masks and walk base metadata. These paths imply nine
semantic validations per shard after initial-state creation; this is source-derived
accounting, not a measured phase-time breakdown or proof of its exact current call.

The next storage-side candidate is an operation-local verified-overlay context that
reuses validated proxies/digests while retaining anchored identity and mutation
checks. It needs identical corpus/objective/schedule/tensor results plus fail-closed
mutation tests and measured stage timing before adoption. No persistent unchecked
hash cache or pinned runtime modification was made. This addresses startup cost
regardless of whether the final value recipe contains SF.

[Source analysis and limits](evidence/500m-continuation-20260921/startup-cost-review.json).


## Evening continuation: implementations published and E preparation active

At September 22 00:12 UTC (September 21 local), D reached 92,488/113,459
updates, approximately 81.5%, and its D–C and D–B matches remained queued.
The active frozen runtime was preserved.

[PR #810](https://github.com/jjoshua2/DeepFin/pull/810) implements the identified
operation-local overlay validation reuse. Independent review and 54 tests pass.
A sequential eight-shard, 65,536-row constructor-stage comparison reduced
semantic validations from 72 to 8 and time from 15.011 to 2.772 seconds (5.42x),
with identical full plans/objective counts and ordered target hashes. This is
not a full-corpus startup or training-throughput measurement. Whole-repository
type-check failures exactly reproduce the frozen baseline's 23 diagnostics.

[PR #811](https://github.com/jjoshua2/DeepFin/pull/811) implements the registered
SF-free E recipe using original native BT4 values. Fifteen tests pass. Independent
review verified all 35 manifests and current teacher metadata pins, plus three
real source families totaling 24,576 rows: D policy and every non-value array
remain exact, while independently computed E WDL matches byte-for-byte.

The full 58,090,688-row CPU preparation launched at 00:10:55 UTC, PID 1755363,
under independently reviewed plan SHA256
`c0e137a3ec1a64bd6bc1fd2d0345901b18d4c2408923c302869cb710cdd40cdc`.
It uses frozen builder commit `4ecf772963967cb9de53d6e9a2cd895172a41e15`,
two CPU threads, nice19, no GPU, one 16-hour cap, 150 GiB disk reserve and
40 GiB available-RAM floor. The same capped job qualifies all 35 output roots.
Failure/STOP/resource tests pass and preserve incomplete artifacts.

Plan, logs and terminal receipt location:
`~/chess-artifacts/operations/factorial58-sffree-preparation-20260921`;
outputs: `~/chess-artifacts/labels/factorial58_sffree_20260921`.
E training is not launched or queued: it still requires completed full-corpus
qualification, reviewed exact training admission and D's remaining matches.
The teacher hypothesis and 500M production qualification remain unresolved.

## Labeling and storage CPU optimizations

The next bounded optimization pass preserved active D and SF-free preparation.
Three independently reviewed changes are published:

- [Ceres encoding #812](https://github.com/jjoshua2/DeepFin/pull/812) reduces binary-history reduction cost. Across three saved panels, encoder CPU time fell 22.7–30.3%, with all 3,072 feed outputs byte-identical. The author and independent reviewer each passed 118 focused tests. This is encoder-component evidence; inference throughput was not measured.
- [BT4 source reading #813](https://github.com/jjoshua2/DeepFin/pull/813) avoids decoding 512-row source chunks four times for 128-row inference batches. An 8,192-row ABBA test measured 3.91x/4.09x faster source reading on NVMe/external ZIP and 1.31x faster total measured CPU preparation. All source/feed/batch hashes matched. Twenty-eight tests and independent review passed. External ZIP exercised the reader helper; full-producer packed-source admission was not added.

[Raw BT4 preparation #814](https://github.com/jjoshua2/DeepFin/pull/814) projects only the eleven fields consumed by labeling and
verification, avoiding allocation of unused Stockfish search payloads. After
integer/nesting/optional-dependency compatibility fixes, the complete 8,236-row
CPU screen measured 8.64s to 5.16s median read-plus-encoding time (1.675x), with
all consumed fields and seven ordered identity/input/legal/feed hashes exact.
The initial ten-field parser-only 33x prototype is not the final result. Sixty-two focused tests passed on the locked dependency version; independent
review verified the implementation, paired results and compatibility cases.

These gains address different CPU stages and must not be multiplied into an
end-to-end speedup. Full GPU-labeling throughput and large cold-working-set
training remain unqualified. New-position generation is a separate bottleneck.
No active frozen runtime was changed to adopt these patches.

## Queued complete labeling check and larger storage qualification

[BT4 prefetch #815](https://github.com/jjoshua2/DeepFin/pull/815) adds one bounded
CPU preparation future ahead of main-thread ONNX inference. It remains off by
default. Seventy-three tests, independent interrupt/cleanup review and exact
8,236-row CPU feed/identity parity pass; these do not establish GPU speedup.

[Complete pipeline screen #816](https://github.com/jjoshua2/DeepFin/pull/816)
compares original serial decoding, projected serial decoding and projected
prefetch in fixed ABCCBA order at batch 128 on the same 8,236 rows. Its deciding
metric includes labeling plus deep output verification, with exact actual input,
raw neural output and stored-array parity. A 5% reduction versus optimized serial
is required for the prefetch screen. Hash observation cost is included. Nine CPU
tests and three actual-producer/verifier call-contract smokes pass, along with
independent code and final launch review. The 20-minute job is queued after D,
D-C and D-B; append-only readback preserved every prior item and active state.

At 01:22 UTC September 22, D was at 105,688/113,459 updates (93.2%). In its latest
100 completed windows, training-thread batch preparation/wait accounted for
1,172.155 of 2,783.746 seconds (42.1%). This phase includes sampler/CPU preparation,
memory pinning and transfer submission. It is not isolated disk I/O or measured
GPU idle time. Concurrent CPU work and cache state prevent causal attribution.
The per-window observations and source snapshot identity are banked in the
[phase snapshot](evidence/500m-continuation-20260921/D-phase-snapshot.json).
This supports prioritizing the data-to-training path, while leaving the storage
comparison to determine the external drive's effect.

The larger storage preparation selects 256 unique shards across 35 cohorts,
1,963,948 rows, for an actual GPU trainer comparison of external ZIP against
NVMe directories. CPU packing and full-stream qualification launched under a
separate reviewed 20-minute cap, two threads, nice 19, disk/RAM reserves and
owned-child cleanup. Preparation is still pending; it is not a GPU result or a
cold multi-terabyte storage qualification. E's separate full-corpus SF-free
preparation continues; E training remains unadmitted.

The first larger-storage qualification stopped after 626.29 seconds because the
strict packed validator rejected preserved root-level `row_provenance.npz` files
present in 224 of the additional source shards. Packing itself passed byte checks
in 229.83 seconds. This is a representation-compatibility failure, not evidence of
storage slowdown or target mismatch. A narrow fix accepts that one root-level
auxiliary member while retaining archive hashing and all other member checks;
33 packed-epoch and CPU-trainer tests pass independently. Existing archives and
failed receipts are preserved. The initial qualification stage took about 396 seconds, completing the directory
stream before the packed-loader eligibility failure. A fresh 20-minute
qualification-only budget was authorized before requalification; no samples, targets or GPU decision thresholds were changed.

The fix is published as [PR #817](https://github.com/jjoshua2/DeepFin/pull/817).
Corrected qualification launched with reviewed plan SHA256
`e321ca4f189ae00ede1f0e367301d85530f21d9499eb2dc4ffcd15f39a295b79`;
all 256 archive bytes and staged bindings were independently verified before
launch. GPU admission remains contingent on both complete streams passing.

Corrected qualification completed in 748.045 seconds, within its new cap; total
CPU preparation including the preserved failed attempt was 1,374.334 seconds.
Both representations consumed all 1,963,948 rows with exact ordered tensor hash
`c39d8f17230b3ab22cbe4203cca15d3e314085a94df795cd12b58fbbf0300707`.
The [full-stream receipt](evidence/500m-continuation-20260921/storage-full-stream-qualification.json)
and [completion receipt](evidence/500m-continuation-20260921/storage-requalification-complete.json)
bank the result. Cached CPU consumer time including digest observation was
318.643 seconds for NVMe and 340.250 seconds for external ZIP. These are
qualification timings, not GPU throughput or cold 500M storage evidence.
The actual GPU comparison is published in [PR #818](https://github.com/jjoshua2/DeepFin/pull/818).
