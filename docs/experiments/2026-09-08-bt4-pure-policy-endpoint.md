# Pure BT4 policy endpoint versus H20

**B100 completed its registered seed-zero epoch at 11:43:28 UTC on September 8. The separately qualified B100-versus-H20 arena coordinator launched at 12:26:02 UTC; the 100-simulation stage started at 12:26:07 UTC. Arena outcomes remain unread in this publication.**

## Completed training and arena launch — September 8

The pure sharpened-BT4 **policy** endpoint completed all **18,910,484 rows,
36,935 updates and 420 windows**, with seed zero and batch 512. Every window had
finite loss and mean gradient norm, with **zero nonfinite skips and CUDA retries**.
The [training completion](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/training/training.complete.json)
binds the final checkpoint `b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62`.
The training charge was **9,258.994 seconds (2h34m19s)**, within its 4.5-hour cap.
This establishes completed training, not improved playing strength. SF-derived
value targets and all non-policy fields remained unchanged.

The [realized schedule](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/training/realized_schedule.json.gz)
completed its CPU verification in **274.70 seconds** and matches H20's canonical
schedule `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
The [independent launch review](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/arena/independent_launch_review.json)
checked the compact training summary, full window accounting, final checkpoint
receipt, schedule and arena binding. Historical purity, committed-config and
sampler limitations remain; this is a seed-zero development comparison.
The full training summary and checkpoint remain external, with locations and hashes
in the [evidence catalog](evidence/bt4-bootstrap/b100-completion-arena-launch-manifest.json).

The [arena operator receipt](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/arena/root_arena_operator_launch.json)
records coordinator **368555** starting at **12:26:02 UTC**. The
[100-simulation process snapshot](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/arena/startup/low/process.json)
records timeout supervisor **368593** and arena **368594** starting at **12:26:07 UTC**.
The [frozen launch manifest](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/arena/B100_H20.launch_manifest.json)
compares B100 with H20's registered final checkpoint using the allocation below:
100-simulation ordered-prefix GSPRT, first 128 pairs then 64-pair steps, capped at
500 pairs; followed by the fixed 128-pair 400-simulation probe on the same opening
prefix, regardless of a valid shallow result's sign. Each arena retains its
1.5-hour hard cap, shared GPU lease and failure/STOP handling. No arena result or
measured end-to-end speedup is claimed.

Actual preparation confirmed prior temperature **1.0** for both players, the full
16-ply histories and identical qualified search settings. Rolling pools are
**256 games at 100 simulations** and **128 at 400**, with evaluator cap **4096**;
actual model flags imply uncapped leaf batches of 4096 and 2048 respectively.
The CUDA arena remains on Python 3.10/Torch 2.11/NumPy 1.26; the separate coordinator's
Python 3.13 CPU environment does not replace that model/search runtime.

### Preserved preparation failure and correction

The [v2 CPU preparation failure](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/forced_opening/actual_cpu_preparation.log)
was a reader eligibility error: registered opening **55** has a legal, valid,
nonterminal 16-ply history with exactly one legal move. The reader incorrectly
required at least two, a condition appropriate to the capacity diagnostic rather
than this match. [PR #564](https://github.com/jjoshua2/DeepFin/pull/564) removed only
that restriction; history, legality, terminal and uniqueness checks remain.
Its [independent review](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/forced_opening/independent_review.json)
and passing local tests/static checks are retained; all PR checks passed.

The [successful v3 preparation](../../scratchpad/bt4_joint20/publication_b100_completion_20260908/arena/qualified_preparation.json)
uses the **byte-identical 500-opening panel**, SHA256
`b4f1d16b488ff9efa0c466706b74595559a807c99a0e4ca75379a4d0745cfb6c`,
including opening 55 and the exact first-128 prefix. No replacement opening,
checkpoint change or new scientific comparison was introduced. Both the failure
and subsequent successful qualification remain archived; this is not a negative
playing-strength observation.


## Historical startup snapshot — September 8, 09:09 UTC

B100 training started at **09:09:09 UTC** (05:09:09 local), after the completed
H20 package and the selected endpoint registration. The
[startup process snapshot](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/training.process.snapshot.json)
records trainer **336571**, timeout supervisor **336570** and coordinator **336548**.
The [startup log prefix](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/training.startup.prefix.log)
confirms all **2,309 shards** were staged and the model was created on CUDA.
That snapshot established startup only; the completed-epoch evidence is recorded above.

The [frozen training manifest](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/B100.training_manifest.json)
uses the merged schema-3 training-only coordinator, commit
`edf646fdfdcb395c57f9a4fca36865149f031622`. The actual trainer remains the qualified
wise-cloud `7ec2615` runtime: Python 3.10.12, Torch 2.11.0+cu128, CUDA 12.8 and
NumPy 1.26.2. Seed zero, batch 512, 16 planning/loading workers, 88-step windows,
optimizer and training command remain matched to H20; only corpus and output paths
change. The [runtime qualification](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/runtime_qualification.json)
and [root launch review](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/root_launch_review.json)
bind actual imports, native hashes, original source/configuration and the new corpus.

The [prospective schedule](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/schedule/B100.prospective_schedule.json.gz)
verified **18,910,484 rows, 36,935 batches and 97,968 games** against the original
source/C witnesses. Its canonical hash remains `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
The CPU-only verification took **483.57 seconds**, with peak RSS **699 MiB**, on
CPUs 6–7 with CUDA hidden. This proves the prospective input schedule; at that startup snapshot, post-training staging, finite windows and realized counters
were still pending. The completed qualification is recorded above.

The [launch consistency check](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/launch_consistency_check.json)
compared recorded child argv/runtime and source pins with the root prelaunch
review and manifest. The earlier [operator receipt](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/operator_launch.json)
correctly retained `training_started:false` when it only knew the coordinator had
started; the later process snapshot supplies trainer-start evidence. Historical
preparation statuses and the [original selected registration](../../scratchpad/bt4_joint20/B100_preparation_v1/training_launch_publication_v1/original_preregistration.md)
are preserved unchanged. The publication author prepared the coordinator and this
consistency check; root supplies separate publication review.

The training-only coordinator dispatches **no arenas**. Its inclusive training cap
is **4.5 hours**, followed by a separately bounded 30-minute CPU schedule check.
The two future arena allowances remain 1.5 hours each, making the selected B100
package cap **7.5 GPU hours**. Their manifests/runtime/readout still require separate
qualification. The separately registered capacity probe is outside that package
cap. That startup record claimed no training completion, promotion or playing result.

The [launch evidence manifest](evidence/bt4-bootstrap/b100-training-launch-manifest.json)
binds these 15 compact artifacts and lossless schedule compression. The full
2,309-shard metadata inventory, corpus summaries and operator source remain external
with explicit hashes. No full corpus replay or training-window outcome read was
part of this publication. B100 remains pure sharpened-BT4 **policy** supervision,
with all SF-derived value targets and other non-policy fields unchanged.

## Question and selection

H20 is the development incumbent, without production promotion. Its registered
package favored H20 over C20T05 at 100 and 400 simulations and over G20T05 at 100.
The aligned higher-search contrast was unresolved. The highest-value next family
question is whether the pure BT4 policy endpoint improves on this hybrid, before
fine-tuning another intermediate mixture.

B100 uses the legal-normalized BT4 policy sharpened at **teacher temperature 0.5**
as the entire policy target. **SF-derived value targets and all other non-policy
fields remain unchanged.** This is not BT4-only value supervision, an unsharpened
teacher or a change to inference prior temperature. The common source remains
the original 18,910,484-row, 2,309-shard, 97,968-game corpus; no G10 transfer is
part of this comparison.

This dated selection supersedes the original *unselected* B100-versus-C fixed-bank
option. The completed H20 package and its original rules remain immutable.
G50 and other alternatives remain available, without an automatic queue.

## Training and controls

Train B100 once from scratch at seed zero, using the qualified one-full-game-epoch
schedule: batch 512, 16 planning/loading workers, 88-step windows, 36,935 batches
and the same optimizer, initialization and numerical semantics as H20. Require
completed finite diagnostics, zero nonfinite skips/CUDA retries and matching
prospective and realized source-normalized row/batch schedules. The canonical
schedule is `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Select the registered final checkpoint, with no intermediate-checkpoint search.

The H20 reference is its already qualified final seed-zero checkpoint, SHA256
`0a711fcf10ff87fc8360d3fd4b3035b170a7c15172616317c4a9687ae99d7017`.
The launch qualification must bind the B100 corpus lineage, policy normalization,
non-policy identity, actual runtime/native/configuration pins, schedules and final
checkpoint before arena execution. Training preserves the historical qualified
Python 3.10/Torch 2.11/CUDA 12.8 semantics; adopting newer arena control flow must
be qualified separately. Existing `valid_control:false` purity, committed-config
and historical-sampler limitations remain.

## Registered match allocation

The prospective allocation follows [PR #547](https://github.com/jjoshua2/DeepFin/pull/547).
It requires the reviewed ordered-prefix implementation and a qualified launcher
and reader before use. Existing fixed-N readers are not sufficient qualification
for a sequential result.

| Comparison | Registered allocation | Decision |
| --- | --- | --- |
| B100 versus H20, 100 simulations | Paired GSPRT: logistic H0=0/H1=+15 Elo, alpha=.05/beta=.10; first look at 128 pairs, then every 64; cap 500 pairs/1,000 games | First crossed boundary favors its separated hypothesis; cap without crossing is inconclusive |
| B100 versus H20, 400 simulations | Fixed 128 pairs/256 games, exactly the first 128 low-budget opening pairs | Report paired score/Elo and nominal 95% interval; retain this probe regardless of the shallow result |

The final 500-pair cap is a declared sequential look even though it falls between
64-pair steps. Only a complete canonical prefix can enter the sequential decision;
every released look is checked in order, and the first crossing is immutable.
Completed speculative suffix games remain banked separately from the deciding
sample. H1 is not a +15 Elo confidence lower bound; H0 is not equivalence. The
GSPRT's nominal error guarantees are asymptotic, and stopped ordinary Elo intervals
are descriptive. An operationally incomplete stage is not a negative result.

Use matched qualified training search shapes, prior temperature **1.0** on both
sides, move temperature 0.1, maximum 300 plies, original development book seed 42 and
16 opening plies. Bind exact full settings, opening histories/endpoints/colors,
execution capacity and implementation hashes before launch. Preserve fresh
confirmation seeds/openings. A separately bounded capacity component measurement
may inform execution capacity; it does not choose policy recipes from playing
outcomes or alter the registered scientific samples. No throughput gain is assumed.

A positive 400 result alone does not show a growing search advantage. If reporting
the secondary interaction, use the same fixed 128-pair core at both budgets and
label it exploratory; use the existing aligned-pair bootstrap convention of 10,000
PCG64 replicates, seed 20260903. Two budgets measure one contrast. Neither extra
games nor a sequential verdict captures training-seed variability. Review the
completed bundle before choosing further refinement or fresh confirmation.

## Resource and execution status

B100 has a **7.5 GPU-hour hard package cap**: training 4.5 hours and each arena 1.5
hours, including termination allowance. A separately registered capacity component
probe has its own **20-minute cap**, before training. The combined maximum is
therefore **7 hours 50 minutes**, not 7.5 hours including the probe. CPU-only metadata
and schedule qualification is outside these GPU allowances; each schedule stage
retains its 30-minute bound and releases the GPU lease.

The registered training is complete, and the separately qualified arena coordinator
has launched as recorded above. Arena outcomes remain unread in this publication.
Preserve shared GPU leasing, surviving timeouts, STOP handling,
owned-process cleanup, durable failure/completion receipts and at least 150 GiB free.
No automatic retries or outcome-dependent extensions. Keep labeling available during
preparation and gaps. Unused allowance does not justify another arm. No production
adoption is implied.
