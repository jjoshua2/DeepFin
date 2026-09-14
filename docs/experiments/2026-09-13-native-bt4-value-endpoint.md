# Native BT4 value endpoint on the matched 35M corpus

Selected September 13, 2026, before reading the running Ceres100 policy match.
Status: the bounded V100 CPU rewrite launched at 17:38:46 UTC. Explicit qualification/training support is implemented; actual completed targets and dataset qualification remain pending. No V100 training run or match has launched.

## Question and control

The [completed 35M comparison](2026-09-13-combined35m-value-transfer.md) favored
50% SF / 50% native BT4 WDL over SF-only WDL by **31.30 Elo [8.53, 54.34]**.
That establishes neither the best mixture nor whether SF improves on BT4-only
value supervision. Test this missing endpoint before refining small mixture doses.

Candidate **Combined35M_V100** changes only `search_wdl` to normalized native
BT4 side-to-move W/D/L probabilities, stored as float16. Use the existing writer's
`--alpha 1`; do not softmax probabilities again or derive them from policy logits.
The direct control is completed **Combined35M_V50**, checkpoint
`6d36f93d040c8babed040159419279af62a444f6466f42b720e12bb72d67ab02`.

Both use the identical frozen **35,314,577 observations / 4,321 shards / 21
sources**: 18,910,484 original observations and 16,404,093 selected G10 observations.
These are observations, not unique FENs. Native BT4 labels already cover this
selection; no new teacher inference or deeper SF search is required.

Retain B100 policy targets at teacher temperature 0.5 and every other stored
array, seed101 initialization, original frozen trainer/config/optimizer, one
game epoch, batch512 and two planner/two loader workers. The expected schedule
is **68,974 updates / 784 windows**, final window70, with canonical identity
`ca3922c459b321dd5e890f8b3e21f6fee6a24aa6c7f0d76653165616aeca67bd`.
Actual completion must establish that schedule rather than inherit its status
from the control. New host launches explicitly cap compiler concurrency at two;
that resource setting does not claim measured memory savings or bitwise equality.

## Head semantics

Both completed 35M controls report WDL target shares search1/SF0/outcome0.
Their categorical target is absent, categorical rebuilding is inert, and
categorical and SF-eval losses were zero in all 784 windows. The SF contribution
being tested is inside stored `search_wdl`, not the trainer's separate SF fraction.
V100 therefore means BT4-only **supervised WDL targets** under this configuration.
The coupled categorical output and shared policy/value features can still change
indirectly. This does not independently supervise every value-related output to
BT4. Preserve the same masks and objective; adding a categorical target or changing
auxiliary weights would be a different experiment.

## Preparation and compute limits

Use the original B100/SF-value input roots and their authenticated native-label
mappings. Do not feed already mixed V50 values back through a writer expecting
the original source. Extend existing admission and coordinator checks explicitly
for V100; a V100 corpus must never masquerade as a V50 receipt.

The supported writer copies complete shards. Account for actual source storage,
output growth and concurrent writers before launch; a value-only overlay is not
an admitted shortcut. The completed metadata census covers these exact 21 roots:
**26.245 GiB allocated**. Allow **32 GiB total output**, plus **8 GiB for other
writers** and the **150-GiB reserve**: require **190 GiB free at launch**. The
accounting snapshot had 201.812 GiB free; refresh before launch. These are planning
allowances, not a compressor bound. Enforce aggregate output and free-space limits
during the operation, and preserve protected controls.

[Compact readiness evidence](evidence/native100-readiness-20260913.json) records
the actual head masks/losses, exact-root metadata census and disk-budget derivation.
The metadata check took 36.114 seconds and read no array payloads; it is not a
dataset qualification or source-content verification.

- Full preparation: at most **4 hours** for sequential rewrites, with low-priority
  two-CPU execution and bounded memory. Completed comparable G10 work extrapolates
  to about 2.1 hours for 35M; this is a planning estimate, not a measured runtime.
- Dataset qualification and prospective schedule: at most **1,800 seconds each**.
- Training: **21,600 seconds**; coordinator **27,000 seconds**, plus at most
  30 seconds of outer termination allowance. One actual epoch, no automatic retry.
- CPU match preparation: **600 seconds inclusive**. Match: **7,200-second stage**
  within **12,030 seconds inclusive** for the enclosing operator.

Keep 48-GiB startup and 32-GiB running available-memory floors, owned-process
cleanup and STOP handling. Exact source/output/implementation pins and concrete
commands must be saved before each launch. Preserve incomplete outputs and
receipts for diagnosis; a failure does not authorize an automatic rerun or a
different cohort. The separate Ceres policy match continues unchanged.

## Deciding comparison

After valid completion, run **V100 versus V50, 512 games / 256 swapped pairs,
400 simulations, priors1, seed20260913**, with the same qualified search runtime,
16-ply openings, maximum300 plies, no tablebases, rolling128 and batch4096.
Reuse development panel
`14470ee9bcf5fdfc822bb19988941ecf4bbe2a1ea739d46487d3663b1735340c`.
It is already used for recipe development, not fresh confirmation. Keep the
existing paired-score estimator and its nominal 95% interval; read the completed
bank, not a rolling score.

A complete paired score interval wholly above0.5 supports V100 in this setting;
wholly below0.5 supports retaining the SF contribution in V50. If it crosses0.5,
the result is unresolved and V50 remains the provisional value recipe. No extra
games, mixture sweep, training seed or promotion follows automatically.

Retain the three historical validity limitations: no held-out purity receipt,
committed rather than current live-config premises, and game-epoch sampling
distinct from the historical replacement control. This one-seed comparison
does not settle 100M transfer, RL behavior, other teacher mixtures, draw-versus-Q
calibration, or value/policy interactions with a different policy teacher.


## Actual bounded CPU rewrite launch

The parent launched the rewrite at **2026-09-13 17:38:46.483384 UTC** (exec61857 / sole observer597), bound to plan `bbabe006…` and frozen preregistration SHA-256 `3ca5d62d491a0064b0950ccdd6d021f9a04a88bf4189fb150d298f6c85fbd0de`. This is an actual outer-command launch, not completed target production or dataset qualification. Startup measured **84,433,498,112 bytes available RAM** and **216,670,490,624 bytes free disk**; these are launch snapshots, not current measurements.

The fixed plan rewrites exactly **21 B100 input roots / 35,314,577 rows / 4,321 shards** into fresh outputs. The original 18.91M stage retains its qualified `5afc1e2f…` writer/runtime and batch 256; the 20 G10 stages retain `c2ce999d…`, batch 1024 and the accepted native-WDL bank mappings, including the four-output run06-large mapping. Both unchanged writers already support alpha 1. Only `search_wdl` changes; B100 policy, all other arrays, source identity checks and observation selectors remain fixed. No new teacher evaluation, source selection or live runtime modification occurs.

The existing sequential operator retains one child at a time, CPUs 6–7 / two numeric threads, hidden GPU and low CPU/I/O priority. The supervisor uses a 12-GiB virtual-address limit; the original stage retains its 12-GiB limit and G10 children retain 4 GiB. These address-space limits are not claimed physical-memory consumption. Startup requires 48 GiB available RAM; ongoing checks require 32 GiB. The whole operation has one **14,400-second absolute deadline**, including imports, preflight, producers, final metadata and cleanup: outer TERM at 14,370 seconds plus 30 seconds to KILL. The original stage is allocated 6,500 seconds; the 20 G10 allowances total 6,680 seconds, followed by a 300-second final reserve. Insufficient remaining budget yields a preserved completed prefix, not an all 21 completion or automatic extension.

One metadata-only accounting pass measured **26.245 GiB allocated across the exact inputs**. The plan reserves **190 GiB free at startup: 150 GiB retained floor + 32 GiB aggregate output allowance + 8 GiB for other writers**. The 32-GiB output check counts allocated blocks across all completed outputs, active `.writing` output, records and directories. It runs periodically at 300 seconds and after stages/finalization; it is a sampled threshold, **not a filesystem quota**. Frequent disk checks retain the 150-GiB floor. The other-writer allowance is reserved capacity, not a guarantee about future write rates.

Parent review caught two preparation defects before any producer attempt: an incorrect 2,180-shard count in the preflight/completion predicates, and an output-accounting race when a writer atomically renames `.writing` to its final path. Both were corrected before launch. Focused fixtures execute the actual predicates (accept 4,321 / reject 2,180) and accounting helper; the latter retries only disappearance, at most three times within the same deadline, and requires a complete scan while other errors remain fatal. Independent preparation review `d7b2dd21…` passed the corrected frozen operator and bindings. There was no invalid producer launch to discard.

The separate V100 source extension was merged in [PR #722](https://github.com/jjoshua2/DeepFin/pull/722), with independent full-source review. That supplies explicit alpha 1 lineage, historical V50 verifier and future training/match support; it does not qualify outputs that have not completed. Actual rewrite completion, V100 corpus/subset/prospective qualification and final training bindings remain required. Failure preserves partial outputs and evidence without automatic retry. The currently selected value recipe remains V50 until the registered completed comparison supplies evidence to change it.

[Compact plan, disk accounting, review and actual-launch evidence](evidence/native-v100-rewrite-launched-20260913.json) retains every stage's exact argv/native binding and hashes the full local plan/layouts. This publication read only immutable launch/preparation records; it did not poll active jobs, scan corpus payloads, rerun source admission or launch training.

## Actual V100 rewrite completed; prospective schedule launched

The CPU rewrite completed with **exit 0 in 7705.348 seconds** (2 hours 8 minutes
25 seconds), within the four-hour allocation. All **21 ordered cohorts /
35,314,577 rows / 4321 shards** completed. The actual alpha-one recipe replaces
`search_wdl` with native BT4 WDL probabilities (SF weight zero), retaining B100
policy and the 16 other arrays under the frozen producer's source/content proofs.

All 35,314,577 rows were reported changed. Maximum stored probability-mass error
was **0.0003662109375**, reflecting stored precision rather than an exact-unit-sum
claim. The largest child RSS in saved per-stage resource receipts was
**401,868 KiB**; this is a child measurement, not aggregate host peak or proof of
a memory improvement. Independent compact review reconciled all 21 completed
stages and 42 rewrite/derive-summary pins, including the copied B100 policy
summaries. It did not repeat payload checks.

At **19:55:27.042 UTC**, the parent launched the prospective ordered-corpus
schedule against those actual completed outputs. Manifest
`446fc6041d25d0efa1a51ad9ba994572c65536e70fa0275198d85260cafae1f2`
binds the exact 21 roots. The CPU-only command uses CPUs 4,5, two numeric threads,
nice 19, a shared 1740-second deadline and an **1800-second outer allocation**.
It requires 48 GiB available RAM and 150 GiB disk at startup, with the existing
32-GiB callback guard and no address-space cap. Startup recorded
88,348,295,168 bytes available RAM and 184,630,718,464 bytes free disk.

[Compact completion and launch evidence](evidence/v100-rewrite-completed-20260913.json)
retains actual terminal/start records and independent completed-rewrite and
final-command reviews. Rewrite completion does not establish a completed
prospective schedule, selected-subset compatibility, training or an arena result.
The frozen original preregistration remains unchanged; the registered V100
versus V50 question is still unresolved.

## Prospective schedule and compatibility completed; V100 training launched

The prospective schedule completed with exit 0 in **371.488 seconds**. All
21 ordered cohorts / 4321 shards contain **35,314,577 rows and 182,188 games**.
The seed-101, batch-512 plan expects **68,974 updates**. Source, B100 and V100
canonical schedules match the prior V50 controls at
`ca3922c459b321dd5e890f8b3e21f6fee6a24aa6c7f0d76653165616aeca67bd`;
the V100 path-specific physical plan is
`4b47ce5ecb8fd272d8b93455107d8926dc485cd42915d707ba0e2beb3dc0f007`.
These are prospective schedules, not realized training receipts.

The subsequent selected-subset compatibility qualification completed with exit 0
in **48.056 seconds**. All 2012 G10 shards marked as globally partial sources
are selected finalized shards; the original 2309 shards are not partial. The
registered `allow_partial_corpus` exception applies to this exact completed
selection. Leak and mixed-history allowances remain false. `search_wdl` covers
all 35,314,577 rows; the unused `sf_wdl` label/value-blend field has zero coverage
as registered. Historical architecture/replay/trainer caveats remain recorded,
and the actual trainer still executes its own compatibility gates.

The parent launched the host at **20:17:18.936 UTC** and the owned training stage
started at **20:17:23.691 UTC**, PID 501378 under timeout supervisor 501377.
The actual startup receipt records the qualified `wise-cloud` runtime with
Python 3.10.12, NumPy 1.26.2, PyTorch 2.11.0+cu128 and CUDA 12.8. Its actual
command selects the 21 V100 roots, seed 101, batch 512, game-epoch sampling,
**two planning and two loading workers**. Compiler workers are capped at two.

The training stage has a **21600-second** ceiling; coordinator and outer bounds
are 27000 and 27030 seconds. Existing GPU ownership, cleanup, 48-GiB startup /
32-GiB running headroom and 150-GiB disk reserve remain. Startup recorded no
GPU compute applications, 89,001,152,512 bytes available RAM and
184,231,911,424 bytes free disk. These are launch samples, not measured peaks.
The single metadata/argv default check exited 0 before launch; parent and
independent final host reviews passed.

[Completed prerequisites and actual launch evidence](evidence/v100-training-launched-20260913.json)
preserve exact report, qualification, host-manifest and review pins. At that launch
snapshot, training and the registered arena were still pending. Their subsequent
completion and launch are recorded below.


## September 14 UTC: V100 training completed; fixed arena launched

V100 seed-101 training completed with **35,314,577 realized rows and 68,974
updates**, using the registered two planning and two loading workers. The
completed receipt verifies actual staging and game-column identity, with realized
physical schedule `4b47ce5e…` and common canonical schedule `ca3922c4…`.
Training charged **14,316.172 seconds**; the host exited 0 at **00:17:14.085 UTC**
after **14,395.149 seconds** including coordination and completion checks.
The checkpoint SHA-256 is `9bf9ceffd4f93fe36eba3f525d69c1a8756e97e4f332973594c8c4eff1a7c836`.

The actual CPU package preparation exited 0 in **68.199 seconds**. It verified
the V100 checkpoint against the historical V50 checkpoint `6d36f93d…`, matched
model architecture, and the full ordered 256-opening history panel. CUDA remained
uninitialized during that probe. Independent review accepted the actual prepared
receipt `9119cbcb…` and its binding to the unchanged final manifest.

The parent launched the fixed arena at **00:30:37.371 UTC**: **512 games / 256
swapped pairs, 400 simulations, seed 20260913, and both prior temperatures 1.0**.
It retains the reused development panel, rolling 128 games, evaluation batch 4096,
300-ply maximum and no tablebases. CPU cores 2 and 3 and two compiler workers
are explicit; the owned arena ceiling is 7,200 seconds within the 12,030-second
inclusive outer allocation. An earlier parent check guessed the wrong prepared
receipt path and failed before invocation; it did not launch or repeat a match.

This is a completed-training and actual-launch record, **not a match result**.
The three historical `valid_control=false` caveats remain: no held-out purity
receipt, committed rather than current live configuration premises, and intentional
game-epoch sampling instead of the historical replacement sampler. The frozen
runtime and preregistration remain unchanged. No value winner, 100M transfer,
independent replication or RL deployment is established by this launch.

[Compact completion and launch evidence](evidence/v100-trained-arena-launched-20260914.json)
retains actual terminal/start receipts, schedule and checkpoint identities, CPU
preparation and independent review. No model, book, training, or CPU preparation
was repeated to publish this record.
