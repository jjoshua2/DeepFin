# Ceres100 policy endpoint preparation

Registered September 13, 2026. Status: full Ceres100 materialization, corpus qualification and prospective epoch qualification completed; the seed-0 training host launched at 14:53:49 UTC. Completed training and the deciding comparison remain pending.

The missing pure-Ceres policy endpoint is a substantive comparison with the previous equal BT4/Ceres policy mixture. CeresB50 trained on the original 18,910,484 rows with original SF values and scored +12.22 Elo, nominal 95% paired interval [-22.88, +47.57], against B100. That unresolved result does not establish whether the BT4 contribution helps or hurts. No completed pure-Ceres policy student appears in the inspected experiment records. The separate Ceres value mixture retained B100 policy and does not answer this question. See the [completed Ceres record](2026-09-11-ceres-weighted-bootstrap.md).

Ceres100 replaces only `policy_target` with the legal-normalized saved Ceres policy at temperature 0.5. The existing producer implements this endpoint with `--bt4-weight 0 --bt4-temperature 0.5 --ceres-temperature 0.5`; zero is the BT4 weight, not the Ceres weight. It preserves all 16 nonpolicy arrays and original SF-derived values. BT4 sidecar provenance and input checks still run, even though BT4 receives zero target mass. No labels, history conversion, sampling algorithm or producer math change.

The full teacher manifest is `teacher_manifest_assembly_v1/completed_snapshot16_v1/policy_manifest.json` under `scratchpad/bt4_joint20/ceres_teacher_readiness_v1/full_corpus_planning_v1`, SHA256 `2eed622b9e9af1907d7aea36fbeeb22bbb33344b4dcde16e2eac336e7861fada`. It binds 2,309 original source shards and 18,910,484 rows, including the saved collector qualification and legacy provenance. The teacher remains C3-768-30-pre8-I8, fixed32 ORT profile `ceres-c3-fixed32-compact-v2`. Native Ceres numerical parity and native-engine strength remain unestablished. Reusing the exact saved profile permits a same-profile endpoint comparison; it does not erase that limitation.

The new `Ceres100` supervisor and recipe admission require weights `{bt4: 0, ceres: 1}` and reject relabeling the half mixture as the endpoint. Existing CeresB50 and value-profile behavior retain their defaults. The output is a fresh `qtemp_0.0005_hist_20m_ceres_policy_C100T05` corpus. Full source/feed/teacher alignment, all nonpolicy copy proofs, output publication and later metadata qualification remain required. Synthetic endpoint tests exercise real target writing and consumer loss, so no additional real pilot is allocated merely to repeat those checks.

The proposed **preparation-only allocation is seven hours inclusive** (25,200 seconds), two numeric threads on CPUs 6–7, GPU hidden, at least 32 GiB Linux available memory and 150 GiB free disk, and a 64 GiB sampled output limit. The existing batch size remains 128. These headroom and output guards do not constitute an RSS cap; no CUDA or model load is part of preparation. The measured comparable B50 materialization took 5.57 hours. Seven hours allows some margin, but concurrent I/O and run-to-run timing remain uncertain. An incomplete job preserves its partial output and failure evidence; there is no automatic extension or resume. The earlier provisional four-hour estimate was corrected before publication or launch.

Full training and evaluation allocation comes after the current matched 35M value results and remains a parent decision. This preparation does not queue a new GPU job, select a training seed or opening panel, or allocate extra games. A future scientific comparison must distinguish the endpoint question from seed variability and retain common SF values and matched search settings. Active training and source generation remain untouched.

Validation: the real endpoint fixture preserves nonpolicy bytes and SF value loss while changing policy gradients; endpoint probabilities ignore the BT4 allocation. Metadata tests reject swapping endpoint/half-mixture profiles and source identities. The existing qualifier fixtures cover the new profile, and supervisor tests cover zero-weight argv, low-memory refusal and the seven-hour cap. The 80-case focused run had one legacy error-message mismatch; the corrected targeted check passed, as did the additional allocation-cap test. Scoped types and whole Ruff/basedpyright/Vulture passed. Test imports reused existing native binaries only after their C/header sources matched the qualified test worktree; no native rebuild or live runtime change occurred.

## Preserved initial import failure and corrected runtime preparation

The first actual preparation attempt failed after 2.363 seconds at 06:09:37 UTC, before manifest/payload admission, because the isolated operating runtime lacked `_lc0_ext`. Its original terminal and log remain under `scratchpad/bt4_joint20/ceres100_policy_preparation_v1`; this was not a corpus-content failure and produced no completed endpoint target.

The corrected preparation adds four existing native-module symlinks to that isolated runtime. All corresponding C/header sources match both the qualified original materializer and the resolved binary-owner worktrees. Actual import resolution passed once in 2.032 seconds, importing the real producer and admission modules without a corpus admission, payload read or model load. Native binary hashes are now explicit input pins. No producer bytes, target mathematics, environment packages or active runtime were changed or rebuilt.

The fresh v2 plan is `ceres100_policy_preparation_v2/plan.json`, SHA256 `1ccd6f8d5d224858c0cfbfd994b27f2ab5a6ae2561b8328f5f6cf7a9cb9f700a`. Its prepared command has a 25,197-second remaining inclusive allocation (25,167 seconds to TERM plus 30 seconds to KILL); three seconds are charged against the original seven-hour allocation for the first attempt. The separate import-only preparation check is recorded as validation. Source, teacher manifest, batch128, CPU6,7, memory/disk/output guards and owned cleanup remain unchanged. After independent v2 review PASS `c34ff502…`, the parent launched the corrected preparation at 06:13:33.883234 UTC (exec77576, sole observer391). This is the actual command-launch snapshot, not a completed producer or qualified corpus. [Compact evidence](evidence/ceres100-preparation-launch-20260913.json) preserves the original failure, successful import-only validation, fresh plan and actual launch pins. No training or evaluation is allocated.

## Full endpoint corpus completed and qualified

The corrected producer completed all **18,910,484 rows in 2,309 shards** at 09:08:37 UTC on September 13, after 10,503.777 seconds (2 hours 55 minutes). The existing metadata qualifier then passed at 09:33:32 UTC: 264.061 seconds for the enclosing command and 261.718 seconds inside the qualifier. Its status is `PASS_REGISTERED_CORPUS_QUALIFICATION`. It checked actual layouts, recipe attributes and stable producer-bound storage identities, inheriting the completed producer's payload/feed/copy checks without decoding the corpus again. [Compact completed evidence](evidence/ceres100-completed-preparation-20260913.json) retains all aggregate producer metrics, complete source/teacher/recipe pins and actual terminal/qualification/review references.

Only `policy_target` changed. All 16 nonpolicy arrays remain unchanged, including original SF-derived `search_wdl` values. Weights are exactly BT4=0/Ceres=1, both recorded teacher temperatures are 0.5, and the original saved teacher profile and provenance remain fixed.

| Stored-policy diagnostic | Full-corpus result |
| --- | ---: |
| Maximum stored mass error | 0.00042724609375 |
| Maximum stored total variation | 0.0002075328771593776 |
| Positive legal move entries rounded to zero in float16 | 1,495 |

The support-loss count is **move entries, not rows**; the summary does not supply an affected-row count. The stored-error bounds passed the registered qualifier. No rounding clamp or target repair was introduced.

The commands reported maximum RSS of 994,156 KiB for materialization and 383,120 KiB for qualification. These are observed command resource measurements, not host-wide peaks or future guarantees. The 2-hour-55-minute materialization is faster than the earlier 5.57-hour B50 observation, but this was not a controlled throughput benchmark: cache state and concurrent I/O differ. The runtime retained the original batch128 producer bytes and did **not** adopt PR707. The initial import failure, subsequent qualified native resolution and original launch snapshots above remain preserved.

Independent completed review passed. This milestone supplies a qualified policy corpus, not a completed prospective training schedule, trained endpoint or playing result. Training/evaluation allocation still follows the current matched 35M value results; no new GPU job is queued by this completion.


## Prospective low-memory training admission

The explicit Ceres100 schema3 training-only profile now requests two planner and two loader workers, and completed-match admission requires that same 2/2 setting plus the pinned complete pure-Ceres recipe and original SF-value lineage. The reader reuses the existing full Ceres recipe checks, including teachers, source, nonpolicy preservation, producer pins and completed shard proofs. Other profiles retain their prior settings: B100 and CeresB50 remain 16/16; the existing Downside 2/2 exception is unchanged. The frozen trainer, sampler, seed 0, batch 512 and one-epoch horizon are unchanged. This amendment allocates no training or evaluation; an actual prospective schedule, final manifest and reviewed host command remain prerequisites after a scientific allocation decision.


## Registered endpoint training and deciding comparison

The parent selected pure Ceres100 as the next substantive policy endpoint before inspecting the combined35M value score. This decision retains original SF values and does not depend on that comparison’s outcome. The exact frozen registration below is also retained at `ceres100_policy_preparation_v2/training_readiness_v1/registered_epoch_v1/preregistration.md`, SHA256 `e6169825d07af0d622cabccc68535928b9c5ac6645a31634029c1974161a3c8f`. The original prospective verifier has been launched under its separate bounded CPU allocation; its completed report and training launch remain pending at this publication snapshot. No GPU training has been launched by this preparation.

### Frozen registration

Registered before any Ceres100 training or evaluation. Parent selected this substantive policy endpoint independently of the combined35M value result; that bank has not been inspected for a score at registration. No new labels or corpus rewrite.

Hypothesis: pure Ceres policy T0.5, retaining original SF values, improves on the existing B100 T0.5 policy endpoint. The deciding comparison is Ceres100 versus the completed seed0 B100 checkpoint b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62. No automatic CeresB50 comparison, fine-tuning or dose grid.

Use the qualified original18,910,484-row/2309-shard Ceres100 corpus (qualification172b060b..., derive6844ed43..., rewriteb640508c...). Only policy_target changes; all16 nonpolicy arrays, including original SF search_wdl values, are retained. Both teacher temperatures0.5, BT4weight0/Ceresweight1. Saved Ceres approximate fixed32 ORT/CUDA profile is unchanged; native-Ceres parity is not claimed.1495 positive legal move entries lost through float16 storage are entries, not affected rows.

Train a fresh model with frozen wise-cloud trainer/config/sampler and seed0 (both torch initialization and game schedule), batch512, one actual epoch:18,910,484 rows,36,935 updates,97,968 games,420 windows (88 updates each, final63), warmup1000. Existing B100 used16/16 workers; the explicit Ceres100 amendment543691257 uses2/2 as an execution difference. Require actual prospective and completed canonical schedule dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f and actual planned=realized schedule; do not merely assume worker equivalence.

Existing coordinator schema3 fixes training allowance16,200seconds; retain enclosing21,630seconds inclusive,48GiB available memory before startup/32GiB while running,150GiB free disk reserve and owned STOP/cleanup. No CUDA virtual-address cap. This is a maximum, not a2h guarantee: Downside2/2 took7832s, CeresB50 took11055s, while different concurrent load can change cost. No automatic deadline extension or restart.

Before training, one original prospective verifier invocation is proposed under1800seconds inclusive (1770TERM+30KILL),CPU6,7,hiddenGPU and two numeric threads (original verifier internally usesTorch2/Blosc1). It reads identity/presence columns and metadata, not features/policies/models. No repeat corpus qualification. No small RLIMIT_AS that prevents CPU Torch mapping. Existing STOP,150GiB disk and32MiB sampled output guard retained; parent checks host memory before launch. Parent approved this CPU allocation after reviewing the concrete plan and independent static review 4240399d on September 13, before execution.

After genuine completion and CPU package admission, one fixed comparison:400simulations,256swapped pairs/512games,priors1,noTB,maxplies300,rolling128/batch4096. Reuse immutable panel14470ee9bcf5fdfc822bb19988941ecf4bbe2a1ea739d46487d3663b1735340c and seed20260913 explicitly as a DEVELOPMENT bank now also used for the35M value comparison, not fresh confirmation. No prior Ceres outcome was selected on this bank. Arena allowance7200seconds stage/12030seconds enclosing, matching existing512-game host; arena adaptation/final bindings remain a later reviewed task.

Deciding statistic: candidate score and nominal95% interval clustered over the256 swapped opening pairs. Entire interval above0.5 supports this endpoint; entire interval below0.5 favors B100; crossing0.5 is unresolved. No automatic extra games, seed replication, calibration or promotion. The reused bank and shared development seed constrain generalization.

Retain all three historical valid_control=false caveats: no held-out purity receipt, committed rather than fresh live config premises, and intentional game_epoch sampling rather than the historical replacement sampler. No training launch is authorized by a draft manifest alone. Actual prospective PASS and final manifest/host review are still required.


## Prospective qualification complete; epoch host launched

The original prospective verifier passed in **289.288 seconds**. Report `ee6c9cff…` and independent review `1208d708…` bind the complete original 18,910,484-row/2,309-shard Ceres100 corpus to the fixed seed-0 canonical schedule `dc687fc3…`, with 36,935 planned updates. This is prospective qualification; actual staging and realized training still require the completed-epoch check.

One existing default command validation returned exit 0 and printed the exact Ceres100 source and fresh output paths, seed 0, batch 512, 88-update windows and two planner/two loader workers. It allocated no GPU work and queued no comparison. Its receipt does not contain an elapsed duration, so none is inferred here. Final schema3 manifest `c680aeca…`, host plan `f1e0f7f2…` and independent final review `ad1ff287…` retain the immutable registration `e6169825…` and frozen coordinator `543691257…`; the original wise-cloud trainer and sampler remain unchanged.

The parent launched the actual host command at **2026-09-13 14:53:49.289165 UTC** (exec 94358, sole observer 480). The saved `started.json` records the requested coordinator command and owned process group. This publication claims a host launch, not an observed optimizer step or completed epoch. The startup guard measured 89,145,061,376 available bytes; parent prelaunch checks reported approximately 83 GiB available memory, zero swap use, 192 GiB free disk and a clear GPU. These are launch observations, not peak or current resource claims. The 48 GiB startup/32 GiB ongoing memory floors, 150 GiB disk reserve, 16,200-second training cap and 21,630-second enclosing allowance remain fixed.

[Compact qualification/default/launch evidence](evidence/ceres100-epoch-launch-20260913.json) retains the actual receipts, final bindings and both independent reviews. The separately registered 512-game/400-simulation comparison against B100 remains unlaunched, with no automatic B50 comparison or extension. Reused development-panel status, approximate-Ceres provenance and all three historical control caveats remain unchanged. No model, active training progress, corpus payload or game bank was read for this publication.


## Explicit fixed-arena endpoint support

The existing fixed512 host/probe/package reader now admits one additional explicit profile: Ceres100 candidate versus B100 reference. It checks genuine original-epoch receipts with the existing matched-training reader (including Ceres1002/2 endpoint recipe/SF-value proof and historical B10016/16), not the combined35M receipt schema. Original receipt input pins and realized schedule are retained in the host’s stability observations. The combined35M branch, owned capture/cleanup and resource guards are unchanged. Both branches retain literal512 games/400 simulations/seed20260913/priors1 and the registered256-opening panel. The Ceres profile labels the reused development panel honestly. Actual Ceres100 completion/checkpoint pins and CPU package remain pending; this code change launches no job.
