# Ceres100 policy endpoint preparation

Registered September 13, 2026. Status: completed Ceres100 versus B100 scored +2.04 Elo [−22.16, +26.26] at 400 simulations. The registered result is unresolved; retain B100 policy.

The missing pure-Ceres policy endpoint is a substantive comparison with the previous equal BT4/Ceres policy mixture. CeresB50 trained on the original 18,910,484 rows with original SF values and scored +12.22 Elo, nominal 95% paired interval [-22.88, +47.57], against B100. That unresolved result does not establish whether the BT4 contribution helps or hurts. At preregistration, no completed pure-Ceres policy student appeared in the inspected experiment records. The separate Ceres value mixture retained B100 policy and does not answer this question. See the [completed Ceres record](2026-09-11-ceres-weighted-bootstrap.md).

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

The existing fixed512 host/probe/package reader now admits one additional explicit profile: Ceres100 candidate versus B100 reference. It checks genuine original-epoch receipts with the existing matched-training reader (including Ceres1002/2 endpoint recipe/SF-value proof and historical B10016/16), not the combined35M receipt schema. Original receipt input pins and realized schedule are retained in the host’s stability observations. The combined35M branch, owned capture/cleanup and resource guards are unchanged. Both branches retain literal512 games/400 simulations/seed20260913/priors1 and the registered256-opening panel. The Ceres profile labels the reused development panel honestly. At that source-support snapshot, actual Ceres100 completion/checkpoint pins and the CPU package remained pending; the code change itself launched no job.


## Endpoint training completed

The actual Ceres100 training host returned **exit 0**, ending at **2026-09-13 17:04:04.307632 UTC** (exec94358 / sole observer480). The enclosing host lasted **7,815.018 seconds**; owned training consumed **7,657.046 seconds**. The realized-schedule verifier reported **152.768 seconds internally**, versus **155.040 seconds** for its owned process stage. These are distinct measured intervals, not throughput or peak-memory comparisons.

Completed receipt `830c73be…` binds checkpoint SHA-256 `8b8fea7b9ae0fba2a836d724a75631873f32d624f015fa28c067b1b19709bd85`. The fresh seed-0 model completed **18,910,484 rows / 2,309 shards / 36,935 updates**, batch512, in **420 windows** with **63 final-window updates**, using two planner and two loader workers. Actual staging and completion matched the prospective canonical schedule `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`; the actual physical plan is `7f4c18df63dd3e11f60fcd6b8aa5a6e766de2b4cd9e57715500a75c29526ec51`. Independent completed-epoch review passed without rereading model or corpus payloads.

The completed recipe remains pure saved Ceres policy T0.5 with all16 nonpolicy arrays and original SF-derived values retained. The C3-768-30-pre8-I8 fixed32 ORT profile remains approximate: this result establishes neither native-Ceres numerical parity nor playing strength. All three historical `valid_control=false` caveats remain: no held-out purity receipt, committed rather than current live configuration premises, and intentional game-epoch sampling instead of the historical replacement sampler. The registered endpoint comparison was selected before inspecting the combined35M value score and continues to isolate policy by retaining SF values.


## Qualified CPU package and actual deciding-arena launch

The real CPU package preparation returned **exit 0 in 67.250 seconds**, including a **64.319-second owned probe stage**. Prepared receipt `d69405e1…` binds the completed Ceres100 checkpoint above and the existing B100 checkpoint `b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62`. The observed CPU probe confirmed matching **61,444,448-parameter** architectures, regenerated all **256 panel histories**, and did not initialize CUDA. Independent actual-package review `9c163bb4…` passed. Neither preparation nor this publication produces a playing result.

The parent launched the actual arena host at **2026-09-13 17:10:20.046924 UTC** (exec77013 / sole observer550), using frozen runtime `/tmp/deepfin-ceres100-fixed-arena` at `50db3e0…`, final manifest `df9945d1…` and the genuine prepared receipt. The immutable outer-start receipt binds the actual argv and absolute deadline. This is an actual host-launch record, not a completed game count or strength claim.

The requested match remains **Ceres100 versus B100, 400 simulations, 512 games / 256 swapped pairs, seed20260913, priors1, no tablebases, maxplies300, rolling128 and batch4096**. Panel `14470ee9…` is the reused development bank also used for the combined35M value comparison; it is not fresh confirmation. The original rule remains: a complete paired score interval above0.5 supports Ceres100, below0.5 favors B100, and crossing0.5 is unresolved. No extra games, automatic B50 match or promotion is implied.

The launch explicitly sets `TORCHINDUCTOR_COMPILE_THREADS=2` alongside two numeric threads. This bounds requested compiler concurrency; it is **not measured memory savings**. Existing 48-GiB startup/32-GiB ongoing available-memory floors, 150-GiB disk reserve, owned cleanup, STOP handling, 7,200-second arena stage and 12,030-second enclosing allowance remain. The outer command retains its registered 12,000-second application deadline plus 30-second termination margin. Training and arena run sequentially.

[Compact completed-training, CPU-package and actual-launch evidence](evidence/ceres100-trained-arena-launched-20260913.json) includes the genuine receipts and independent reviews, with selected realized-schedule fields and identities for the full saved metadata. No model, corpus payload, opening book or active game bank was reread for publication, and no active job was polled. Approximate teacher provenance and all three historical control caveats remain unchanged.


## Completed deciding comparison: unresolved

The full registered **512-game / 256-swapped-pair** comparison finished with Ceres100 scoring **50.29297%**, or **+2.04 Elo with nominal 95% paired interval [−22.16, +26.26]**, against B100. Both used 400 simulations and prior temperature 1.0. The interval crosses equality, so the preregistered hypothesis is **unresolved** and **B100 remains the policy reference**. This does not establish equivalence, reject the Ceres family, or trigger extra games, a dose sweep or promotion.

| Pentanomial category | Pairs |
| --- | ---: |
| Two candidate wins | 29 |
| Candidate win and draw | 52 |
| Two draws or split wins | 95 |
| Candidate loss and draw | 53 |
| Two candidate losses | 27 |

The middle category includes both double draws and split wins. All 256 pair scores are retained in the [compact completed evidence](evidence/ceres100-policy-result-20260913.json). The prior CeresB50 policy mixture scored +12.22 Elo [−22.88, +47.57] against B100 and was also unresolved. Comparing those two point estimates does not establish that B50 beats Ceres100; no direct B50 match follows automatically.

The owned GPU stage completed successfully in **2,031.458 seconds**. The enclosing host returned **exit 0 in 2,034.914 seconds**, ending at **2026-09-13 17:44:14.960426 UTC** (exec77013 / sole observer550). Its completed process record reports `process_complete=true` and an actual observed arena command matching the qualified CPU package. No missing-command recovery or rerun was needed. The generic reader's `launch_qualification_verified=false` remains explicit: the separate CPU-package review, immutable parent launch and owned process/runtime/cwd/prepared bindings supply the external launch evidence rather than changing that flag.

Independent completed review `73c06e19…` checked the complete small bank once: its content hash, all 512 unique pair halves, panel opening FENs, colors, seeds, settings, results and every published pair score. Independent arithmetic reproduced the score, standard error, Elo and pentanomial counts. The already completed full reader checked checkpoint contents and original-epoch training lineage; those model checks were not repeated during review. The actual command retained seed 20260913, the reused development panel, rolling 128 / batch 4096, maximum 300 plies, no tablebases and the original search shape. Compiler concurrency was explicitly set to 2; this is neither a bitwise-equivalence nor measured memory-savings claim.

The scope remains one original 18.91M-row training seed and a reused development bank. Native-Ceres numerical parity is unestablished for the approximate C3 fixed32 profile. All three historical `valid_control=false` caveats remain: no held-out purity receipt, committed rather than current live-config premises, and intentional game-epoch rather than historical replacement sampling. This comparison does not settle 100M transfer, independent replication or future RL strength. The separately selected native-BT4 value endpoint addresses a different question and preserves B100 policy.
