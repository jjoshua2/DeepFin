# G50 versus B100: original-epoch policy-dose comparison

Prospective registration draft, 2026-09-08. No G50 training or match has launched. B100 is the current development leader after its completed comparison with H20. This next comparison asks whether retaining a 50% Stockfish policy contribution improves on the pure BT4 policy endpoint while preserving the same SF value supervision.

G50 is already materialized: `data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_global_G50T05`, 18,910,484 rows in 2,309 shards. Its policy is the legal-normalized global arithmetic mixture of 50% stored SF policy and 50% BT4 at temperature0.5; B100 uses 100% of that same BT4 component. Both preserve the original value/history/nonpolicy targets. Reuse the [completed materialization evidence](2026-09-07-bt4-hybrid-endpoints.md), rather than repeat the cached teacher audit or rewrite the corpus.

## Training and matched baseline

Train G50 from fresh initialization with seed0, matching B100's seed and original epoch protocol. Fresh initialization does not mean an independent-seed confirmation. Use the qualified original trainer/config/sampler and Python3.10/NumPy1.26.2 runtime, batch512, game-aware sampling without replacement,16 plan/load workers,88-step windows,36,935 updates and420 finite windows. Require zero skipped/retried batches and the original canonical schedule `dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`. Preserve `--allow-invalid-control` and its historical caveats.

The new training run is `runs/armB/qtemp_0.0005_hist_20m_bt4_global_G50T05_epoch_v1`; reference is the completed B100 checkpoint `runs/armB/qtemp_0.0005_hist_20m_bt4_global_B100T05_epoch_v1/checkpoint.pt`, SHA256 `b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62`. Use schema3 `training_only` with profile `G50`; qualify realized completion before separate arena preparation. No warm start from B100, extra epoch, larger corpus or value-target change is introduced.

Training has a4.5-hour GPU-stage cap (16,200seconds). B100's matched training took9,258.994seconds; that supports the existing ceiling, not a promised G50 duration. Prospective/realized schedule verification remains CPU-bounded at1,800seconds per invocation. These original-protocol16-worker schedule settings must be included in the launch resource check alongside active preparation; they are not a two-core data-preparation job.

## Prespecified comparison and interpretation

Use the merged `matched_original_epoch` [recipe screen](../bt4_recipe_screen.md): candidate G50 minus reference B100, qualified old-CUDA overlay, full search, prior temperature1, original PGN loader, seed42,16-ply histories and the same canonical500-pair development panel.

- Low:100 simulations, ordered SPRT Elo0/+15, alpha.05/beta.10, first128 pairs then64-pair looks, cap500 pairs; rolling256 and evaluator4096.
- High:400 simulations, fixed first128 paired openings, rolling128 and evaluator4096. Run after any valid low result, including H0 or inconclusive; operationally invalid low halts and preserves evidence.
- Each arena has a5,400-second hard stage cap, arena deadline5,340seconds, TERM at5,370 and30-second kill grace. Training plus both arenas has a27,000-second (7.5 GPU-hour) ceiling; prior B100 training is reused, not charged as a new stage.

Low H1 supports further consideration of G50 under this stopping rule; H0 favors retaining B100, and an uncrossed cap/deadline remains inconclusive. Report stopped low Elo/intervals as descriptive. Use the existing paired-opening sample-variance score/Elo interval for the fixed high panel. For the aligned first128 high-minus-low score interaction, use10,000 paired percentile bootstrap replicates with PCG64(seed20260903). A high interval spanning zero or conflicting budgets must remain visible; no automatic promotion follows a single result.

This same-seed development comparison isolates a meaningful50% versus100% global policy dose before a finer grid. It does not establish a globally optimal mixture, independent confirmation or general search/data scaling. Keep the corpus and one-epoch horizon fixed here; larger G10 data and extra epochs answer different questions and remain separate adaptive options. Soft-SF remains a distinct prospective target family, not a mandatory queue following G50.

## Remaining preparation

Bind the existing completed producer and publication receipts to the compact G50 qualification schema, retaining their distinction between all-shard process/metadata proof and the independently decoded first-shard check. Verify the prospective G50 physical schedule once using the existing canonical verifier. Then freeze the training-only manifest/runtime and register the final launch command. G50's training checkpoint and CPU arena preparation pins remain absent until their stages complete; draft manifests deliberately contain invalid pending hashes.

No full-corpus payload scan, corpus regeneration, teacher inference, training or match was performed to author this registration. The inherited training-control/purity limitations remain in force; development selection does not create a held-out claim.
