# Ceres selected-label guard cadence: fixed 4,096-row GPU screen

**Readout, 2026-09-27.** Recursively scanning aggregate bank and state size at every guard call consumed measurable wall time. In a fixed four-arm A/B/A/B screen, using a five-second size-scan cadence reduced mean arm launch-to-exit wall from **41.815 to 34.390 seconds (17.76%)**, with identical saved raw heads, Ceres feed, and rebuilt policy/WDL target bytes. This is a provisional result for the selected 4,096-row bank. It is not a measured gain for a full Selected-E labeling run, source generation, or the 500M pipeline.

## Hypothesis and fixed comparison

The control A scanned aggregate bank and state size at each guard call (zero-second cadence); B scanned at a five-second cadence. Both retained identical first-call hash and exact-set checks and final-call exact-set, stat, inode, device and mount checks for the **same 13 observed mapped ORT/CUDA/WSL libraries**. The v3 one-session CUDA probe had found those 13 actually mapped libraries; its provisional 30-candidate list was held. The v4 and v5 successors were also held for missing-set validation and stale gate documentation, respectively. The v6 packet bound the observed closure and passed independent source admission before the GPU launch.

The arms ran serially as A1/B1/A2/B2 on one GPU, with independent state and output directories, one shared GPU lock, 1,200-second per-arm and 6,000-second whole-test process-group caps. Each arm used the same pinned model and source selection: **4,096 rows, 64 fragments of 64 rows, 128 fixed-32 neural calls, no padding**. The producing runtime, model, selected roster and target-reconstruction code were SHA-pinned. A post-arm CPU readback checked all selected row identities and rebuilt the selected-only and dual-input float16 policy and WDL targets from the saved raw heads. The comparison criterion was paired launch-to-exit wall with exact raw/feed/target parity; there was no preregistered threshold for production adoption or a formal power calculation.

| Pair | A: zero-second scan | B: five-second scan | B saving |
| --- | ---: | ---: | ---: |
| 1 | 42.846 s | 34.495 s | 8.352 s |
| 2 | 40.783 s | 34.286 s | 6.497 s |
| Mean | 41.815 s | 34.390 s | 7.424 s (17.76%) |

All four processes exited zero. Independent terminal review reconciled the plan, completion, child timing, profile and scan receipts; all **14 aggregate stream hashes** matched across the four arms. They cover raw policy/value/value2 logits, legal indices and offsets, saved and rebuilt feed, input and row identity, and selected-only/dual-input policy and WDL float16 target bytes. Each arm produced exactly 64 complete fragments within the 128 MiB output cap and no partial fragment remained. The reviewer found the 13 planned libraries in both first and final mapping snapshots for every arm, with no deleted relevant mapping. Child PIDs were gone and the cooperating GPU lock was free at review time.

The neural-session sums stayed near 23–24 seconds per arm (A1 23.217, B1 24.050, A2 23.783, B2 23.615). Recorded guard-scan work fell from 6.669/6.501 seconds in A to 0.071/0.089 seconds in B. The result is consistent with removing guard-scan overhead, not faster neural inference. The full supervised four-arm test including serial CPU readback and between-arm work took 160.849 seconds; the percentage above uses paired **arm** launch-to-exit wall.

## Interpretation and next decision

Two A/B pairs do not support a useful uncertainty interval, and order, cache state, and ambient contention can affect their wall times. The first/final loaded-library proof is intact, but the five-second version samples intermediate aggregate bank/state size less often. Per-call deadline, STOP and disk guards remain; exact byte parity establishes this bank's output equivalence, not an assurance about every future file change. The reviewer could verify the cooperating lock and descendants after completion, not retrospectively prove exclusivity against unrelated GPU users.

Keep the previous cadence available as a control. A source-reviewed production-path change and a representative Selected-E same-row cost run must precede any full-labeling speedup claim. Physical admission of that 35-cohort route is separately held on a larger-than-expected file roster and a frozen stage-reader pin. No source, target, corpus, owner, or 500M throughput credit follows from this screen.

The [compact machine-readable readout](evidence/ceres-selected-guard-cadence-20260927.json) records the four arm timings, source/plan/completion/readback digests, decision and 14 parity hashes. Its SHA-256 is `aba157aec9b36f886b5b3711014c53fdd00a6536a3641eb83e08aa584aba2530`. The immutable source packet is `f78ce28ab1c9c5e483931c1a16450a2cc715d65920e7c2d20f1b8bc2db7c606e`; the complete local result is `ea7d70608278d965035be7fa1e55e37f0ca53edf66ddb91fbabc6270b46052fd`. Independent source and terminal reviews are `f66e86f551b18d80dcae45886f5448a04e87c135ecbe831096af8a72e79cbaf2` and `8a0ee3428fb8b3d33901f57973b4f3a4287e3d7807e9cbeba303821568b49706`. The compact publication omits machine-local paths and the 4× full output bank; these digests identify the retained artifacts.
