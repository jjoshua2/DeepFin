# NumPy huge-page advice at the overlay hot loader: bounded negative screen

Date: 2026-09-23. This was a child-process-only comparison on eight qualified B schema-2 shards (65,536 rows) using the same PR #810 overlay-validation-reuse source in all arms. The only deliberate intervention was `NUMPY_MADVISE_HUGEPAGE=1` (control) versus `0` (candidate). The host kernel's transparent-huge-page `enabled` and `defrag` modes were both `[madvise]` throughout the recorded children; the runner read and checked those settings and did not change them.

The frozen order was control, candidate, candidate, control. All four children passed exact parity of the complete epoch plan, ordered eight-record roster, every decoded array shape/dtype/byte hash, and ordered target hashes. Requested NumPy advice matched each child's realized mode. The frozen plan SHA256 was `8d3b0c7bac25db0cd3c09bbfd9640e0c4ce95899dadf9bef33c88ddd9210f558`, guarded runner SHA256 `f0de6e270b2c7ba554c23074907ee4a90c7c274885026b0cc8987d16b10d5c08`. The plan pinned the original eight-shard receipt, source commit/files, Python/package versions, kernel modes, bounds and exact decision rule. Admission was separate and fresh.

| Arm | NumPy advice | Sum of actual `_load_one` seconds | Whole-child wall seconds |
| --- | --- | ---: | ---: |
| 0 | on | 3.738455 | 9.797364 |
| 1 | off | 4.715370 | 11.924423 |
| 2 | off | 4.589864 | 11.519935 |
| 3 | on | 3.248547 | 8.755328 |

Median hot-load time was **3.493501 s with advice on** and **4.652617 s with advice off**; turning it off was **33.18% slower**. Median whole-child wall time was **9.276346 s on** and **11.722179 s off**, **26.37% slower**. Both adjacent comparisons favored advice on. The preregistered benefit gate required at least 10% lower candidate hot-load time, both adjacent directions favorable, no more than 5% whole-child wall regression and exact parity. Correctness passed; the benefit gate failed. Retain NumPy's current advice setting for the next loader-only GPU comparison. Do not carry this toggle into that comparison.

The four children used 49.659 CPU seconds total, had about 1.0 GiB peak RSS, and completed within the 180-wall/120-CPU-second per-arm bounds, two cores (16-17), nice 19, no visible CUDA, 40 GiB memory and 150 GiB disk floors. Eight focused synthetic tests passed. The independent reviewer rechecked all four receipts, arithmetic, source/plan/advice and kernel pins, and found the negative decision correct. The original untracked prototype runner/tests in the NumPy worktree were preserved; the guarded executed copy and its tests, plan, receipts and compact publication manifest are under `scratchpad/numpy_thp_hotload_20260923_postreboot_v1/`.

This was a warm, small B-shard loader screen with a synthetic row-count objective census. It does not measure E-corpus or full-trainer throughput, memory-pressure behavior, physical huge-page realization, or the external drive. The earlier overlay validation-reuse CPU diagnostic remains positive and is the reason to test that loader change in a separate matched GPU trainer window.

Compact frozen [plan](evidence/numpy-thp-hotload-20260923/prepared-thp-plan.json), [four-arm summary](evidence/numpy-thp-hotload-20260923/run01/summary.json), and [artifact manifest](evidence/numpy-thp-hotload-20260923/publication-manifest.json) are banked with this record. Whole-repository lint was attempted: Ruff passed; local Basedpyright reported 2,160 errors and 1,226 warnings amid missing dependency/import resolution. This does not claim a repository-wide type-check pass.
