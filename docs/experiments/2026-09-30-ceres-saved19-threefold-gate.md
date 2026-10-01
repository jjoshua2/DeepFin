# Ceres saved19 negative-threefold CPU gate readout

**Readout, 2026-09-30.** The history-sensitive natural-outcome check was a measured cost in saved Ceres root preparation. A prospective helper tries to prove that a threefold draw is unavailable before falling back to the ordinary claim-aware outcome check. This bounded, one-pass CPU replay used 19 completed full-history games and 2,022 selected roots from one pinned archived source. It changed no live Ceres source or model.

The helper proved the negative-threefold case on **1,956/2,022 roots (96.736%)** and used the fallback on 66. For every saved root, its natural outcome equaled `board.outcome(claim_draw=True)` without changing board state. Separately, the incumbent `prepare_published_root` reconstruction matched the archived full-history input tensor, feed hashes, input keys, position fingerprint, compact and Leela legal support, played-move support and index, and prepared-root identity. All 19 terminal results and terminations matched the strict gate with its 400-ply cap, `rule50_match_v1` decision, six-piece limit, and recorded Syzygy inventory metadata. The tablebase files themselves were **not** byte-hashed; metadata parity is the scope of that check.

The measured natural-outcome component totaled **0.371460706 s** for the incumbent and **0.271017207 s** for the helper, a **0.100443499 s** difference on these same roots (27.04% lower component time; 1.3706× baseline/candidate). The helper was faster in 18/19 per-game component totals. Call order alternated by row index, but this is one diagnostic pass without an uncertainty interval. The incumbent whole-root preparation summed to **2.730943294 s**; a candidate whole-root path was **not measured**. The owned replay elapsed 8.024 s, which includes more than this component. The roughly 0.10044 s component difference is not a whole-root benchmark, end-to-end source gain, or month-scale projection.

| Saved game | Roots | Negative proof / fallback | Incumbent → helper gate (ms) | Terminal |
| ---: | ---: | ---: | ---: | --- |
| 1 | 125 | 125 / 0 | 21.452 → 15.442 | syzygy |
| 2 | 87 | 87 / 0 | 16.335 → 8.224 | natural |
| 3 | 47 | 47 / 0 | 8.627 → 2.795 | natural |
| 7 | 115 | 115 / 0 | 21.155 → 13.711 | natural |
| 11 | 130 | 127 / 3 | 21.833 → 17.521 | natural |
| 12 | 118 | 118 / 0 | 19.165 → 13.763 | syzygy |
| 13 | 64 | 64 / 0 | 12.334 → 5.075 | natural |
| 15 | 117 | 114 / 3 | 21.608 → 14.135 | natural |
| 18 | 118 | 118 / 0 | 21.212 → 13.996 | syzygy |
| 19 | 128 | 128 / 0 | 29.980 → 44.168 | syzygy |
| 20 | 77 | 77 / 0 | 16.594 → 6.391 | natural |
| 22 | 137 | 97 / 40 | 23.682 → 22.028 | natural |
| 23 | 109 | 109 / 0 | 21.443 → 12.637 | natural |
| 24 | 56 | 56 / 0 | 11.701 → 3.916 | natural |
| 25 | 125 | 125 / 0 | 20.985 → 15.144 | syzygy |
| 26 | 118 | 98 / 20 | 18.575 → 15.632 | syzygy |
| 27 | 142 | 142 / 0 | 21.728 → 19.314 | syzygy |
| 28 | 94 | 94 / 0 | 22.912 → 13.654 | natural |
| 29 | 115 | 115 / 0 | 20.141 → 13.471 | syzygy |
| **Total** | **2,022** | **1,956 / 66** | **371.461 → 271.017** | 11 natural / 8 Syzygy |

The [portable per-game evidence](evidence/2026-09-30-ceres-saved19-threefold-gate.json) records all 19 exact receipt SHA-256 values, row ranges, proof/fallback counts, both component times, baseline-only whole-root times, and terminal outcomes. The archived source ZIP is 1,530,905 bytes with SHA-256 `09bd17dd73a7adaefc9c2fbde289174eceba365c835b50cd9a069f4c15d94e67`; its selected source is `filled32_opening_v1`, under the source namespace pinned in that evidence. The plan SHA-256 is `6906185f465567eb418244618ded2a87da164ae9fcca425d470f1df0f8fd7f26`. The negative-proof source SHA-256 is `655a7acc41c1a957c0b05216aca2012c77d7c1e85b23ee70063b24d6a84d5a86`, the CPU runner is `2a1c910cd3d2305e2bac68b6fc5405d6866f15e9b52edeaa76de6ca104032aed`, and the owned supervisor is `0704af969eb00eb39b1a1fcfbbda678725a46b7dcdc6ff20cf1c06f66137ea51`. The child terminal SHA-256 is `e52edbf645f6e6cb94e129f532dfbda40d6f4771eadb0d039c5f3390b9c509a3`; the independently monitored session complete receipt is `8f960b6881b8ca38da4b7cad7a57e563f87c90c4904b14e36d34b6fb1a59bac0`. The complete receipt records exit 0, 8.024 s wall time, and 325,580 KiB peak child RSS.

This is a CPU-only diagnostic result on a fixed saved cohort. The candidate source and runner were temporary runtime files, not reusable production code in this repository; nothing was adopted into the live GPU generator. There is no new generated-corpus, target, training, Elo, CI-qualified throughput, or 500M completion credit. A whole-root speed claim would require timing a candidate whole-root implementation against a matched control.
