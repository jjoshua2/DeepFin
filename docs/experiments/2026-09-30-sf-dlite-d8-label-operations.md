# Selected-E D-lite depth-8 CPU label operations — 2026-09-30

This is a dated operational amendment to the [frozen value-screen preregistration](sf_dlite_selected_e_value_prereg_20260930.md).
It changes resource handling and source ordering only. The 2,500,000-row
roster, Stockfish profile, scalar calibration, candidate formula, unchanged
Selected-E policy, direct depth-12 audit, training comparison, and paired
strength decision remain frozen. The failed full-label attempts below have
zero admitted label, target, training, and Elo credit.

Paths in this public record are relative to the local chess artifact root, or
portable placeholders. They are not literal commands for another host. The
exact private command, working directory, environment and authorization bytes
are preserved in the cited launch proof bundles and their detach receipts.

The first eight-worker full-label launch stopped after 6.301 seconds at its
6 GiB conservative summed-RSS cap before sealing any block. The failed
receipt is `labels/sf_dlite_legacy_g10_d8_full01_20260930/LAUNCH-1790791752482419454-FAILED.json`
(SHA-256 `33b45342d9977b34dafeecebd1710c335971ea27c61d34ed43414096735972c4`).
Worker v7 (SHA-256 `8e00e10566a72ef60ea0bd101d1780b7d2fda0bb6e41470821475f9c456dd335`)
filters each worker's owned source IDs before sorting. A complete roster
proof found the exact same ordered indices as the original global sort
followed by ownership filtering in all eight partitions (receipt SHA-256
`8cbe01662a539d451afcf18de4d8ab8c57d55722ccfa32df73819ce525fd47ab`).
An eight-worker full-roster, real-engine **no-label** screen measured
6,990,480 KiB summed RSS, 2,593,658 KiB proportional set size, and
94,544,276 KiB host MemAvailable (receipt SHA-256
`7440b3fbadb4c5716a8c211384a46a456223b32eac5bfb69d5e80ba21e3e10bf`).

A reviewed second attempt used launcher v8 (SHA-256
`c125d36aecfe860fbe958e99f0033ffdd1e2774e3b907ecbf29ab709d325f61a`),
detach shim (SHA-256 `5a23b94eb35f59bf45bffa8ae3abc7783ad5749bee43b59068ba0939c8f35994`),
and root authorization (SHA-256
`e80640201c0b4ce77ae0f47b2826f0039fc254e8f39d3e7eaf3ccb843a422b03`).
It retained summed RSS with an 8 GiB aggregate cap, added 2 GiB per worker
tree and a 32 GiB host MemAvailable floor, and kept the four-hour whole wall,
600-second unsealed block, 30 GiB output, and 100 GiB shared-host physical
I/O bounds. Its portable command shape was:

```bash
<qualified-python> <reviewed-full02-detach-shim> --audit <artifact-root>/operations/sf-dlite-legacy-g10-independent-audit-run01-20260930/TERMINAL.json --audit-sha256 159648aa60fa3cdc4f11e79c0ff629a47d2a538316cc8011bee5aa1030865d6a --authorization <artifact-root>/operations/sf-dlite-legacy-d8-full02-failure-proof-20260930/full02_root_authorization.json --authorization-sha256 e80640201c0b4ce77ae0f47b2826f0039fc254e8f39d3e7eaf3ccb843a422b03
```

That `full02` run stopped after 20.53 seconds at 8,679,284 KiB summed RSS.
Each worker tree was below 2 GiB and host MemAvailable was 94,891,800 KiB.
Two source-bound blocks totaling 766 rows had been sealed; the failed root
still has **zero admitted labels** and will not be reused. Its failed receipt
is `labels/sf_dlite_legacy_g10_d8_full02_20260930/LAUNCH-1790793082744959855-FAILED.json`
(SHA-256 `45501de076a7f8487873405a84199498ef78fecc40992a4c999d67487dca5993`).
The 13-file failure evidence bundle is
`operations/sf-dlite-legacy-d8-full02-failure-proof-20260930/MANIFEST.json`
(SHA-256 `d9f5fea2b837006c80392b6d1581b44546c144ce3cb4a940a42a4778a090b317`).

An eight-worker no-label screen then hashed and fully decoded the first two
source files owned by each worker: 16 pinned files, 8,204–8,415 raw rows per
file, with no Stockfish search or labels. Engine initialization had a transient
7,414,548 KiB summed-RSS peak; the decode stage peaked at 6,996,280 KiB
RSS and 2,605,999 KiB proportional set size, with at least 94,839,588 KiB
host MemAvailable. The decoder streamed rows rather than accumulating a
source file. Terminal SHA-256
`f4a4ae0e8dfb19b7f0e6e9e54dd4918f752ba2ec06fad490010d6a016ddd2005`
and durable 34-file proof bundle
`operations/sf-dlite-legacy-d8-eight-decode-screen-proof-20260930/MANIFEST.json`
SHA-256 `f953f2ab89695618bc76e7a195877760479d53d696f6306402776bfd61c45363`
preserve the measurement. The `full02` excess arose during row replay,
search, or result accumulation; this screen does not isolate those components.

The next reviewed capacity choice is six workers under the **existing** v8
8 GiB aggregate / 2 GiB per-worker / 32 GiB host-free caps. A 2.5M-row
order proof for six source-ID partitions passed (receipt SHA-256
`885fd8a522f99e664cb69d6ea82cc5c100b333129ca484fa7e0e917b19e122e9`).
Worker row counts are 445,856, 426,294, 408,800, 398,927, 422,278,
and 397,845. Scaling the pilot's conservative eight-worker 8,184-second
allowance by the maximum owned-row ratio yields about 11,198 seconds, below
the fixed four-hour cap; this is an operational allowance, not a guarantee.
The fresh six-worker `full03` shim is prepared at SHA-256
`1e70d26c85ee5b5993c96860414d2d328306534927c0104273213cc0fcb25137`.
Root reviewed that exact shim and issued authorization, later archived as
`operations/sf-dlite-legacy-d8-full03-launch-proof-20260930/full03_root_authorization.json`, SHA-256
`ebc99bbe9d56d122f7f81637bfb9ca4d7bbb22736789345b5e7d36c60158e54b`.
Its portable command shape was frozen before launch:

```bash
<qualified-python> <reviewed-full03-detach-shim> --audit <artifact-root>/operations/sf-dlite-legacy-g10-independent-audit-run01-20260930/TERMINAL.json --audit-sha256 159648aa60fa3cdc4f11e79c0ff629a47d2a538316cc8011bee5aa1030865d6a --authorization <artifact-root>/operations/sf-dlite-legacy-d8-full03-launch-proof-20260930/full03_root_authorization.json --authorization-sha256 ebc99bbe9d56d122f7f81637bfb9ca4d7bbb22736789345b5e7d36c60158e54b
```

The invocation awaited the preceding shared-I/O owner's explicit lease
release. A successful complete label terminal and independent all-label audit
remain required before any target, GPU, or strength credit.

The shared lease was released and this exact `full03` command launched. Its
detach receipt is
`labels/sf_dlite_legacy_g10_d8_full03_20260930/DETACH-1790794489780088489.json`;
its six-worker launch receipt is
`labels/sf_dlite_legacy_g10_d8_full03_20260930/LAUNCH-1790794489942036291.json`.
The ten-file launch evidence bundle is
`operations/sf-dlite-legacy-d8-full03-launch-proof-20260930/MANIFEST.json`
(SHA-256 `cb107a102eff4deabe01b9be4accb6b40707d0a03a25977edee8a67424314e59`).

`full03` stopped after 66.212 seconds at the unchanged 8 GiB summed-RSS
guard: measured aggregate RSS was 8,527,060 KiB, with individual worker
trees at 1,328,372–1,553,548 KiB and host MemAvailable at 95,357,548 KiB.
It had sealed eight source-bound blocks covering 11,970 physical rows. These
blocks are retained as failed-attempt evidence only; **the complete `full03`
root has zero admitted labels, targets, training, and Elo credit**. Its failed
receipt is
`labels/sf_dlite_legacy_g10_d8_full03_20260930/LAUNCH-1790794489942036291-FAILED.json`
(SHA-256 `e15fda45a2001dafe2992b2b09e4dfe8403c54172f05f3d8cf380055b05a660f`).
The 24-file failure proof bundle is
`operations/sf-dlite-legacy-d8-full03-failure-proof-20260930/MANIFEST.json`
(SHA-256 `3da409360bae6d88aaba9e19e9bb5883a0f848bd22a5feee06041adee22aea52`).

A separate, reviewed two-worker **no-credit** memory diagnostic then used the
unchanged real-row replay, Stockfish depth-8 search, hash-8 and six-man
protocol. It capped each worker at 8,192 selected rows and the parent at
300 seconds, with 12 GiB diagnostic summed RSS, 32 GiB host MemAvailable,
10 GiB shared physical I/O and 100 MiB output guards. Its seven-file frozen
source bundle is
`operations/sf-dlite-d8-warm-search-diagnostic-v2-source-20260930/MANIFEST.json`
(SHA-256 `19cc5d873cf5ab8758b7aced2086c8a568984b873ca77700626e9e54e1b0a878`).

The diagnostic's sampler stopped after 77.832 seconds on a process-exit race
while reading one Stockfish `/proc` memory file. One worker had reached 8,192
rows and the other 7,552; the terminal is **FAILED_DIAGNOSTIC_ZERO_CREDIT**
(SHA-256 `044d5ae256e535d4141882622acaf53957beea84961fd6b84e968fdeaa87fc34`).
The 69 preceding per-second memory samples and child logs are preserved in
`operations/sf-dlite-d8-warm-search-memory-diagnostic-v2-run-20260930/MANIFEST.json`
(SHA-256 `f66988fb4680847da0d5495450ca94b26576cf71fbc6e678d2177394b18d8a01`).
The observed two-worker peak was 3,712,652 KiB summed RSS and 2,811,735 KiB
proportional set size; the sampled host MemAvailable minimum was
94,608,560 KiB (about 90.2 GiB).

Detailed `/proc` mapping samples identify retained Syzygy tablebase pages as
the main source of growth: for one worker, tablebase RSS rose from zero before
search to 315,860 KiB at 2,048 rows and 881,320 KiB at 7,552 rows, while its
Python RSS stayed near 733,000 KiB after the first block and Stockfish anonymous
RSS rose only from 158,108 to 199,976 KiB. The second worker's tablebase RSS
reached 743,536 KiB at 8,064 sampled rows. These measurements do not establish
a full-corpus memory ceiling. A reviewed cache-lifetime or physical-memory
envelope decision is needed before another full attempt; no fourth launch or
resource-guard change is authorized by this readout. The experiment formula
and preregistered decision rule remain unchanged.

A prospective launcher v9 is prepared for root review (source SHA-256
`f9d0de29d50cc0de707b711325131ec479f482602c8f75adee8323ffd7ab11f7`).
It would retain six persistent Stockfish engines and all source, depth,
calibration, block, wall, output and physical-I/O limits, but use all-owned-
process proportional set size below 24 GiB and per-worker anonymous memory
below 2 GiB as admission guards. The 32 GiB host MemAvailable floor remains;
summed RSS becomes recorded diagnostic data. The fresh `full04` detach shim
(SHA-256 `74fc44c30ece016db60c382f49228398da63ecab12f3cf78f7d8c14267e01c77`)
targets a new output root and requires a new root authorization sealing those
limits. Six focused CPU memory-meter fixtures pass. This proposal has **not**
launched or admitted labels; source review, authorization and a dated exact
launch record must precede any fourth attempt.
