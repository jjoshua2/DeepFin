# Bounded shared verification for tri-source sort runs

**CPU-only synthetic follow-up, 2026-09-30; zero corpus, target, external-drive throughput, and 500M-row restart credit.** The [checkpointed sort foundation](2026-09-30-tri-source-checkpointed-sort-foundation.md) rehashed and reparsed ancestor parts on each nested verification call. This change gives one merge invocation a shared verification session. A fresh invocation still checks the complete required run DAG from its sealed bytes.

The session caches only checked run receipts and small part summaries, with explicit limits of 128 runs and 4,096 parts. A part is read into a bounded buffer of at most 8 MiB, hashed and parsed before it contributes rows; each of at most four merge readers also checks the part receipt and payload against the preflight proof before yielding any row. The session binds root, source, config, code, sort kind, row cap, byte cap, and fan-in. Reuse under a changed store identity refuses. The cache is discarded after the invocation; the next invocation rehashes retained parts rather than trusting path names or timestamps.

| Synthetic source runs | Distinct physical parts | Final-run payload bytes | Earlier no-op verified extent bytes | New no-op payload-file bytes read | New build payload-file bytes read |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 6 | 308 B | 924 B | 616 B | 1,232 B |
| 16 | 32 | 1,244 B | 9,952 B | 3,732 B | 8,708 B |
| 64 | 160 | 5,036 B | 75,540 B | 20,144 B | 50,360 B |

For the 64-run no-op, the new reader consumed each distinct part once, including old ancestors: 20,144 payload bytes, four times the final run's 5,036 bytes. Earlier figures count repeated *part extents entering verification*, while new figures count actual payload bytes read through the bounded part reader. They are not a measured storage-device speedup. The new counts exclude claim/receipt metadata, directory operations, and any external-drive latency. First-build output still reads more than one pass, as the table shows.

Twelve focused Python 3.10 tests pass: 4/16/64-run stable no-op receipts and sorted rows, old-ancestor refusal on a new invocation, changed payload or receipt after preflight before any row is merged, source/config/code session-identity refusal, bounded session caps, and the existing real SIGKILL/resume witnesses. Focused Ruff, Vulture, and Pyright pass. The full-repository Pyright run remains nonzero at the **same 188 errors and 8 warnings** as the parent revision, with zero new relative-file/line/message diagnostics. The [evidence index](evidence/2026-09-30-tri-source-shared-verification.json) pins the frozen code, tests, metrics, and validation receipts.

This is a bounded prototype: it rejects a session exceeding 128 runs or 4,096 parts, and a new full verification still reads the whole ancestor DAG once. On mutable local and Windows-mounted external drives, a prior receipt cannot establish that old bytes remain unchanged. Production source cursors, paired-wave comparison, winner/target joins, segmented sort-frontier manifests, independent all-row readback, and measured external-drive restart time remain separate work. No production path or GPU run was changed.
