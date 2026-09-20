# Factorial58 CPU preparation overlap

Status: independently reviewed wrapper adopted; bounded CPU sidecar launched
2026-09-20. This changes scheduling and storage preparation only. The 58,090,688-row
cohort roster, teacher recipes, training seed, and frozen producer/runtime are
unchanged. This is not a playing-strength result.

## Measured probe

A disjoint probe used the existing frozen preparation functions on cohort 03:
262,079 rows across 32 shards, CPUs 2–3, nice 15 and idle I/O priority. Limits were
30 minutes,32 GiB process RSS,32 GiB available-memory floor,80 GiB free-disk floor,
and no GPU visibility.

| Work | Observed |
| --- | ---: |
| Seal immutable base | 17.5725s |
| Build and verify B/C/D target overlays | 119.6837s |
| Output content bytes | 101,128,095 |
| Allocated output bytes | 106,098,688 |
| Output files | 2,465 |

[Raw probe receipt](artifacts/2026-09-20-preparation-overlap/probe.json).
Proportional extrapolation suggests about 22GiB allocated for all 58M overlay rows;
full-corpus I/O, label coverage and scheduling can change throughput. It does not
establish an exact completion ETA. An earlier probe failed before sealing because
a helper import shadowed the frozen producer package; the corrected isolated
import has a fresh-process regression test. Failed artifacts remain host-local.

## Adopted scheduling

The sidecar can seal all 35 immutable bases and build the 20 already labeled
cohorts containing at most 2.1M rows each. The 19M cohort's build stays with final
preparation. Small ready cohorts go first. Each stage is capped at 40 minutes;
queued preparation can wait at most 45 minutes for the current complete stage,
then retains its original 16-hour preparation allowance. The registered outer
budget is 60,420 seconds. A durable handoff request prevents later sidecar stages.

Queued preparation invokes its original pinned main function in-process. A
module-local subprocess adapter passes the ownership-lock descriptor to its
new-session workers; killing the coordinator cannot unlock an active writer.
Existing source/input pins, STOP checks and resource guards remain enforced.
Final preparation verifies and reuses complete sidecar outputs; partial outputs
remain fail-closed for explicit recovery.

The actual queue command and source/config pins are checked before every sidecar
stage. Source roots must remain at their canonical locations after sealing.
No active corpus was moved, and no caches were dropped.

The adoption transaction preserved all other queue entries and inserted the
separately reviewed 15-minute BT4 batch benchmark immediately before preparation:
remaining Ceres collection → BT4 batch benchmark → preparation → existing training
and arena sequence. [Adoption receipt](artifacts/2026-09-20-preparation-overlap/adopted.json)
and [exact config](artifacts/2026-09-20-preparation-overlap/config.json) are retained.

Sidecar launch PID 1023632 used CPUs 2–3. Host evidence and append-only stage timings:
`/home/josh/chess-artifacts/operations/factorial58-preparation-overlap-20260920/`.
The live sidecar source was pinned at commit 3995e6471. Its source SHA is
`6af0b06e16bc704e4406bfe0031ed6da24218716c596e6ea70e6288026d114cf`.

## Validation

Eight focused tests pass, including real multiprocess handoff, persistent restart
refusal, sidecar and queued-parent SIGKILL lock retention, STOP cleanup, queue
registration binding, cohort-size admission and isolated probe imports. Ruff
passes. An independent reviewer reran the tests and validated the actual operator
against the proposed 113-pin descriptor before adoption. No claim of final
factorial preparation completion is made by this launch record.

Initial production readback: cohort 03 sealed in 15.95s and completed target
construction plus verification in 111.06s; cohort 01 then sealed in 19.55s.
[Initial stage receipts](artifacts/2026-09-20-preparation-overlap/initial-production-stages.jsonl)
confirm useful production work beyond the disjoint probe. The final 35-root
qualification remains pending.
