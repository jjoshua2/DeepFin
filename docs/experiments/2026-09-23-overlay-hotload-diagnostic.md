# Overlay validation reuse: measured hot-load benefit

Date: 2026-09-23 UTC. Eight qualified B schema-2 shards, 65,536 rows.

PR [#810](https://github.com/jjoshua2/DeepFin/pull/810) reduced the median sum of
actual `GameAwareEpochBuffer._load_one` times from **10.925 s to 4.110 s**,
**62.38% less time (2.66x throughput for this stage)**. All four arms had identical
complete plans, ordered records, decoded array shapes/dtypes/byte hashes, and
ordered target hashes. This passes the preregistered 10% reduction threshold and
warrants a later full-trainer comparison.

| Order | Arm | Hot load seconds | Child measured wall seconds | Hot semantic validations |
| --- | --- | ---: | ---: | ---: |
| 0 | Control | 11.8465 | 33.0326 | 40 |
| 1 | Candidate | 3.4850 | 9.4934 | 0 |
| 2 | Candidate | 4.7357 | 11.9380 | 0 |
| 3 | Control | 10.0041 | 28.1692 | 40 |

Startup semantic validations were 72 per control and 8 per candidate. Candidate
contexts retain the completed semantic validation while still checking source
identity; avoiding repeated semantic scans explains the measured reduction.
Only `chess_anti_engine/replay/target_overlay.py` differs within the package between
the two frozen source commits. The fixture and all execution pins are in the
[plan](evidence/overlay-hotload-diagnostic-20260923/prepared-hotload-plan.json).

## Scope and resource evidence

This is an ABBA CPU diagnostic using the actual loader method with a synthetic
row-count objective census. It is not a full trainer, an E-corpus measurement,
or a cold external-drive throughput test. Constructor and census stages precede
loading, so the measured loads are warm. E continued concurrently; its recent
window was 24.128 s with 8.301 s input wait at admission, and a later observed
window was 26.580 s with 10.727 s input wait. Those isolated windows do not
establish the size of any training interference. No GPU benchmark was launched.

Each child was limited to two low-priority CPU threads on cores 16-17, 180 wall
seconds and 120 CPU seconds, with CUDA hidden. All four exited successfully,
using 105.904 child CPU seconds in total and 149,373 bytes of arm/summary output.
The host had 87 GiB available RAM and 843 GiB free disk at admission. The source
plan stays prepared; a separate fresh admission receipt authorized this single
run. Reruns need a new receipt and output directory.

The runner passed 10 independent synthetic tests. Parent review independently
compared all receipt parity fields, recomputed arm totals, and checked the source
diff. [Review and artifact hashes](evidence/overlay-hotload-diagnostic-20260923/parent-review.json),
[summary](evidence/overlay-hotload-diagnostic-20260923/summary.json), and
[admission](evidence/overlay-hotload-diagnostic-20260923/admission.json) retain the evidence.

## Next decision

Keep validation reuse ahead of minor gather optimizations for the next trainer
comparison. Preserve the current E runtime and finish its queued paired arena
before GPU testing. A trainer test must measure end-to-end time and preserve
actual objective masks, row order, augmentation, and checkpoint/optimizer parity.
This diagnostic alone does not determine training speed at 500M positions.

The exact [executed runner](evidence/overlay-hotload-diagnostic-20260923/executed_runner.py) is retained as provenance. Publication review found its tooling branch was based on stale local main and scoped typecheck errors remain; reusable tooling is being rebased and repaired separately. These do not change the banked execution or the exact-parity result. No claim of a clean full lint gate is made.
