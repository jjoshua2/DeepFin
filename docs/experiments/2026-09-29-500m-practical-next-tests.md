# 500M reassessment and practical next tests

The current candidate remains cheap BT4/Ceres policy and value labels, an explicit
SF-origin position share, and sparse SF corrections tested as a separate change.
The recipe is defensible; its optimal fractions and month-scale capacity remain
unproved. This record updates the execution order in the
[earlier candidate](2026-09-29-500m-source-target-candidate.md).

## What the results support

The original E–D comparison on the shared 576-opening bank found E−D −12.37 Elo
[−27.18,+2.39]. Both use the same per-row 50/50 BT4/Ceres policy. E removes SF
from value and redistributes its weight between the neural teachers, so the
comparison does not isolate SF's contribution. The zero-Elo preregistered
result is unresolved; it does not justify declaring SF useless or selecting an
optimal mixture. Selected-E versus E passed its −20-Elo gate to measure cost;
Selected-E versus D remains inconclusive under its −30-Elo gate. See the
[direct E–D readout](2026-09-29-e-vs-d-576-strict.md).

The audited old-cohort four-arm label-cost comparison saved 38.89% complete-arm
wall with one selected teacher instead of both. That warrants testing the larger
current-source bank. It is not a second saving to subtract from capacity estimates
that already assume one teacher per row. See the
[cost readout](2026-09-29-full512-old-cohort-abba-cost.md).

## Execution order

1. Run the source-reviewed conditional label-cost comparison on all 58,820
   within-bank deduplicated neural-source rows, with S1/D1/D2/S2 controls.
   Report teacher calls, conversion, target writing and forensic readback separately.
   Its equal-opening-family sampling frame is explicit. It need not await the
   global legacy index, final production source fractions, packing or training:
   those are different claims with different prerequisites.
2. Test larger labeling batches on fixed current rows. The archived Ceres batch-512
   experiment reached 1,968 rows/s on an 8,192-row prepared batch and 1,070 rows/s
   on a 32,768-row stream, but failed its original numerical-equivalence limits.
   Top-move agreement was 99.707%; policy and value targets changed. Verify the
   current model/runtime compatibility, then measure speed and target differences
   against bracketed fixed-batch controls. Treat it as a numerical profile variant,
   with a quality test before adoption. Exact byte equality is required for a
   byte-preserving claim, not for every candidate worth testing. Also screen BT4
   batch coalescing across games rather than accepting batch-32 diagnostic calls as
   its production optimum. Archived evidence is from [PR #760](https://github.com/jjoshua2/DeepFin/pull/760),
   head `7b4b37b53a266c3086b38d6e648f23030d3df82f`.
3. Measure the Ceres proof-writer change during matched live generation. Its
   saved-game CPU result cuts the measured actor-to-sink publication interval
   34.48%, preserving decoded arrays and metadata. It does not establish a 34.48%
   overall source-speed gain. Use enough games to exercise normal flushes and the
   tail, with unchanged engine, model, seed, schedule and strict Syzygy support.
   See [PR #933](https://github.com/jjoshua2/DeepFin/pull/933).
4. Complete the cheap banked-data S7 holdout before spending training time on SF
   correction. Its first 8,462-row shard selected 7.26% of rows and captured 11.13%
   of ordinary d9 >300-cp BT4 deficit mass, against 7.52% for same-count random.
   That is below the frozen 20% capture requirement, but one shard cannot decide
   the complete 16-shard holdout. Do not retune the selector after this observation.
   Any later trained correction comparison must hold source and neural value
   targets fixed and compare both an unchanged baseline and equal-cost random
   correction. A selector metric alone is not playing strength.
5. Finish replay/history dedup of the existing three-source slice, materialize
   selected targets, pack them, and make the real trainer consume the pack from
   the external drive. Use generation needed for scaling to accumulate the
   matched SF-origin versus neural-origin source contrast, instead of generating
   days of data solely for a likely small source effect.

GPU work is serial, including benchmarks and training. CPU work can proceed
independently when its actual resource use fits. Source generation and matches
retain rule50-aware six-man Syzygy WDL and DTZ support.

## Storage screen: dense targets are a diagnostic format

A CPU-only exploratory screen reopened the audited old-cohort S1 selected target
payload: 12,288 rows, FP16 policy width 1,858 plus three FP16 search-WDL entries.
It used 4,096-row blocks. Nonzero policy counts averaged 26.09, with median 29
and maximum 61. Sparse representations decoded to exactly the same FP16 bytes;
no probabilities were removed or retuned. Sizes below exclude container headers,
input features, row identities and other training fields.

| Target layout | Logical payload | zlib level-1 payload | Compressed bytes/row |
| --- | ---: | ---: | ---: |
| Dense | 45,735,936 B | 1,420,665 B | 115.61 |
| Padded sparse | 3,031,040 B | 933,144 B | 75.94 |
| Exploratory CSR | 1,380,888 B | 830,216 B | 67.56 |

Dense zeros already compress well. The practical padded-sparse improvement over
compressed dense was 34.3% in this bank, rather than the much larger reduction
against uncompressed dense. The repository already has padded sparse policy
conversion in `replay/shard.py`; integrating and measuring that path is preferable
to inventing a second codec. CSR is an exploratory geometry comparison, not an
implemented trainer format. One-shot CPU timings and this old cohort do not
establish cold-drive throughput or production pack size. The screen is
self-reviewed with exact parse/decode and compression round trips; no independent
payload rerun is claimed. [Compact evidence](evidence/2026-09-29-policy-target-storage-screen.json)
records the source hash, counts, sizes, timing observations and limitations.

## Capacity decision

The previous illustrative sum is still about 37 ideal GPU-stage days before
SF CPU work, dedup, packing, cold reads and retries. Its historical label rates
are assumptions for the current producer, and its trainer rate is cache-affected.
A month is a target to test, not a commitment supported by current receipts.
Larger batches and compatible generating-owner label reuse offer direct ways to
reduce the GPU terms. Sparse storage and buffered publication address CPU and
I/O terms, but their stage savings must not be counted as equal overall savings.

Do not require an entirely fresh corpus merely because accepted legacy profiles
have small differences. Record their input/history, model, calibration and target
profiles, and evaluate the intended use. Reuse changes a byte-preserving claim
only when exact identity was promised; otherwise it is a recorded data-profile
choice that needs an appropriate quality comparison.
