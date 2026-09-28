# Numeric hash map versus a dense scan at a shared slot budget

## Preregistration

Continue #876 at 59079d479e22b9cf6fe583c815a8e091775b1b2b. The original U64Map
source is unchanged. Hypothesis: hashing reduces numeric lookup/update work as
occupancy grows, but a simpler scan may win for small tables or clustered keys.
This is a mechanism screen, not selection of the fastest possible map or cache.

Control: ScanMap uses the SAME imported U64Map.Table and U64Map.Slot types and
U64Map.new allocation/half-full admission. Its entries are dense, lookup scans
only occupied slots, and erase swaps the last entry into the removed position.
It shares result decoding/admission helpers but not hash lookup or backshift.
Both reserve the same physical buckets with the same concrete slot representation
and hold the same keys/values. This matches retained table storage, NOT process
RSS, transient allocation or every possible scan representation. Neither is a
replacement for Base.Map or a production transposition/evaluation-cache baseline.

Cases: random full-width keys at 8, 64 and 256 entries; high-half-only keys at
64 entries; 64 deliberately colliding high-half-only keys with a wrapped cluster.
For each distribution/size, use hits, misses, and churn. Each fixed 256-operation
cycle restores its initial dictionary. Churn includes rejected full insertions,
updates, returned previous values, delete/miss and reinsertion. The deliberately
bad cluster measures a known noncryptographic-hash limitation, not its likelihood
under real chess keys. Synthetic deterministic keys are not sampled chess hashes.

A separate build changes only the Impl import in the operation driver. Table
construction, parsing, population and final output occur outside its IO.now
operation timer. The timed section includes map calls, tape traversal, result
checksumming and loop overhead. The runtime-supplied tape prevents compile-time
constant folding of a known result. It is not a pure probe-count measurement.

Before performance, run BOTH implementations against the existing independent
Python dictionary's 32 fixture traces in native-target and UBSan builds. Then
check each benchmark case at zero, one and three cycles in both builds: validate
return-value/disposition checksum and the complete final dictionary, including
duplicates and high halves. Long-repeat expected checksums use an independently
tested affine composition; only state-preserving cycles may use it. A checksum
alone is not a proof; original per-operation traces and final-state checks remain.

For each case, double a COMMON cycle count from 32 until both arms reach 75 ms,
the slow arm reaches 1,500 ms, or 16,384 cycles is reached. Calibration is retained
and excluded. Run six alternating matched pairs at that fixed count. Admit ratios
only if every measured observation is at least 50 ms. Call a >5% improvement or
regression only when all six paired ratios agree beyond that threshold; otherwise
report inconclusive. This is a descriptive single-host screen, not a confidence
interval, p-value or many-workload-adjusted significance claim. Keep all results,
including scan wins, below-floor cases and failures. No speed threshold is CI's
correctness gate. No rerolls to obtain a favorable timing outcome.

Budget: one hosted CPU panel, 240 seconds for calibration/measurement plus bounded
compilation and correctness tests; one native build at a time, two Torch threads,
no model, GPU or live access. This adds benchmark tools/control only; no map/search
integration, dynamic growth, default change, compiler update, merge or deployment.
Recovery: original map unchanged. Self-review only; no independent review or proof.

## Reproduction

```sh
bash native/bend_engine/install_ci_bend.sh
python -m native.bend_engine.u64_map_probe.benchmark \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-map-screen --measure
```

Omit --measure for reference and benchmark-driver correctness only. Use a fresh
output directory. The report preserves raw calibration/measurement observations,
full generated workload definitions, source/input/output hashes and compiler
identity. Raw stdout/stderr files are retained beside it. Timing is opt-in and
separate from ordinary pytest. No historical Actions artifact is a runtime input.

## Local checks

Both operation drivers compiled with the unchanged pinned compiler and Clang 17.
All fifteen cases matched checksum and complete final state for three cycles on
both implementations. All 33 new Python tests passed with --noconftest (the local
container does not have the repository's locked global test dependencies). These
are not the full hosted old-fixture or sanitizer qualification. Local source
construction needed parser helpers and explicit copy annotations; the original
map and reference expectations were not modified. No local timing comparison is
used for the performance readout.

## Readout

The completed matched-work screen is recorded below.


### Completed hosted qualification

Run https://github.com/jjoshua2/DeepFin/actions/runs/36012260177 passed all preceding gates. All 67 focused Python cases passed without skips (33 new). Focused static and repository-wide Ruff/Basedpyright/Vulture passed. Both implementations passed all 32 existing dictionary traces (7,042 operations each) under optimized native-target and UBSan builds. All 180 zero/one/three-cycle benchmark-driver checks matched checksum and complete final dictionary. Modes repeat fixtures, not independent datasets.

All 180 measured observations passed the same checksum/final-state tests. Calibration samples are retained but excluded from paired comparisons. The generated workload definitions and raw stdout/stderr remain in the run artifact; source-qualified workload/input/output hashes, every sample and compact results are committed. Shared backing slot storage does not imply identical RSS or transient allocations.

| Workload | Cycles/sample | Hash median ms | Scan median ms | Median paired scan/hash | Screen decision |
|---|---:|---:|---:|---:|---|
| random-8-hits | 16384 | 120.5 | 175.0 | 1.448 | hash_over_5pct_faster |
| random-8-misses | 8192 | 88.5 | 130.0 | 1.469 | hash_over_5pct_faster |
| random-8-churn | 16384 | 134.0 | 220.0 | 1.642 | hash_over_5pct_faster |
| random-64-hits | 8192 | 79.0 | 366.5 | 4.646 | hash_over_5pct_faster |
| random-64-misses | 8192 | 116.0 | 645.5 | 5.586 | hash_over_5pct_faster |
| random-64-churn | 8192 | 116.0 | 506.0 | 4.353 | hash_over_5pct_faster |
| random-256-hits | 8192 | 82.5 | 1232.5 | 14.958 | hash_over_5pct_faster |
| random-256-misses | 4096 | 60.0 | 1200.0 | 20.008 | hash_over_5pct_faster |
| random-256-churn | 8192 | 116.5 | 1769.0 | 15.185 | hash_over_5pct_faster |
| high-half-64-hits | 16384 | 150.0 | 729.0 | 4.860 | hash_over_5pct_faster |
| high-half-64-misses | 8192 | 134.5 | 645.0 | 4.799 | hash_over_5pct_faster |
| high-half-64-churn | 8192 | 115.0 | 507.0 | 4.404 | hash_over_5pct_faster |
| cluster-64-hits | 2048 | 243.0 | 92.5 | 0.380 | scan_over_5pct_faster |
| cluster-64-misses | 1024 | 235.0 | 81.0 | 0.345 | scan_over_5pct_faster |
| cluster-64-churn | 2048 | 413.5 | 126.0 | 0.305 | scan_over_5pct_faster |

Ratios above one favor hashing. They describe this operation loop versus the simple dense-scan control, not Base.Map, a tuned flat hash table, real chess keys, end-to-end search or a neural cache. The deliberately clustered input is synthetic and not a measured real-world frequency. No implementation is integrated or selected as a universal winner. Default map, engine, compiler and live settings are unchanged. Self-review only. Nothing merged or deployed.
