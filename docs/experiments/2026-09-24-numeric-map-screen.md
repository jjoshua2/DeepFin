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

Pending hosted execution.
