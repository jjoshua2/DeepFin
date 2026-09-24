# Matched numeric-map timing with CBoard-derived keys

## Preregistration

Source baseline: PR #876 at `89bd90c461ebca9654556ecdf5ac652285cd28a1`.
Its three current workflows passed before this continuation. The earlier
CBoard-key replay qualified behavior but did not time those workloads. The
older synthetic-key performance screen used a different CPU target. This is
a separate measured panel, not a reinterpretation of either earlier record.

Question: does the hash map's operation-loop advantage over the dense scan
remain on the already-qualified chess-derived key sets? Keep U64Map.bend,
ScanMap.bend, benchmark.bend, benchmark.py, CPU selection, dictionary oracle,
fixtures and native key producer unchanged. This experiment adds no new map,
cache consumer, engine behavior, compiler update or live configuration.

Before timing, reconstruct the existing four legal-walk corpora from the current
checkout's compiled CBoard and require their record hashes, source identities
and six key-identity relationships to match the committed replay record. The
extension binary is rebuilt and recorded, not presumed byte-identical across
build paths. No new seeds, key filtering, bucket-based selection, or corpus
budget expansion is permitted for this measurement.

There are twelve fixed operation workloads: hits, misses and churn for opening,
castling-rich, en-passant and promotion corpora. Each holds 64 distinct keys in
128 identical retained slots; the scan searches only its occupied dense prefix.
Each cycle contains 256 operations and restores the same logical dictionary.
The churn mix includes rejected full insertions, replacement, deletion, misses
and reinsertion. Repeated cycles can change placement; the operation-loop result
includes that behavior, not a claim about a static probe layout.

The keys come from legal positions, but the scheduled operations are synthetic.
These are not captured production accesses, sampled selfplay games, full search
trees or neural-evaluation-cache traffic. The dense scan is a simple control,
not Base.Map, a tuned flat hash table, or the strongest alternative. Identical
retained slots do not imply identical process RSS or transient allocation.

## Controls, measurement and decision

Use the existing `benchmark --chess --measure` command, the unchanged verified
Bend compiler pin and Ubuntu Clang 18. Both arms select explicit
`-march=x86-64 -mpopcnt -mbmi2`, after baseline CPU capability and instruction
checks, with `-O3 -ffp-contract=off`. UBSan is a correctness mode, not the timed
configuration. One Bend worker, two Torch threads, one compilation at a time,
no GPU or real model.

The command first qualifies both implementations on all 36 dictionary replay
cases in the explicit target and UBSan, then all twelve operation drivers at
zero, one and three cycles (144 executions). Each timed execution must match
its expected returned-result checksum and complete final dictionary. Compilation
diagnostics, a semantic mismatch, or an incomplete measurement panel fail the
experiment. Repeated modes use the same fixtures, not independent datasets.

Retain the existing calibration and decision rules: double a common cycle count
from 32 until both arms reach 75 ms, the slower arm reaches 1,500 ms, or 16,384
cycles is reached. Exclude and retain calibration. Then run six alternating
matched pairs per workload (144 measured observations total) at the fixed common
count. Ratios are scan time divided by hash time, so above one favors hashing.
Report a ratio only if every measurement in that workload is at least 50 ms.
All six ratios above 1.05 qualify as the practical hash advantage; all six below
1/1.05 qualify as the scan advantage; otherwise call it inconclusive at 5%.
A valid negative result is retained, not a failed CI speed gate. No timing rerolls,
post-result threshold relaxation or selection of a favorable subset.

The timer excludes input parsing, table allocation/population and final reporting.
It includes native map calls, tape traversal, result checksumming and loop
execution. Paired loop measurements on one host are descriptive; no confidence
interval, production speedup, cache hit-rate, memory saving or Elo inference.
Do not pool with historical `-march=native` observations or compare absolute
millisecond times across hosts. Preserve raw samples, phases, work counts,
input/output hashes, full workload definitions and compiler/host identities.

Budget: one hosted CPU panel, the existing 240-second calibration/measurement
budget and 60-second child bounds, within a 15-minute workflow. No full broad
CPU suite needs repeating for a documentation-only continuation; focused map
contracts/static checks and native comparison are the recorded prerequisites.
Failure recovery is to retain the failed outputs and leave existing source and
live settings unchanged. Review is self-review only; no independent review or
formal proof. No merge or deployment is authorized by this experiment.

## Reproduction

```sh
uv sync --locked --extra dev --extra cpu
. .venv/bin/activate
bash native/bend_engine/install_ci_bend.sh
python -m native.bend_engine.u64_map_probe.benchmark \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-chess-map-timing --chess --measure
```

Use a fresh output directory. The existing persistent correctness workflow does
not gain a timing gate. The measurement workflow is isolated from the feature PR.

## Readout

The completed single-host panel is recorded below. No source or measurement rule changed after preregistration.


### Completed qualification

[Run 36064709080](https://github.com/jjoshua2/DeepFin/actions/runs/36064709080), job `107851645834`, completed every stage on the first attempt. All 102 existing map Python tests passed without skips/failures/errors; focused Ruff/Basedpyright passed. The corpus source, four record hashes and six identity examples matched the already-qualified record before timing. The two map implementations each passed all 36 replay cases / 12,061 operations in explicit BMI2/POPCNT and UBSan builds. All 144 zero/one/three-cycle driver checks passed. Build modes repeat fixtures, not independent games. No new tests or full broad CPU-suite run is claimed by this documentation-only continuation.

The single timed panel contains 144 measured observations and 168 separately retained calibration observations. Every observation passed the returned-result checksum and complete final-dictionary comparison. All input/output hashes and paired summaries were recomputed from downloaded records before publication.

| Corpus/workload | Cycles/sample | Hash median ms | Scan median ms | Median paired scan/hash | Decision |
|---|---:|---:|---:|---:|---|
| chess-opening-hits | 2048 | 88.0 | 499.0 | 5.677 | hash_over_5pct_faster |
| chess-opening-misses | 2048 | 100.0 | 891.5 | 8.911 | hash_over_5pct_faster |
| chess-opening-churn | 2048 | 114.5 | 592.0 | 5.192 | hash_over_5pct_faster |
| chess-castling-hits | 2048 | 78.0 | 498.0 | 6.452 | hash_over_5pct_faster |
| chess-castling-misses | 2048 | 103.5 | 894.5 | 8.648 | hash_over_5pct_faster |
| chess-castling-churn | 2048 | 98.5 | 595.0 | 6.004 | hash_over_5pct_faster |
| chess-ep-hits | 2048 | 84.5 | 501.0 | 6.185 | hash_over_5pct_faster |
| chess-ep-misses | 2048 | 99.5 | 891.5 | 8.960 | hash_over_5pct_faster |
| chess-ep-churn | 2048 | 97.0 | 593.5 | 6.121 | hash_over_5pct_faster |
| chess-promotion-hits | 2048 | 82.0 | 500.0 | 6.091 | hash_over_5pct_faster |
| chess-promotion-misses | 2048 | 103.5 | 894.0 | 8.662 | hash_over_5pct_faster |
| chess-promotion-churn | 2048 | 98.0 | 591.5 | 6.021 | hash_over_5pct_faster |

Ratios are medians of paired scan/hash times, not ratios of the unpaired median columns. Above one favors hashing. Read the decisions against the preregistered all-six-pairs and duration rules; a below-floor or inconclusive result is retained rather than promoted. This compares the native operation loops at 64 stored keys, not whole-engine throughput, cache reuse safety, or a tuned alternative map. The structural-key/history/collision distinctions from the preceding replay remain prerequisites for any consumer.

The practical use of this result is to choose between these two candidates for comparable numeric-key workloads. It does not justify transplanting the map into the production structural graph or neural cache without its separate identity, capacity and lifecycle contract. Neither historical synthetic results nor this panel establishes a universal fastest map.

All 456 contract/calibration/measurement observations are committed in [samples.csv](evidence/chess-map-timing/samples.csv), with compact [summary and source identities](evidence/chess-map-timing/summary.json), generated-code/binary fingerprints and host details. Full workloads, legal paths, raw stdout/stderr and JUnit are retained in artifact `10834874698`; ZIP SHA-256 `760e2a50d6de23b53ec42c2b5073639114c988b5be1d5f7dcee18df5a4c358c5`. The source and native extension fingerprints describe this run; the extension is not presumed binary-identical to an earlier build. Artifact retention is finite, but generators, compact results and raw timing observations are committed.

The isolated measurement/publication workflows are absent from the feature diff. No map, benchmark driver, engine key function, compiler, production workflow, default or live setting changed. All three workflows on parent `89bd90c4` had passed before this experiment; new-head checks are separate. Self-review only; no independent review, formal proof, real-model/GPU result, merge or deployment.
