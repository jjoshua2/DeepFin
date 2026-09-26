# Legal-move cache: equal-work reuse, saturation and lifetime costs

## Preregistration

Source baseline: PR #876 at `ead1abf29f6bbe6245d7ea8428f5fa0b45d6054f`.
MoveCache and every board/map implementation remain unchanged. This screen asks
whether avoiding legal generation pays for full-identity hashing/lookup and list
retention, and what happens when a cache cannot admit the requested positions.
It is not an engine speed or playing-strength experiment.

The native driver parses every board, builds attack tables and primes both arms
before its monotonic millisecond timer starts. Direct and cached execution use
the same runtime-selected loop, process the same boards in the same order, and
consume every returned move into an order-sensitive checksum. The measured loop
never calls a second independent generator. Parsing, reference checks, printing
and final-owner destruction are outside the operation timer. Cold-lifetime cases
explicitly include cache destruction/reconstruction between eight-request cycles.
The runtime dispatch, list traversal/checksum and IO continuation overhead remain
inside both arms; this is not an isolated hash-table operation timer.

Five warm cases use the SAME eight measured boards and an eight-record cache.
Priming selects 0, 2, 4, 6 or 8 of those boards and fills the remaining slots with
other distinct boards. The resulting 0/25/50/75/100 percent hit shares are checked,
not inferred from timings. Uncached misses always retain ordinary move generation.
The changing primer is outside the operation timer, but remains part of process
peak memory. Three additional cases cover cold eight-board lifetimes, eight warm
boards with maximum preallocation, and a cached empty checkmate move list.
These fixed legal fixtures are not self-play access traces or a natural hit rate.

Before timing, compare full diagnostic move lists against an independently built
CBoard legal oracle, including special-move flags. Preserve native order using the
direct driver's diagnostic as reference. Repeat diagnostics and zero/three-cycle
checksum/count/route contracts in optimized and UBSan builds. A deliberately
shortened native loop must compile and exit normally, then fail the work oracle.
Fewer requested operations must never look like a speedup. Long-run checksums use
independently tested affine composition; neither checksums nor tests are proofs.

Measure with `-O2 -ffp-contract=off` and the existing explicit BMI2/POPCNT target,
after its baseline capability and actual-instruction checks. No `-march=native`
and no compiler update. All builds are sequential. Use one complete local CPU
panel, with calibration/measurement bounded to 240 seconds and each child bounded
by the existing 120-second process-group timeout. No models, GPU or live jobs.

For each case double a COMMON repeat count from 32 until both arms reach 150 ms,
the slower arm reaches 2,000 ms, or 65,536 cycles is reached. Keep calibration
but exclude it from reported comparisons. Then take six alternating matched pairs
at that fixed count. Ratios require every measured sample to last at least 50 ms.
A descriptive improvement/regression requires all six ratios to agree beyond 5%;
otherwise report inconclusive. There is no performance threshold in correctness
CI. Retain slower-cache, below-floor and failed cases; no favorable-run selection.
This rule is not a statistical confidence interval or a cross-host guarantee.

Each invocation has a fresh GNU time `%M %x` record. RSS is that native child's
peak in KiB, including startup, parsing, priming, tables, runtime and cache storage.
It is not a live/steady-state byte counter, isolated payload cost, parent Python
memory, GPU memory or a full-population cache bound. Never difference cumulative
child high-water marks. Report raw values, median and range, without fitting a
universal bytes-per-entry formula.

## Reproduction

```sh
python -m pytest tests/test_bend_map_cache_benchmark.py
python -m native.bend_engine.u64_map_probe.move_cache_benchmark \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-cache-cost --measure
```

Omit `--measure` for driver/reference correctness only. Use a fresh directory.
The report retains workload definitions, all diagnostic/calibration/measurement
observations, source hashes, native order, native build identity and negative
control outcomes; raw output, per-child RSS and command records accompany it.
The benchmark is explicit/opt-in; no old workflow or algorithm is changed.

## Completed local readout

The single predeclared timing panel completed. All 96 diagnostic/contract executions
passed using optimized and UBSan builds, with 17 distinct boards checked by the
independently compiled CBoard oracle. There were 170 retained, excluded calibration
observations and 96 measured observations (six alternating pairs per case). The
shortened native workload compiled, exited normally and failed the unchanged work
oracle; seven invalid native configurations received their exact rejection.

| Scenario | Direct median ms | Cached median ms | Median paired direct/cached | Decision |
| --- | ---: | ---: | ---: | --- |
| warm-0pct-hits | 263.5 | 264.5 | 0.998 | inconclusive_at_5pct |
| warm-25pct-hits | 260.5 | 223.0 | 1.176 | cache_faster |
| warm-50pct-hits | 255.5 | 175.5 | 1.452 | cache_faster |
| warm-75pct-hits | 525.5 | 279.0 | 1.880 | cache_faster |
| warm-100pct-hits | 525.0 | 184.0 | 2.838 | cache_faster |
| cold-eight-once | 257.0 | 476.5 | 0.537 | cache_slower |
| warm-eight-large-arena | 524.5 | 181.0 | 2.874 | cache_faster |
| warm-empty-move-list | 29.0 | 17.0 | not admitted | below_measurement_floor |

Ratios above one favor caching. Common work counts can differ between scenarios;
compare paired arms within each row, not raw milliseconds across rows. Warm hit
shares use identical timed boards, but which boards hit is selected, not randomly
sampled. A 25% request hit share is not a universal break-even rate: actual hits
could have cheaper or more expensive legal generation.

Caching was beneficial on this working set from the tested 25% hit case upward.
The saturated zero-hit case was inconclusive at the predeclared 5% criterion.
The cold eight-request lifecycle used about 1.86 times as much operation-loop time
with caching. This supports preserving reuse across requests instead of repeatedly
creating tiny short-lived caches; it does not select a production lifetime policy.
The empty-list samples remained below 50 ms even at the maximum repeat count,
so their apparent ratio was not admitted and was not rerun with a larger budget.

| Scenario | Direct peak RSS median MiB | Cached peak RSS median MiB | Cached range MiB |
| --- | ---: | ---: | --- |
| warm-100pct-hits | 2.180 | 2.312 | 2.262–2.324 |
| warm-eight-large-arena | 2.170 | 7.420 | 7.402–7.449 |
| cold-eight-once | 2.160 | 2.156 | 2.152–2.184 |

The maximum-capacity allocation contains only eight live lists here. Its larger
peak RSS is not a bound for a fully populated 32,768-list cache, and small RSS
differences are subject to allocator/page-layout effects. No per-node byte cost,
steady-state allocation guarantee, or live-training memory setting is inferred.

### Provenance and execution limits

This was local Linux with Clang 17 and the unchanged 84-file Bend compiler pin,
not hosted Clang 18 qualification or the locked Python 3.13 development environment.
The source archive was the verified `1a1129ec` snapshot; the only later native
dependency, MoveCache.bend, was restored from the current PR and checked against
Git blob `877a0f86beb51ae6098047c852a7eb004dab7a5d`. A GitHub compare confirmed that
all other native dependencies were unchanged at `ead1abf2`.

The outer tool limit stopped the first combined command after all 96 correctness
executions, during the final negative-control build. It produced **no calibration
or timing observations**. Those outputs and source hashes were independently
reconciled, the interrupted negative control was rebuilt and executed, and the
remaining stages completed separately using the same generated positive binaries.
The initial report is retained as interrupted, not relabeled an end-to-end pass.
There was exactly one timing panel and no performance reroll. The recovery script,
raw reports, outputs, command records, per-child RSS and frozen preregistration
are retained in the accompanying local evidence bundle.

The compact summary and all 362 observations are included in the proposed patch
under `evidence/move-cache-cost-screen/`. Local trace readback checked every hash,
parsed work/count/route observation, RSS record and the recomputed summaries.
The 43 new Python tests passed with global conftest disabled; repository-wide
Ruff/Basedpyright/Vulture and hosted CI have not run on this patch. Existing test
globs select the new Python module, but native benchmarking remains explicit.

No cache/search implementation, compiler pin, workflow, dependency, live setting
or previous benchmark record changed. No full-engine, playing-strength, formal
proof or independent-review claim. Self-review only. GitHub publishing actions
were unavailable in this session; this is an apply-ready local continuation,
not a pushed or merged commit.

Raw combined report SHA-256: `b459721b9c01a245ae0edf9d336c3bc0d6b66b2519dffffe09765dbdd7c47a00`.


## Hosted correctness and publication follow-up

The preceding local readout and all 362 historical observations are preserved,
including the interrupted pre-timing attempt, cold-cache regression and below-floor
empty-list result. They remain Clang 17 measurements; no timing panel was rerun.

Run https://github.com/jjoshua2/DeepFin/actions/runs/36220257392 completed all validation stages before publication. The locked CPU environment
and Clang 18 built the unchanged measured Bend driver from the same pinned compiler.
All 457 map Python cases passed without skips, including 51 in the new module.
Focused Ruff/Basedpyright and whole-repository Ruff/Basedpyright/Vulture passed.
The native entry point ran without --measure: 32 diagnostic and 64 zero/three-cycle
contracts matched the independently built CBoard oracle and exact native ordering;
all 96 outputs, RSS records, work/route counts and hashes were reconciled. The
shortened-work mutation ran normally and was rejected, as were seven invalid
inputs with the exact expected diagnostic. Generated C matches the banked driver's
hash; original native/board dependencies remain source-identical.

The reusable Python harness now includes the same seven invalid-input cases that
previously ran only in the local recovery script. Eight additional admission tests
cover their validation; a tuple-construction style edit is semantically unchanged.
No cache algorithm, native driver, historical sample, or expected move changed.
The corrected reference recomputes the banked summary from all 362 CSV rows.

The compact hosted-validation.json records tested source hashes, fresh correctness
counts and raw-report identities separately from the historical experiment. Its
source manifest identifies the pre-readout documentation; this appended section
and that evidence file are documentation-only. This follow-up validates the
reusable entry point, not the historical timing on another compiler or full-engine
performance. Source-built ordinary PR CI remains separate. Nothing was merged,
deployed or enabled in a production frontend. Self-review only.
