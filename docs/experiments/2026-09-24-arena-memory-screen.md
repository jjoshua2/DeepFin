# Bounded arena memory and initialization-cost screen

## Preregistration

Continue #874 at `de48b98e1e3343033041f622407c0542ea550c03`. Larger fixed
arenas pass search qualification, but their resident memory and initialization
costs have not been measured. This screen informs whether dynamic growth is
worth investigating; it does not implement it or choose a live memory setting.

Use the unchanged qualified #874 generated C (SHA-256
`8f9d1712af22a1ec40914ce2e53e60d5a3fde264b92c34a8561d4c3edb254b72`) and
unchanged deterministic callback/worker. Rebuild with Clang 18 `-O3` and
`-ffp-contract=off`. Fix physical batch four, 146 channels, synchronous mode,
one Bend worker, depth eight. No compiler or application changes, model, GPU,
production UCI, training or live configuration. Historical generated C must match
its exact hash; different source is a new experiment, not a silent replacement.

Capacities: 4,096; 4,097; 8,192; 16,384; 65,536. The 4,097/8,192 pair reserves
identical physical storage and tests the rounding cost, rather than pretending
one more logical slot costs one more physical node. Root counts: one and sixteen.
Workloads: one simulation/root (startup-dominated, NOT pure initialization) and
64 simulations/root (equal-work short search). Start positions cycle over the
four opening histories used by the earlier callback screens.

For each root-count/workload, independently verify the default diagnostic run
using CBoard/Python-chess and actual input traces. Every larger arena's complete
trees, chronological events and all non-time work must match that reference.
Then verify diagnostics-off runs preserve summaries/work. Every measured run
must finish its requested simulation count, without a capacity stop. Smaller
memory or elapsed time caused by less work is a failed comparison, not a win.

Measure each native child separately with GNU time `%M %x`. `%M` is that child's
peak resident set in KiB, not virtual address reservation or cumulative Python
child high-water mark. Never difference `RUSAGE_CHILDREN.ru_maxrss` between runs;
a preceding large child must not contaminate a later small child's observation.
Use a new output directory and RSS file per invocation, check its exit code and
native report, and retain output hashes. GNU time's child is the native binary;
Python's reference/Torch memory is not part of that measured process.

One excluded diagnostics-off warmup per cell, then five measured cycles with
rotating capacity order so each capacity appears at each order position once.
This is 20 diagnostic executions, 20 warmups and 100 measured child processes.
Report all five RSS observations, median and range per cell, plus whole-process
wall and internal coordinator medians. Whole-process time includes allocation,
reporting and exit; it is NOT an isolated allocation timer. Short timing samples
remain descriptive, with no speedup threshold, precision ratio or statistical
significance claim. RSS includes the runtime, policy tables, input/output buffers
and allocator behavior, not just the node arena. No RSS-to-node-byte conversion.

Keep all observations including surprises. Correctness is the deciding gate;
there is no pass/fail memory or speed target and no performance reroll. A partial
panel or changed work is a failure. At most 240 seconds of screen execution plus
build/static work, 60 seconds per child, one native build at a time and two Torch
threads for the separate reference verifier. Maximum tested allocation is the
already-admitted sixteen roots times 65,536 node slots, not a larger cohort.

An OS process peak on one Linux host is not a memory safety budget for live
training, GPU VRAM or a real model. The maximum/RSS and timing results cannot
establish how much memory another process will retain or how lazy/growable
storage would perform. No settings are adopted. Self-review only.

## Reproduction

```sh
CC=clang-18 CXX=clang++-18 bash native/bend_engine/multi_root/build_arena_memory.sh QUALIFIED_ARENA.c NEW_BUILD
python -m native.bend_engine.multi_root.benchmark_arena_memory \
  --binary NEW_BUILD/runner --oracle NEW_BUILD/oracle --output NEW_SCREEN
```

`NEW_SCREEN/report.json` retains the complete panel and failure status. The
binary's stdout/stderr and independent RSS records remain under that directory.
Historical C is available in arena qualification artifact `10806529575` (finite
retention), ZIP SHA-256
`f21aede54a4aa00a4689163d2a2e418f9479221fe3990f5480bc44a19bc523a5`.
Source-only correctness CI remains independent of that historical artifact.

## Local checks

The 42 new parser, equal-work, panel-order and per-child measurement tests pass
locally with `--noconftest`; the container lacks python-chess required by the
repository's global conftest. This is not the locked hosted regression result.
The exact historical C compiled with local Clang 17 after two bounded command
attempts timed out; neither timeout produced performance observations. Hosted
Clang 18 and the complete locked-environment screen remain the recorded gate.

## Readout

Pending hosted execution. No measured memory or initialization verdict yet.
