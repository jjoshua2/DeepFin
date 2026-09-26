# Native CPU batch service profiles

This opt-in tool measures the existing native `deepfin_model_run_batch` callback
and compares **equal amounts of immediately available work** across separately
bound fixed-batch packages. It neither changes Bend scheduling nor runs self-play.
Do not use it to infer GPU behavior, accepted search EPS, Elo or production defaults.

## Scope of the clock

Each `steady_clock` interval starts immediately before the synchronous callback
and ends after it returns. It includes the existing bridge's zeroing, copy,
model execution and output packing. CPU output is physically ready on return.
Input preparation, reference checking, buffer poisoning and report formatting are
outside that interval. Trace and buffer-audit options are rejected during timing.
The benchmark, not a search program, owns these caller buffers.

Model copy/hash/loading time is reported separately. The first configured sweeps
are explicitly labeled warmups and retained in the raw observations; summaries
exclude them. All real occupancies `1..physical_batch` are measured in rotating
order. Package order also rotates across repeated processes. The input is a fixed
prefix of sixteen public chess root tensors, repeated across calls, not a sample
of production leaf arrivals. Cache, CPU scheduling, allocator and frequency effects
remain part of this single-host microbenchmark.

Every forward, including every warmup, must match an independent eager singleton
reference within the existing CPU-F32 tolerances (2e-6 absolute, 2e-5 relative).
All real logits must be finite and the unused output tail must remain untouched.
Failures emit no successful native profile; the driver preserves stderr and writes
a failed report. Packages, checkpoints, inputs, references and executable hashes
are checked and recorded. Only exact compatible CPU-v3 targets may be compared
in one run. This is a trusted-artifact test, not execution of untrusted model code.

## Build and measure

Use the repository's locked CPU environment, immutable trusted packages and a new
output directory for every target. The build wrapper calls the existing strict
batch binding and native CMake build; it does not regenerate Bend or modify the
live engine. It refuses CUDA binding. Models are exported separately.

```sh
bash native/bend_engine/service_profile/build.sh /tmp/service-b1 \
  /path/to/model-b1.pt2 /path/to/libtorch/share/cmake
bash native/bend_engine/service_profile/build.sh /tmp/service-b4 \
  /path/to/model-b4.pt2 /path/to/libtorch/share/cmake
python -m native.bend_engine.service_profile.profile \
  --target /tmp/service-b1/build/deepfin-service-probe /path/to/model-b1.pt2 \
  --target /tmp/service-b4/build/deepfin-service-probe /path/to/model-b4.pt2 \
  --checkpoint /path/to/checkpoint.pt --out /tmp/new-service-profile \
  --warmups 4 --samples 24 --repeats 3
```

The complete raw timings, warmup observations and per-process medians are retained
in `report.json` and each child's stdout. Nanosecond units do not promise
nanosecond measurement accuracy. The p95 is a nearest-rank sample statistic.
Executed rows per service second is real rows divided by summed callback time;
padding is separately counted and accepted rows/useful EPS stay null.
Process wall time includes startup/warmup/checks and is **not** callback latency.

## Offline comparison, not automatic dispatch

For ready queues of 1/2/4/8/16 real rows, the report compares plans that complete
exactly that many rows. Each alternative uses one fixed package for all calls;
there is no unmeasured hot-switching between independently loaded packages.
Dynamic programming permits measured nonmonotonic occupancy costs rather than
assuming the largest batch always wins. No occupancy is interpolated, no new
arrivals are assumed, and no row is dropped to make a deadline appear satisfied.

The cost is the **sum of per-call sample p95 values**, not a measured whole-queue
latency, the p95 of that sum, a calibrated deadline bound or a guaranteed optimum.
The lowest estimated-cost package is an experimental comparison only. Host/model
load, worker queues, encoding/backup, gather/scatter, package residency and game
quality require separate end-to-end checks before runtime adoption. No profile is
automatically loaded by the live runner.

## Source-only checks

```sh
python -m pytest tests/test_bend_service_profile.py
python -m native.bend_engine.service_profile.qualify --out /tmp/new-service-check
```

The latter explicitly compiles the same native timer/probe against a deterministic
test callback at five batch sizes and both input widths, plus two UBSan builds.
It tests timing/sample identity, exact I/O bounds and rejected bad outputs,
backend failure, trace/audit contamination and malformed arguments.
That callback is never linked into `build.sh`'s real model target.
The normal pytest cases require no new model compilation or native benchmark.

## Completed measurement evidence

[The experiment record](../../../docs/experiments/2026-09-25-native-service-profile.md)
retains the first CPU fixture measurement, full raw timing report, per-process
variation, source identities and limitations. It is not a production recommendation.
