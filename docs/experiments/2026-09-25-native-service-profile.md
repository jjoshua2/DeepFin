# Native batch service-time screen

## Preregistration — September 25, 2026

Parent #884: `f53c908a0af23267f61cee303944d11b878620de`.
Continue PR5's measured dispatch work, not live training or a new search policy.

Question: how does native synchronous CPU callback latency change with bound
physical batch and actual occupancy? Can a provenance-bound offline comparison
identify distinct equal-work choices without assuming that larger is faster?

Control: same saved untrained 5,043,005-parameter 175-plane CPU-F32 checkpoint
SHA256 `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
Reuse its singleton package SHA256
`9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
Export one batch-four package from those exact weights and encoding. No GPU or
private/trained checkpoint, architecture change, training, or match arena.

Plan: three native processes per package, rotating package order; four warmup
sweeps and 24 measured sweeps per process. Each sweep measures all real occupancies
1..B once in rotating order, using fixed root-tensor prefixes. For B1+B4 this is
420 forwards and 924 real rows including warmup, with 360 measured forwards and
792 measured real rows. These are repeated synthetic-workload measurements, not
independent games or 792 search simulations. Every output, including warmups, is
checked against eager singleton reference outside its timing interval.

Deciding gates: model/encoding/CPU/binary/input/reference identities reconcile;
all native and parser failure controls pass; every row stays within inherited
CPU-F32 tolerances; all prescribed samples are preserved with separate warmup
labels and physical/real/padded counts. Any failure stops a successful report.
There is no speedup threshold or favorable-run selection; keep all timings and
process variation. A timing screen cannot establish Elo or end-to-end EPS.

Deliverables: a separate native benchmark using the unmodified model bridge;
strict profile parsing/aggregation; offline equal-work fixed-package plans from
measured occupancy curves; reproducible checks and compact results. The comparison
uses summed sample p95 as a descriptive cost estimate, not a probabilistic
deadline guarantee. It does not switch packages in the running engine or modify
the scheduler. No assumption that a CPU result transfers to the 5090.

Budget: one bounded hosted CPU job, two Torch threads and one compiler job.
No Bend generation is needed; only the small C++ benchmark and existing bridge
are compiled. First run the cheap parser/native/static controls. Reuse existing
artifacts, avoid repeating successful model work for estimator-only corrections,
and retain any failed evidence. No original oracle or tolerance may be relaxed.

Independent review is not available in this tool environment; self-review and
limits must be explicit. No merge, deployment or live-setting change is authorized.

## Completed readout — September 26, 2026

**[Run 36216701536](https://github.com/jjoshua2/DeepFin/actions/runs/36216701536),
job `108333965512`, completed every stage successfully.** The measured staging
commit is `4c98604cb737872cf4d0909672e66687bd36a2e8`; its exact eight-file
post-correction manifest and source patch identify the executable/tool bytes.
Publication preserves those bytes, adding only the readout, experiment-index link,
compact evidence and a separate source-only regression workflow. #884 remains
unchanged at `f53c908a0af23267f61cee303944d11b878620de`.

This implements the measurement prerequisite for PR5's dispatch policy, **not
automatic runtime policy adoption**. No Bend/search/worker/model-bridge source,
production setting, compiler pin, model architecture or weights changed. Nothing
was merged, deployed or run on live training hardware.

### Actual CPU service measurements

The host reported AMD EPYC 9V45, four logical CPUs exposed, with two Torch threads
and one interop thread. The same saved untrained 5,043,005-parameter, 175-plane
CPU-F32 checkpoint was exported at batch four; its old singleton package was reused.
A cell below pools 72 measured forwards: 24 per process across three processes.
All prescribed observations, including excluded warmups, are retained.

| Bound physical batch | Real positions per call | Median callback time | Sample p95 | Real rows / summed service second |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.5383 ms | 0.5598 ms | 1,949 |
| 4 | 1 | 1.0417 ms | 1.1801 ms | 942 |
| 4 | 2 | 1.0403 ms | 1.1688 ms | 1,886 |
| 4 | 3 | 1.0481 ms | 1.1722 ms | 2,807 |
| 4 | 4 | 1.0510 ms | 1.2153 ms | 3,737 |

Within the tested batch-four package, callback cost is approximately constant as
real occupancy increases: padding does not make its physical tensor smaller.
A full call supplies four real results at about the same measured cost as a
one-real-row call. Conversely, padding a lone position to four is substantially
more costly in this screen than using the singleton package. These are
descriptive backend observations, **not a measured change to engine speed or Elo**.

Per-process median singleton time ranges from 0.4693 to 0.5481 ms. For batch four
with one real position the corresponding range is 1.0028–1.1386 ms, and for a full
batch it is 1.0109–1.1408 ms. This variation is retained, not removed as outliers.
The host was not affinity/frequency isolated, and the samples are short and
correlated. No significance test or universal optimal batch size is claimed.

Across all six model processes, the run executes **420 forwards and 924 real rows**:
**360 measured forwards / 792 real rows / 1,224 physical rows / 432 padding rows**,
plus **60 excluded warmup forwards / 132 real rows / 204 physical rows / 72 padding
rows**. The warmup flag, sweep/occupancy order and every duration are checked before
aggregation. Model open times range from 17.66 to 18.63 ms and are separate from
callback time; external whole-process time includes startup, warmup and validation.

Every real output in every warmup and measured forward passes the independent
eager singleton reference outside the timed interval. Maximum absolute logit
error is `4.172325134277344e-7` for the singleton package and
`4.76837158203125e-7` for batch four, within unchanged 2e-6 absolute / 2e-5 relative
tolerances. Output finiteness and the unused output tail are checked on every call.
These are raw policy/WDL logits, not search acceptance or playing-strength tests.

### What the offline plans do, and do not, say

The report compares equal real work for ready queues of 1/2/4/8/16 rows using the
sum of per-call sample p95 costs. Every alternative uses **one fixed package**;
there is no unmeasured package switching or mixing of packages within a plan.

| Ready rows | Singleton estimated cost | Batch-four estimated cost | Lower estimate |
| ---: | ---: | ---: | --- |
| 1 | 0.5598 ms | 1.1801 ms | Batch one |
| 2 | 1.1196 ms | 1.1688 ms | Batch one, narrow difference |
| 4 | 2.2392 ms | 1.2153 ms | Batch four |
| 8 | 4.4784 ms | 2.4307 ms | Batch four |
| 16 | 8.9568 ms | 4.8613 ms | Batch four |

**These sums are neither measured whole-queue latencies nor the p95 of those
latencies, and they do not guarantee a deadline.** The two-row difference is only
about 49 microseconds and is smaller than observed process-level variation;
it is not a robust production recommendation. Waiting for future arrivals,
queue fairness, encoding/backup, worker transport, package loading/residency and
the chess-search consequences are absent from this estimate. The DP supports
nonmonotonic occupancy curves, but it does not establish that those curves are
noise-free. No profile is read automatically by the live runner.

The input workload uses fixed root-tensor prefixes: batch one repeats the first
fixture, while batch four uses up to the first four. They share exact checkpoint
and encoding identity, but this is **not an identical empirical position mix** or
a representative production leaf-arrival distribution. The offline comparison
assumes these measured costs approximate those of other rows; that assumption
still needs a matched-workload and end-to-end test. CPU results cannot qualify
the trained network or a 5090.

### Correctness and continuing checks

- **266 focused Python tests passed**, including **61 new service-profile cases**,
  without failures, errors or skips. Tests reject malformed identities/durations,
  reordered/missing samples, contradictory warmup flags, invented accepted EPS,
  incompatible targets and missing occupancy costs. Equal-work plans cannot
  silently drop rows; explicit nonmonotonic examples test the DP.
- Whole-repository Ruff/Basedpyright/Vulture and explicit tool/test lint pass with
  zero type errors or warnings. Shell syntax checks pass.
- The same new native timer/probe compiles and executes against a deterministic
  test backend at five batches and two input widths, plus batch-four UBSan at
  both widths: **12 configurations, 350 nominal callback calls and 252 rejected
  controls**. The controls include failed startup/forward, exceptions, nonfinite
  and mismatching output, a touched output guard, trace/audit contamination,
  malformed arguments and corrupt/truncated/extra input/reference bytes.
- The real CPU probes were built by the actual new `build.sh` wrapper against the
  unchanged native model bridge. No Bend generation or full UCI/cohort rebuild
  was needed or performed. LibTorch/model internals are not sanitizer-qualified.
  The test callback is not linked into real-model probes.

The new source-only service regression workflow compiles the normal/UBSan probe
controls without model weights, Torch installation, historical build artifacts or
Bend generation. Its first PR execution is separate from the completed dedicated
qualification above; ordinary pytest runs the cheap parser/plan tests. Existing
engine/live/completion CI is unchanged.

### Recovery and provenance

The first run, 36216500780, stopped at static checks: seven subprocess calls lacked
explicit `check` arguments, and the optional error value needed explicit None
narrowing for the locked type checker. The next attempt, 36216609520, stopped
because the correction placed `check=False` before a positional argument.
The final correction keeps `run(command, check=False, ...)`, preserves explicit
return-code/failure checks and adds syntax compilation before the hosted gates.
No native timing code, sample plan, oracle or numerical tolerance changed.

**Neither failed attempt exported a model or collected timings.** There was one
batch-four export and one complete measurement panel, not performance rerolls.
The failed runs remain recorded as failures. Local 61 parser cases and the
normal/UBSan 12-configuration native panel also passed; hosted counts above are
the deciding version-matched evidence. A local duplicate pytest-module-name
collision was fixed by using an isolated test filename, not changing assertions.

The completed environment was Python 3.13.15, locked Torch 2.14.0+cpu, uv 0.12.10,
Bun 1.4.2 and Clang/Clang++ 18.1.3 on Ubuntu 24.04. CMake's existing -O1 and
-ffp-contract=off flags apply. Exact build logs/dependencies are retained. Native
products have no libpython link; this is dependency evidence, not a chroot test.
**Self-review only, not independent review or formal proof.** No ThreadSanitizer,
new whole-engine sanitizer run, trained checkpoint, GPU, self-play, game match,
fixed-wall search comparison or production-readiness claim.

Artifact **10897737515**, `deepfin-native-service-profile`, expires October 26,
2026. Its ZIP SHA256 is
`c7ecfb78fd0223f762e959031bd3f32de69bd11a4035be2a2834752276270db8`.
It contains the source patch/manifests, build/static/JUnit logs, all model timing
reports and raw stdout, public fixture/reference tensors, and test-only probe
binaries. **It contains no trained weights, model package or real-model binary.**
No binaries or raw tensor files are committed to Git.

Compact complete raw timing reports, binding/source identities, native-control
summary and publication provenance are committed under
`docs/experiments/evidence/native-service-profile/`. An independent download
recomputed every cell, offline plan, timing sum and warmup/physical-work count,
matched each native stdout record and verified input/reference hashes.

- Qualified patch SHA256: `64016aba402f56cd2543415d027a7ce1d7f0705c3e052d505d088490e7c75fad`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Singleton package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
- Batch-four package SHA256: `e08421109e5eb3b0ee0b1d911bfff82bc49de6b8a327dda3d696dc2f26e0384a`.
- Singleton binary SHA256: `dbf7151048a1001bb70dddabf2ff5676acbc7777be84f0a5fb6b0790972fd26a`.
- Batch-four binary SHA256: `12f57271ab562ed5e1bffc4e495942bcd3fba912107e69fc160d2af86c69bb41`.

Remaining PR5 work is a measured policy integrated into Bend and qualified at
equal neural work and wall time, plus persistent self-play/data generation.
This profiling PR supplies evidence and tooling, not completion of those tasks.
