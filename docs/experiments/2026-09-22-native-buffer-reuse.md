# Native policy-map and inference-buffer reuse

## Scope and preregistration — September 22, 2026

PR2 of the neural-search scaling plan, on PR1 #828 at
`ba18279fffdb13beb894e1d4f49cd60601f38583`. No training, live configuration changes,
merge, deployment, search-policy change, compiler upgrade, CUDA or strength claim.

Hypothesis: immutable policy-map construction, float-list marshaling, and repeated
bridge input-tensor allocation can be removed without changing any selected-leaf
input, legal policy mapping, probability, completion count, or chosen move.

Control: the parent's list-based numerical converter and existing selected-leaf
oracle; reuse the exact saved untrained CPU-F32 batch-one package from PR1's
qualification artifact rather than exporting another model. One resource-bounded
hosted CPU qualification, two Torch threads, one Bend runtime thread, and small
local compiled probes. No arena, training or production GPU compute.

Success requires unchanged numerical tolerances; bit-identical list/array numeric
cases; rejection of malformed capacities/counts/nonfinite outputs; stable addresses
across changing inputs and repeated roots; zero per-forward bridge input-tensor
allocations after startup; unchanged material UCI/perft; quiet/diagnostic raw traces
and search results identical. Any functional mismatch blocks qualification. No
speed threshold is claimed from this fixture or shared runner.

## Ownership and implementation

The Bend application holds a linear `NativeEvaluation.Cache` across go, stop,
position changes and ucinewgame. Its maps are built once for the neural process;
material mode has a disabled cache and does not construct them. Only the geometric
mapping tables persist. Legal entries and history-dependent input are fresh on every
leaf. Full input capacity is cleared before encoding, including unused cells.

`ModelBuffers.bend` declares a narrow synchronous effect. Input and output buffers
are consumed by the effect and returned after the model finishes. The C transport
checks logical input size separately from exact physical capacity, requires the
pinned compiler's packed scalar representation, and rejects aliasing. It never
constructs per-float input/output lists. `LogitReply.convert_buffer` checks capacity,
exact output count and all 1,861 finite values before legal gathering; unused output
capacity is not part of that shape. It returns the output array for the next call.

The C++ bridge creates one model input tensor, then copies new input into that tensor.
It still copies AOTI outputs into the distinct Bend workspace and validates their
shape/device/dtype. The saved package contract and fail-closed behavior are unchanged.
No promise is made about allocations inside AOTI, legal-policy arrays, temporary
history encoding, or Bend's generic runtime. This is not zero-copy or async-safe
multi-slot scheduling. No buffer may be reused before synchronous completion.

Detailed per-leaf protocol diagnostics are now explicitly opt-in. The existing
external neural verifier enables them; ordinary native play need not serialize a
path and a pair of numbers for every legal move. Rule-draw notices and final PR1
work accounting remain available. Raw traces stay separately opt-in, exclusive-create,
owner-only, and outside the normal execution path.

## Local checks

The unchanged pinned Bend 2.0.21 + U64 compiler accepts the complete application.
Normal and UBSan compiled probes each pass 16 transport round trips (both 146- and
175-plane widths, diagnostics off/on), seven transport/configuration failures,
11 bit-identical list/array logit cases and four additional capacity/needs failures.
The test-only transport backend checks every input value and actual input/output
pointer reuse. It is not linked by any engine build and is not a model fallback.

The array-logit cases poison spare capacity with NaN; valid logical outputs still
pass. In-range NaN/infinity and missing/extra logical outputs still fail. Original
numeric probe files and their tolerances are unchanged.

Local full C generation exceeded the 4 GiB container budget; no compiler pin or
code-generation settings were weakened. Hosted qualification supplies full native
execution evidence when complete. Local self-review is not independent review.

## Reproduction

Use the repository's locked CPU development environment and verified compiler.
After emitting C from `buffer_probe.bend`, link it with `buffer_probe.c` and
`-DDEEPFIN_BEND_NATIVE_MODEL`; do not use that test backend in the application.
Emit/compile `logit_probe.bend` and `buffer_logit_probe.bend` without that define.
Then run:

```sh
python native/bend_engine/standalone/verify_buffers.py \
  --transport /path/to/buffer-probe --array-logits /path/to/buffer-logits \
  --list-logits /path/to/list-logits --report /tmp/buffers.json
python -m native.bend_engine.standalone.verify_neural \
  --checkpoint /path/to/fixture.pt --package /path/to/fixture.pt2 \
  --oracle /path/to/cboard-reference --report /tmp/neural.json \
  --command /path/to/deepfin-bend-neural --threads 1
python -m native.bend_engine.standalone.verify_reuse \
  --package /path/to/fixture.pt2 --reference /tmp/neural.json \
  --report /tmp/reuse.json --command /path/to/deepfin-bend-neural --threads 1
```

No permanent workflow, recurring model export, automatic native traversal, binary,
model package, or raw trace is part of this change.

## Completed qualification and readout — September 22, 2026

Executable feature commit: `0958be53ea09a1fbe414a34bb9cde97a7eb13777`.
The later readout edit changes only this document. The feature stacks on #828 at
`ba18279fffdb13beb894e1d4f49cd60601f38583`; nothing merged or deployed.

The initial hosted run [35765779079](https://github.com/jjoshua2/DeepFin/actions/runs/35765779079)
passed the full build, whole-repository and explicit native-verifier static gates,
89 accounting/benchmark/broker tests (no failures/skips), both compiled-probe modes,
original material UCI/perft and actual neural oracle. It then failed because the
new reuse test inherited a client that rejects *any* stderr, including its own
requested audit. The quiet run had already observed 42 forwards with zero input
or output address changes and one bridge input-tensor allocation. No candidate
was published from the failed run.

Corrected the new test with a narrow audit-aware subclass: only a single precisely
formatted audit line and zero process exit are allowed. The original client and
its other tests remain untouched. Six fault controls reject nonzero exit, missing,
malformed, extra or duplicate stderr. A two-line bridge follow-up also expresses
output offsets as byte offsets rather than typed float-pointer arithmetic into
packed scalar storage.

Final [run 35767013700](https://github.com/jjoshua2/DeepFin/actions/runs/35767013700),
job `106879001747`, is green. It verified all source hashes, reused the unchanged
Bend-generated C from the first build, rebuilt/relinked the final C++ bridge, and
reran whole-repository/native lint and actual-model checks. This was a bounded
recovery run, not another Bend code-generation or model-export attempt.

### Functional evidence

- Full checked compiler/build: Bend 2.0.21 + U64 at
  `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, Clang 18.1.3.
  Locked Python 3.13.15 / Torch 2.14.0+cpu, two Torch threads, one Bend thread.
- Final whole-repository Ruff/Basedpyright/Vulture and explicit native verifier
  lint pass with zero type errors/warnings. The unchanged accounting/benchmark/
  broker code passed 89 pytest cases in the first run.
- Normal and UBSan probes each pass 16 buffer round trips, seven rejected
  transport/configuration cases, 11 bit-identical list/array cases and four
  additional capacity/needs rejections. These probes did not change in recovery.
- Original material regression: 10 fixture positions, 137 exact children,
  51 searched roots, 23 rejected transactions, readiness/single-stop checks and
  perft 8,902 / 97,862 / 43,238. Its binary did not change in recovery.
- Final neural oracle: 18 searches, 50 traced forwards/reconciled replies,
  1,141 legal priors, six automatic-draw replies and four zero-forward terminal
  searches. Maximum raw-logit absolute error is 5.364418029785156e-7 and maximum
  probability error 2.9802322387695312e-8, within unchanged predeclared tolerances.
  Same-board/different-history inputs stay distinct; three startup failures fail closed.
- Final reuse verifier: 32 searches across quiet and diagnostic modes, 42 forwards
  in each process. Both show zero input/output address changes and exactly one
  bridge input-tensor allocation. Quiet/diagnostic raw input/output traces are
  byte-identical. Repeated roots, invalid position rollback and ucinewgame preserve
  the cache correctly; legal entries and history-dependent tensors do not leak.

UBSan evidence is for the small compiled transport/numeric probes, not the whole
LibTorch/model stack. No-Python chroot isolation was not rerun; ldd shows no
libpython dependency. No trained production weights or GPU execution were tested.

### Bounded performance screen

The final run preregistered a descriptive screen before executing it, retained as
`screen-preregistration.txt`: same saved model package, four full-history positions,
four blocks with alternating engine order, one excluded warmup per process,
32 real evaluations and 100 ms wall time, profiling enabled. Three arms separate
storage/transport changes from per-leaf diagnostic suppression. No speed or Elo
threshold was used to declare functional qualification successful.

Hosted hardware: four logical CPUs on AMD EPYC 9V45. There are only four distinct
positions and four blocks; these are not 16 independent chess-strength samples.

| Arm | Fixed-32 observations | Mean command-to-bestmove | Accepted rows / external decision second |
| --- | ---: | ---: | ---: |
| PR1 saved executable, diagnostics on | 16 | 36.76 ms | 870.59 |
| PR2 reusable storage, diagnostics on | 16 | 32.12 ms | 996.28 |
| PR2 reusable storage, diagnostics off | 16 | 28.67 ms | 1,116.08 |

Every fixed-evaluation observation meets the budget/comparability checks and all
paired bestmoves agree. Each arm executed/accepted 512 real rows. Aggregate external
decision time decreased 12.62% with diagnostic output held constant, and 22.00%
with diagnostics off. Corresponding external-denominator throughput increases
are 14.44% and 28.20%. These are descriptive fixture measurements, not production
speed guarantees, statistical significance or Elo estimates. They do not use the
internal EPS denominator to hide command preparation or result-output latency.

**The fixed-wall-time comparison did not qualify.** The existing 4,096-node arena
stopped all 16 quiet runs before 100 ms (68–98 ms internal time), seven diagnostic
reuse runs and one baseline run. Other runs exceeded the preregistered 10 ms
external overrun allowance. Only 3/16 baseline and 5/16 diagnostic runs were
individually comparable; quiet had 0/16. The driver correctly returned code 2.
All 96 observations, including failures to reach the wall budget, are retained;
no fair three-arm fixed-wall-time improvement or strength conclusion is claimed.
A larger/adjustable, separately qualified arena and deadline/output work are needed
before that measurement becomes informative. This PR does not alter those limits.

### Evidence and limitations

- First artifact `10712880174`, `deepfin-pr2-reuse-validation`, holds original
  build/probe/material results, 89-case JUnit, initial source manifest and failed
  audit-test readout. ZIP SHA256:
  `b5c1bc851b57ea24b1eaf4f9599773e9cbff764f11563ad45c056fe5669a2632`.
- Final artifact `10712263467`, `deepfin-pr2-reuse-final`, holds final source
  manifest/patch, lint, six audit controls, neural/reuse reports, all benchmark
  observations/configuration/corpus, screen script and final executable identities.
  ZIP SHA256: `9ca394931555faede07b6242034b7982bac5a65bc70d68c5d9f5895533780de9`.
- Qualified patch SHA256:
  `09d4530bb931265aa9585eab7098d5149602a513edea32fb9441e4dfb9a931af`.
- Final executable SHA256:
  `97cbf10c4c336bd109e4a95a1cd8b5fc193a1311efac2bbad45e295c280c0661`.
- Exact saved package from PR1:
  `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`;
  untrained 5,043,005-parameter CPU-F32 batch-one fixture. No re-export.

Artifacts have 30-day retention. Raw neural traces and trained/private models
are not uploaded in this PR's artifacts. Temporary workflow/payloads and generated
binaries are absent from the feature diff. Self-reviewed, not independently
reviewed or formally proven. The model backend remains transitional native
LibTorch/AOTI; this is not Bend-authored neural-network math or asynchronous batching.
