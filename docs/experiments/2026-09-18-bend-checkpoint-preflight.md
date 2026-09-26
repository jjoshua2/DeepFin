# Bend real-checkpoint readiness, artifact retention and reuse

## Decision and scope

Follow #781 (`d7517e5823d5b01239fae71472907e174ec591fb`) without another language,
search, perft or synthetic-model milestone. No connected CUDA host or trained
checkpoint is available in this environment. The next real experiment should not
export before discovering a broken toolchain, erase its evidence on failure, or
require recompiling a successful model package just to retry a later stage.

Inspection found all three in the initial runner: source verification occurred
inside Bend compilation after export; its unconditional TemporaryDirectory removed
exports on both success/failure; native-start failures closed the stderr temporary
file before the caller could read the underlying loader diagnostic.

## Change

Add an explicit no-forward preflight, optional NEW retained work directory and
exact-checkpoint package reuse. Preserve the default disposable behavior. Reuse
validates package hash/version, checkpoint identity/state selection, resolved model
config, encoding, bucket and target, then reruns all numerical/search checks.
Progress records distinguish preflight readiness from actual qualification. They
retain prior completed groups and bounded native stderr, record per-stage elapsed
time, and restore the calling process's compilation-cache environment. No model,
Bend, CBoard, math tolerance, test depth or trained/GPU qualification is changed.

Preflight strictly loads and checks weights on CPU, verifies the pinned compiler
and required tool executables, and checks the requested target. It does not export,
execute a model forward, compile/start a native worker, or run search. CUDA queries
may initialize a context. A free-memory snapshot is neither a reservation nor a
prediction of peak compilation/inference demand; the caller still needs a safe
compute window. Executable discovery does not guarantee a later build succeeds.

## Confirmation plan (before hosted execution)

Keep the current saved-transformer fixture purely as regression evidence, not a
new model milestone. Export from one immutable fixture checkpoint in a retained
directory, run the existing control/batched/cancellation checks, then reuse the
same package from the same checkpoint in a NEW directory. Both must pass unchanged
CPU logit tolerances and complete search comparisons. Assert identical per-root
structural outcomes; native call counts may vary with batch arrival timing.
Verify the package hash is unchanged and the second report has no export stage.

Exercise the no-forward preflight, all old cheap contracts plus new identity,
workspace, path-alias, early-failure, atomic-report and stderr-cleanup cases. A
changed checkpoint/package/batch/state selection must fail before native execution.
Retain one deliberate post-export failure and verify package survival. Record
preflight as not_run, never as neural/CUDA evidence. Keep the original native
qualification, single-row/batching/session regressions when confirming the modified
shared startup wrapper. No added recurring exports/reuse runs: only cheap tests
join ordinary pytest; one bounded, temporary hosted confirmation handles the
new execution path. The permanent path-scoped workflow's depth/cost defaults stay.

Budget: one CPU hosted confirmation, 15-minute job cap, one compilation worker,
two inference threads. Additional attempt only to correct observed validation
findings. No GPU purchase/allocation, training, live process access, checkpoint
download, production merge or deployment. Recovery is discarding this isolated
branch, never resetting earlier work. Self-review only.

## Hosted readout: PASS

[Run 35419948876](https://github.com/jjoshua2/DeepFin/actions/runs/35419948876),
job **105835597821**, passed all stages, including source publication. The exact
validated non-workflow source commit is
`101b63106e01ab064bd30594de844a1f3b669392`. The final feature tree removes temporary
development files, applies the already-checked permanent workflow, and adds this
readout; executable source bytes are unchanged and match local blob hashes.
Compiler remains `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.

Locked Torch **2.14.0+cpu**, Bun 1.4.2 and Python 3.13.15. The checkpoint is the
existing **untrained** two-layer, 5,043,005-parameter transformer fixture, not a
trained production network. Its normal weights, encodings and model implementation
are unchanged from #781.

| Execution | Bend modes | Search epochs | Real rows | Native forwards | Maximum absolute logit error |
| --- | --- | ---: | ---: | ---: | ---: |
| Fresh export, retained workspace | generic, portable, native, UBSan | 68 | 500 | 332 | 5.364418029785156e-7 |
| Exact saved package reused | native | 17 | 125 | 82 | 5.364418029785156e-7 |

Together: **85 epochs, 625 real input rows, 414 native forwards**, including ten
deliberately cancelled partial epochs and 75 normal completions. Both paths pass
the existing per-reply numerical/board/tree checks and control/batched/recovery
comparisons. Fresh-native and reused-native epoch summaries are identical. Reuse
retains the identical package hash, has no export stage, and rebuilds its worker
and Bend executables before executing validation again. It is NOT a cached PASS.
UBSan covers Bend/chess, not LibTorch or the compiled model package.

No-forward preflight produced `status=preflight_passed`, `qualification=not_run`,
no groups, no export stage and no native calls. A changed checkpoint identity was
rejected at package verification before execution. A CUDA request on the CPU-only
runner failed at preflight with `no CPU fallback`; its report also says not_run.
Neither of these negative checks is a GPU pass.

Ruff, Basedpyright (zero errors/warnings), and **148 cheap tests** passed: 26 new
readiness/reuse/report cases plus 122 inherited checkpoint/batch/neural/session
contracts. Tests demonstrate package retention after a simulated C++ build failure,
source/package/manifest/planned-artifact alias protection, refusing an existing
work directory, atomic JSON replacement, cache restoration on failure, no execution
in preflight, and preserving stderr while reaping a failing native process.

The initial hosted attempt passed all 148 tests and Ruff but stopped on two
Basedpyright warnings about Iterator annotations on generator context managers.
Explicit `Generator[..., None, None]` annotations fixed those without changing
runtime behavior, suppressing checks, or relaxing any tolerance.

The fresh export stage reported 26.803 seconds; the reused run has no export stage.
These are diagnostic stage durations, NOT an engine-performance comparison:
full-run modes and validation work differ, and the new code changes no search
algorithm or model kernel. No end-to-end throughput conclusion is drawn.

## Local regression readout

Torch 2.10.0+cpu / Clang 17 passed native fresh export and native package reuse,
each 17 epochs/125 real rows/82 native forwards, with identical epoch outcomes and
max absolute logit error 5.960464477539062e-7. The final 148 cheap tests and Ruff pass.

After the startup/report changes, all original regressions were rerun locally:
- four-mode session validation, 38 sessions per mode and 207 python-chess oracle positions;
- four-mode TinyNet singleton integration, 40 epochs/640 native calls, 64 encoding
  checks plus different-history inputs, and all native negative cases;
- four-mode TinyNet batching integration, 68 epochs/500 real rows/337 native calls,
  all control/cancellation/recovery checks and six malformed native inputs.

The final path-scoped PR workflow retains those original native steps plus the
existing saved-transformer test. Only the new cheap-test path/commands were added;
there is no extra recurring preflight/export/reuse run. Broader PR checks are
separate from the successful focused confirmation, not assumed green.

## Evidence

Artifact **bend-readiness-confirmation**, ID **10577496412**, 30-day retention.
ZIP SHA-256: `3b4beb832bb6e19c679703e31ac4dcdb5cc72adc75becf60fd10167ff2404a3c`.
- Fresh report: `f0acfdb4c861595df356f69a6c85a4c1679dfe79be412e258b38c8ba0994a29e`.
- Reuse report: `c084cbe88d287b911010dbe9b21039f4447e5782c054e5079658be5f33e5ab0c`.
- Preflight report: `4f5eb741997050289b8de27e998c35dce06053b76d7db602249cfb57e043768d`.
- Checkpoint mismatch report: `32bb82bc4ae5d72d26663b2d8222fa12f5a6d415498ea5e6c082da3be171117a`.
- CUDA unavailable report: `fd165cd6a543b87525783b3c1169f392451e2c274644f17a3b25a5f5a1c49b45`.
- Executed/reused package: `97bc4edeced4fa7a398021ff85269849a90f2faca195332619c95042aaabca32`.
- Fixture checkpoint: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.

Only JSON reports/identities were uploaded, not model weights or native packages.
These hashes identify tested artifacts, not guaranteed reproducible archive bytes.

## Run and next decision

In an isolated checkout and prepared CPU/CUDA environment:

```sh
# Strict CPU-loaded checkpoint and compiler/device readiness; no inference.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/copied/trainer.pt --device cpu --preflight-only \
  --report artifacts/bend-preflight.json

# Export and validate, retaining artifacts for diagnosing any later failure.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/copied/trainer.pt --device cpu \
  --work-dir artifacts/bend-run-01 --report artifacts/bend-run-01/report.json

# Retest that EXACT checkpoint/package without recompiling the model export.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/copied/trainer.pt --device cpu \
  --reuse-package artifacts/bend-run-01/checkpoint.pt2 \
  --work-dir artifacts/bend-run-02 --report artifacts/bend-run-02/report.json
```

Work directories must be new. For CUDA use the existing explicit device index and
predeclared tolerance arguments on a compatible host in a safe resource window.
Preflight may initialize a CUDA context and is not a memory reservation or a fit
prediction. Only trusted immutable checkpoints/packages should be used; sidecar
hashes are integrity/identity checks, not authentication or hardware-portability
proofs. Reuse still reruns numerical validation and complete diagnostic searches.

No trained checkpoint or CUDA execution was available here. The next meaningful
result is that real target-host run, not another synthetic model feature. This
change adds no compiler feature, production Gumbel parity, UCI, root reuse, draw
adjudication, performance optimization or universal proof. Nothing is merged or
deployed. Review is self-review, not an independent review.
