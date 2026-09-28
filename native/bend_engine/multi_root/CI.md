# Completion and cohort regression CI

The completion-notification speedup was qualified in PR #870, but the ordinary
CI and Bend chess probe did not compile its new native wait tests or exercise
its held-control regression. `.github/workflows/bend-completion.yml` makes these
checks persistent on relevant pull requests and main pushes. Manual dispatch
is available after the workflow lands. Permissions are read-only; no historical
Actions artifact, baseline binary, GPU, model export or benchmark is required.
The existing opt-in performance/comparison scripts remain separate.

## Native gate

```sh
CXX=clang++-18 python -m native.bend_engine.multi_root.ci_completion native \
  --output artifacts/NEW-completion-native
```

This compiles the real native worker and completion tests in normal and
ASan/UBSan configurations. JSON coverage is checked, not just the exit code.
The deliberately no-notification build must exit with the wakeup diagnostic;
a crash, timeout, missing executable or empty report is not accepted instead.

The completion-test harness releases its deliberately held callback before
joining the worker during exception unwinding. An injected assertion while
held must exit with its expected message in both builds, not hang on the
worker destructor. This changes test cleanup only, not application shutdown.

## Actual coordinator gate

Activate the repository's locked CPU environment and install its pinned compiler:

```sh
bash native/bend_engine/install_ci_bend.sh
revision=$(python -c 'import json; print(json.load(open("native/bend_engine/standalone/toolchain.json"))["revision"])')
BUN=bun CC=clang-18 CXX=clang++-18 \
  python -m native.bend_engine.multi_root.ci_completion cohort \
  --compiler-root "build/bend_toolchain/sources/$revision" \
  --output artifacts/NEW-completion-cohort
```

The gate checks the compiler fingerprint, generates C from the current Bend
coordinator, and calls the existing `qualify_async.sh` and `qualify_deadlines.sh`
without changing their oracles or tolerances. This includes the ten batch/width
configurations in sync/async modes, coordinator UBSan, held cancellation/stop/
quit, and deadlines. The previously caught buffered quit acknowledgement is
therefore exercised on the actual program, not a source-text approximation.
Every one of the 33 expected verifier reports is required; empty directories
and partial matrices cannot produce a passing summary.

Each command retains stdout, stderr, status and duration. Failures retain a
failed summary; a timed-out command kills its own process group, including
compiler or test children. Use a fresh output directory so old evidence cannot
be silently overwritten or mistaken for new results. CI uploads compact logs,
reports and compressed generated C, not binaries. Repeated modes reuse the
same fixtures; sanitizer coverage is not a proof for all interleavings.

No throughput assertion is made or gated here. The earlier callback performance
observations retain their original source identities and limitations. This is
continuous regression coverage, not a new performance experiment or deployment.

## Arena regression extension

After the existing 33 default-capacity reports, `ci_completion.py cohort` invokes
`verify_arena.py` using the same freshly generated callback executables. It checks
fixed-array boundaries and ticket sentinels, invalid environment settings, three
compiled semantic mutations, and 37 real coordinator cases with quiet repeats.
An 8,192-node start-position search must both pass the independent numerical/tree
oracle and complete 256 simulations using more than 4,096 nodes. Missing/partial
arena reports fail the gate. No historical artifact input or throughput assertion
is added. The default coordinator/cancellation/deadline matrix remains intact.
