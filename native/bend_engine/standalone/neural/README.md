# Opt-in Bend-owned neural leaf evaluation (CPU)

This entry point connects actual native search tickets to a **complete Bend-built
input and legal policy list**, a persistent native AOTI model, and **Bend-owned
legal-policy/WDL softmax**. It reuses the standalone UCI parser/state, chess, rules,
history, search and result selection. The material executable remains the default.
It does not launch, embed or call Python, NumPy or CBoard.

The narrow native adapter copies raw float32 words, verifies/copies the exact
build-bound model package, pads one real row to its declared static batch, executes
LibTorch/AOTI and returns raw policy/WDL logits. It does not decide legal moves,
chess outcomes, UCI commands, probabilities, scheduling or backups. Model kernels
are still compiled from PyTorch: this is **not** Bend-authored neural computation
or a Python-free training/export pipeline.

## Build and run

First retain a trusted CPU float32 package and its `.json` manifest from the
existing `neural_probe.checkpoint_probe` qualification. That external Python
export/qualification is transitional build tooling, not an engine runtime.
The new build does not itself export a model or infer its configuration.

```sh
bash native/bend_engine/standalone/neural/build.sh \
  build/bend_native_model /absolute/path/checkpoint.pt2 /absolute/path/libtorch/share/cmake
DEEPFIN_BEND_PACKAGE=/absolute/path/checkpoint.pt2 \
  ./build/bend_native_model/deepfin-bend-neural --threads 1
```

`LIBTORCH_CMAKE_PREFIX` may also point to the `torch/share/cmake` directory of an
existing CPU development installation. Neither Python discovery nor an interpreter
is invoked by the build. Bun, Clang/Clang++, CMake, OpenSSL development files,
LibTorch and shell/Git tools are build prerequisites. Optional arguments four/five
select the source-verified compiler checkout and generic/portable/native/UBSan mode.
Existing output paths are refused; previous compilers and binaries are untouched.

Supported metadata: v3 checkpoint package, compact policy1858, CPU float32,
row-independent fixed batch1/2/4/8/16, root/root-legacy-meta history, v1/v2_threats,
corrected repetition. The matching package SHA, configuration, checkpoint identity
and Torch version are retained at build time. The native library checks its release
number; same-build ABI/hardware compatibility still must be qualified on the target.
This does not authenticate arbitrary code or establish that an untrusted manifest
accurately describes a package. Only trusted, matching exported packages are valid.

The package is opened as a regular non-symlink file, copied into a private temporary
directory while hashing, and only the matching copied bytes are loaded. Native
input guards are enabled (`AOTI_RUNTIME_CHECK_INPUTS=1`); the tested exporter emits
them. Actual input size/finiteness and both output shape/device/dtype/finiteness
are checked. Failure terminates with stderr and nonzero status, **not** a material
fallback or fabricated bestmove. Guard availability and output compatibility must
be requalified for a different exporter.

## Ownership and limitations

Bend reconstructs each selected leaf's actual played-plus-search history and
adjudicates automatic draws before inference. It prepares both inputs and indices
from that Game and checks their exact move-key order against the outstanding
native ticket. It validates all1861 returned words, gathers only current legal
slots and uses max-shifted softmax; a model-predicted draw remains nonterminal.
The model and generated maps persist across normal searches and position changes.
No leaf-only FEN cache, eager reference search or external controller is involved.

Inference and encoding are **synchronous**. `isready`, `stop`, EOF and quit are
processed between search steps, not during a blocked native call, initialization
or output. There is no hard-stop deadline or asynchronous cancellation guarantee.
No cross-search batching, CUDA, subtree reuse, production Gumbel parity or optional
claim action is added. Search/UCI bounds are inherited from the standalone engine.
The shared result label now says experimental PUCT rather than incorrectly naming
the material evaluator when this opt-in backend is active.

Deployment is **not the single static-file material deployment**. It needs the
executable, matching native libraries/loader, model package and writable extraction
space. Tests must enumerate that native dependency set and keep all Python clients
and references outside it. CPU feature data may be supplied as ordinary platform
data; no interpreter, libpython or Python controller belongs inside.

## Opt-in external validation

Set `DEEPFIN_BEND_TRACE` to a NEW file to capture actual input/raw-output records.
Bend also emits identity-bound probability traces. Tracing is absent by default;
it is diagnostic output, not a supported external inference protocol.

```sh
DEEPFIN_BEND_PACKAGE=/absolute/path/checkpoint.pt2 \
DEEPFIN_BEND_TRACE=/tmp/new-bend-trace.bin \
python -m native.bend_engine.standalone.neural.verify_neural \
  --checkpoint /absolute/path/trainer.pt --package /absolute/path/checkpoint.pt2 \
  --trace /tmp/new-bend-trace.bin --report /tmp/new-bend-report.json \
  --command ./build/bend_native_model/deepfin-bend-neural --threads 1
```

The verifier uses independent Python chess transitions, the unchanged C/Python
encoders, eager singleton model outputs, probability conversion and the existing
reference PUCT. It does not feed reference tensors or probabilities into the
candidate. This trace gate checks requested ancestors, visit accounting and chosen
moves, not an independent dump of every internal node's final statistics.
`response_check.bend` and `bind_model.test.js` cover malformed-response/build
contracts separately. All native execution remains opt-in; no deeper routine
perft, additional recurring native job or model export is installed by this feature.
