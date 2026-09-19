# Checkpoint-owned Bend evaluator qualification

Unlike the seeded TinyNet commands, this entry point requires an actual supplied
`trainer.pt` with `model` tensors and embedded `arch` metadata. It never substitutes
random weights, searches unrelated params files, reads live YAML, or tolerantly
skips missing/shape-mismatched tensors. It does not infer that a file is trained
from its name or shape. Use a trusted, immutable copy of your checkpoint.

```sh
# In an isolated checkout/environment, not the running training checkout:
uv sync --locked --extra dev --extra cpu
bash native/bend_engine/bitboard_probe/install_toolchain.sh
uv run python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/trainer.pt --out-dir artifacts/bend-checkpoint-cpu \
  --device cpu --dtype float32 --batch 4

# On an appropriately provisioned NVIDIA host, after checking existing GPU jobs:
# Select the CUDA environment instead of the mutually exclusive CPU profile.
uv sync --locked --extra dev --extra cu130
uv run python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/trainer.pt --out-dir artifacts/bend-checkpoint-cuda \
  --device cuda:0 --dtype bfloat16 --batch 4
```

The output directory MUST NOT exist. Success writes `model.pt2`, its `model.json`
sidecar, and `qualification.json`. A failed export may leave a diagnostic directory
without a valid manifest/report; its existence is not a pass. The command owns a
separate temporary compiler cache. Nothing replaces production AOT packages or
rebinds their constants. This exports a **frozen single-checkpoint tuple package**,
not the existing dictionary-output, weight-rebindable `chess_bN.pt2` artifacts.
Those are intentionally not claimed interchangeable.

The saved configuration is authoritative. The normal strict architecture decoder
checks unknown/new schema fields; history, extra features, repetition fix and
policy format must be explicit. `lc0_root`/`lc0_root_legacy_meta`, v1/v2 features,
and compact lc0_1858 are supported. Bare state dicts, legacy history and models
requiring a separate dynamic-relations input are rejected. The only config override
is disabling gradient checkpointing for inference. All saved model tensors must
load strictly and be finite. Nonpersistent buffers are reconstructed by the pinned
project implementation. No Trainer or optimizer is instantiated.

## Device and artifact identity

CPU runs F32. CUDA requires an explicit local device index and either F32 or BF16;
an unavailable CUDA device is an error, never a CPU fallback. Float32 wire inputs
are converted on the selected device to the declared model dtype. The exported
wrapper returns F32 policy/WDL logits; native CUDA copies complete before reply.
The same explicit device index is passed to `AOTIModelPackageLoader`, using the
API already exercised by the existing CUDA parity bridge.

Version-three sidecars bind the checkpoint hash, effective full ModelConfig/hash,
Torch version, Torch CUDA build, C++ ABI, device/dtype, CUDA compute capability,
batch, input planes and output order. The loader verifies them before native
startup. Fingerprints provide identity, not a trust signature: AOTI packages are
native executable code. Do not load an untrusted package or edit a sidecar to
bypass a mismatch. Changing device/runtime/architecture calls for re-export and
requalification. Batch-one/v2 smoke packages retain their old CPU contract.

## What is tested

Five Bend searches preserve their individual histories and share the existing
bounded batch queue. One-real-row padded control, normal batching, and queued/
in-flight cancellation + restart all compare complete search structures/visits
and best moves. Every accepted reply is checked by the existing diagnostic PUCT
reference. The full model's actual policy/WDL outputs drive search. Existing
session deadlines, depth/capacity bounds and one request per tree are retained.

Two different comparisons are deliberately separate:

1. Native C++ versus Python execution of the **same fixed-shape package**, on the
   same device and input bits, must be bit-exact for every output including padding.
   Any discrepancy stops qualification; no tolerance conceals a wire/order/weight bug.
2. Each real row (even a cancelled one) is compared with independent **eager
   singleton** execution of the strictly loaded checkpoint. F32 retains the smoke
   logit gate of atol 2e-6 / rtol 2e-5. BF16 instead gates the maximum **per-row**
   legal-policy and WDL total-variation distances. Defaults are 0.01 for each,
   declared engineering limits, NOT calibrated playing-strength equivalence or a
   substitute for the production AOT numerical gate. `--policy-tv-limit` and
   `--wdl-tv-limit` are explicit and recorded. Absolute logit error is also reported.

These tests assess checkpoint-specific compilation, batching, and input/output
composition. A numerical pass is not proof of strength, production Gumbel parity,
or an end-to-end speedup. Tiny differences in near-tied decisions can still fail
strict search-control comparisons; investigate rather than loosening them silently.
This diagnostic PUCT still lacks full draw adjudication and root advance. Repeated
history fixtures exercise encoding, not real-game draw termination.

## Evidence and cost

The explicit CPU fixture command creates a temporary, **untrained reduced
transformer** (32-wide, two layers, four heads) with split QKV, per-layer Smolgen,
relation bases, ARC position adapter and DeepNorm. It saves/reloads actual weights
through this same interface, including a deliberate parameter change that a fresh
same-seed reconstruction would miss. It does not download or fabricate trained
weights or run an optimizer:

```sh
uv run python -m native.bend_engine.neural_probe.checkpoint_smoke \
  --report artifacts/bend-checkpoint-transformer-smoke.json
```

This is not the full production model size. Its default Bend modes are native and
UBSan; `--modes generic portable native ubsan` qualifies all four. UBSan covers
Bend, not LibTorch/model kernels. Actual CUDA and the user's full trained artifact
require the explicit host command; CPU compilation cannot establish GPU execution.
Export and validation hold multiple model copies (eager + Python package + native
package), so budget memory and compile activity alongside existing jobs.

No checkpoint smoke, model compilation, inference, search or perft benchmark is
added to ordinary pytest or recurring workflows. Only small loader/metadata
contract tests are added there. Existing perft depths and compiler pin are unchanged.
