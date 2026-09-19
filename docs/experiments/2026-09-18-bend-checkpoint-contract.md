# Bend: explicit checkpoint and device contract

## Decision

Follow #779 on `4ab0836c69f7d60ae41c84fa069527a8c731f047`. Stop adding synthetic
scheduler features; make the existing search/batching path accept a specified,
self-describing checkpoint without random fallback or another compiler change.
The user need not rewrite C encoding or C++ inference in Bend.

No trained .pt/.pt2 artifact was present in the accessible repository snapshot;
the repository has no published releases. The local runtime is Torch 2.10.0+cpu
with no CUDA device. Therefore a trained-model/GPU execution claim is unavailable,
not replaced by presenting smoke weights as a trained checkpoint.

## Implementation / scope

Strict embedded architecture + state-dict loading, explicit history/policy
identity, frozen tuple-output package, runtime/dtype/CUDA-device identity checks,
and full-package bit parity precede interpreting the search result. The native
worker supports CPU F32 and explicitly selected CUDA F32/BF16. CPU mode keeps
prior package formats compatible. The existing bounded coordinator gains an
optional full-batch comparator; its ordinary smoke path is unchanged.

The checkpoint-specific command compares native and Python package outputs exactly,
then compares real rows to the original eager singleton model. Default maximum
legal-policy/WDL TV limits are 0.01 (engineering bounds, not strength calibration);
F32 also retains atol 2e-6 / rtol 2e-5 logits. Full control/batched/cancelled search
state agreement remains required. No benchmark or production Gumbel claim.

No live checkout, configuration, checkpoint, deployed package, compiler, chess core,
perft depth or routine native-test schedule is changed. New output directories
must not exist. A failure never publishes a qualification report.

## Local exploration

A temporary untrained reduced transformer has 32-wide/two-layer/four-head trunk,
split QKV, per-layer Smolgen and relation bases, ARC adapter and DeepNorm. It is
saved with full checkpoint metadata, then strictly reloaded. No training performed.
Native + UBSan Bend modes passed 34 search epochs (30 normal and four cancelled),
250 real inference rows. Native C++ outputs were bit-identical to Python loading
the same package. Maximum absolute eager logit error was 4.470348358154297e-7;
max legal policy TV 5.21540641784668e-8; max WDL TV 1.4901161193847656e-8.
All cancellation/recovery/control search comparisons passed.

An initial local orchestration timeout interrupted export; it produced no result
report and is not counted as successful validation. A subsequently bounded run
completed. No numerical threshold was relaxed. Focused Ruff and 113 small tests
passed after scalar guard/import cleanup; no model forward in ordinary pytest.

## Hosted confirmation plan (before hosted execution)

One bounded 15-minute CPU confirmation on the locked project environment. Run
Ruff/Basedpyright and focused cheap tests. Retain original four-mode session,
batch-one neural, and batch-four neural checks after changing the native worker.
Run the new reduced-transformer checkpoint smoke in all four Bend modes. Every
native/Python same-package output must match exactly, eager differences must pass
the stated gate, and complete search states must match. No timing acceptance gate.
Do not download trained weights, rent GPU capacity, or launch training.

Recovery: discard the isolated branch; preserve all prior PRs. CUDA remains an
explicit unexecuted gate. Self-review only unless an independent review is recorded.

## Hosted readout: PASS

[Confirmation run 35410207092](https://github.com/jjoshua2/DeepFin/actions/runs/35410207092),
job **105808161649**, passed every validation and publication step. The exact
executed implementation was published as clean commit
`733866cb98c1f07c34961bab4f77d1ee659a1a83`, directly on #779's head. This later
readout modifies documentation only. All eleven feature files were SHA-256
checked against the local final source before execution.

Locked Torch **2.14.0+cpu**, Bun 1.4.2, Python 3.13.15. The saved model is explicitly
an **UNTRAINED reduced transformer**, not the full production network or a trained
artifact. Checkpoint ownership is exercised by saving a deliberately changed
parameter; the loader never regenerates weights from a seed.

All four Bend CPU builds passed: generic C, forced-portable U64 helpers,
native-target C, and UBSan. The new transformer qualification ran **68 epochs**
(60 normal completions, eight deliberately cancelled partial epochs), **500 real
input rows**, and **333 native batch calls**. Every native output was bit-exact
against Python loading the SAME package, including padded output rows. Every real
row was also compared with independent eager singleton inference.

| Maximum discrepancy | Observed |
| --- | ---: |
| Native versus Python same-package values | 0 (bit-exact) |
| Compiled versus eager absolute logit error | 4.172325134277344e-7 |
| Legal-policy total variation | 5.029141902923584e-8 |
| WDL total variation | 1.4901161193847656e-8 |

Full tree structure, node boards, visits and best moves matched one-real-row
controls; per-reply statistics matched the diagnostic reference. Both cancellation
cases kept exactly two completed simulations, and their new epochs recovered.
Each mode exercised real batch fills 1/2/3/4. Calls and arrival-dependent fills
are correctness observations, not a throughput speedup.

| Mode | Normal batch calls/rows | Fault/recovery calls/rows | One-real-row calls/rows |
| --- | ---: | ---: | ---: |
| Generic | 22/40 | 20/45 | 40/40 |
| Portable | 22/40 | 20/45 | 40/40 |
| Native | 22/40 | 20/45 | 40/40 |
| UBSan | 22/40 | 25/45 | 40/40 |

The existing regressions also passed after the worker changes:

- 38 persistent-session cases per mode, 207 python-chess oracle positions;
- 40 batch-one TinyNet epochs / 640 native calls, 64 independent encoding checks
  plus the same-board/different-history case, six malformed native inputs;
- 68 batch-four TinyNet epochs / 500 real rows / 341 native calls, six malformed
  batch-four inputs and all cancellation/control checks;
- Ruff, Basedpyright and **113 small tests** (33 new checkpoint contracts plus
  80 existing batching/boundary/session cases).

The first hosted attempt stopped at static validation before native execution:
a hash helper was newer than the configured Python target, Torch device annotations
required a type-based CPU/CUDA branch, and the final report needed an explicit
type. Streaming hashing and the annotations were corrected without suppressions,
removing checks, changing model weights, or relaxing numerical thresholds.

Evidence: artifact **bend-checkpoint-confirmation**, ID **10574046851**, 30-day
retention. ZIP SHA-256:
`b0a7e0c76665463f74fb5338c4668be2de0a473b15a44ae95db9d448bdb042ac`.

| Report | SHA-256 |
| --- | --- |
| Transformer | `d696c1642f5bb3fea874ec1fbdc912745ac56b5d846994640fb127d8a4508dbe` |
| Session regression | `6cf81318c6835d0391b5ef58fdc561dd25113150bf4bb986f5f3c98eb2b5c1c6` |
| Batch-one regression | `56b017de4c7bf6e8318701cd3699d1efb2b948cc320451d728497c92cecb5388` |
| Batch-four regression | `a2708418b9d0eb4886bb59816f8831191c52056957624aa2e18d2bc90499ae82` |

Executed checkpoint SHA-256:
`8142a2331fd1d6ed6a332e6d723aabcb6faba29442b16359bde1b136535e028d`.
Executed package SHA-256:
`6cad044b1dfe53b75b4b4ba052f9c7e6bc2e953af39cd04ad4042b6c7bcdcc22`.
ModelConfig SHA-256:
`188f1231306b343ce08947299ada2c0c94bf28716096593ffa0bc0b12ab815c7`.
Packages/checkpoints were disposable smoke artifacts, not committed. Archive hashes
identify this execution; re-export is not a reproducible-archive-hash promise.

## Remaining decision and review

The report explicitly records `cuda_executed: false`. The CUDA/BF16 route is
implemented and CPU-compiled, but **actual CUDA, a full-sized trained checkpoint,
and end-to-end throughput remain unqualified**. The next decisive test is the
explicit supplied-checkpoint command on an isolated appropriately provisioned
host, not another synthetic scheduler feature. This freezes new v3 packages;
it does not adopt or rebind existing production `chess_bN.pt2` files.

No new recurring model-export/search workflow or perft-depth increase. Temporary
confirmation files are absent from the review branch. No production merge or
deployment. Self-review only: strict architecture/state loading, checkpoint/package
identity, dtype/device routing, comparator separation, cancellation and resource
cleanup inspected. No independent reviewer, formal proof or playing-strength claim.

Commands, artifact semantics and remaining limits:
[`neural_probe/CHECKPOINTS.md`](../../native/bend_engine/neural_probe/CHECKPOINTS.md).
