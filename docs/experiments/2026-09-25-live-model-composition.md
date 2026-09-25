# Persistent live runner: actual CPU model composition

## Preregistration — September 25, 2026

Base: #884 at `52175f1bf2cb53732e774ed1e5ad32f8c3703748`. All four existing
PR workflows passed before this continuation. No runtime, search, model, compiler,
live configuration or production process changes are planned.

Hypothesis: the persistent Bend owner composes with the exact saved CPU-F32 AOTI
model across root replacement, explicit removal/re-admission and shared-root
forwards without changing leaf inputs, numerical results, final trees or physical
work. The earlier blocked-callback qualification is not actual-model evidence.

Control: saved untrained 5,043,005-parameter, 175-plane PR1 checkpoint, SHA256
`bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
Reuse its batch-one package, SHA256
`9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
Export at most one CPU batch-four package from the same checkpoint and encoding
using the existing exporter. No trained/private checkpoint, GPU, training or arena.

Deciding checks: fourteen sequential fixture generations in one process, including
remove/re-add, plus a two-root shared case. Run each at physical batch one and four,
with diagnostics/traces on and off in separate native processes. Require exact
lifetime accounting, stable native bridge buffers and one bridge input allocation;
all selected inputs, logits, legal policy/WDL and populated final tree fields must
pass the existing independent oracle. Keep raw-logit tolerances 2e-6 absolute and
2e-5 relative, and legal probability tolerances 2e-7 / 3e-6. No tolerance relaxation.
Require an observed multi-root physical forward at batch four, not only a tensor
shape. Mutated nonfinite trace output must fail the same oracle. Quiet results and
physical counters must equal diagnostic results. Failed checks stop publication.

The verifier maps emitted (slot, generation) identities into unique oracle root
ordinals. Original trace bytes, node fields, paths, reply values and counters are
not replaced with reference-generated values. This projection rejects cancelled,
deadlined, stale, missing or reordered generations; it does not certify cancellation
with real models. Existing deterministic cancellation evidence remains separate.

Budget: one hosted CPU lane, two Torch threads and one compiler job, eight native
model processes, at most one batch-four export. Reuse the hash-checked generated
live C from #884 because no Bend/C/C++ runtime changes are planned. Record source,
compiler, C, executable, model and report identities; do not mistake this for a fresh
Bend generation. No fixed-wall performance or strength metric is tested.

Only Python verification helpers/tests and this record are changed. Existing test
client defaults remain unchanged; optional environment and diagnostic arguments
support isolated model processes. Local parser checks are not native evidence.
The full repository lint gate and focused regressions run in the locked hosted
Python/Torch environment. Compact final reports should be committed; no model,
binary or raw trace should enter Git history. Self-review only unless another
actual reviewer is obtained. Nothing is merged, deployed or run on live hardware.

## Readout

Not yet qualified. Completed observations and remaining limitations belong here
only after the actual model and all deciding checks execute successfully.
