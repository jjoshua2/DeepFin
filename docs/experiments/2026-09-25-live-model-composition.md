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

## Completed readout — September 25, 2026

**[Run 36170337509](https://github.com/jjoshua2/DeepFin/actions/runs/36170337509),
job 108188042803, completed every stage successfully.** Staging commit
`b15626b14843b727836c67a2152c46a25635d378` plus the exact three-file manifest in
the artifact identify the tested sources. Publication applies the complete tested
four-file patch on #884 at `52175f1`, verifies its hashes and adds only this
readout, documentation links and compact evidence. No runtime, compiler, model
architecture, weights, production setting, merge or deployment is changed.

The previously open composition gate is now passed for this specific CPU fixture:
the actual persistent Bend owner calls the existing native asynchronous worker and
LibTorch/AOTI model across successive root generations and shared-root forwards.
This is not a substitute Python search or the earlier deterministic callback.

### Observed results

| Case | Physical batch | Generations | Real/accepted rows | Forward calls | Padding rows | Legal priors / final nodes compared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential reuse | 1 | 14 | 34 | 34 | 0 | 670 / 485 |
| Sequential reuse | 4 | 14 | 34 | 34 | 102 | 670 / 485 |
| Two live roots | 1 | 2 | 7 | 7 | 0 | 231 / 154 |
| Two live roots | 4 | 2 | 7 | 4 | 9 | 231 / 154 |

Sequential reuse completes each root before installing the next, including an
explicit removal/re-add at generation eight. It tests persistent model and buffer
lifetime, not batching efficiency: its batch-four calls intentionally have three
padding rows each. The two-root batch-four run actually dispatches `[1,2,2,2]`
real rows, demonstrating shared model calls. Fewer calls do not establish faster
execution; no timing or playing-strength deciding metric was measured.

Each row above has a separate diagnostics-off repeat, for **eight native model
processes and 164 real executed rows** in the completed run. The **82 traced rows**
pass independent CBoard feature-bit, raw-logit, legal-policy/WDL and complete tree
checks; quiet runs match per-generation outcomes and physical-work counts. There
are **1,802 legal-prior and 1,278 populated final-node comparisons** across the
traced cases. Reused fixture searches are not independent games.

The model is the saved untrained **5,043,005-parameter, 175-plane CPU-F32** checkpoint.
Maximum absolute raw-logit error is `5.960464477539062e-7` in both reuse cases and
`4.76837158203125e-7` in both shared cases. Inherited tolerances remain 2e-6 absolute /
2e-5 relative for raw logits and 2e-7 / 3e-6 for legal probabilities. Existing
history, rule-draw, terminal, promotion and en-passant fixtures remain authoritative.
Each reuse case preserves six automatic-draw replies without fabricating neural
execution for them. Nonfinite mutations of the captured native output are rejected
by the same independent oracle after its unchanged nominal trace passes.

Per-generation results agree exactly across batch one/four in these fixtures.
**Full real-model tree bit hashes differ across batch sizes**; every populated field
passes its numerical/reference check. Do not promote this to arbitrary-checkpoint
bit identity. Original model trace bytes, paths, replies and node fields are never
replaced by reference-generated values: only `(slot,generation)` diagnostic identity
is projected into unique oracle ordinals, with strict stream/lifetime checks.

All eight native processes report stable bridge input/output addresses and exactly
one bridge input-tensor allocation. That observation does not cover internal AOTI/
LibTorch allocations or establish a whole-process memory or throughput improvement.
The native products have no libpython link; dependency inspection is not chroot
isolation. Python is used for test control and the independent oracle, not native
search or model execution.

### Static checks, failures and resource use

Whole-repository Ruff/Basedpyright/Vulture and explicit verifier lint pass with zero
type errors or warnings. **477 focused Python tests pass, including 45 new cases,
with no failures, errors or skips.** Local corrected parser tests pass 161 cases
(45 new plus 116 existing live cases); local Ruff and the explicit-typeshed
Basedpyright invocation pass. The default local type-checker launcher initially
lacked its standard-library stubs; no source suppression was used. A local direct
fixture import lacked the built CBoard extension; hosted qualification supplies
that native dependency rather than substituting a mock oracle. Two full local
14-generation callback transcripts also pass the corrected identity projection;
they are not additional model runs.

The first [run 36169578809](https://github.com/jjoshua2/DeepFin/actions/runs/36169578809)
passed its static gates and executed the first model reuse session, but its new
projection rejected the legitimate `pending-remove` acknowledgment. Added a checked
pending-removal state plus eight positive/negative removal cases: premature, stale,
duplicate, missing and unfinished handshakes cannot silently pass. No engine change,
removed scenario, reference change or numerical relaxation was needed. That run's
patch-retention step also failed because shallow checkout omitted the diff base;
the corrected checkout fetches history. Both failures remain visible; artifact
10879493006 retains the initial failed qualification report.

The deciding run reused the saved singleton package and exported one batch-four
package. **There were two batch-four exports across the two attempts**, not one in
total, because the first export was not retained after its failed run. The first
attempt's incomplete model session is not included in the completed-run counts
above. No trained/private checkpoint, GPU, training run or game arena was used.

Hosted environment: Python 3.13.15, locked Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2,
Clang/Clang++ 18.1.3, two Torch threads and one compiler job. The exact #884 generated
live C was recompiled and linked, **not newly generated by Bend**. Its source hashes
were checked against the earlier qualification manifest. Compiler selection and
all Bend/C/C++ runtime sources are unchanged. The preserved failed attempt and the
completed rerun are separate evidence, not favorable timing rerolls.

### Scope and remaining work

**Replacement/removal in this real-model gate occurs after the old root finishes.**
No live cancellation or deadline was injected while this model executed. Earlier
held-callback tests cover those transitions independently; this run does not turn
them into real-model cancellation evidence. It also does not cover trained-network
fidelity, CUDA/5090 execution, concurrent forward slots, self-play/replay generation,
adaptive dispatch, warmed throughput, fixed-wall strength or Elo. The live process
still has one physical slot and one pending lifecycle change. Graceful shutdown
coverage is not a guarantee for every fatal exit. Self-review only; no independent
review, new sanitizer run, race proof or formal proof is claimed.

The new verifier is opt-in; ordinary pytest exercises its cheap parser/control
cases, not model loading/export. Existing source-only callback/live CI remains
unchanged. For a fresh build use `build_live.sh` with the exact trusted package;
the numerical gate accepts the resulting binary, checkpoint and legal oracle:

```sh
python -m native.bend_engine.multi_root.verify_live_model \
  --binary /path/to/deepfin-bend-live --package /path/to/model.pt2 \
  --checkpoint /path/to/checkpoint.pt --oracle /path/to/legal-oracle \
  --report /tmp/new-live-model-report.json
```

### Evidence identities

Compact reports and provenance are committed under
`docs/experiments/evidence/live-model-composition/`. Artifact **10879714453**,
`deepfin-live-model-composition`, expires October 25, 2026. It retains exact source
patch/hash records, model/binding reports, JUnit, lint and dependency evidence,
**not model weights, binaries or raw traces**. Raw trace hashes are in the reports;
new reproduction requires the trusted model/fixture and matching toolchain.

- Artifact ZIP SHA256: `6109b78bafa187af65b04fe58c417740e7d0dcbfe5e2c8e2e02b6a0cdb8c2479`.
- Qualified patch SHA256: `cfbbd644f7cc71b873b190ec77e6181a16f30c00fb714010655f5da2d7aad028`.
- Generated live C SHA256: `0febc12b7a6a8a27ddf894cf7bfdc01115413f2792a3be5c93fa814ba610e471`.
- Batch-one binary SHA256: `91544d8b8bc5875df61425a3d6b1f7154e545159c03f5beef00b27d394ed4c7d`.
- Batch-four binary SHA256: `1c7ffbf25884324ff915ded34f073b231c133f601e134c1d3962a5d71202aae5`.
- Batch-four package SHA256: `36b1f6a8da1456e942e5047e99eedd0d2eaa28fef896b34129fc5331c774c4e9`.
- Checkpoint and singleton-package SHA256 identities remain those preregistered above.

Ordinary PR checks on the published follow-up are separate from this completed
model qualification. Nothing was merged, deployed or changed in live training.
