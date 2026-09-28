# Native fixed-batch boundary qualification

## Preregistration — September 22, 2026

Scope: PR3a, the native batched tensor boundary ahead of CUDA/BF16 qualification.
Base: PR2 #830 at `4bf90fed14b3ad0a192a4e93cc978b8f102a7b3c`.
Do not merge/deploy, change live jobs, alter search selection, or imply the GPU
part of PR3 is complete. No trained private model is used or uploaded.

Hypothesis: a single-owner native backend can reuse bounded Bend storage across
full and partial fixed batches without padding/row/layout contamination. No
throughput or strength hypothesis is tested in this slice.

Controls: existing singleton CPU package and saved PR2 generated engine source;
existing checkpoint exporter, model weights and encoding; independent eager
singleton outputs; all returned raw F32 bits compared to bridge trace. Full/partial
batches and repeated first rows use different padding. Normal UCI binding remains
batch-one-only. A separate ATen test invokes the actual packing helper with bad
tuple counts, shapes, dtype/device, null/invalid tensors and invalid row counts;
rejection must not partially write caller storage.

Acceptance: all declared bucket sizes and both input widths pass the bounded
transport test double (including failure propagation); real saved-model packages
at batch one and batch four pass full-logit comparison with predeclared 2e-6
absolute / 2e-5 relative tolerances; padding is exact +0; output tail is untouched;
buffer addresses stay constant and one bridge input tensor allocation is observed.
The unchanged singleton search path must pass the existing selected-leaf verifier
when linked against the modified bridge. No tolerance relaxation on failure.

Budget: one new batch-four export of the saved untrained CPU fixture, bounded
CPU-only native builds and 5 model forwards per batch probe, plus the existing
18-search singleton oracle. Two Torch threads, one compiler job and one Bend
runtime thread. No training, match arena, live GPU or model re-export at other
batch sizes. Small normal/UBSan test-double probes qualify transport, not LibTorch
or compiled-model internals. CI uses the locked Python 3.13/Torch CPU environment.

Recovery: failed gates do not publish a qualified feature branch. Preserve compact
reports and source hashes; fix the failing layer without regenerating unrelated
Bend application code or exporting a different checkpoint. Self-review only unless
an actual independent review is separately obtained.

## Readout — September 22, 2026

The preregistered functional acceptance rule passed. No speed, strength, production
checkpoint or GPU qualification was attempted or inferred.

[Qualification run 35771405323](https://github.com/jjoshua2/DeepFin/actions/runs/35771405323),
job `106893742241`, completed successfully on the exact executable-source feature
commit `3b6aa7415127ea6f24b8be6720ebc9838876f8cf`. The subsequent documentation commit
changes only this readout. The source manifest downloaded from the run matches
all 16 locally authored source/test/document files. Nothing merged or deployed.

Environment: locked Python 3.13.15, Torch 2.14.0+cpu, uv 0.12.10, Bun 1.4.2,
Clang 18.1.3, Linux x86-64. Two Torch threads, one compiler job and one Bend thread.
Compiler remains `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`;
its unchanged 84-file fingerprint is
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

### Completed checks

- Whole-repository Ruff, Basedpyright and Vulture passed, plus explicit native
  verifier checks: zero type errors or warnings. **113 pytest cases passed** with
  zero failures/skips, including 24 new cheap parser/failure controls and the
  existing accounting/benchmark/broker regressions. **65 Bun contracts passed**
  (101 assertions), including 32 new binding cases and unchanged singleton/pin tests.
- Compiled transport passed all **20 configurations**: normal and UBSan modes,
  batches 1/2/4/8/16, and both 146/175 input widths. There were **100 successful
  calls and 100 rejected negative controls** across these configurations. The test
  double checks every input value, row mapping, persistent host addresses, output
  tail, invalid row/capacity requests and propagated backend failure. This is not
  actual-model qualification at every batch/width.
- The actual ATen output-packing helper passed **5 valid noncontiguous batches and
  18 malformed tuple/shape/dtype/device/row/null controls**. All rejected cases
  left destination storage unchanged; all valid cases preserved the complete tail.
- Real native execution used the same **untrained 5,043,005-parameter checkpoint**
  as PR1/PR2. The singleton package was reused exactly. Only batch four was newly
  exported, using the existing exporter and the same weights/encoding. Actual
  model inputs here are deterministic synthetic tensors, not selected chess leaves.

| Bound physical batch | Calls | Real rows | Physical rows | Padded rows | Maximum absolute logit error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 5 | 5 | 0 | 4.172325134277344e-7 |
| 4 | 5 | 13 | 20 | 7 | 5.960464477539062e-7 |

Each real row was compared with an independent eager singleton evaluation under
unchanged 2e-6 absolute / 2e-5 relative tolerances. Actual traced input bits match
the prescribed inputs; physical padding is exact +0 after full and partial calls;
Bend-returned logits are bit-identical to the native trace. Full/partial calls
return the same first-row result within those tolerances. Both processes show
zero input/output address changes and exactly one bridge input-tensor allocation.
Each actual-model executable also rejects four invalid row/buffer controls.
Accepted neural rows and useful EPS remain **null**: this backend has not applied
replies to a search tree.

### Singleton compatibility

The saved PR2 generated engine C was hash-verified, not regenerated or modified.
It was recompiled/relinked against this PR's final bridge and newly generated
singleton header. The existing neural verifier passed **18 searches, 50 traced
forwards/reconciled replies, 1,141 legal priors, 6 automatic-draw replies and 4
zero-forward terminal searches**. Three startup-failure controls still reject.
Maximum raw-logit error is 5.364418029785156e-7 and probability error is
2.9802322387695312e-8 under unchanged tolerances. Same-board/different-history
inputs remain distinct. No perft, full no-Python chroot or full-model UBSan rerun
is claimed; the small transport UBSan checks have the scope stated above.

### Self-review and limits

This is self-review, not independent review or formal proof. A local probe-formatting
issue made larger output records unnecessarily quadratic; reversing the traversal
and prepending fixed-sized fragments removed that problem without touching the
engine. The revised probe passed the complete hosted matrix. The local pinned
compiler/type/probe checks were useful, but local Python static tooling lacked a
usable complete environment; the locked hosted lint and tests supply that evidence.
No validation rule, original chess oracle or numerical tolerance was weakened.

Normal UCI binding remains batch-one-only; CUDA/BF16 remains explicitly rejected.
The backend is single-owner/synchronous and does not batch search leaves, schedule
buckets, launch concurrent workers, infer accepted work or move chess logic into
C++. Runtime inputs still copy into one model-owned tensor. AOTI's own allocations
are not counted by the bridge input-allocation audit. The 4,096-node arena and
fixed-wall comparability limit from PR2 are unchanged.

PR3b remains: native CUDA/BF16 and device/stream/staging behavior, numerical
qualification with trained production weights on the 5090, and appropriate actual
batch buckets. Later scheduler/search PRs must carry physical versus real rows
into PR1 accounting and preserve linear buffer lifetime through asynchronous work.

## Evidence and reproduction

Commands and contracts: [batch backend README](../../native/bend_engine/batch_backend/README.md).
Artifact **10714064067**, `deepfin-pr3a-batch-qualification`, contains source patch
and manifest, JUnit results, binding/transport/helper/model/singleton reports,
compiler and executable identities, and the exported batch-four sidecar. Retention
is 30 days. Raw traces, models, generated binaries and staging workflows are absent
from the feature diff and this qualification artifact.

- Artifact ZIP SHA256: `2b936ff0d1e732c23a9df8dd794bed4ef52fefa03223db59fc716ee5d837aaf9`.
- Qualified source patch SHA256: `f47e2ed3b2974d9fb82f10c9c61e6d7ebd164d778e73c0dcf97dd150b06f72f6`.
- Checkpoint SHA256: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Reused singleton package SHA256: `9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.
- Batch-four package SHA256: `5ddf5079193988d31563634676235e6c9484ad8978a0b7331210899ad175b77d`.
- Batch-one probe executable SHA256: `d95b4fc2e8dec28ec3061adb689d5d8fd8dda23ddd7a4811628149b60b705228`.
- Batch-four probe executable SHA256: `93596955689f5e7facca808b985ce0330e2d645e25f2666be687fa03aba4e7ab`.
- Relinked singleton executable SHA256: `eb98fd58643dfd7909a0935aeb0832e68d33751345f8f2e7b6e850b009eacaa9`.
- Reused PR2 generated C SHA256: `28bed3f939b8cf3d2e764043f482d545e047704adfbe5985ee79484b605d3965`.
