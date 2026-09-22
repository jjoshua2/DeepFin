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

## Readout

Pending the explicit hosted model qualification. Local compiler type checking,
Bun binding tests and transport probes are separate from a real model pass.
