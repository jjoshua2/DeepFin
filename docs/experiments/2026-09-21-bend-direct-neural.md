# Direct Bend-owned search-to-model evaluation

## Preregistration

Base #803 / dfa0cda0498d45494e1c52e8271fbf20ebf37871. Compiler remains
Bend 2.0.21 + U64 at aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.
Question: can real selected leaves use the complete Bend-owned input/policy API,
a native model forward, and Bend-owned legal-only softmax/backups, without a
Python engine controller or encoder? Reuse the existing transformer fixture and
exporter, not a new architecture. This is deployment composition, not trained
strength, performance, CUDA or a Bend-authored transformer/training claim.

Bend owns request history, rules, tensor construction, legal indices, finite-logit
validation, normalization, ticket identity and tree updates. Native code only
marshals floats, verifies the immutable package identity and tensor ABI, pads a
static batch, invokes LibTorch/AOTI, and returns raw policy/WDL logits. No native
chess semantics, legal gathering or softmax. One persistent loader, one synchronous
real-row call at a time; optional trace is off normally. No Python runtime path.
Model export remains a separate Python build-time migration dependency.

Acceptance: compare every traced real leaf tensor/policy index to the original
encoders, all native raw logits to eager singleton outputs, and Bend probabilities
to independent stable softmax. Preserve prior raw-logit tolerance 2e-6 absolute /
2e-5 relative. Probability tolerance is predeclared 6e-7 absolute / 2e-5 relative
for F32 versus F64 reductions. C input comparison is exact bits; the original
Python pawn-storm exception remains only planes 173/174 at absolute 1.2e-7.
Use the existing independent diagnostic PUCT reference with the actual checked
F32 replies to compare every final node and result counter. Cover both turns,
played history, promotions, castling/EP, automatic terminals and several successive
UCI positions. Ensure fixed package identity and encoding are bound together.
Bad manifests, changed packages, shape failures and nonfinite logits must fail
without a model-to-material fallback. Build unit probes for malformed reply sizes,
wrong legal ordering, duplicate/out-of-range slots, finite extremes and large
illegal logits (legal-only normalization), including WDL and request IDs.

Use only the prior untrained transformer checkpoint for the CPU integration gate,
exported once, with two inference threads and single compiler jobs. Run the normal
material control separately. Budget: one bounded 15-minute hosted qualification,
additional attempts only for concrete findings. No GPU, training, production config,
main/live branch change, merge, deployment or deeper default perft. Native integration
is opt-in, not another recurring model export. Preserve previous work and use a
new artifact directory. Independent review is unavailable; mark self-review.

No claim of hard real-time behavior: synchronous LibTorch work and optional trace
output can delay stop/isready until that call/serialization completes. Infinite
remains bounded. The neural runtime needs the compiled package, LibTorch/OpenSSL
and OS shared libraries, unlike the static material-only executable. Test it in
an isolated filesystem without Python/Bun/compiler/shell executables, with external
Python/C references outside. No all-Bend model computation claim is made.

## Readout

Pending actual qualification. Do not infer runtime validation from module checking.
