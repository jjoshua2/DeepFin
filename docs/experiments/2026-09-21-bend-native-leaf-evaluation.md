# Bend-owned selected-leaf evaluation with a native CPU model

## Plan before model execution

Base #803 / dfa0cda0498d45494e1c52e8271fbf20ebf37871. Preserve source-pinned
Bend 2.0.21+U64 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.
Question: can actual selected leaves use the existing complete Bend input/policy
component, execute a pre-exported native model and return validated probabilities
to Bend search without Python orchestration or encoding?

Add an opt-in native-model entry point, retaining the material executable. Reuse
its UCI parser/state, rules/history, search, finish and mapping/encoding modules.
Bend owns selected-leaf history, full input, legal indices, stable legal softmax,
WDL softmax and the identity-bound search reply. Narrow native C/C++ owns raw
U32 transport, package integrity/copying, LibTorch tensors and AOTI execution.
Neither native layer implements chess, UCI decisions or a batching scheduler.

Scope CPU float32, supported 146/175 layouts, corrected repetition semantics,
row-independent static batch packages with one real row and zero padding. Bind
package SHA and metadata at build time, verify/copy identical bytes before loading,
reject missing/mismatched models without a material fallback. Model export remains
an external Python build tool, not an engine runtime or Bend-authored kernel claim.
Use only the existing untrained saved transformer fixture for this gate.

Predeclared checks: raw response conversion independently exercised with extreme
logits, nonfinite values, missing/extra words, invalid slots and mismatched moves.
Real selected leaves must match exact external C input/policy maps and Python
history (only the inherited pawn-storm rounding exception), eager singleton logits
at atol2e-6/rtol2e-5 and independent probability conversion at atol2e-6/rtol2e-5.
Reconstruct search ancestors independently and compare request/node identities,
visit accounting and selected moves with the existing diagnostic PUCT reference.
Check actual prior played history, arbitrary/forward roots, rule draws without
model calls, no-legal-move roots, normal reset/stop and model persistence. Native
output faults and package mismatch must fail before a bogus bestmove.

Run with an interpreter-free native dependency root when feasible: unlike the
material static executable, this one legitimately needs LibTorch, a loader,
model package and writable extraction space. Enumerate them; never claim that
only one executable is sufficient for this model backend. Python oracles remain
outside. No Python subprocess or libpython can be used by the candidate.

Synchronous calls are not interruptible; readiness/stop may wait for inference,
encoding, initialization or output. No asynchronous batching/cancellation or
CUDA qualification, speed, strength, production Gumbel parity or training claim.
Model kernels are compiled from PyTorch, not authored in Bend.

Budget: isolated CPU, two inference threads, one compiler/build worker; hosted
confirmation bounded to 15 minutes with retries only for concrete findings. No
training, GPU, model downloads, live-process/YAML changes, merge or deployment.
Native verifiers remain opt-in, no deeper perft or expensive recurring tests.
Recovery discards only the isolated feature. Self-review, not independent review.

## Results

Pending model integration. The Response and Backend modules pass local Bend
checking. The opt-in Response test passes native UBSan for 15 cases, including
three accepted conversions and twelve semantic rejections. Eighteen new cheap
package-binding contracts pass. These are component results, not yet model/search
execution, full-engine compilation or evidence of an interpreter-free model run.
