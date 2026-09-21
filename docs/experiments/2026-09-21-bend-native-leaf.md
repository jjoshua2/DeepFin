# Bend-owned selected-leaf neural evaluation

## Before execution

Base #803: dfa0cda0498d45494e1c52e8271fbf20ebf37871. Keep verified compiler
 aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae; no production or core search change.
Connect actual standalone selected leaves, not only diagnostic input commands,
to the complete Bend input/policy component and a pre-exported CPU native model.
Bend owns history/rules, input, legal-logit gathering, stable softmax/WDL and the
reply consumed by Search.resume. Native code is only bounded tensor transport,
artifact integrity/version validation and LibTorch execution. Python is an external
export/test tool, never the running controller. This is a migration backend, not
Bend-authored transformer computation or a Python-free training toolchain.

Acceptance: preserve material default/regressions; compare actual selected paths,
full input tensors and raw model logits against existing external encoders/eager
reference; compare normalized legal priors/WDL and search traversal against the
existing diagnostic reference. Fail malformed/nonfinite outputs before tree update.
Reject unsupported manifest, version, hash, target, batch or encoding at startup
or binding, never silently fall back to material. Only v3 CPU F32 batch-one compact
policy and corrected root-oriented inputs initially. Use the existing saved untrained
transformer fixture, no downloaded/trained checkpoint or new learning experiment.
Predeclared raw-logit tolerance remains absolute 2e-6 / relative 2e-5. Normalized
probabilities allow absolute 2e-7 / relative 3e-6. Tensor comparison uses exact C
bits and only the previously declared two storm-plane Python rounding exception.

Test isolated deployment with native libraries/package but no Python, Bun, shell
or helper process. Unlike the material build this is not a static one-file engine.
Synchronous model calls are nonpreemptible; readiness/stop can wait for a forward.
Maps rebuilt per leaf and list marshaling are correctness-first, not optimized.
Do not infer strength, throughput, CUDA correctness, or full production Gumbel parity.

Budget: isolated CPU, one compiler at a time and two LibTorch threads. One bounded
hosted confirmation (15-minute cap); follow-ups only for observed faults. Focused
local component checks where memory allows. No deeper perft, new permanent workflow,
training, live-process edits, deployment or merge. Preserve unrelated branches.
Self-review only unless an independent reviewer is actually obtained.
