# Bend real history/policy and native CPU neural composition

## Question / baseline

Follow-up to #777's deterministic evaluator. Parent is
`c8bf8679fe68523effc7d971def5cd27ad35d4c9`; compiler remains
`57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`. No production adoption or compiler change.

The prior tree was flexible enough for persistent evaluator IO, but requests lacked
history paths and their packed move words were not neural policy IDs. Perft tuning
cannot expose either mismatch. Test an actual native model runtime against real
CBoard inputs and an eager reference before spending further effort on throughput.

## Qualification contract / hosted confirmation plan

The hosted gate is correctness, not a speed screen. Accept only if:

- every feature float matches the actual CBoard encoder bit-for-bit for all
  history modes 0/1/2, 34/63/67 extra planes and repetition-history flags;
- every legal dense/compact policy ID matches the existing independent Python
  move encoder, with action order preserved;
- the real production ChessNet class, in a small seeded **untrained** configuration,
  exports to `.pt2` and executes in native LibTorch AOTI;
- full native logits and legal/WDL probabilities match eager execution on each
  requested leaf (FP32 absolute/relative tolerance 5e-5; BF16 optional 0.025);
- only native outputs feed Bend, and all final tree fields agree with the diagnostic
  tree reference, including reset and recovery after an injected backend failure;
- invalid requests fail before publishing partial output, and existing four-mode
  persistent-session checks remain green after the path protocol extension.

Budget: one path-scoped CPU hosted job, 15-minute timeout, two inference threads,
one compiler worker, eight roots x four 12-simulation epochs. No GPU, training,
network checkpoint downloads, production changes or perft depth increases.
No timing verdict. On failure, keep qualification incomplete; do not relax the
reference or route Python oracle outputs into Bend. Rollback is isolated branch
removal, not resetting prior work.

## Architecture / boundaries

Bend emits the leaf and ancestor move path; native CBoard replay qualifies the
history and computes actual input planes. Native C++ AOTI loads a CPU batch1 tuple
package, then gathers compact logits and normalizes policy/WDL. Python is still a
**test relay**, not an embedded interpreter in either native library. The native
libraries are not yet linked directly into the Bend IO effect. No Python-free
production-pipeline claim is made.

The encoder is shared production code through two entry points, so its byte parity
is an integration guarantee, not independent mathematical proof of the encoder.
Move mappings also compare to Python `move_to_index`; python-chess checks candidate
board transitions. Full raw logits are checked against the eager version of the
same seeded model. Package/policy-map hashes and the actual output treespec prevent
silent artifact or tuple-order substitution. Training quality is not measured.

## Local observations

Clang 17 / Torch 2.10.0+cpu, pinned Bend:

- 612 root configurations and 936 search children: exact feature bytes and action
  mappings. Both colors, EP, all promotions and castling included. Same current
  board/counters with different history produces different input.
- Six invalid feature/path/action requests rejected without output writes; action
  order reversal preserves tensors and reverses mapped IDs. The same matrix passed
  a separate undefined-behavior-sanitized feature library.
- Eight roots x four epochs, **312 native AOTI evaluations**, real 146-plane FP32
  untrained ChessNet. Deterministic reset and recovery passed. Max raw/eager absolute
  error 4.76837158203125e-7. Three invalid native model calls rejected without output
  commits; a following valid call succeeded on the same model handle.
- The path-aware original session gate passed all 38 sessions on generic,
  forced-portable, native-target and UBSan builds, including 207 python-chess
  positions. No search algorithm or chess-rule changes.
- 50 isolated cheap tests (35 new, 15 prior), focused Ruff pass. Full Basedpyright
  and hosted confirmation are pending at this record's initial publication.

## Limits / next gate

Only CPU batch1 is executed here. No trained checkpoint, CUDA qualification,
production Gumbel/C MCTSTree parity, batching/virtual loss, inference deadline or
in-flight cancellation, history snapshots for arbitrarily long games, draw
adjudication, or performance claim. The 146-plane FP32 smoke is the default; extra
neural dtype/width options are unqualified unless separately reported. Feature
contexts are serial because the underlying C encoder uses global mode state.

Self-review only, not an independent reviewer. Review concentrated on parent-path
order/bounds, immutable-root history, black/promotion policy identity, no partial
output writes, F32 normalization direction, native ownership and test isolation.
The next useful gate is native batch scheduling and actual target-package execution,
not treating this untrained CPU graph as proof of production readiness.
