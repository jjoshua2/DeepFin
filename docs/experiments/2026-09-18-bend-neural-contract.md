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
  and hosted confirmation were pending at this record's initial publication.

## Hosted readout: PASS for the bounded CPU composition contract

Executable commit: `17ed2108feb88c4ea4c397469e048834835a9c7f`.
Confirmation run: https://github.com/jjoshua2/DeepFin/actions/runs/35401914108
Job: `105783452466`, successful on 2026-09-18.
Staging revision: `f079b9c270b99927a9eeafc47508cb72361a371d`; the workflow checked
source hashes before executing and published a clean feature commit on #777 only
after success. This subsequent readout commit does not change executable sources.

Hosted environment: Clang 18.1.3, Torch 2.14.0+cpu, Bun 1.4.2, Python 3.13.15.
No package was downloaded from a live training run. The gate exported its own
seed-1909 untrained production ChessNet, width32/layer1/head2, no Smolgen,
146 input planes, FP32, fixed CPU batch1. One retained search-model instance served
all eight roots; an additional diagnostic load tested a deliberately bad sidecar.

- Native history/features and legal-action IDs matched exactly for all **612 root
  configurations + 936 search children**. Six invalid requests were rejected;
  action-order reversal and distinct-history/same-current-position checks passed.
- **312 actual native AOTI calls** supplied the search responses across 32 epochs.
  Each root ran normal/repeat/backend-failure/recovery epochs. Full raw policy and
  WDL logits, legal priors and WDL probabilities matched eager execution within
  maximum absolute error **4.76837158203125e-7**, without relaxing the 5e-5 gate.
- Full diagnostic-tree comparisons and deterministic reset/recovery passed. The
  injected error at request three retains only two completed simulations.
- Three invalid model inputs were rejected without output commits. A following
  valid call succeeded. Loading without required runtime input checks was rejected.
  A same-hash model with deliberately incorrect 179-plane sidecar was rejected by
  the actual package's generated input checks, not merely sidecar validation.
- The history-aware original session regression passed **38 sessions per build**
  in generic, forced-portable, native-target and UBSan modes; python-chess agreed
  across **207** distinct oracle positions. This four-mode result covers the Bend
  session, not four differently instrumented LibTorch builds.
- Focused Ruff and Basedpyright passed, and all **50 cheap contract tests** passed.

Compact per-root neural observations (12 completed simulations in the normal
session; 39 native evaluations across its four epochs):

| Root | Nodes | Max depth | Best packed key | Max absolute eager error |
| --- | ---: | ---: | ---: | ---: |
| startpos | 245 | 3 | 1544 | 3.5762786865234375e-7 |
| repeated | 242 | 3 | 1032 | 3.8743019104003906e-7 |
| black_history | 323 | 3 | 3390 | 4.76837158203125e-7 |
| ep_history | 334 | 3 | 771 | 4.76837158203125e-7 |
| captures | 363 | 3 | 771 | 4.76837158203125e-7 |
| castle_white | 254 | 3 | 64 | 3.5762786865234375e-7 |
| promote_black | 104 | 3 | 4104 | 4.76837158203125e-7 |
| promotion_capture | 136 | 3 | 15984 | 4.76837158203125e-7 |

These moves reflect an untrained network, not a chess-strength assessment. Different
Torch versions may initialize weights differently despite the same seed; parity
compares each run's packaged weights with its own matching eager reference.

Artifact: `bend-neural-confirmation`, ID `10571536264`, retained for 30 days.
Neural JSON SHA-256: `3784db0de56dbf54e6701123f65a4c6eb7551ea02777c83d704b9f4610db3a63`.
Session JSON SHA-256: `cb36167c84913cc62defe79412e266ac3c7a9ffe4c76166c7430f63c61cd9bd4`.
Hosted generated package SHA-256: `f36189d4414ed5aafe714d90be6895bc8940867cb39eecee1634759a90480bad`.
Canonical policy-map SHA-256: `cc28489f3fdeafbbebd31de7254949861a3843a93639f6b6c03afd87ee89951a`.
The generated package/cache are intentionally temporary, not repository artifacts;
repeat the export command rather than assuming this exact binary remains available.

The first hosted attempt stopped at four type-checker findings before neural
execution. Compatibility hashing/context-manager annotations fixed them without
suppressions. Self-review also added mandatory `AOTI_RUNTIME_CHECK_INPUTS=1` and
negative actual-package input tests. The successful rerun retained all positive
and negative checks. Validation duration includes export, builds and references
and is not an inference-throughput measurement.

A separate optional local 175-plane BF16 fresh-export attempt exceeded its
120-second process budget before producing a report. It remains **unqualified**;
no numerical failure, BF16 support verdict or CUDA conclusion follows from that
incomplete attempt. The completed neural qualification here is 146-plane FP32.

## Limits / next gate

Only CPU batch1 is qualified here. No trained checkpoint, CUDA qualification,
production Gumbel/C MCTSTree parity, batching/virtual loss, inference deadline or
in-flight cancellation, history snapshots for arbitrarily long games, draw
adjudication, or performance claim. Native feature contexts are serial because the
underlying C encoder uses global mode state. Existing perft depths are unchanged;
only cheap contracts enter ordinary pytest, with native work path-scoped.

Self-review only, not an independent reviewer. Review concentrated on parent-path
order/bounds, immutable-root history, black/promotion policy identity, no partial
output writes, F32 normalization direction, native ownership and test isolation.
The next useful gate is direct native evaluator integration, batch scheduling and
actual target-package execution, not treating this untrained CPU graph as proof
of production readiness.
