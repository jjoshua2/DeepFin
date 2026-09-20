# Bend-owned policy boundary

## Predeclared scope

Base #801 (fe658a2680ec7941742c045feff0ccb806a898ad). Continue the
Bend-everywhere migration: generate policy maps and translate legal Bend moves
without NumPy, Python, CBoard or generated table files in the engine.

The fork advanced to 806b373a7a4509479054da74a82271c6a4c25014 on September 20,
adding U64 reference laws. Adopt its verified source fingerprint separately
from application changes; keep native compiler/kernel bytes unchanged. Run the
existing standalone checks on that pin before adding the new policy source.

Acceptance: exact complete full/compact/mirror lookup equality; legal move
indices and reverse resolution for both colors, castling, EP and every promotion;
no fallback for a nonlegal or out-of-range index/key. All metadata validation
precedes diagnostic output. Same-board/different-history positions must have
identical policy mappings, unlike their history tensors. Read-only commands
preserve the complete played history, reject during active search, and work in
a static-only empty runtime. Preserve unchanged encoding/rules/UCI/perft tests.

This does not add missing classical input planes, normalize neural logits, run a
model, or change material search. Claims and null moves are outside model policy.
No C runtime addition, proof of compiler correctness, speedup or strength claim.

Budget: isolated CPU, at most two compiler processes, one engine thread; one
ten-minute hosted confirmation with reruns only for concrete failures. New native
checks remain opt-in; no perft increases or new recurring CI workload. Nothing
is merged, deployed, trained or changed in live production. Recovery is dropping
the isolated branch. Self-review only unless an independent reviewer is obtained.

## Implementation

Policy.bend constructs the compact vocabulary by enumerating all 4672 full
geometry slots in order. It owns full-to-compact, compact-to-full and square-pair
lookup arrays, with logical bounds distinct from physical capacity. Mirroring
flips files in oriented coordinates; Black's perspective separately flips ranks.
Normal/queen-promotion rays and dedicated N/B/R promotion slots remain distinct.
Policy.Space is an explicit sum type. Resolution uses the current generated legal
entries, retaining the original Ply flags, not a numeric-ID fallback move.

PolicyDiagnostic exposes read-only table, legal-list, UCI/key and full/compact
lookup commands. It validates command shape and prepares all legal entries before
emitting a response. A nonexistent legal match is an error; null/claim identifiers
are never converted into real moves. It builds disposable maps for each diagnostic;
this is not an optimized inference hot path or a model integration. The reusable
typed map owner may be retained by a future Bend inference controller.

No authored C/effects, model/search/evaluator, historical encoding or rule code
was changed. Default material search does not call the policy diagnostic. Native
array lookup and code generation still belong to the trusted compiler boundary.

## Local preliminary observations

Before adding policy code, the unchanged standalone verifier passed on the new
806b373... pin: 137 exact legal children, 51 searches, 23 invalid transactions,
canonical perft counts and eight legal UCI-client plies. The native compiler,
checker implementation and effect files compare byte-for-byte with the previous
fd1df817... compiler snapshot; the reference Base definitions differ as expected.
The source-law runner u64_proofs.js passed, including its negative controls. This
is evidence for the fork's source laws, not a proof of the new policy adapter.

The initial native policy run passed all 17,156 scalar table values, 2963 legal
move comparisons over 173 position fixtures, 2156 exact UCI/key/full/compact
resolution checks, 663 rejected requests and all 176 distinct promotion choices
(8 files, legal forward/capture directions, 4 pieces, 2 colors). All tables and
legal IDs matched the existing pure Python reference exactly. That preliminary
run had no C reference; final C/native/isolation results are recorded separately.

The Bend checker rejected computed tuple destructuring, a missing copy annotation
and an unsupported forward reference during implementation. Helper boundaries and
explicit ownership fixed those without compiler changes. A raw integer space tag
was replaced with Policy.Space before final compilation. Ruff findings in the
external test were corrected without relaxing assertions. The existing pin test
was updated to assert the new exact revision/fingerprint, not bypassed.

## Hosted/final readout

Pending. Preserve exact revisions, reports and limitations after execution.
