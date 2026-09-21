# DeepFin slider-index laws and checked U64 toolchain adoption

## Scope and precommitted acceptance

User-approved continuation, stacked on history-encoding PR #801 at
`fe658a2680ec7941742c045feff0ccb806a898ad`. No merge, production deployment,
model inference/export, training, playing-strength experiment or perft-depth
increase. Keep Bend ownership and the no-Python executable runtime.

Adopt `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` and its matching
U64 proofs. Verify the actual compiler fingerprint, prove ordering/bounds against
production functions, require meaningful broken implementations to fail, and
requalify existing table/UCI/rules/encoding checks without relaxing assertions.
Failures stop adoption; they are not converted to success or hidden.

Budget: one isolated local source-proof development environment, bounded hosted
CPU qualification, no GPU/model work. The full finite source certificate checks
128 segments/107,648 generated subsets, with a 30-minute checker timeout. Native
qualification uses the existing three perft counts, not new depths or arenas.

## Implementation

The compiler manifest pins Bend 2.0.21 plus the reviewed U64 extension and the
upstream cyclic-template fix. All 84 compiler/effect files have fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Three verbatim proof files and their license are tied to the same revision by
content hashes and immutable Git-object comparison.

All pre-existing engine Bend source files are byte-for-byte unchanged, including
Tables.bend and Sliders.bend. Proof-friendly expressions live in Spec.bend, with
inductive/definitional proofs linking them to the actual production loops and
reader. No arithmetic algorithm, native intrinsic, compiler file, UCI/rules/
encoding/search algorithm, or independent oracle changes.

Ten application contracts include nine quantified theorems and a closed
exhaustive computational certificate. Generic induction connects actual
`Tables.fill`/`Tables.tables` arrays to PEXT-indexed writes. Generic bit induction
proves mask invariance; U32 losslessness retains the explicit high-zero premise.
The finite certificate verifies every canonical generated subset's index,
high word, segment bounds, non-wrapping sum, used address interval, and terminal
cycle. See `native/bend_engine/standalone/proofs/README.md` for precise limits.

Seventeen controls cover checker-level implementation mutations, cyclic false
proofs, unsafe-success warnings, missing proofs/imports, holes, and dependency
drift. The existing 12 compiler-pin tests remain separate. A path-filtered,
read-only CI job runs the complete source gate; no recurring native encoding,
perft, model export or new ordinary-pytest traversal is added.

## Development observations (not final hosted qualification)

The generic production-loop bridge and controls passed locally with Bun 1.4.2
and the exact pinned checker. An initial seven-condition finite walk certificate
passed all canonical segments/subsets in 843.77 seconds with peak RSS 1,295,256 KiB.
The final predicate additionally ties each address to an expression proved equal
to the actual `Sliders.lookup` read;
final source/native results must be recorded separately, not inferred from that
initial check. The initial standalone pin-only candidate builds with Clang 17.0.0; its existing
three IO/serve unsafe dependencies are outside the pure proof graph.

A local attempt to factor runtime helpers hit the 4 GiB container memory limit
during concurrent compiler/proof work. That runtime refactoring was discarded,
not accepted on the strength of source proofs. The final candidate leaves the
engine sources unchanged and requires its own native qualification.

## Recovery of the interrupted qualification

The earlier isolated candidate was not published: run 35524263059's native job
106113571173 passed, but source job 106113571113 exhausted its 30-minute checker
timeout. Pins, generic proofs and its 15 negative controls passed first; the
complete finite certificate did not. The publish job was correctly skipped.

A proof-only evaluator now avoids repeated zero-padding in Word.pext. It gathers
selected bits into a list and pads once. `Fast.word_agrees` proves agreement at
arbitrary word width; `Core.step_agrees` proves equality of the entire original
eight-condition predicate for arbitrary arguments. The added public law forces
that bridge to check. Corrupting either the extractor or its predicate invalidates
the proof, bringing the negative-control count to 17. All pre-existing engine
Bend sources, compiler inputs, native adapters and independent oracles remain
unchanged. No timeout, predicate, compiler gate or numerical assertion is relaxed.

The recovered candidate's banked native qualification includes all five builds,
the unchanged engine/rules/encoding suites, full table comparison and static
empty-runtime check. Its source tree is
`00bb12ce6f6baf4f2a0b3ddc485d96085bf6f064`; source/native artifact IDs are
10609517108 / 10609279201. The final repaired source proof and publication
identities are recorded separately after execution, not inferred from that run.

## Validation readout

The new generic reflection proof and all 17 rejection controls pass locally.
The complete ten-law certificate is being qualified separately. The permanent
source-only workflow will check the published PR tree, without substituting
compiled native answers for a proof. Final result identities belong in this
record and the PR discussion once available.

## Review and trust boundary

Self-review only at authoring; no independent reviewer was available in this
session. Review specification/CI edits, not merely the theorem names. The
closed certificate enumerates the canonical domain; it is not a universal
carry-rippler theorem for all 64-bit masks. The universal array bridge is about
the actual loops, not the entire build wrapper, all Array algebra, ray geometry,
legal chess, inference, or the native compiler. Preserve independent tests.

The Bend fork's inherited strict-TypeScript failures remain documented and were
not suppressed. Main/live branches and existing running processes are unchanged.
