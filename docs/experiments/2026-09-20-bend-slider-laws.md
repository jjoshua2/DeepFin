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

Eighteen controls cover checker-level implementation mutations, cyclic false
proofs, unsafe-success warnings, missing proofs/imports, holes, and dependency
drift. The existing 12 compiler-pin tests remain separate. A path-filtered,
read-only CI job runs the complete source gate; no recurring native encoding,
perft, model export or new ordinary-pytest traversal is added.

## Recovered development notes (not acceptance evidence)

The interrupted workspace included notes about earlier reduced-condition and
runtime-refactoring experiments. Those notes are not used as a PASS certificate.
The acceptance result must come from the unchanged eight-condition predicate,
the full ten-law module, and the exact native-input identity recorded below.

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

## Proof packaging repair

The first reflection repair still exceeded the local 30-minute limit in the
complete module: using the closed equality as a theorem argument repeatedly
normalized the full finite predicate during type comparison. Small-domain timing
was diagnostic only and was never substituted for the 128-segment acceptance.

`Spec.CheckedTables<n,key,at>` now packages the exact equality as its required
constructor field. Generic elimination functions recover that equality for the
same production-loop and ordinal theorems. This does not remove any condition,
change the checker, admit an axiom, or raise the timeout. The closed constructor
proof must still evaluate the complete predicate. An 18th rejection control
tries to forge a certificate of an explicitly false terminal cursor and fails.

## Fresh native qualification and source status

Hosted run **35556227949**, native job **106200288255**, passed on the exact
reflection candidate tree `a7640c87ee9d6955f873f458a4cdcd25e8981f6d`.
Its artifact **10620212310**, `slider-law-native-qualification`, has ZIP SHA-256
`2bc7623c88a243e498cb4bb3213bf4338d2774399e0471d9f4d0974fa0e00895`.
The certificate-packaging repair changes only proof/gate/documentation files;
compiler inputs, manifest, engine source, adapters, qualification commands and
independent oracles are identical to that native-tested tree. Reuse this result
rather than repeat the same five-build traversal for a proof-only change.

All five environments (generic, forced-portable, native, UBSan and static in an
otherwise empty chroot) passed the unchanged suites. Per environment: all
108160 used table entries match; 137 legal children, 51 searched roots, 23 invalid
transactions and eight independent UCI-client plies pass; rules compare 116
position/history facts and 11 search cases; encoding compares 505 complete
Python tensors and 501 C tensors, with four explicit Python-only wide-clock cases.
Canonical perft remains 8902 / 97862 / 43238. The logical tensor is the same
112-plane partial input, not a new complete neural model input.

Generated C SHA-256:
`1312a1e83b545786afd8a79697b6f8e208b977ba3059f48ecd7719f8f3c64fb4`.
Static ELF SHA-256:
`598a5fcf8339b82f5d8fe9f5e45a580178d03038fc40dc42eac83304370a88c0`.
The static runtime inventory contains only `deepfin-bend`. All table bytes hash
`37cf5ae16f1709ef3220c30f60fbd142bc32be6315016dd037a6a8e631c5a663`;
ordered tensors hash
`cb7a7e6688a73c5bfbf734769cd6c5f1d3a04609167bf019a4cbfa79d5eb6bbf`.
These identities equal the preceding native-tested candidate. No speed or
playing-strength claim is inferred.

The boxed generic bridge and all 18 rejection controls pass locally with the
exact pinned compiler. **Full source qualification remains pending at this
staging revision**; controls-only results do not establish the complete laws.
Record the completed source run and publication identities after it finishes.

## Review and trust boundary

Self-review only at authoring; no independent reviewer was available in this
session. Review specification/CI edits, not merely the theorem names. The
closed certificate enumerates the canonical domain; it is not a universal
carry-rippler theorem for all 64-bit masks. The universal array bridge is about
the actual loops, not the entire build wrapper, all Array algebra, ray geometry,
legal chess, inference, or the native compiler. Preserve independent tests.

The Bend fork's inherited strict-TypeScript failures remain documented and were
not suppressed. Main/live branches and existing running processes are unchanged.
