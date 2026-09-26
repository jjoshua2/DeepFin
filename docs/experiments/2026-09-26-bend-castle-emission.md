# Actual castling emission and guard provenance

## Scope and acceptance

Base: PR #889, `8608169d517a3eefeffed459bb7d7c0631f7c4c4`.
Pinned compiler: `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, unchanged 84-source fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Target the actual output-list theorem, not another coordinate-only model: show that
`Chess.castle_side` returns its exact input tail or one guarded, correctly formed new
move followed by that tail. Derive membership provenance and actual update consistency
from the true producer guard. Keep existing accepted laws and production code unchanged.

Acceptance is a complete six-law public source consumer with meaningful semantic and
policy controls, bounded current-pin native producer checks against an independent
coordinate/set reference, original compiler/pin tests and unchanged repository lint.
Tests are opt-in and CPU-only; no routine perft increase, GPU, model or training work.
No merge or deployment. Self-review only unless a separate review actually occurs.

This record is written after successful local source construction and the first
native report, before final whole-suite and hosted qualification. It is not presented
as a prospective performance experiment or retroactively preregistered measurement.

## The six contracts

The new suite lives at `native/bend_engine/standalone/proofs/castle_emission/`.
Its public LAWS file imports actual Chess and the independent Board invariant.
PROOF explicitly imports LAWS and discharges every obligation.

1. Actual output is the input tail or a single exact guarded move prepended to it.
2. A result member absent from the input tail entails the input producer guard.
3. That new member equals the complete computed castling Ply.
4. A true guard supplies initial rook-landing freshness in the independent predicate.
5. Applying a new member preserves input Board consistency, composed with the existing
   public castling update theorem rather than assuming output consistency.
6. A false guard preserves the complete actual `(table, tail)` pair.

The guard contains the selected castling-right bit, owned source king, owned corner
rook and empty between path. A source proof, not a textual source guard, connects it
to the actual function. Structural Word proofs reflect actual masked occupancy into
independent freshness and narrow that freshness to the rook landing square.

## Domains that must remain distinct

The input tail is arbitrary and can already contain flag-two moves or duplicates.
The membership results require absence from the input tail; a closed counterexample
with an existing castling entry and false current guard is retained. The extension
law is occurrence-aware even when values duplicate the tail.

Source statements quantify arbitrary actual affine tables. Positive output-list
refinement does not prove table-content preservation or lifetime; false-guard
rejection separately preserves the complete pair. A nonempty source witness uses a
zero-filled arbitrary table, establishing satisfiability, not attack-table correctness.

Native tests build actual current Tables.build and call actual castle_side. The
independent reference checks geometric start/transit attacks, not a candidate-provided
answer. Destination-attacked cases are intentionally retained by castle_side; later
legal_moves filtering and its semantic king-safety theorem remain separate.

No claim covers the whole legal_moves list, valid metadata history, king counts,
legal reachability, final king safety, legal-generation soundness/completeness,
native ownership/lifetime, performance or new Python application migration.

## Execution status

Six public laws and their importing consumer passed locally with exact safe output.
The first native report passes 852 distinct producer requests, all four build modes,
and three compiled/executed actual producer corruptions. Complete final qualification
and any failed intermediate attempts will be recorded below with their exact sources.


## Completed hosted qualification

Hosted run **36249795989** passes all six public laws, importing consumer, seventeen classified controls, four native producer modes, three compiled/executed actual-code corruptions, original compiler source/pin checks and unchanged repository lint on source `f3e5db1fb2e7b671fcfefaf7ea03d52b0f88a536`.

The producer-output gap is now checked against imported Chess.castle_side, not only source anchors. The actual list is its exact input tail or one exact guarded castling Ply prepended. New-member guard and coordinate proofs require absence from that tail. A structural occupancy/freshness proof derives the input rook-landing condition, then the existing public castling-update theorem yields actual child-Board consistency. The caller supplies input consistency, not output consistency or desired target emptiness.

Arbitrary actual affine arrays remain inputs to the source statements and real calls. Positive list observations do not prove returned-array identity or native lifetime; false-guard rejection separately preserves the complete pair. Arbitrary input tails may contain old flag-two moves and duplicates. The explicit old-tail/false-guard counterexample remains, as does a satisfiable nonempty source example using an arbitrary zero table rather than a claimed correct attack table.

All852 distinct native requests execute actual Tables.build and Chess.castle_side, checking complete ordered move lists and1,469 complete raw child Boards per mode (27,911 U32 Board fields). There are432 true guards and269 emissions. Four destination-attacked examples intentionally remain in the castle_side result; later legal_moves destination filtering is outside this theorem. Seven malformed batches are rejected per mode. Modes repeat the same fixtures, not disjoint or exhaustive legal games. Inherited tail children are compared exactly but are not certified legal or consistent.

The external native reference uses coordinate/set ownership, path clearance and independent geometric attacks. Candidate input contains raw Boards, side choices and fixed tail-selector modes, not expected decisions. Guard bypass, omission of transit from the path mask and wrong destination compile/run before incorrect output is rejected. No old legal_probe compiler pin is bypassed; no full perft, benchmark, search, model or GPU work is added.

Eight source controls fail for intended semantic/refinement reasons; eight enforce manifests/import safety; one synthetic zero-exit warning tests the wrapper and is not a compiler execution. Crashes, missing imports, affine/parser errors and timeouts are not semantic success. One local model-mutation draft reached the implementation-link proof before the expected guard-extraction location; the corrected harness targets that actual source connection without changing any theorem. Earlier affine/erased-argument construction errors remain historical.

All499 inherited native source entries remain byte-identical; all511 candidate entries match before publication. Exact-source160/470 parent evidence plus newly executed6/17 yields modular166 laws/487 controls, not a complete aggregate run. Original compiler16-laws/seven-controls and12 pin tests pass. Hosted locked Python3.13/uv0.12.10 CPU tools pass unchanged lint; the original local missing-tools failure is not relabeled successful. C compiler selection is scoped to native checks.

Only documentation,index and evidence follow source qualification. A one-byte base64 transport transcription was explicitly corrected and exact chunk,patch and complete Git-tree identities verified before source application. It is a transport correction, not a code/checker change.

Self-review only. No production function, prior accepted law, compiler input, permanent workflow, routine perft or live process changed. No additional Python application responsibility moved into Bend. Export,external references,data/control/training and transitional C++/LibTorch/AOTI remain dependencies; pinned checker/Base,native lowering/storage,ABI,toolchain,OS and hardware remain trust boundaries. Existing compiler/snapshot-lowering/closed-builder limitations are unchanged.

Next acceptance is compositional provenance through both castling sides and the final actual legal-move filter, plus semantic start/transit/destination attack correctness and independent metadata rules. Source king/rook bit membership and rights/path guard provenance do not establish historical rights legality, king counts, FIDE reachability or generator completeness.
