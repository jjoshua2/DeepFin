# Whole-board round trip and ordered abstract squares

## Scope and acceptance

This continuation refreshes #878 and the newer #879, then starts directly from
#879 at `d8629e31a55ceda25edef836f8f34a895416dba4`. The complete source archive
and raw commit were recovered; all 404 inherited native-source entries match.
Repository guidance, development rules, branch lifecycle and experiment index
were read. The work is isolated, with no live checkout or production change.

This record follows constructive local proof work and precedes hosted
qualification; it is not a backdated preregistration. Acceptance requires the
five-law importing consumer, all classified controls, bounded four-mode actual
Board observation tests, original compiler source/pin checks, and unchanged
whole-repository lint. Parent 127/350 evidence may be retained only with exact
source/report and successful-job identities. No combined aggregate execution is
implied. Routine tests/perft budgets remain unchanged. No GPU, model or training.

## Reconciliation with the saved local candidate

Saved local commit `c2d327dfd1d6d814554a211f7f6ee4ab7e2c922b` remains preserved in
the supplied occupied-decoding patch/review package. It used `proofs/decode/`,
while the already-published #879 uses `proofs/decoder/`. These are not byte-identical
patches. The local `square_partition` corresponds to published
`square_projection_valid`; both prove valid actual per-square observations.
The local abstract round trip and occupied-decoder result correspond to the two
published contracts of those names, on their explicitly consistent domains.
The published guarded result additionally handles empty squares. The saved
explicit empty-observation conclusion is now registered and derived from those
published producers in this suite; it is not silently omitted or called new
mathematics. All previously published laws and both saved implementations remain
untouched. Counts start from #879's 127/350, not the saved variant's 127/349.

## New result

For every consistent actual Board, observing the two ordered 32-square lists and
three metadata values, then restoring, returns the entire original Board.
The output is not merely a list of valid tags. The length theorem establishes
exactly 64 observed positions. A separate bounded-square theorem ties the list
orientation to actual bit observations even for inconsistent boards. Equal
snapshots identify equal consistent Boards. Empty-square observations are
reconciled with the saved local contract, without treating raw king fallback
as an empty tag. All five contracts are universally quantified.

Snapshot restoration is total on malformed lists but no unrestricted two-way
Snapshot bijection is claimed. Cell.Invalid and out-of-range raw kind values
remain outside any proposed well-formed abstract-board inverse theorem.
Metadata is copied exactly, not validated as legal chess state.

## Source and native evidence are different

The structural snapshot functions are source-proved. A direct attempt to compile
them natively failed with `an arity over 255`; exact source and diagnostic are
in `evidence/bend-whole-board-roundtrip/lowering-*`. This limitation is retained,
not reclassified as a successful native test. The new successful native path
observes actual Chess operations across all 64 squares and reconstructs the
Board in the independent external reference. It does not execute the structural
proof model or supply expected labels to the candidate.

The 1,026 bounded Board cases include all 768 actual fresh square/kind/color
insertions, 192 mixed consistent inputs, 64 metadata updates, empty and start.
Each native build checks all 19 raw fields and 128 guarded kind/color observations.
Modes repeat fixtures. Two deliberate corruptions (actual wrong pawn tag and
skipped probe square) compile and execute before being rejected for wrong values.
These are regressions, not discovered production defects. No perft/table build.

## Development failures retained

Constructive source drafts required explicit annotations on dependent column
constructors, corrected the order of an existing insertion-law invocation and
metadata helper arguments, and moved a late import into the import section.
None changed a theorem domain or checker. The first control run stopped at a
nonunique mutation selector; the selector was narrowed to the actual constructor
line. An outer 90-second tool invocation ended without a completed report;
subsequent checks retain streamed logs and explicit process receipts. An optional
streaming tool was unavailable before executing its command. Original failures
remain in the review package, separate from valid semantic controls.

## Remaining obligations

Whole-board consistency, ordering and reconstruction do not establish legal
reachability, metadata validity, FEN freshness, move application or generation.
Next P3 work should derive freshness from actual parser traversal, then prove
invariant-preserving removal and moves, including special moves. Existing P2
closed-builder and compiler limitations remain separate. Source checks trust the
pinned checker/Base; native lowering, runtime storage, ABI, toolchain and hardware
remain trust boundaries. Self-review only, not independent review. No application
responsibility moved from Python; export/references/data/control/training and
transitional C++/LibTorch/AOTI remain dependencies.

## Qualification status at construction

Public consumer and all-square native tests passed locally. The complete final
focused gate and hosted repository qualification are recorded separately when
completed. No failed or partial gate is counted as success here.


## Completed local checks

The final five-law consumer and all 17 controls passed in one focused command
(45.768 seconds). Native actual-board observations passed all four modes on
1,026 boards per mode, with 65,664 square observations and 150,822 output-field
comparisons. Two corrupted programs compiled and executed before rejection.
The original compiler source suite and all 12 pin tests passed; local repository
lint failed for missing tools and remains unqualified. The full aggregate was
not run. Structural snapshot native lowering remains unqualified as documented.


## Completed hosted qualification and explicit lowering limit

Hosted run **36085950814** passes the five-law focused source gate and all17 controls, four native actual-observation modes and both corruption checks, original compiler source/pin tests and unchanged repository lint on source `92737251af190ebddba5b83d146e462ae1049ec6`. The separately reproduced structural snapshot C-lowering limitation remains unresolved.

All five laws are universal: complete Board reconstruction under partition consistency; exactly64 stored square observations; injectivity on consistent Boards; the saved explicit empty-square observation; and correct coordinate order for every bounded square. Metadata remains arbitrary. This is a left inverse on actual consistent Boards, not a bijection over unrestricted Snapshot lists or valid chess positions.

The focused command executes its consumer and all17 controls. Eight are semantic/refinement failures at intended locations, eight enforce manifests/import safety and one is a synthetic warning-output unit, not a compiler execution. No crash, missing import, affine error or timeout counts as successful semantic rejection. The exact source closure and all control records match the final local report.

Exact-source parent127/350 is retained on404 manifest entries, its completed focused/report evidence and successful hosted job. Modular coverage is132 laws/367 controls; the full132-law wrapper was not run. No old law or gate was changed. The alternate saved local decode implementation remains preserved rather than silently replaced by this decoder-based continuation.

Native generic/portable/native-target/UBSan each pass1,026 actual Board cases,65,664 square observations and150,822 output fields. The driver reconstructs all eight bitboards from actual occupancy-guarded decoder output and preserves raw metadata, comparing independent square arrays and actual returned Board fields. All768 fresh square/kind/color insertions,192 random Boards,64 metadata cases,empty andinitial are included. Nine malformed requests per mode are rejected. The complete native report matches local except cc. Modes repeat fixtures, not disjoint or exhaustive board/game datasets.

One actual decoder mutation returns the wrong pawn tag; one probe mutation skips observation squares. Both compile and execute before wrong-value rejection. The latter is explicitly a test-traversal mutation, not a production change. The native probe does not import or execute structural Spec.observe/restore.

The archived direct structural-snapshot probe was retried with the pinned compiler and again failed C generation with Error: an arity over255. Its nonzero receipt and exact source are committed. This is an unresolved lowering limitation, not an accepted source rejection control, native execution or a passing structural-snapshot round trip. The source theorem and actual-decoder native checks are separate evidence layers.

Hosted Bun1.4.2/Clang18.1.3; Clang is scoped to native probes. Locked Python3.13/uv0.12.10 CPU tools use the normal extension compiler. Unchanged Ruff/Basedpyright/Vulture passes, resolving only the local missing-tools lint gap. Original compiler16-laws/seven-controls and12 pin tests pass. No compiler source or protected checker was modified.

All416 native-source entries match before evidence publication. Only documentation,index and evidence follow qualification. Earlier explicit binder/import-order construction failures, a nonunique mutation-site assertion and an interrupted outer execution remain historical; the final local streaming-to-file supervised focused run completed successfully in45.768seconds. No failed or incomplete attempt is counted as a proof pass.

Self-review only, not independent review. Source/hash agreement is provenance rather than a separate reviewer. Pinned checker/Base,native lowering/storage,ABI,toolchain,OS and hardware remain trust boundaries. No production runtime, earlier accepted law, permanent workflow, routine perft, search, model/GPU, training or benchmark changes. No additional Python application responsibility moved into Bend; export,references,data/control/training and transitional C++/LibTorch/AOTI remain dependencies. No merge,force push,deployment or live-process action.

Next decisive P3 work is proving actual parser traversal supplies fresh squares, then invariant-preserving removal and move/special-move operations. Snapshot validity in both directions, metadata validity, legal reachability, king safety and legal-generation soundness/completeness remain separate.
