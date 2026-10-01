# Complete-generator table preservation

## Question and baseline

Continue PR #904 at `1915e4f283b1e3d5daf7e26683d5a6c2ef48a121` without altering its two castling-sequence
laws, their evidence, or production code. The missing compositional input is that
ordinary generation and optimized filtering retain the same table, not only that
an isolated initialized castling pipeline preserves it.

This increment adds three public storage laws under
`native/bend_engine/standalone/proofs/table_preservation/`. It is local-only at this
record: no GitHub update or new PR was performed. The connector exposed read actions
only and direct Git transport failed DNS. A prepared non-force update script and
incremental bundle accompany the external review package; the script was not run.

## Three public contracts

| Contract | Actual operation and guarantee |
| --- | --- |
| `ordinary_scan_preserves_table` | `Chess.scan` returns the original complete array, for arbitrary source-key lists, Boards, and input tails. |
| `optimized_filter_preserves_table` | `Chess.filter_prepare` returns the original complete array, including the initial check, full-filter branch, ray/blocker preparation, and optimized branch. |
| `complete_generator_preserves_table` | The array component of actual `Chess.legal_moves(table, board)` equals `table`. |

The public domains are actual `Array<U64>` values and arbitrary actual Boards. There
is no initialized-mask, balanced-array/depth, representation-consistency, singleton,
valid-turn, rights, or bounded-source premise for these **source storage equalities**.
The native test domain is narrower and is stated below. No native safety for
unsupported raw arguments follows from a source value equality.

The consumer specializes the full-generator result to the existing actual symbolic
initializer, with arbitrary construction parameters and seed. This instantiation
does not certify the initialized values: accepted geometry/initialization results
remain separate, unchanged, and reusable.

## Proof construction

`Bind` eliminates an actual returned pair using a proved equality of its array
component. It transports a continuation to the same stored array without assuming
the returned value or changing production ownership. `Query` reuses the accepted
arbitrary-storage read theorem for every actual piece class selected by the engine,
including queen's two sequential reads and the full attacked/in_check callback
chain. These helpers constrain storage, not attack values.

`Scan` follows the actual piece decoder's six Boolean cases, pawn targeting,
destination enumeration and recursion. It does not assume that an inconsistent
Board is decoded as a legal position. `Castle` handles both producer-guard outcomes.
`Filter` handles both values of the required-check flag, recursive full and fast
filtering, and actual blocker preparation. `Generator` composes actual scan,
kingside, queenside and `filter_prepare`. `Lift` constructs an existing reification
certificate for every actual input Array, so the public caller does not need to
supply a proof-only storage image or balanced shape.

The unchanged pinned CLI checked the complete importing consumer and producer
bodies. This is not a theorem about a replacement `legal_moves` implementation,
source-token matching, or an equality whose preservation premise is supplied by
the caller. It is also not an equality specifying the returned move list.

## Completed local source and native checks

The final full source command passed all three laws, its consumer, and all 17
controls together. The consumer returned exactly `All terms check.` with exit 0
in 4.052 seconds. Its complete proof/test/source manifest
contains 47 entries. Eight controls are intended source/refinement
rejections, eight guard law/proof/import inventories and unsafe/foreign/hole/symlink
boundaries, and one is a synthetic zero-exit warning-output test. The latter is not
a compiler execution. Missing imports, syntax/ownership failures, process kills,
crashes, and timeouts do not count as semantic rejection.

The source mutations target actual read corruption, attack-reducer storage loss,
scan state alteration, unchecked-filter state alteration, rejected-castle state
loss, and full-generator table reinitialization, plus an incorrect public result
and a missing continuation certificate. Each semantic rejection is checked for
expected/observed diagnostics at the intended new proof function.

A supplementary queen-mask mutation changes `or_result` from union to intersection.
The complete storage consumer still passes. That result is expected: **wrong attack
answers can preserve storage**. It is not counted as another law or rejection
control and does not authorize a move-correctness claim.

The native verifier passed generic, forced-portable, native-target, and UBSan modes.
Each mode executes **29 context/Board/tail requests** from eight base cases and
compares **2,638,112 logical U64 slots / 5,276,224 U32 limbs** across the four
post-operation checkpoints. Counts include repeated slots and requests; modes
repeat fixtures, not independent datasets or exhaustive Boards.

Every logical slot is observed before operations and after actual scan, both castle
calls, optimized filtering, and complete `legal_moves`. The five balanced contexts
are one nonzero leaf, 8 patterned slots, 512 patterned slots, the real 131,072-slot
`Tables.build()` with an unused nonzero marker, and 131,072 patterned slots that
are intentionally not valid attack tables. The host independently computes each
synthetic pattern. For the real builder, comparisons use the observed pre-operation
contents rather than claiming an independent all-mask initialization oracle.

Eight malformed bounded requests are rejected per clean mode. The probe may
produce output before rejecting a later malformed field; transactional validation
is not claimed. Native square inputs remain bounded and the fixtures have moving
kings. Arbitrary ragged source arrays are not a separately qualified native layout.

Three actual-code mutations erase slot 131071 at the scan, unchecked-filter, or
full-generator stage. All three compile and execute with generic flags, and **every
ordered move list agrees with the unmutated baseline**. Logical-slot comparison
still rejects all three, first observing high limb 0 instead of 826366246 at the
respective scan/filter/legal checkpoint. This catches state corruption that move
answers alone miss. It is a deliberate regression, not a found production defect.

The native driver observes complete ordered lists but independently specifies only
storage patterns and preservation. Empty-tail explicit-pipeline/full-generator
list parity uses the same underlying implementation, not an independent legal-move
oracle. Full logical-slot snapshots are stronger than one marker read but do not
prove heap identity, pointer identity, absence of temporary writes, allocation,
concurrency, or native ownership/lifetime safety. Raw bulk stdout is streamed to
temporary files; the full report retains snapshot hashes, ordered lists, counts,
and first mismatch diagnostics, not every raw slot line.

## Qualification boundaries and retained history

Original compiler checks passed: 16 source laws, seven rejection controls, and all
12 pin tests. Compiler source remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 inputs with fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Local Bun is 1.4.2. The native report records the actual C compiler.

Unchanged repository lint exited 1 because Ruff, Basedpyright and Vulture are not
installed. No locked-environment or hosted lint pass is claimed for this increment.
No fresh hosted qualification, independent review, or full aggregate wrapper was
performed. Self-review only. The retained parent has modular 193-law/623-control
records; adding this locally executed 3/17 gate gives **modular local evidence for
196 laws / 640 controls**, not 196 independent discoveries or a complete aggregate
execution. All 620 inherited native-source files remain byte-identical.

Initial source drafts included a raw numeric dispatch whose unreachable fallback
could not be reduced by the pinned checker. The replacement helper uses the
existing six-piece type; the actual decoder is proved to select those cases, so
the public scan/filter/full-generator domains are unchanged. A native draft failed
affine binder checks before its corrected probe compiled. An initial supplementary
boundary mutation changed a knight read address; the fixed proof term rejected it
at that changed address even though preservation could be reproved. The final
supplementary test instead changes queen union to intersection, isolating an answer
change without a changed storage proof. The initial gate was not called a pass.
Original available failure logs accompany the external review package. No old proof,
production function, or compiler input was weakened to repair these drafts.

## Handoff

The complete generator's source table-preservation gap is now established. Next,
combine this generic equality with the initialized geometric/castling results and
actual accepted-member/filter-path certificates. That remaining argument still
needs to specify accepted move values and safe stages, not just retained storage.
Universal forward/reverse attack correspondence, historical rights and complete
move-generation soundness/completeness remain separate obligations.

No production runtime, prior proof, compiler input, permanent workflow, model/GPU,
training, search, benchmark or routine perft budget changed. No new application
responsibility moved from Python. Pinned checker/Base, native lowering/storage,
ABI, toolchain, OS and hardware remain trust boundaries. Existing compiler,
snapshot-lowering and literal closed-builder limitations are unchanged.
