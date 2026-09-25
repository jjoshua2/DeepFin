# Public FEN consistency and exact result assembly

## Baseline, reconciliation and scope

Based on PR #882, `feat/bend-parser-frontier-20260925` at
`3ddb2be9d49d46d3848a35e15732adf1ec370c14`. The downloaded complete qualified
parent evidence tree supplies all 444 unchanged native-source entries; its later
publication commit is documentation-only. Hosted qualification must check out the
exact final parent and preserve that final documentation too.

The saved local freshness commit `f6047bffa22804fc7086a76dbaa4072659221a20`
contains five contracts, whereas the published frontier implementation contains
four. Their directories and proof implementations differ. Both are preserved.
The saved public `fen_result_is_consistent` result is not among the frontier's
four registered laws. This continuation adapts that final monadic composition to
the existing published placement producer, rather than duplicating all freshness
proofs. Two further contracts expose failed-placement rejection and exact Game
construction from validated actual field outcomes.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 plus
U64, 84-source fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Repository instructions, development/branch-lifecycle guidance and experiment
index were read. Work occurs only in an isolated proof/test checkout. No production
function, prior accepted law, compiler input or routine perft budget is changed.
No merge, deployment, model/GPU, search, training or benchmark is authorized here.

This is a constructive development record, not a backdated preregistration.
Acceptance requires exact safe source-consumer and closed-witness success;
classified source/behavioral/policy controls; four native modes and actual-code
mutations; original compiler checks; and unchanged repository lint. Failed
attempts remain failures. The full inherited aggregate is separately opt-in;
retained exact-source parent evidence does not imply a fresh aggregate run.

## Statements and assumptions

| Contract | Domain and result |
| --- | --- |
| `fen_result_is_consistent` | All six arbitrary input Strings; actual None or returned Game Board satisfies the existing independent partition invariant. |
| `rejected_placement_rejects_fen` | Actual initialized placement is None; actual public FEN is None regardless of all remaining fields. |
| `validated_fields_construct_exact_game` | Actual subparsers supply the specified placement/side/rights/EP/decimal results and clocks meet the actual bound/positivity checks; the complete actual Game equals the specified metadata-applied Board, exact clocks and Nil history. |

The first is unconditional on caller-supplied freshness or Board values. Its
proof consumes the published initialized-placement consistency and metadata
preservation results. The third is explicitly conditional on actual validator
outputs; it is not an independent proof of their lexical correctness or an
assumed equality of the final Game. Board, metadata, clocks and history are all
retained in the returned value.

The consumer includes symbolic Strings/certificates and a genuinely accepted
nonempty FEN with different clocks and nondefault metadata. Closed witnesses
separately ensure nonvacuity and invalid-placement rejection. The candidate is
not permitted to pass solely by returning None for all inputs.

## Qualification status before hosted execution

Final local/hosted receipts will be appended only after their checks complete.
The initially accepted source consumer is not a claim of complete gate success.
Earlier synchronous gate invocations exceeded their enclosing command limits
while comparing symbolic corrupted functions. The final classified control design
uses closed actual-output witnesses for selected implementation mutations;
those are not mislabeled as isolated universal theorem failures. Universal
bounds/premise controls remain separately required. The public statements and
compiler are unchanged. A JavaScript duplicate variable and native String binder
error were fixed before final gate execution; those failed drafts receive no
semantic-control or native-pass credit.

## Remaining work and trust

This closes the optional public-FEN consistency composition, not exact arbitrary
placement contents, full lexical acceptance completeness, semantic metadata
validity, legal reachability, king safety or move correctness. Parser freshness
was already proved by the qualified frontier parent and is not counted again.
The next P3 targets remain exact parsed-square correspondence and invariant-
preserving removal/move/special-move operations.

Self-review only, not independent review. Checker/Base, native lowering/storage,
ABI, toolchain, OS and hardware remain trust boundaries. Existing compiler and
snapshot-lowering limits remain unresolved. No additional Python application
responsibility moves into Bend; export, references, data/control/training and
transitional C++/LibTorch/AOTI remain dependencies.
