# Supplementary actual en-passant update refinement

This is new source-checked progress on actual flag1/promotion0 `Chess.make_move`,
not part of the nine saved public laws or hosted150-law/436-control totals.
The exact source files are archived as text for deliberate promotion into their
own gated suite. No production or existing proof file changed.

## Source claims and domain

`Actual.actual` proves that the entire actual Board result equals explicit
clearing and typed reinsertion, with the existing metadata update behavior.
The removal includes the source, target and additional capture square
`destination XOR 8`. `Actual.preserves` proves the resulting per-square partition
is consistent whenever the input Board is consistent. Both quantify over arbitrary
actual Boards and raw U32 source/destination arguments; the first does not need
consistency. No source-pawn, color, destination-emptiness, last-double-push or king
safety premise is supplied, and those legality properties are NOT established.

`Mask.same` structurally proves the actual three-square mask can use the earlier
arbitrary-mask deletion/target-freshness result. The proof reuses the existing
clearing, typed insertion, source-piece selection and complete Board bridge.
Metadata uses the unchanged Ordinary.rights/ep helpers. It is not a separate
independent chess-metadata specification. Source equality includes all eight
bitboards and all three metadata fields, not just an invariant Boolean.

The importing consumer uses both universal results. It also checks a white e5xd6
update removing d5 with an untouched distant rook, a black e4xd3 update removing
d4, and an arbitrary empty-source consistency consequence. The examples are
raw board operations, not legal-position reachability certificates.

The final source consumer completed in 2.860 seconds
with exit0 and exactly `All terms check.`. The complete 28-file source/probe
manifest is in report.json. Proof imports were traversed and checked for regular
files, no symlinks, no foreign/unsafe dependencies and no holes. Compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,84 source inputs,fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

## Fresh local native/reference checks

Generic and UBSan builds each passed **4,316 complete Board cases /82,004 U32
field comparisons**, plus nine invalid requests. There are4,096 raw source/destination
pairs,28 pawn-coordinate examples,64 arbitrary metadata cases and128 inconsistent
input diagnostics;4,188 inputs satisfy the representation partition. The last
128 cases compare exact behavior outside that premise, not invariant preservation.
The total counts describe fixture rows, not a claim that every Board/input is unique.
Modes repeat the same inputs and are not exhaustive board/game coverage.

The independent reference uses64-square sets of kinds/colors and clears the
three designated squares before inserting the selected kind. It compares every
bitboard and metadata field. The native probe imports only Base and production
Position/Text/Chess; it does not execute the proof representation or consume
reference-generated answers. Native square inputs are bounded0..63, narrower
than the source laws. Only generic and UBSan were executed for this supplement,
not the four-mode primary-suite qualification.

Common output SHA-256:
`3240f45616a9844ef16889cb211afea9bbbcbdea716d1185a364d7d8b5159131`.

Two disposable changes to ACTUAL capture selection were rejected at
`Actual.unfold` with ordinary expected/observed source diagnostics: omitting the
remote capture by using destination itself, and using XOR16 instead of XOR8.
Both compiled and executed before the external native reference rejected row0,
field1, observed257 rather than1. The old ordinary and promotion consumers still
passed each mutation, because their flag0 domains do not exercise this branch.
This demonstrates additional behavioral coverage, not a deficiency of those
correctly scoped previous proofs. Neither diagnostic increases registered counts.

The final portable Python driver completed without warnings. Its earlier version
emitted a Python invalid-escape SyntaxWarning in its diagnostic-regex literal;
the regex was changed to a raw string and the entire unchanged proof/native/mutation
sequence reran successfully. This was a driver warning, not a failed Bend proof,
C compilation, or ignored native sanitizer message. Original warning report/log
remain in the conversation review package. The initial source-only check also
passed; final sources reuse the existing metadata helpers instead of copying them.

## Reproduction

In a disposable checkout of the qualified primary source
`36c2ac370a00ea91f8f7e136d0167e0abcc24fd1`, copy the four `.bend.txt` files here to
`native/bend_engine/standalone/proofs/en_passant_review/`, stripping only `.txt`.
The archived verifier is ordinary Python source despite the text suffix:

```sh
python verify_review.py.txt /path/to/checkout /path/to/pinned/bend \
  --bun /path/to/bun --cc clang --report /tmp/ep-review.json
```

The driver checks compiler identity, safe source success, complete generic/UBSan
Board values, invalid probe requests, both source/native mutations and the two
previous source consumers on those mutants. Temporary mutation copies are deleted;
the input checkout and compiler remain untouched. A subprocess failure or timeout
is an error, not semantic rejection. Source hashes in report.json bind every
checked dependency and the probe; the driver hash is recorded separately.

## Limits and next acceptance

Self-review only,not independent review. These two derived statements and native
checks are local supplementary evidence, not separately hosted-qualified public
laws. Promotion into a registered LAWS/PROOF suite with its complete safe gate,
additional native modes and independent review remains next. Castling and the
connection from legal move generation to update preconditions/king safety remain
separate. Existing compiler,structural-snapshot and closed-builder limitations
remain. No application responsibility moved into Bend; no perft,model/GPU,training,
search,benchmark,production change,merge,deployment or live-process operation.
