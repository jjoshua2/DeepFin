# Placement-parser prefix safety

This opt-in suite is based on PR #880. It starts the parser-freshness argument
with properties of the **actual** `Position.layout` traversal. It does not yet
prove that all accepted traversals write fresh squares or preserve board
partition consistency.

```sh
bun native/bend_engine/standalone/proofs/parser/focused.js /path/to/pinned/bend --report /tmp/parser-focused.json
bun native/bend_engine/standalone/proofs/parser/verify_native.js /path/to/pinned/bend --report /tmp/parser-native.json
# Expensive full chain, not implied by focused/modular evidence:
bun native/bend_engine/standalone/proofs/parser/verify.js /path/to/pinned/bend --report /tmp/parser-aggregate.json
```

The exact compiler and source fingerprint are checked before and after the gates.
Success requires status zero and exactly `All terms check.`; raw zero with unsafe
warnings is insufficient. No earlier accepted law or gate is changed.

## Five universal source contracts

| Contract | Actual guarantee |
| --- | --- |
| `layout_append_exact` | Parsing concatenation equals parsing the suffix from the actual prefix result, including the full Board, cursor and validity flag. |
| `invalid_layout_rejected` | Any suffix from an invalid Layout is rejected by actual `layout_result`. |
| `accepted_layout_has_valid_prefix` | Acceptance of a concatenated parse entails a true validity flag after the chosen prefix. |
| `layout_preserves_metadata` | Arbitrary actual placement traversal leaves turn, castling rights and en-passant fields unchanged. |
| `typed_piece_transition` | All six typed pieces and both colors have the exact actual one-character insertion, cursor increment and validity update. |

All quantify over actual `Position.Layout`/Board values and Strings. Metadata and
raw U32 cursor fields are arbitrary in the source statements. Typed piece tags
come from the existing board specification; the actual raw insertion is retained.
The append and typed-transition equalities include the complete resulting Layout,
not a scalar observation or a surrogate parser's state.

`Dispatch` proves a motive-based elimination of the real character dispatch. Its
33 disjoint Boolean cases exhaust a U32 character: slash, or the first differing
bit from slash. This is not a host-generated answer table. The branch helpers are
connected to the actual function by the checker; a mutated slash transition is
rejected at that connection. `Actual` inducts over String constructors for exact
composition and sticky invalidity. `Metadata` separately composes actual insertion's
accepted metadata frame. `Typed` eliminates the twelve piece/color encodings.

## Preconditions are not parser-wide validation

An arbitrary input Layout may already contain pieces, a bad cursor or an invalid
flag. The laws cover those raw states without pretending they are valid FEN input.
A valid prefix need not be a complete accepted placement. The proof of accepted
prefix validity is a safety prerequisite for cursor reasoning, not proof of
nonoverlap, freshness, rank bounds or legal reachability.

The raw parser can mutate its Board before reporting failure. The consumer shows
that parsing `x` from the empty start state sets a black occupancy bit while
`layout_result` returns None. Thus rejection is not a rollback guarantee for the
raw Layout object. This is documented actual helper behavior, not a demonstrated
bug in the outer FEN parser, which discards the failed result.

The occupied insertion at an out-of-range raw cursor follows the actual U32
arithmetic and saturating U64 bit constructor. It is not assumed to wrap modulo64.
No six-field FEN validation, metadata validity, parser-to-abstract-board theorem,
king count, check, move legality or reachability follows from these contracts.

## Native and rejection checks

The native probe executes actual `layout` and `layout_result` on prefix, split
suffix and concatenated strings. It exposes the entire three resulting states:
file/rank, validity, acceptance, eight bitboards and three metadata fields (69
U32 fields per row). The independent reference uses square sets of piece kinds
and colors plus a character/cursor interpreter. It does not supply the candidate
with expected states or invoke proof code in native execution.

Fixtures include every square/piece/color single-piece placement, every split of
empty/start layouts, invalid prefixes, malformed characters, Unicode, boundary
cursors and arbitrary metadata over three different Board seeds. Native inputs
are Unicode strings without embedded NUL, total length at most512, initial file
0..9/rank0..8, and at most64 rows per invocation. The bounded probe rejects nine
malformed requests; these are not additional raw-parser or whole-FEN theorems.
Modes repeat the same 1,295 fixtures, not disjoint or exhaustive board/String sets.

Seventeen source controls are eight ordinary semantic/refinement failures, eight
manifest/import safety checks and one synthetic unsafe-warning unit. The latter
is not a compiler execution. Semantic controls require ordinary diagnostics at
specified locations; crashes, missing names/imports, affine errors and timeouts
are not accepted. Native controls mutate real Position code: reviving invalidity
on a digit and discarding all characters. Both must compile and execute before
an independent wrong-state assertion rejects them.

The full wrapper retains the 132-law/367-control parent then adds five laws and
17 controls. Retained exact-source parent evidence may be used for modular
137/384 qualification, but must not be called a newly executed full wrapper.

No production code or Python application responsibility moves in this increment.
Source results trust the pinned checker/Base; native lowering, storage/lifetime,
ABI, toolchain and hardware remain separate trust boundaries. The existing
structural snapshot lowering limitation remains unchanged and outside this probe.
