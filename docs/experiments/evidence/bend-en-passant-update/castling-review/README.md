# Supplementary actual castling update and conditional partition preservation

These are newly checked derived source results, archived outside the public
en-passant suite. They do not add registered laws or controls to152/453. No
production function, old proof or compiler input changed.

## Exact guarantees

`Actual.actual` proves complete actual `Chess.make_move(b, Ply{src,dst,0,2})`
equality for arbitrary Boards, raw U32 source/destination and arbitrary metadata.
The result equals clearing source, destination and the selected rook source,
inserting the actual selected source kind at the destination, then inserting
a rook at the actual midpoint and updating metadata through existing helpers.
All eight bitboards and three metadata fields are included. The specification
is this raw-update composition, not an independently proved legal-chess update.

`Preserve.actual` proves partition preservation for arbitrary consistent Boards
and each of four coordinate routes:4->6,4->2,60->62,60->58, provided the **initial
rook destination is empty**. The expected rook targets are5,3,61,59. Route choice
does not prove the board has the matching side to move. The theorem's premise
observes initial occupancy only; it does not assume the desired output or a
post-insertion invariant. A structural Word lemma proves that clearing and the
first disjoint insertion preserve freshness of this rook destination. Existing
clearing/insertion producers then establish output consistency. Four finite
coordinate certificates discharge target separation and mask normalization.

The input-condition difference from ordinary updates is real: the production
castling branch adds the rook destination without first clearing it. The consumer
proves an actual consistent input containing a black bishop at f1 yields an
inconsistent result after the raw white kingside update, and separately proves
that this input fails the rook-destination freshness condition. This is a
counterexample to unconditional representation preservation, **not a demonstrated
production bug**: make_move is documented for legal-move producers. The consumer
also checks all four exact king/rook relocations and shows an empty-board raw
request may satisfy the sufficient consistency conditions without being legal.

No source-king/source-rook existence, legal turn/rights, other path-square
emptiness, attack-free king path, king safety, reachability or clock/history
correctness is asserted. The freshness condition is sufficient, not necessary.
No theorem about arbitrary flag combinations or promotion with castling follows.

## Checks actually executed

The importing consumer completed with exit0 and exactly `All terms check.`.
The final portable JavaScript review driver repeats that source check and both
native modes. Its completed receipt reports50.902seconds. An earlier enclosing
invocation was interrupted after both native modes, without a final receipt;
it receives no complete-gate pass credit. The bounded supervised rerun completed.

Generic and UBSan each pass4,360 distinct complete-Board inputs and82,840 U32
field comparisons, plus nine malformed requests. Cases comprise all4,096 raw
source/destination pairs,256 mixed-board route cases with empty rook destinations,
four blocked-rook cases and four empty-source cases. All4,360 inputs are partition
consistent;264 satisfy the preservation theorem's coordinate/freshness conditions.
All those264 results are consistent. Among all raw cases,4,175 outputs are
consistent and185 are deliberately outside-premise inconsistent results. The
whole suite does not assert unconditional preservation. Modes repeat fixtures;
no portable/native-target or exhaustive Board/game result is claimed here.

The candidate probe executes only actual Chess/Position operations, receives
raw Board and coordinate inputs, and does not import proof models or answers.
An external64-square set model checks all eight bitboards and metadata. Its
metadata policy follows the existing raw API, not separately proved chess rules.
Common output SHA-256:54a64d1f1021eb892853f42241552727fda47f492be4c7a1f7289a8b8df61547.

Two actual flag2 corruptions, omitting rook placement and selecting the wrong
rook source, are rejected ordinarily at `Actual.unfold`. Both compile/run and
fail independent native values (first test row, rook field:0 vs32 and160 vs32).
The first failing fixtures are intentionally blocked-rook inputs; the exact
update theorem covers those inputs even though preservation does not. The old
ordinary, promotion and en-passant consumers still pass both mutants, demonstrating
why those correctly scoped flag0/flag1 contracts cannot cover the flag2 branch.

Dropping the initial rook-freshness premise is rejected at `Preserve.actual`.
Missing imports, malformed terms, affine/termination errors, crashes and timeouts
are not accepted semantic rejections. These three source diagnostics and two
native corruptions are supplementary only, not additions to the public17 controls.
The source dependency closure contains28 regular hashed files with no
unsafe/foreign/hole imports; Base is supplied by the unchanged pinned compiler.

## Reproduction and trust

Copy all archived `.bend.txt` and `.js.txt` files to
`native/bend_engine/standalone/proofs/castling_review/` in a disposable checkout
of qualified source59acb49cb4395884741803b448842d2100054b1d, removing only `.txt`.
The `.js` verifier must retain that suffix so its module location is correct.
Then run:

```sh
BEND_NO_TELEMETRY=1 TERM=dumb bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/castling_review/consumer.bend
BEND_NO_TELEMETRY=1 TERM=dumb bun \
  native/bend_engine/standalone/proofs/castling_review/verify_review.js \
  /path/to/pinned/bend --report /tmp/castling-review.json
```

Pin:aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,84 inputs,fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Local Bun1.4.2/Clang17. The exact new source, full closure manifest, driver and
consumer/review receipts are preserved here. The driver creates disposable
mutants and verifies source/compiler identities afterward; it changes no original
source. Reproduction can vary timings and compiler identity without changing the
stated source results or fixture outcomes.

Self-review only, not independent review. Pinned checker/Base, native lowering,
allocation/storage, ABI, toolchain, OS and hardware remain boundaries. Existing
compiler, snapshot-lowering and closed-builder limitations are unchanged. No
application responsibility moved from Python. No model/GPU, search, training,
benchmark, added perft, merge or deployment occurred. Next acceptance is deliberate
public-suite promotion of these castling contracts, then deriving their premises
from legal-move producers and proving king safety and independent metadata.
