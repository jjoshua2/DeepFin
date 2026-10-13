# Actual pawn intersection counts through generation and replay

This increment is stacked on checked PR1079 head
`7cddd6019094421cb5e3624cf5544b58e1535609`. It proves two more actual
`Pos.valid` structural fields: `U64.popcount(pawns & white) <= 8` and
`U64.popcount(pawns & black) <= 8`.

`Fusion.bend` proves that the actual removal mask, which includes the
destination, lets the two updated planes fuse into one updated intersection.
This clears ghost pawn bits before a nonpawn move repaints a destination.
`Selector.bend` connects the actual promotion/piece selector to source pawn
membership; nonzero promotions cannot insert a pawn. `Intersection.bend`
combines source pawn and active color membership. `Actual.post` then uses
PR1079's structural count lemmas: deletion of the owned joint source pays
for at most one insertion, and inactive or nonpawn intersections only delete.

The generic child theorem needs only an owned, in-range source, in-range
destination, and noncastle full Ply. Arbitrary raw turns, overlapping planes,
coincident source/destination, EP metadata and other piece planes are allowed.
The active branch is exactly `turn == 1`, matching actual `make_move` and
source selection. No board partition, canonical turn or target geometry is
assumed. Source/range/noncastle certificates are reused from the exact base's
actual producers; zero rights excludes their castle branches.

`Bounds.bend` connects Nat intersection counts to the actual U32 comparator
through count-value cases, rather than bitboard enumeration. `Root.valid`
extracts the two initial bounds from actual `Pos.valid`. `consumer.use_child`
requires no initial count bounds. `consumer.use_replay` additionally takes
the two initial pawn bounds and proves, for actual ordered `Protocol.moves`,
the exact input table, zero rights, both bounds and Nat counts no larger than
the initial counts for every surviving `Some`. `None` remains conditional
failure. Replay processes every command cons, including duplicate tails;
lookup retains the first complete matching Ply and its promotion and flag.

`Initialized.use_replay` composes the same actual result with PR1079's
certificates, retaining metadata, zero rights, back-rank pawn exclusion,
both color bounds of 16, exact `Init.run`, depth 17, 128 tables and extra 64.
It adds both initial pawn bounds. King count preservation, full child
`Pos.valid`, replay acceptance and full legal-move correctness remain outside
this increment.

Fixtures check actual multibit targets 0/16/63 and a duplicate-containing
tail with 12 full-Ply occurrences, exact ragged table identity, fuel/empty
behavior, promotions 1–4 with flag 0, and ordinary promotion 0 with
`Bool(pawn && destination == ep)`. They check EP removal of destination and
XOR-8 victim, arbitrary turn 2, promotion removing a pawn, ghost destination
clearing and overlapping colors. Counterexamples show why unowned raw moves
and arbitrary castle plies cannot support unconditional joint-count claims.

From `native/bend_engine`, run:

```sh
python3 -m standalone.proofs.pawn_counts.qualify_pawn_counts CHECKER \
  --checker-manifest CHECKER_TREE_JSON --report FRESH_REPORT \
  --evidence-dir FRESH_EVIDENCE
```

The qualifier requires Bend 2.0.21+U64 at
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2 and verified
84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
It checks three entries and 37 isolated semantic mutants, with 86400 seconds
per check, CPUs 1 and 3, two threads, 6 GiB address-space/RSS ceilings and
16 MiB output caps. Every run has before/after source, snapshot, Git and
compiler guards; reused dependencies must match the exact base's Git blobs.
Controls include destination clearing, decoder/joint ownership, initial
bounds, exact table, producer fields, EP, duplicates and disconnected replay.
Parser, inference, affine-use, resource and timeout errors receive no semantic
credit. A positive-only run is an intermediate milestone, never completed
qualification. Evidence stays external to the source worktree.
