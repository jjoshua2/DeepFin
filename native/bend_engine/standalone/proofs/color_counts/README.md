# Actual color counts through generation and replay

This increment is stacked on the exact checked PR1075 head
`9d6b4a9524cb34d2a635a31f4b3c5b76db31143a`. It adds two previously
unproved structural fields of actual `Pos.valid`: the white and black
`U64.popcount` bounds of 16.

`Words.bend` proves deletion cannot increase count, deletion of an owned bit
pays one count unit, and singleton insertion costs at most one unit.
`U64Count.bend` connects these structural lemmas to the actual U64 operations,
bit constructor, and test-bit selector. `Actual.post` proves both Nat counts
cannot increase for an owned, in-range source, in-range destination, and a
noncastle full Ply. The ownership branch is exactly `turn == 1`, matching
both actual `Chess.color` and `Chess.make_move`; arbitrary raw turn values
are allowed. Color planes may overlap, the source and destination may
coincide, and the EP metadata and other piece planes are unrestricted.

`Emission.bend` derives those certificates from actual `bit_squares`,
`destinations`, `scan_after`, `scan`, and legal filtering. The destination
scanner retains its actual fuel and coherent empty flag. `scan_after`
consumes its actual `(table, targets)` pair. A pawn destination on ranks
0 or 7 emits promotions 1 through 4 with flag 0; other destinations use
promotion 0 and flag `Bool(pawn && destination == ep)`. Each physical
destination cons and each promotion is preserved, including duplicates
already in a certified tail. Zero rights excludes the two actual castle
producers; no general castle color-count preservation is claimed.

`Bounds.bend` connects the structural Nat count to the implementation's
actual U32 comparator, using count-value cases (0 through 64 and 0 through
16), not a bitboard enumeration. The actual initial bounds plus count
nonincrease imply the actual child bounds. `Root.valid` extracts the
initial bounds from actual `Pos.valid`.

`consumer.use_child` proves count nonincrease for any generated legal
member under zero rights, without initial count bounds. `consumer.use_replay`
additionally requires both initial actual color counts to be at most 16.
For the actual ordered `Protocol.moves` result it preserves the exact
input table; every surviving `Some` result has zero rights, both bounds,
and Nat counts at most the initial counts. `None` is conditional failure,
not evidence of move acceptance. Lookup uses the first matching complete
Ply, including its promotion and EP flag. Replay induction processes every
physical command cons, allowing duplicate-containing arbitrary tails.

`Initialized.use_replay` composes this same actual replay result with all
of PR1075's initialized certificates. It retains the conditional metadata,
zero-right and back-rank pawn premises and all four original initialization
certificates: exact `Init.run`, depth 17, 128 tables, and extra 64. The only
added structural premise is the two initial color bounds. This increment
does not establish full child `Pos.valid`, king counts, pawn counts at most
8, replay acceptance, or full legal-move correctness.

`Fixtures.bend` checks multibit targets 0/16/63 with a duplicate-containing
tail, all 12 physical occurrences and complete fields, ragged table identity,
fuel exhaustion, empty targets, back-rank promotions versus EP, raw turn 2,
EP removal of both destination and XOR-8 victim, and first full-Ply lookup.
It also checks counterexamples showing why an unowned raw move and a
castle with a missing rook cannot support an unconditional count claim.

Run the frozen qualifier from `native/bend_engine`:

```sh
python3 -m standalone.proofs.color_counts.qualify_color_counts CHECKER \
  --checker-manifest CHECKER_TREE_JSON --report FRESH_REPORT \
  --evidence-dir FRESH_EVIDENCE
```

The qualifier requires Bend 2.0.21+U64 at
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified
84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
It checks three entries and 30 isolated semantic mutants, with 86400 seconds
per check, CPUs 1 and 3, two threads, 6 GiB address-space/RSS ceilings,
and 16 MiB output caps. It checks source, snapshot, frozen Git tree and
compiler hashes before and after every run. Parser, inference, affine-use,
resource and timeout failures receive no semantic-rejection credit.
Evidence is external to the source worktree. A positive-only run is an
intermediate milestone and never a completed qualification.
