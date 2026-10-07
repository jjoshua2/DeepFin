# Actual parsed FEN row certificate

This increment supplies the remaining caller row-partition premise of PR1068 from the actual parser. For arbitrary six input strings, `Rows.fen` produces `Root.rows(Position.fen(...))`: `None` has a `Unit` certificate; every `Some{game}` has `board/Spec.valid(Position.board(game)) == True`.

The placement producer already exists in `frontier/consumer.arbitrary_placement`. This suite connects it through the exact initialized `Layout{Position.empty(),0,7,True}` and the actual side, rights, EP, halfmove number/bound, fullmove number/positive/bound and metadata binds. Invalid temporary layout states are rejected; no certificate for their raw temporary boards is asserted. Metadata preserves the piece/color partition. This is not a new proof of the cursor induction.

`consumer.use_fen_step` and `consumer.use_fen_singleton` consume this certificate internally and call the existing checked PR1068 consumers. For arbitrary FEN strings, move text and actual array `a`, their only proof inputs are the exact initialized-array equality `a == initialized_attacks/Build.run(d,seed,n,extra)` and `d == 17n`, `n == 128n`, `extra == 64n`. No caller parsed rows, FEN acceptance, canonical turn, king presence, board geometry or legal-position premise is added.

The resulting contracts are the existing `fen_validated_full/Spec.step` and `singleton`: actual `Protocol.apply_one` and singleton `Protocol.moves`, after actual `Protocol.validate(Position.fen(...))`, equal full filtering of the exact ordered actual candidates and actual first matching complete Ply, with the same array, text, game, history and counters. Both initial-check branches and parser/validator failure branches remain included. The prior full-Ply occurrence proofs preserve all duplicate occurrences.

This connects one actual parser/validator/command path under an explicit initialized-table contract. It does not prove default `Tables.build` conversion, arbitrary multi-command replay, child board invariants, target generation completeness, reachability, or full legal-move correctness. A syntactically accepted empty board demonstrates why parser row partition differs from `Position.valid` and chess legality.

Qualification uses the pinned Bend 2.0.21+U64 compiler revision `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified 84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. The driver checks `Rows.bend` and `Fixtures.bend` (which imports and calls both public consumers), then isolated declaration mutations. Each positive and negative check gets 86400 seconds, CPUs 1 and 3, a 6 GiB address-space/RSS cap and 16 MiB output file caps. Only a single typed mismatch at the declared obligation counts as a negative rejection; inference, syntax, import, ownership, timeout and resource failures do not count.

Run from an isolated clean stacked worktree with fresh report and evidence paths:

```sh
python3 -B -m native.bend_engine.standalone.proofs.parsed_fen_rows.qualify_parsed_fen_rows /tmp/deepfin-king-away-checker-aaeb9bc --checker-manifest /path/to/checker-tree.json --report /path/to/Q001.json --evidence-dir /path/to/Q001
```

The report records exact Git base/head/tree, per-entry dependency closures, complete source/compiler snapshots and hashes, bounded commands, raw checker output, resource use, before/after guards and isolated mutant inventories. This suite is stacked on verified PR1068 head `3bd6053b18559193c6aab5a0daa2f75e713ce94b`, tree `1d28a886f6e473ef3939a324e63865599453a426`. All imported dependencies must remain exact Git blobs from that base. Checked local candidates and independent review receipts are kept outside the source tree; publication is draft only and does not merge the stack.
