# Actual back-rank pawn exclusion through replay

This increment is stacked on verified PR1073 head `ff1aada6940831ec381a9638eb64e7b285f03e41` (tree `804645b8691cd5498ddac25f049606b1f28eef23`). Production code and inherited proofs are unchanged.

`Mask.safe(b)` is the exact last `Position.structural` conjunct: the pawn mask intersected with `U64.from_parts(4278190080,255)` is zero. `Emission.legal_member` proves this predicate for the actual `Chess.make_move` child of every full-Ply legal-list occurrence, with parent exclusion as its sole board premise. It allows arbitrary arrays, turn values, rights, EP metadata, overlapping planes, and duplicate occurrences. It does not assume row validity, canonical turn, EP coherence, source ownership or target geometry.

The small obligations retain actual pawn-first decoding and the actual promotion decision. Every scanned pawn destination on rank 0 or 7 emits promotions 1–4 with flag 0. Otherwise promotion is 0 and the flag is `Bool.to_u32(pawn && dst == ep)`. A nonpawn cannot add to the pawn plane. `Rank` checks the 64 square cases from the previously checked nonempty-ctz range; that bound is derived internally from the actual mask. Arbitrary fuel and coherent empty flags follow `destinations` exactly. The caller tail is checked at every physical cons, preserving repeated full-Ply occurrences. `scan_after` receives and returns the actual `(table, targets)` pair. Castle sources are 4 or 60 even for noncanonical turn, so parent exclusion implies nonpawn decoding there. The proof does not assume they decode as kings.

`Mask.post` checks the exact pawn update including ordinary, EP and castle deletion masks. Deletion preserves exclusion; the generated insertion certificate rules out back-rank pawn addition. Both actual castle producers and the full/fast legal filter retain the property. `Apply` transfers it to the actual first full-Ply text match and `Position.apply`. `Replay` follows every ordered command, including duplicates, empty input, failed lookup and `None`, and retains the exact complete array. `consumer.use_replay` accepts any conditional safe input game.

`Root.valid` extracts the parent premise from actual `Position.valid`, via the actual structural conjunction. `Initialized.use_replay` adds back-rank exclusion to the identical actual replay result certified by PR1073's metadata and zero-right consumer. It retains parent rows/canonical turn, zero rights, exact initialization, depth 17, table count 128 and extra 64; back-rank exclusion is an explicit additional parent premise. It does not require parent EP coherence. The generic theorem has none of these initialization premises.

Fixtures check a multi-bit target mask with both back ranks and an ordinary EP target, a repeated caller tail, complete promotion/flag fields, ragged-table preservation, exhausted fuel, empty targets, first-match full flags, arbitrary duplicate command lists and both raw boundary failures. A raw ungenerated pawn move can add a back-rank pawn; an unsafe parent can retain one. Neither boundary is claimed safe.

This closes only the back-rank pawn-exclusion condition. King/material counts, surviving-right home-piece frames, full `Position.valid`, replay acceptance and full legal-move correctness remain outside the claim.

Qualification uses the existing pinned Bend 2.0.21+U64 checkout `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2 and verified 84-file fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`. The serial checker has 86400 seconds per positive or negative check, CPUs 1/3, two threads, 6 GiB AS/RSS limit and 16 MiB output cap. Evidence stays outside the worktree and includes exact snapshots, complete hash maps, raw commands/logs/rusage and single-declaration mutants. A control earns credit only for a semantic type mismatch at the intended obligation; parser, affine, inference, resource and transport failures do not count.

```sh
cd native/bend_engine
BUN="$HOME/.bun/bin/bun" python3 -m standalone.proofs.backrank_pawns.qualify_backrank_pawns \
  /path/to/pinned/checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh/qualification.json --evidence-dir /path/to/fresh/checks
```
