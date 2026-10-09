# Initialized knight source through actual scan_step counting

This opt-in increment stacks on PR #1010 head
`7e4ddc1d9e1ca20245b85fc077d4ae2800fe729a` (tree
`aa9bf30b956682bf68c007b5ab435dff62bf0c30`). It adds one bounded
production-caller composition for a knight source.

## Checked composition

`KnightScan.piece_targets_knight` uses the already-qualified initialized
non-slider query result from `attack_geometry/consumer.bend`. Its table is the
actual `Tables.tables` call followed by 64 actual `Tables.extras` iterations
over the 17-level allocation. Under that explicit actual initialization path,
`Chess.attack(1, ...)` at the selected source returns the independent
coordinate-defined knight mask. The implementation's `Chess.piece` selector
must return engine piece id 1, and the source square is below 64.

The actual `Chess.piece_targets(False{}, ...)` result is then exactly

```
G.mask(G.Knight{}, source)
  & ~(Chess.color(board, Chess.get_turn(board)) | Chess.get_kings(board))
```

This matches `Chess.exclude_targets`: friendly occupied squares and all king
squares are removed. `Chess.occupied(board)` is passed into the actual attack
query; the knight route reads its initialized slot and is independent of
occupancy. No other target filtering is inferred.

`scan_step_knight_closed_count` composes the target equality with actual
`Chess.scan_step`, actual `Chess.scan_after`, and PR #1010's full-Ply
occurrence-count theorem. For any query and arbitrary tail, the count is the
tail's full-Ply count plus the contribution of the queried destination if it
is in the computed target mask and all emitted fields match. The target is
in-range by construction; the formula keeps the explicit range check.
Because this is a knight, the generated promotion and flag are both zero,
independent of EP metadata. `query_emission_count` retains all four query
fields; duplicate tail occurrences remain counted.

## Premises and limits

- `source < 64`, `Chess.piece(board, source) == 1`, and the source pawn bit
  is clear. The latter is stated separately to select the non-pawn branch of
  actual `Chess.scan_step`.
- The table uses actual `Tables.tables` plus full extras at depth 17, with
  arbitrary seed and loop parameters; the already-qualified initialized-knight
  theorem discharges that slot query.
- No global Board validity, king uniqueness, reachability, or legal-move
  premise is added. Other overlapping planes are not ruled out beyond the
  explicit source conditions.
- This proves pseudo-attack target emission and a one-source scanner count. It
  does not prove `Chess.scan`, `legal_moves`, king safety, or legal-move
  soundness/completeness. It does not resolve the separate zero-argument
  `Tables.build` normalization issue or claim that issue is fixed.

## Qualification

Pinned Bend 2.0.21 + U64 revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`, and
Bun 1.4.2. The fail-closed gate checks the public consumer, hashes its import
closure and suite sources, caps positive checking at 86,400 seconds / two CPUs
/ 6 GiB / 64 KiB per output file, and requires intended failures for wrong
knight geometry, omitted friendly/king exclusion, wrong ordinary promotion/EP
fields, a dropped duplicate tail, and a disconnected scan consumer.

Run from the repository root:

```sh
python3 -m native.bend_engine.standalone.proofs.knight_scan_count.qualify_knight_scan_count \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /tmp/deepfin-pr1000-qualification-033774f6fecacbb079180705c16667ee609322b3/snapshot/native/bend_engine/standalone/proofs/bitboard_membership/evidence/2026-10-03/checker-tree.json \
  --report native/bend_engine/standalone/proofs/knight_scan_count/evidence/2026-10-03/qualification.json \
  --evidence-dir native/bend_engine/standalone/proofs/knight_scan_count/evidence/2026-10-03
```

The exact command, positive/negative stdout and stderr, timings, RSS figures,
compiler fingerprint, and transitive source hashes are retained in the evidence
directory. Prior checked `KingAway` and singleton inventory results are reused
as context only; neither is re-proved here.
