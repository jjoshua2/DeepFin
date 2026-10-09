# Actual initialized King scan_step query count

This increment stacks on PR #1012 head
8c69ac67e6eeeb73a2fde1d7f1b64b68ee14d63a (tree
210f51bf49e467d2d86932166da3ae18f2a730ac). Its local proof run pins the
tree-equivalent commit d48c6da394f3663876ac6fcff01fa9e859bd51ed.

## Checked composition

KingScan.piece_targets_king composes the generic initialized-leaper geometry
result at G.King{} with actual Chess.attack(5, ...), then follows
Chess.exclude_targets to remove friendly-side occupancy and all king-plane
squares. Chess.scan_step is specialized to the non-pawn branch with an explicit
source-pawn-clear premise.

The source conditions are local and explicit: its square is below 64, the King
plane is set there, and the actual piece selector returns ID 5. No global board
validity condition is introduced. For an arbitrary full-Ply query and an
arbitrary duplicate-containing tail, the count is the full-Ply tail count plus
one exactly when the query destination is in range and in the actual filtered
target mask and its source, promotion, and flag fields match. King emissions
have promotion 0 and flag 0; duplicate tail occurrences remain counted.

This proves pseudo-attack target membership and a one-source scanner count.
It does not prove legal-move correctness, king safety, the full Chess.scan, or
the separate zero-argument Tables.build normalization issue.

## Qualification

Pinned Bend 2.0.21 + U64 revision
aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae, verified 84-file fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4, Bun
1.4.2. The gate checks the public consumer under 86,400-second / two-CPU /
6 GiB / 64 KiB output limits and requires intended failures for wrong King
geometry, omitted friendly/King exclusion, wrong ordinary fields, a dropped
duplicate tail, and a disconnected scan consumer. The runner pins local base
commit d48c6da394f3663876ac6fcff01fa9e859bd51ed and tree
210f51bf49e467d2d86932166da3ae18f2a730ac.

The exact command, stdout/stderr, resource records, output hashes, and
import-closure source hashes are retained under evidence/2026-10-03/. The
checked knight caller remains separate; this increment adds only the King
source route.
