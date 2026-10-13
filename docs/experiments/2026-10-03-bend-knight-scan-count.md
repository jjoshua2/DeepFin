# Actual initialized knight scan_step query count

This increment now stacks on corrected draft PR #1010 head
`7e4ddc1d9e1ca20245b85fc077d4ae2800fe729a` (tree
`aa9bf30b956682bf68c007b5ab435dff62bf0c30`). Its initial qualification was
run on PR #1010 head `8956feb8a1d94d0bd58a682ec69e060aa00cd8b0` (tree
`66651f5aeda4fc76c1e95e3ef05709e96dc7ca8a`), which had the same proof-source
tree before evidence-only path redaction. The refreshed qualification reruns
the complete gate against the corrected parent. It composes the prior closed
full-Ply actual `scan_after` count with one real production caller:
`Chess.scan_step` for a source whose engine piece selector returns knight id 1.

The actual initialized knight attack query yields the independent coordinate
mask. Actual `Chess.piece_targets` removes friendly-side occupied squares and
all king squares; it retains the entire table component. The checked count then
filters that mask by the query destination and adds the exact full-Ply count of
an arbitrary caller tail, preserving duplicates.

Required local conditions are source square below 64, actual
`Chess.piece(board,source) == 1`, and a clear pawn bit so `scan_step` takes its
non-pawn branch. The table is the actual depth-17 `Tables.tables` loop followed
by 64 `Tables.extras` iterations; its initialized knight slot is provided by
the already-qualified attack-geometry theorem. No global board-validity,
reachability, king uniqueness, or target-geometry assumption is added.

Knight emissions have promotion 0 and flag 0 regardless of EP metadata; the
query factor still retains every full-Ply field. This establishes pseudo-attack
target membership and the one-source scan count, not `Chess.scan`,
`legal_moves`, king safety, or legal-move correctness. The zero-argument
`Tables.build` normalization issue remains open.

The pinned qualification runner, import-closure source hashes, positive
checker output, RSS/timing, and semantic negative controls are in
`native/bend_engine/standalone/proofs/knight_scan_count/evidence/2026-10-03/`.
An independent review checked the proof route and the actual `and_not`
operand order. No merge is requested.
