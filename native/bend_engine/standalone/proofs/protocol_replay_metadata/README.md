# Nonempty actual protocol replay bootstraps ready metadata

Stacked on exact PR1070 head `d8d5cd8caea9923296cb0be096d6d503c5fdf1a9`.
Production source and all prior proof/control files remain exact base blobs.

The new generic consumer follows actual
`Protocol.moves(Con{text,tail},(table,maybe))` for arbitrary first text and arbitrary
ordered, duplicate-containing command tail. Conditional parent row partition and
canonical turn suffice. No input EP bound, `Position.valid`, king-presence or
target-geometry premise is supplied. The result preserves the complete input array
and, for every successful returned Game, proves row partition, canonical turn and
EP equal to sentinel `64` or strictly below `64`.

`Apply.legal` uses the actual `child_metadata.Source.legal` affine continuation for
source bounds and full-Ply fields on the same legal-moves pair as table preservation.
PR1070 `Find.all` transports those occurrence certificates through the exact first
`Position.find_move` match. `child_metadata.Post` proves the actual selected update's
rows, flipped turn and bounded output EP; PR1070 `Apply.board` connects it to actual
`Position.apply`. Canonicality is repackaged after each move instead of claiming
fixed parity relative to the initial side. `Replay.tail` follows every subsequent
actual `apply_one` and exact returned pair; `None` and failed lookups remain covered.

The nonempty command shape matters: an empty replay preserves arbitrary input EP.
Fixtures include a raw row-consistent, canonical-turn Game with EP `65`, demonstrate
that empty replay retains it, and check actual first-move replacement with EP `20`.
Two successful commands preserve complete counters/history and produce turn `1`
and EP `44`; first/second failures and initial `None` are covered. Literal zero-leaf
fixtures exercise the actual path without certifying initialized attack tables.

The public FEN consumer obtains initial rows and canonical turn internally from
`Rows.fen` and `Turn.fen`, follows actual `Protocol.validate`, and retains the explicit
initialization equation and literal `d == 17n`, `n == 128n`, `extra == 64n` in its
output. Validation alone does not canonicalize arbitrary raw Games. No closed
`Init.run` normalization is repeated.

This adds a necessary metadata premise toward replay equivalence. It does not prove
EP coherence, child `Position.valid`, king presence, replay acceptance, legal-move
correctness or arbitrary per-command fast/full equivalence. EP range and coherence
are separate properties. All prior PR1070 controls remain unchanged; the new strict
controls check actual source-bound coupling, canonicality, first/recursive application,
EP values, parity, parser producers and the initialization contract.

Pinned checker: Bend `2.0.21+U64`, commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, verified 84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Checks run serially on CPUs 1 and 3, two threads, 6 GiB address-space/RSS cap,
16 MiB output cap and 86400-second positive/negative wall timeout. Fresh snapshots,
raw streams, timing/RSS, exact mutants, commands and hashes live outside checkouts.
Frozen input/runtime/Git guards and one-declaration strict type rejection are required;
parser, import, resource, timeout and inference failures do not count as controls.

From `native/bend_engine` in the isolated worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 BUN="$HOME/.bun/bin/bun" taskset -c 1,3 \
  python3 -m standalone.proofs.protocol_replay_metadata.qualify_protocol_replay_metadata \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /absolute/new-evidence/checker-tree.json \
  --report /absolute/new-evidence/qualification.json \
  --evidence-dir /absolute/new-evidence/fresh-checks
```
