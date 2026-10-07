# Actual castling-right irreversibility through replay

This suite proves properties of the actual `Chess.make_move` rights field for
arbitrary boards, U32 rights and full promotion/flag-bearing Ply values. It then
follows actual `Position.apply`, `find_move`, `Protocol.move_apply`, the returned
`legal_moves` pair, `apply_one` and `moves`. Every successful replay result has
rights that are a bitwise subset of the starting rights, and the complete input
array is retained exactly. Commands can be empty, arbitrary, repeated or fail.

`Words.bend` uses structural Word induction over all 32 U32 bits, including high
bits. `Rights.exact` checks the actual update against its exact loss expression:
the union of source/destination corner bits and the pre-move decoded king mask.
`Rights.source` and `destination` revoke those corner bits; `Rights.king` revokes
3 when the implementation's turn compares equal to 1, otherwise 12. Its premise
is the actual `Chess.piece` decoder returning 5. Empty-source fallback decoding
and noncanonical turns are deliberately covered; this is not a board-validity
or legal-move theorem. Promotion and flag do not replace the decoded kind.

`consumer.use_game` produces the unconditional subset result for any initial
Game. `consumer.use_zero` needs conditional zero input rights, then certifies zero
output rights and each actual `Position.valid_right` check (1/7/4/1, 2/0/4/1,
4/63/60/0, 8/56/60/0). Both preserve exact array contents and shape. First text
selection retains the complete actual Ply; rights hold for every raw Ply, so no
generated-move membership or target-geometry premise is needed for this result.

`Initialized.use_replay` combines zero-right replay with PR1072's existing
initialized nonempty replay metadata consumer. It retains exact initialization
and literal depth 17, table count 128 and extra-row count 64 in the public output.
The existing conditional parent row partition and canonical-turn premises remain;
conditional zero rights is added. Returned Games then have row partition,
canonical turn, EP range/coherence and the four right-validity checks. No parent
EP-coherence premise is added. Structural king/material/pawn counts and back-rank
pawn exclusion remain open, so this does not establish full `Position.valid`.

Fixtures check all four corner bits, high-bit preservation, decoded-source versus
promotion, raw turn 2, shared-text full-flag first matches in both orders, empty
and duplicate commands and `None`. A raw pawn 16→8 with EP flag 1 removes the home
rook at 0 without revoking bit 2; its surviving `valid_right` check is false. This
is an actual raw-update boundary witness, not an initialized legal-list witness.
Validity of surviving rights still requires separate home-piece frame proofs.

The stacked base is PR1072 published head
`0ac15707bf21f90c14cc22cddd295b152978a2a4`, tree
`287f58a0868db13ceead8a371d39c5580cd6e64f`. Production code, prior providers and
controls are exact base blobs. No GPU, live checkout, merge or external reviewer
is involved.

## Reproduction

Use the already pinned checker, not the auto-updating Bend launcher. Bend is
`2.0.21+U64`, commit `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2.
The 84-file checker fingerprint is
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

Run from `native/bend_engine` on this task's isolated branch, with new evidence
and report paths outside all checkouts:

```sh
ulimit -v 6291456
ulimit -f 16384
export PYTHONDONTWRITEBYTECODE=1
export BUN="$HOME/.bun/bin/bun"
taskset -c 1,3 python3 -m standalone.proofs.castle_rights.qualify_castle_rights \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest "$HOME/chess-artifacts/deepfin-castle-rights-20261007/evidence/2026-10-07/checker-tree.json" \
  --report "$HOME/chess-artifacts/deepfin-castle-rights-reproduction/qualification.json" \
  --evidence-dir "$HOME/chess-artifacts/deepfin-castle-rights-reproduction/checks"
```

Three positive entries and 25 exact mutations are checked serially, with 86400
seconds/check, two threads, CPUs 1 and 3, 6 GiB address-space/RSS and 16 MiB output.
Controls must fail by a type inequality at their declared obligation; parser,
affine-use, inference or resource failures receive no credit. Guards bind sources,
snapshot, mutants, exact Git head/tree, dependencies and all checker files before
and after every check. The frozen report, raw streams, commands, timing/RSS,
snapshots and independent review are banked under the evidence path above.
