# Ordinary original-side no-check preservation

This increment stacks on PR1034 commit
`65a17ba07d28725d54d31419469bf6c285ddd315` (tree
`34375611bffe023ef71a6d46df7beb43a4a2ea26`). It proves the actual
`Chess.in_check` returned pair after an ordinary `Chess.Ply{src,dst,0,0}`
candidate, at the original side's preserved king, using the exact input table.

`use_no_check` requires actual candidate membership, `Board.valid`, a nonempty
original-side king plane, canonical turn 0/1, the actual prepared
`filter_requires == False` bypass, actual initial `in_check == (table,False)`,
and separate whole-pair old and new rook/bishop lookup contracts at that king.
The post-move contracts use post occupancy and the original enemy side. These
are conditional lookup contracts, not a theorem about the table builder.

`Fixed` follows the production pawn/knight/king queries (kinds 0/1/5), including
the reverse pawn orientation and the exact returned table. Their masks ignore
occupancy. `Prefix` combines these masks with the ordinary enemy-kind subset
results from the king frame and proves preservation of enemy-masked hits.
`Compose` connects the fixed query lemmas to the actual callbacks and combines
the empty enemy-masked prefix with PR1034's actual zero-prefix slider callback.
The consumer derives its source/destination certificate from actual candidate
membership and composes king selection, initial no-check, both preservations,
and the full production `in_check` call for the original side.

The arbitrary 512-cell fixture has all enemy piece kinds and an actual knight
move. Friendly pawn and king hits keep the raw prefix nonzero after the move;
its enemy-masked prefix is empty. The original-side post check is False while
the check selected by the flipped metadata turn is True. A wrong pawn query
orientation also produces True. No post `Board.valid`, king uniqueness, target
geometry, unconditional full legality, promotion, castle, or builder-validity
claim is made.

Qualification uses Bend 2.0.21 + U64 at
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified
84-file checker fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Positive and negative checks each have 86400 seconds, two CPUs, a 6 GiB address
space/RSS bound, and a 16 MiB output-file cap. Evidence paths must be fresh and
outside the checkout. The harness binds source hashes and Git blobs to the
exact base and immutable published head.

The 27 controls comprise 21 contract-coupling checks and six concrete false
witnesses. They cover query kind/orientation/table identity, retained raw hits,
all three enemy-kind transfers and their color mask, canonical turn, production
callback connection, actual candidate/board/bypass/initial/new-lookup premises,
original-side selection, both preservation consumers, ordinary promotion scope,
the friendly-prefix/opposite-side/orientation fixtures, and inherited full-Ply
duplicate-tail and EP-flag fixtures. Each control requires a single typed
mismatch at its exact named declaration; parse, import, linearity, resource and
incidental dependency failures do not count. The smaller structural controls
check their actual module entry; consumer controls check the full consumer.

```sh
python -B -m native.bend_engine.standalone.proofs.ordinary_nonslider_preservation.qualify_ordinary_nonslider_preservation \
  /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```
