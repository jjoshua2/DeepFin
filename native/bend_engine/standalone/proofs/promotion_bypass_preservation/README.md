# Promotion original-side no-check preservation

This increment stacks on PR1035 commit
`6f22ed9d9ec555f1c0c621ec7100708bf5b37aff` (tree
`077f717e4c2cea71597c1af170bca79093169112`). It proves the actual
`Chess.in_check` returned pair after a generated typed promotion
`P.ply(src,dst,choice)`, with tags 1..4 and flag 0, for the original side
using the exact input table.

`use_no_check` requires full-Ply membership in actual `C.candidates`,
`Board.valid`, a nonempty original-side king plane, canonical turn 0/1,
the actual prepared `filter_requires == False` bypass, actual initial
`in_check == (table,False)`, and separate whole-pair old and actual promotion
post rook/bishop lookup contracts at that king. Post contracts use actual post
occupancy and the original enemy side. They are explicit conditional lookup
contracts; the initialized table builder is not proved here.

`Raw` follows the existing actual promotion dispatch and typed tag selection.
Its color planes and occupancy equal those of the ordinary comparison board.
The promotion never inserts a king. Equality of the selected original-side
king also uses the source-not-king fact derived from actual bypass, source
ownership, source bounds and board consistency. `Enemy` proves all six
enemy-masked kind planes equal the ordinary comparison after the same removal,
using canonical turn. Unmasked kind planes can differ.

`Compare` combines these derived equalities into equal enemy-masked prefix and
slider hits. It transfers the actual promotion-post whole-pair lookup contracts
to the ordinary comparison by occupancy equality, then connects both sides to
the production pawn/knight/king and slider callbacks. The consumer derives
source/destination certificates from actual promotion membership, projects
their field-independent predicates to the ordinary comparison, invokes
PR1035's certificate consumer, and returns the actual promotion-post check
pair for the original side. The ordinary comparison move need not be a
candidate; no shadow candidate membership is assumed.

The arbitrary 512-cell fixture contains all enemy piece kinds and generates
exactly the four promotions 49 -> 57. It checks all four memberships, selected
piece tags, actual bypasses, lookup contracts and original-side no-check
results. The ordinary comparison candidate count is zero. Friendly inserted
knight hits make the raw prefix differ between boards, while enemy-masked
prefix hits agree. Original-side check is False, and the check selected by
the flipped metadata turn is True. Separate fixtures show that flag 1 always
forces checking, while raw flag 2 alone can bypass.

## Remaining actual bypass obligations

`Chess.filter_requires` tests flag 1 and the sensitive source; it ignores
promotion. `Chess.put_move(True,...)` emits tags 1..4 with flag 0, including
when its EP input is True. EP flag 1 always forces. An owned in-range king
source forces every promotion/flag through the existing sensitive-mask lemma.
Actual guarded castle emissions use that king source and existing mask and
pipeline certificates; no blanket claim for arbitrary raw flag 2 is made.
PR1035 covers ordinary tag 0/flag 0 bypasses, and this suite adds typed tags
1..4/flag 0 bypasses under the stated common premises.

Remaining work toward actual fast/full equivalence is exhaustive full-Ply
classification over actual scan and guarded castle lists, per-occurrence
composition with the existing ordered-list equivalence while preserving
duplicates, and derivation of the explicit old/post slider lookup contracts
from the initialized table builder and actual occupancies. Board consistency,
canonical/nonempty king and actual initial-check branches remain explicit.
This suite does not prove full move legality, post `Board.valid`, king
uniqueness, target geometry, EP/castle preservation or blanket fast/full
equivalence. `castle_chain.Spec.ordinary` means flag != 2 and must not be
confused with PR1035's tag 0/flag 0 scope.

## Qualification

Use Bend 2.0.21 + U64 at
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified
84-file checker fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Positive and negative checks each have 86400 seconds, two CPUs, a 6 GiB address
space/RSS bound and a 16 MiB output-file cap. Use fresh report/evidence paths
outside the checkout. The harness binds hashes and Git blobs to this exact
base and immutable published head and checks the compiler identity again.

The 29 controls comprise 20 contract-coupling checks and nine concrete false
witnesses. They cover actual promotion selection and EP flag, source-not-king,
inserted kind, canonical enemy masking, occupancy/prefix/slider connections,
exact returned table, actual membership/board/bypass/initial/post-lookup
premises, original-side selection and both proof consumers. False witnesses
cover absent shadow candidates, promotion tags, original-side output, differing
raw prefixes, opposite-side output, EP forcing, raw castle bypass and inherited
duplicate-tail/EP-field results. Each requires exactly one typed mismatch at
its named declaration. Parse, import, linearity, resource and incidental
dependency failures do not count. Structural controls check their actual
module; consumer controls check the full consumer.

```sh
python -B -m native.bend_engine.standalone.proofs.promotion_bypass_preservation.qualify_promotion_bypass_preservation \
  /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```
