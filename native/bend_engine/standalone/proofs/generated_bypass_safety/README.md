# Actual generated-candidate bypass safety

This increment stacks on PR1038 head
`02a2620006e5c10ae69139785b0d878894a6fc81`, tree
`a90a94238771b306eeda87f7100ecfdcbe737441`.

`Origin` preserves actual full-Ply scan membership through both guarded castle
producers. An actual prepared-mask bypass excludes the guarded castle alternative.
`Safety` classifies that scan occurrence using PR1038, excludes EP, and calls the
existing ordinary or typed-promotion no-check consumer on the exact full Ply.
Promotion tags 1..4 retain flag 0; no tag-0 shadow member is substituted.

The explicit contracts are Board.valid, a nonempty original-side king plane,
exact canonical turn `get_turn(b)==Bool.to_u32(white)`, initial whole-pair
`in_check(input,b,get_turn(b))==(input,False)`, and the old rook/bishop whole-pair
lookup facts at the original king index, original side and original occupancy.
For each supplied occurrence, the affine lookup callback supplies new rook and
bishop whole-pair facts at that original king index, opponent side xor1 and
`occupied(Chess.make_move(b,m))`, conditioned on actual scan membership and actual
prepared-mask bypass False. The callbacks contain lookup facts, not safety
answers. Duplicate occurrences carry separate callbacks. Equality facts use a
reusable record; no affine function is copied.

`Forced` derives EP and actual guarded-castle full filter-step equality at the
actual prepared mask, preserving the exact supplied accumulator pair. It also
constructs their lookup callbacks from the contradictory bypass antecedent, so
these forced cases need no moved-board lookup facts. A raw flag 2 is insufficient:
the castle certificate includes the actual producer guard and full-Ply equality.

`Lift` derives the existing list theorem's bypass-safety premise occurrence by
occurrence. `consumer.member_list_fast_full` covers arbitrary ordered lists of
actual generated members, including duplicates, with an identical arbitrary
retained tail. `generated_fast_full` uses the exact actual prefilter list.
`actual_legal_full` composes actual legal_moves/prepared_equal with the existing
prepare/full result. All equalities include the unchanged input table. No extra
king uniqueness, target geometry or post-board validity premise is introduced.

`Fixtures` invokes this new bridge for an actual ordinary scan member and all
four actual promotion members. Reordered promotion duplicates and repeated
ordinary values retain exact order and an unclassified duplicate tail. Separate
fixtures call generated-list and actual legal_moves/full composition, plus actual
EP and guarded-castle full steps. The retained tail requires no classification.

These results retain the stated board and lookup assumptions. They do not derive
initialized Tables.build contracts, full chess legality, legal-move completeness,
or unconditional fast/full equivalence. The initial-check True branch already
uses full filtering; the no-check bridge explicitly retains initial False.

## Qualification

Use Bend 2.0.21+U64 at `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bun 1.4.2, and the verified 84-file checker fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Every positive and negative check permits 86400 seconds on CPUs 1/3 with two
threads, 6 GiB address-space/RSS and a 16 MiB cap per output file. Run checks
serially; evidence and immutable source/mutant snapshots live outside checkout.

The qualifier classifies 17 controls as ten contract-coupling checks and seven
concrete false witnesses. They test actual scan/consumer/query connections,
actual moved occupancy, original versus flipped check side, promotion dispatch, prepared rays, forced EP/castle
contracts, output tags, returned table, order, duplicate multiplicity and tail
identity. Credit requires exactly one intended typed mismatch with differing
expected and observed types. Parser, import, linearity, resource or incidental
dependency failures receive no negative credit. The full fixture entry imports
the actual consumer and every new proof module; no declaration is removed.

```sh
python -B -m native.bend_engine.standalone.proofs.generated_bypass_safety.qualify_generated_bypass_safety \
  /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```
