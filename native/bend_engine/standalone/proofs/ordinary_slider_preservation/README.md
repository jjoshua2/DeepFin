# Actual ordinary candidate slider preservation

This suite composes geometric post-ray subsets, enemy erasure, and the separate old/new lookup contracts at the original side's preserved king. It is stacked on PR1033, commit `3c10ae2dbf2b19d1f8f830d42f227cdc511d4bbe`, tree `9bad197d7d28dca4664da13679ef8e28c02c848e`.

`consumer.use_slider` uses one actual `Ply{src,dst,0,0}` occurrence in `C.candidates(O.pack(c),b)`. The same table is used for preparation, initial no-check, and both old/new query contracts. Candidate provenance supplies source bounds/ownership and destination king exclusion. `Board.valid`, a nonempty moving king plane, canonical turn `Bool.to_u32(white)`, actual prepared `filter_requires == False`, and actual initial `Chess.in_check == (O.pack(c),False)` remain explicit.

The result is the **whole actual rook/bishop suffix pair** `Chess.attacked_rook(post,preservedKing,originalTurn xor 1,U64.zero(),actualRookLookup) == (O.pack(c),False)`. It checks the original side after move metadata flips. The zero prefix makes this a slider-only result. It does not prove full post-move `in_check == False`, non-slider attack preservation, promotion safety, castling safety, king uniqueness, attack-table builder validity, or full legal-move correctness.

The ordinary fields are explicit: bypass False excludes EP flag 1 but does not exclude promotion values 1-4. The checked promotion fixture is an actual back-rank pawn candidate that bypasses. Enemy erasure requires canonical turn because `Chess.color` selects black for every U32 except 1; a checked turn-2 candidate introduces a slider attack when that premise is absent. No extra target geometry or board validity of the moved board is assumed.

Small structural lemmas compose equality-valued inclusions (`Algebra`), combine four production rays for each slider (`Rays`), expand the actual initial callbacks with storage-preserving arbitrary fixed lookups (`Check`), and transfer empty intersections through enemy rook/queen and bishop/queen erasure (`Hits`). Ray fuel is 7, both flags are False, and the accumulator is zero, as in `Tables.slider`.

The positive fixture has an actual knight candidate and a nonempty opponent rook plane. Separate checked counterexamples expose missing bypass, noncanonical turn, and an occupancy-sensitive bad post lookup. The arbitrary Cells table is not assumed to be builder-produced. Imported scan checks retain the exact table, promotion/EP fields and duplicate-tail multiplicity.

Run the fail-closed gate with the already qualified Bend 2.0.21+U64 source at `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and its verified 84-file manifest:

```sh
python3 -B -m native.bend_engine.standalone.proofs.ordinary_slider_preservation.qualify_ordinary_slider_preservation /tmp/deepfin-king-away-checker-aaeb9bc --checker-manifest /external/checker-tree.json --report /external/Q001.json --evidence-dir /external/Q001
```

Positive and all 22 negative checks have an 86400-second default, two CPU cores, 6 GiB address space/RSS, and a 16 MiB per-output-file cap. Fourteen controls require a named contract-coupling rejection; eight are concrete false-witness rejections. Parser/import/linearity/resource failures receive no qualification credit. The published-head gate binds every qualified source to its HEAD Git blob and every reused dependency to the exact base, records raw stdout/stderr/time/exit/RSS and mutation hashes, and rechecks source/compiler identity after all controls. Durable evidence belongs outside the checkout. Independent internal review precedes a stacked draft PR; no merge is authorized.
