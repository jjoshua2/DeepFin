# Whole-table preservation through actual move generation

Five public laws establish whole **source-array value** identity for actual attack
queries, ordinary scans, either castling producer, prepared filtering, and the full
`Chess.legal_moves` pipeline. They quantify over arbitrary actual affine arrays,
including ragged shapes, arbitrary contents, arbitrary Boards and raw U32 metadata.
There is no initialized-table, consistent-board, single-king or valid-side premise.
The ordinary-scan law also admits arbitrary source-square lists and accumulators.

This does NOT establish correct masks, correct moves, legal positions, no duplicate
moves, generator soundness/completeness, physical pointer identity, allocation success
or native ownership/lifetime. In particular, an empty/no-op generator could preserve
its table. Correctness of answers is supplied by separate existing and future laws.

## Source proof

`Core` transports through the existing affine-to-Data reification certificate;
`Queries` reuses the accepted `storage/Read.read` result and composes every read
callback. The 153 disjoint Boolean clauses eliminate all32 constructors of raw U32
kind dispatch, including its default branch. They are control-flow cases, not
host-generated masks. `focused.py` additionally checks disjoint coverage of all2^32
raw values. `Checks` composes actual attack and in_check callbacks. `Targets`,
`Castles`, and `Filters` prove the real recursive loops and both optimized filter
branches. `Generation` joins the actual ordinary scan, both wings and final prepare
call. No assumed per-query preservation certificate is in the public interface.

The complete consumer imports all producer bodies. Its ragged, nonuniform example
makes clear that a chess-sized initialized array is unnecessary for this property.
All earlier proof bodies and production operations remain unchanged.

## Reproduce (bounded, opt-in)

```sh
export BEND_NO_TELEMETRY=1
python3 native/bend_engine/standalone/proofs/table_preservation/focused.py \
  /path/to/pinned/bend --report /tmp/table-focused.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/table_preservation/verify_native.py \
  /path/to/pinned/bend --report /tmp/table-native.json
```

Compiler revision: aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,84 inputs and unchanged
fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.

The source gate runs all five laws then eight actual-code/refinement mutations,
eight manifest/import guards and one synthetic warning-output wrapper unit. Crashes,
missing files, parse/ownership errors and timeouts are not valid semantic rejections.

## Native observations

Each of generic,forced-portable,native-target,UBSan executes173 distinct input tuples.
Every before/after buffer is read completely:2,313,088 U64 cells /4,626,176 limb
comparisons per mode. This includes17 full131,072-cell buffers plus156 small buffers
of64 or1024 cells. Raw buffers start with independent index-distinct values; actual
Tables.build buffers are compared against their actual pre-operation snapshot.
These are whole-value preservation tests, not independently re-proved initialized
mask contents. Candidate input supplies only context,operation and Board fields.

Operations cover raw attack,in_check,ordinary scan,both castling producers,full
filter,forced-fast filter,prepared filter and complete legal_moves. Returned move
fields are observed, but no full independent move-set oracle is claimed here.
Clean modes repeat fixtures; the source theorem covers arbitrary shapes while these
native cases use complete arrays only. Seven malformed requests are rejected.

Three actual-code mutants overwrite unused slot131071 during scanning, unchecked
filtering, or queen-mask combination. All eight initialized full-generator outputs
remain identical to the clean move lists; affected complete buffers are nevertheless
rejected. Mutations compile/run with generic flags only. Complete cell reads do not
prove native lifetime, pointer identity, or absence of unrelated memory corruption.
