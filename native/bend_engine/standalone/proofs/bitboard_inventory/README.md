# Actual bitboard inventory at zero and singleton masks

`Inventory.bend` proves an exact, bounded result for the actual optimized
`Chess.bit_squares`: zero maps to `Nil`, and every in-range `U64.bit(sq)` maps to
exactly `U32.from_nat(sq) <> Nil`. The singleton theorem case-splits over all
64 bounded squares; the `64 + p` branch contradicts the caller's `sq < 64`
premise through the existing checked `Domain.impossible` proof. This does not
claim coverage or distinctness for arbitrary multi-bit masks.


ClearStep.bend adds the actual, assumption-free clear_lsb_popcount_step for
any U64 value. It proves the actual U64.clear_lsb result reduces the
Word.count(64, U64.to_word(mask)) measure by one when that structural word is
nonzero, and lifts the result to actual U64.popcount. The importing consumer
uses the public theorem; a negative control substitutes a different actual bit
clear and must fail.

This is the one-step cardinality foundation only. It does not prove that
U64.ctz(mask) is in range or names a set bit, connect U64.is_zero to the
structural zero test, or establish arbitrary multi-bit Chess.bit_squares
membership, ordering or exact cardinality.

`OneBit.bend` connects the inventory to actual callers. `scan_one_bit`
uses the existing checked factorization of actual `Chess.scan_after` and then
the new singleton inventory to produce exactly one actual `Chess.put_move`.
Its table, source, pawn bit, en-passant square and Ply tail are arbitrary, and
the complete array component is preserved. `scan_empty_targets` proves the
zero-target call returns the unchanged pair. `scan_step_onehot` and
`scan_step_empty` specialize the real `Chess.scan_step` when the actual
`piece_targets` producer returns respectively one bounded bit or zero; they do
not prove attack or piece-target geometry.

`scan_one_source` additionally assumes the actual moving-side mask is one
bounded bit and the actual target producer returns one bounded bit. It proves
the real `Chess.scan` result is that one emitted move and the unchanged array.
`legal_moves_one_source` rewrites the real optimized `Chess.legal_moves` call
to its actual `filter_prepare` and both actual `castle_side` calls around that
one-move scan result. It keeps the full pipeline in the equality and preserves
the arbitrary input table as it is threaded into the pipeline. It does not claim
that later filter/castle stages return that same array, or prove how those functions
transform that list, nor claim complete legal-move correctness.

All positive terms are consumed by the importing `consumer.bend`. There are no
new laws, axioms, holes, unsafe annotations, initialized-table assumptions or
attack-result assumptions. The consumer keeps the previous destination
factorization layer together with its four checked contracts and five negative
controls; `qualify.py` reruns that earlier gate before the new checks.

## Reproduce

Use the isolated WSL source checkout and the unchanged compiler pin:

```bash
BUN=bun BEND_NO_TELEMETRY=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 \
  nice -n 10 taskset -c 0,1 python3 -m \
  native.bend_engine.standalone.proofs.bitboard_inventory.qualify \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --report docs/experiments/evidence/bend-bitboard-inventory/source.json
python3 -m unittest native.bend_engine.standalone.proofs.bitboard_inventory.test_qualify
```

The gate verifies the compiler's 84-file source fingerprint, hashes its complete
proof/import closure and relevant gate helpers, rechecks the inherited 249
source identities, runs each positive checker call within 180 seconds, and
requires five intended semantic negative controls within 120 seconds each.
Affinity and numerical thread settings are capped at two; the gate limits
cumulative child CPU to 5,400 seconds. Timeouts, missing files, warnings,
backend/type/kind failures and unrelated diagnostics receive no proof or
negative-control credit. The machine-readable report records every command,
output hash, source SHA-256, compiler identity and control diagnostic.
