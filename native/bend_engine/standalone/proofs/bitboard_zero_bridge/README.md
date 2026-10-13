# Actual U64 zero predicate and structural word zero

`Bridge.u64_is_zero_word` proves the assumption-free equality

```text
U64.is_zero(a) == ClearStep.is_zero_word(64, U64.to_word(a))
```

for every actual `U64` value. `ClearStep.is_zero_word` is an LSB-first structural
predicate: the empty word is zero, a `False` head recurses into the tail, and a
`True` head is nonzero. The bridge does not equate that predicate by name with
`Word.zero`; instead, `cmp_zero` proves it equal to
`Cmp.is_eq(Word.cmp(n, word, Word.zero(n)))` by structural induction. The proof
then follows the actual Base definitions of `U32.is_zero` and `U64.is_zero`, and
uses `Word.join` to combine the low and high 32-bit words.

The importing consumer checks the public 64-bit theorem. There are no premises,
new laws, axioms, holes, unsafe annotations, or initialized-table assumptions.
This establishes a zero-predicate bridge only; it does not prove `U64.ctz` is
bounded or selects a set bit.

## Reproduction

Use the pinned Bend 2.0.21 + U64 source at commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (84 files; fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`). From
the repository root, set `BEND_COMPILER_SOURCE` to that source checkout and run:

```bash
export BUN="${BUN:-bun}"
export BEND_NO_TELEMETRY=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2
REPO="$(git rev-parse --show-toplevel)"
"$BUN" "$REPO/native/bend_engine/standalone/verify_compiler.js" \
  "$BEND_COMPILER_SOURCE"
nice -n 10 taskset -c 0,1 "$BUN" --smol \
  "$BEND_COMPILER_SOURCE/bend2/main.ts" \
  "$REPO/native/bend_engine/standalone/proofs/bitboard_zero_bridge/Bridge.bend" --check-only
nice -n 10 taskset -c 0,1 "$BUN" --smol \
  "$BEND_COMPILER_SOURCE/bend2/main.ts" \
  "$REPO/native/bend_engine/standalone/proofs/bitboard_zero_bridge/consumer.bend" --check-only
```

The qualification receipt at
`docs/experiments/evidence/bend-bitboard-zero-bridge/qualification.json`
records the exact theorem sources, compiler identity, two successful checks,
and two semantic controls. The first control disconnects the importing consumer
and fails at `use_actual_u64_zero_word`. The second changes the actual
`U64.is_zero` body to `True` in a disposable compiler copy and fails at
`Bridge.u64_is_zero_word`. Both controls exit 1 without timing out; neither
mutation touches the pinned compiler or proof sources.
