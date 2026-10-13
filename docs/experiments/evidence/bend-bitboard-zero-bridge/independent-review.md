# Independent review: actual U64 zero bridge

Result: **No findings** in the bridge or its importing consumer.

The unconditional public theorem is

```text
U64.is_zero(a) == ClearStep.is_zero_word(64, U64.to_word(a))
```

for every `U64`. The structural predicate is LSB-first: `WNil` is zero, a false
head recurses, and a true head is nonzero. `cmp_zero` proves this predicate is
equivalent to `Cmp.is_eq(Word.cmp(n, word, Word.zero(n)))` by induction, so the
predicate's structural interpretation is explicit rather than implicit.

The reviewer traced the pinned Base definitions: `U32.is_zero` delegates to
comparison with zero; `Word.cmp` recurses through the tail and then compares
heads; `Word.zero` is all-false; `Word.join` concatenates the low and high
words; `U64.to_word` uses that join; and `U64.is_zero` is the conjunction of
both 32-bit zero checks. `u32_is_zero_word` and `join_zero` compose those exact
definitions into the 64-bit result. There are no boundedness premises, added
axioms, holes, unsafe shortcuts, or table assumptions.

The pinned qualification receipt
`docs/experiments/evidence/bend-bitboard-zero-bridge/qualification.json` has
SHA-256 `c686368f20c3b028b5582aba70118232a492e3897717dff5847ff076f8187abf`.
It records the verified 84-file compiler pin, matching current source hashes,
two successful positive checks, and two non-timeout semantic controls. The
disconnected consumer fails at `use_actual_u64_zero_word`; mutating the actual
`U64.is_zero` definition in a disposable compiler copy fails at
`Bridge.u64_is_zero_word`.

This closes only the zero-predicate bridge. `U64.ctz` range and selected-bit
membership remain separate obligations.
