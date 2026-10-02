# Actual U64 zero-predicate bridge, 2 October 2026

Outcome: **PASS for one unconditional actual-source theorem**. This stacked change adds a structural Word characterization of the actual `U64.is_zero` predicate and an importing consumer. It does not duplicate the bitboard inventory proofs in [PR #990](https://github.com/jjoshua2/DeepFin/pull/990).

The dependency/base is PR #990's branch `proof/bend-bitboard-inventory-20261002`, at head `aa9b0e1b9b34606b38e84ae473a4561a2af194e0` when this stack was created. This draft targets that branch. PR #990 itself targets `main`; the ultimate destination is therefore `main` after its dependency is resolved.

## Theorem and assumptions

`Bridge.u64_is_zero_word` proves, for every actual `U64` value and with no premises,

```text
U64.is_zero(a) == ClearStep.is_zero_word(64, U64.to_word(a))
```

The structural predicate is LSB-first: an empty word is zero, a false head recurses into the tail, and a true head is nonzero. `cmp_zero` proves it equivalent to equality of the actual structural `Word.cmp` result against `Word.zero`. The 64-bit proof follows the actual Base definitions of `U32.is_zero`, `U64.is_zero`, `U64.to_word`, and `Word.join`.

The checked consumer imports the bridge. No axioms, holes, unsafe annotations, initialized-table assumptions, or nonzero premises are used.

## Reproduction and evidence

The proof was checked with unchanged Bend 2.0.21 + U64 at compiler commit `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` (84 files; source fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`). The exact commands are in [the proof README](../native/bend_engine/standalone/proofs/bitboard_zero_bridge/README.md); its gate uses a two-CPU cap and verifies the pin before checking.

Both `Bridge.bend` and its consumer returned exit 0, no timeout, and exact `All terms check.\n`. Two semantic controls exited 1 without timeout: disconnecting the importing proof call fails at `use_actual_u64_zero_word`; mutating the actual Base `U64.is_zero` implementation fails at `Bridge.u64_is_zero_word`.

The qualification receipt is [qualification.json](evidence/bend-bitboard-zero-bridge/qualification.json), SHA-256 `c686368f20c3b028b5582aba70118232a492e3897717dff076f8187abf`. The independent read-only review found no findings and is recorded at [independent-review.md](evidence/bend-bitboard-zero-bridge/independent-review.md), SHA-256 `8f05e592ea47fecac50fa1ba66aaba04464b900762e35c1fc6b26af0a9baa0a8`.

## Remaining ctz obligation

This is one component needed by the general multi-bit inventory proof, not that proof itself. The pinned actual definition is `U64.ctz(a) = U64.popcount(U64.sub(U64.lsb(a), U64.one()))`. The exact remaining theorem obligations are still to prove, under `U64.is_zero(a) == False`, that `U64.ctz(a) < 64` and that `U64.test_bit(a, Nat.to_u32(U64.ctz(a)))` (with the repository's exact index conversion) is true. The latter requires connecting the actual `U64.lsb` arithmetic to the first set bit of its structural Word representation. The checked zero bridge supplies the separate actual-zero-to-structural-zero link; neither ctz result is claimed here. Exact multi-bit coverage, order and multiplicity of `Chess.bit_squares` remain open.

PR #990 has its own separate generator-contract CI timeout recorded in its description. This stacked theorem does not depend on that timeout for its local checker receipt and does not claim it is resolved.
