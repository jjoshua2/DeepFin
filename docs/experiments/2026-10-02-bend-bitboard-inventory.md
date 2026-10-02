# Actual bitboard inventory at zero and singleton masks, 2 October 2026

Outcome: **PASS for ten checked Bend definitions**, comprising the exact bounded zero/singleton bitboard inventory, eight actual scan/caller refinements, and one general actual clear_lsb/popcount count-step theorem. This is a source-proof increment, not arbitrary multi-bit enumeration or whole-generator correctness. Base: `bend-bitboard-inventory-20261002` at `269105298285b6098ffbf80405cb18ec33186b38`.

## Theorem and assumptions

[`Inventory.singleton_indices`](../native/bend_engine/standalone/proofs/bitboard_inventory/Inventory.bend) proves, for any natural `sq` with `Nat.is_lt(sq,64n) == True`, the exact actual-source equality

```text
Chess.bit_squares(64n, U64.is_zero(U64.bit(sq)), U64.bit(sq), Nil)
  == U32.from_nat(sq) <> Nil
```

The proof case-splits over all 64 bounded square values. Its `64+p` case derives a contradiction from the caller's bound using the existing checked `Domain.impossible`. `OneBit.empty_indices` separately proves that the actual helper maps zero to `Nil`. These are the zero and all 64 singleton bitboards; no coverage or distinctness statement is made for an arbitrary multi-bit mask.

[`OneBit.bend`](../native/bend_engine/standalone/proofs/bitboard_inventory/OneBit.bend) composes this inventory with the existing checked destination factorization. `scan_one_bit` proves actual `Chess.scan_after` emits precisely the actual `Chess.put_move` for the singleton destination. Its table array, source square, pawn bit, en-passant square and arbitrary Ply tail are preserved. `scan_empty_targets` proves the zero-target actual call returns the unchanged array/tail pair. `scan_step_onehot` and `scan_step_empty` specialize actual `Chess.scan_step` under an explicit equality premise for the actual `piece_targets` result.

`scan_one_source` adds a bounded one-hot premise for the moving-side source mask and proves the actual `Chess.scan` singleton result when the actual target producer returns one bounded singleton. `legal_moves_one_source` rewrites the actual optimized `Chess.legal_moves` to its real `filter_prepare` and both real `castle_side` calls around this result. The equation keeps the caller's arbitrary array as input to the actual threaded pipeline; it does not claim those later stages return the same array. It does not prove the behavior of those filtering/castling stages, target geometry, attack answers, or full legal-move semantics. No board-validity, initialized-table or attack-result premise is assumed.


ClearStep.clear_lsb_popcount_step proves, for every U64 mask, that the actual
U64.popcount(mask) equals the U32 conversion of the count of the actual
U64.clear_lsb(mask) plus one when the independently structural Word
representation is nonzero. Its body proves the reduction against Word.count,
the actual U64.clear_lsb definition, the existing checked U64 subtraction
bridge and actual U64-and bridge; the result is then consumed by consumer.bend.

This closes one arbitrary-mask count step, not the full arbitrary-mask
inventory requested for Chess.bit_squares. The remaining exact obligations are
to prove actual U64.ctz is bounded and selects a set bit, connect actual
U64.is_zero to the structural word-zero predicate, then use those facts to
prove the scanner's output has exact membership and cardinality without
collapsing duplicate list tails.

## Reproducible qualification

The unchanged checker is aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae (Bend 2.0.21 + U64), with 84 compiler/Base inputs and fingerprint d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4. Qualification ran in the isolated Ubuntu WSL worktree with CPU affinity 0,1, numeric thread settings 2 and nice 10. The gate took 70.704 child CPU seconds and 53.546 wall seconds; each positive checker call was bounded at 180 seconds and each semantic control at 120 seconds.

All 4 positive entries returned the exact All terms check output:

- Inventory.bend: exit 0, no timeout, 5.773s; output SHA-256 3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84.
- OneBit.bend: exit 0, no timeout, 6.482s; output SHA-256 3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84.
- ClearStep.bend: exit 0, no timeout, 0.300s; output SHA-256 3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84.
- consumer.bend: exit 0, no timeout, 6.530s; output SHA-256 3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84.

All 5 actual-source and disconnected-consumer controls rejected semantically at their intended obligations:

- actual-bit-square-key: semantic rejection at singleton_indices (0.499s; diagnostic SHA-256 7d38b306f47dbd9ba2932b31cfe05cb46fd17d577b2a7ac9885df6b51e4d3c57).
- actual-destination-budget: semantic rejection at cells (6.518s; diagnostic SHA-256 3a0255cda8e86175c729c63096edc99b517f7d4e205795e33bd924232f07038c).
- actual-scan-step-target-source: semantic rejection at scan_step_onehot (6.499s; diagnostic SHA-256 4060349c860447c4771c913ff8c5341b2418256fd27a96bb105b744ab4f9e7d7).
- actual-clear-lsb-count-step: semantic rejection at use_clear_lsb_popcount_step (6.526s; diagnostic SHA-256 b75809399c972eb7334040da5cfe1383f62ed9dbc3dd1146355006c5f57cd42f).
- consumer-proof-call-disconnected: semantic rejection at use_actual_legal_moves_one_source (6.522s; diagnostic SHA-256 d42b562701257ba919cafe1b62cf2cf203c605d27106ddeb2541b685e93a2d81).

The gate freshly reran the preceding destination-factorization gate: 4 contracts passed and all 5 controls rejected. Its inherited receipt SHA-256 is 61a20248cba5a65e6c7e823d6e4ef1a47f42075a3555e83c8643497b91e80042. All 249 earlier source identities remain unchanged.

The primary qualification receipt SHA-256 is 1d09bf1f9fafc819228886784dcf93296795390aa5ffdb5f1584d8740a6da271 and binds 56 proof, implementation, compiler-pin and gate-helper identities. It records commands, source hashes, exact checker outputs, negative diagnostics, compiler identity, CPU/wall accounting and the final no-drift check. The host-validation receipt SHA-256 is b0a5e77cead71a7818c6ebdeb45923c45f4a2b3a3be34f762f7ea0f9e385c783; focused tests passed under normal Python, -O and -OO, as did ruff, basedpyright, vulture, py_compile and staged/unstaged whitespace checks.

The independent review is recorded in evidence/bend-bitboard-inventory/independent-review.md and covers the exact checked proof, its bridges to actual Base operations, gate controls, receipt hashes and stated scope. PR #990 received a follow-up portability correction after CI exposed maintainer-specific Bun paths. The bounded gate was rerun against refreshed source identities; the exact CI path-leak test and focused ruff check pass.
