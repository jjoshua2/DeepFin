# Bend multibit bit-membership qualification

## Result

`actual_bit_squares_frequency` proves the exact occurrence count for the actual
optimized `Chess.bit_squares` consumer. For every `U64` bitboard and every `U32`
query with `U32.to_nat(query) < 64`, the number of occurrences of `query` in
`Chess.bit_squares(64, U64.is_zero(bb), bb, Nil{})` equals
`Bool.pick(Nat, U64.test_bit(bb, U32.to_nat(query)), 1, 0)`. Thus each set square
occurs exactly once and each clear square occurs zero times; the statement retains
the list's multiplicity rather than only its support.

The actual-consumer theorem has only the query-range premise. It does not construct
a table, evaluate attacks, or assume a popcount bound. Its induction establishes
the fuel bound from the structural U64 word count. Supporting checked theorem
`IndexEquality.u32_eq_nat_eq` connects the actual U32 comparator to natural index
equality. The induction consumes the actual `U64.clear_lsb`, `U64.test_bit`, and
`U64.ctz` bridges.

The portable sources are in
`native/bend_engine/standalone/proofs/bitboard_membership/` in the candidate tree.
They are an import-path adaptation of the checked scratch sources; theorem bodies
and premises are preserved. `Cardinality.bend` retains the exact previously checked
source hash `41422a1d4148f8bcbe9ada9917038e0a0ea5096f441e722a7da288ff372d654`.
The original unfinished `Membership.bend` source remains unchanged at
`c9161e10dcf89cd2ec26ef683859c40965c3e76a142191ae1f1e14b6f7ed53ec`.

## Source and checker identity

- Candidate source snapshot: GitHub PR #992 head `0908d932a0fa9c7d2690c9d33dacaea8cc8ff774`.
- Downloaded source archive SHA-256: `fb83d26cb827041d338ad9404c087bf88eb63fc8c4e4df8d546a7ed3c6a649df`.
- Bend checker source: `/tmp/deepfin-king-away-checker-aaeb9bc/bend2/main.ts`, SHA-256 `34c69df407a8abff02752f26821ee2ff7c0a303e97a32b8b5e0082b142e63f59`.
- Checker `base.bend` SHA-256: `86747736186e77ed9cb02555385026ef2eebe4e84efbcd29773e3b51b5f571a1`.
- Runtime: Bun 1.4.2; checker reported Bend 2.0.34.
- Each check used `timeout 86400s`, `taskset -c 0,1`, and `OMP_NUM_THREADS=2`.

## Checks

All nine portable entries returned `All terms check.` with exit code 0:

| Entry | Wall seconds | CPU | Peak RSS KiB |
| --- | ---: | ---: | ---: |
| `BitSquaresStep.bend` | 3.53 | 132% | 523236 |
| `ClearModelBit.bend` | 0.51 | 174% | 174352 |
| `ActualClearBit.bend` | 3.46 | 135% | 333252 |
| `RemainingBudget.bend` | 0.32 | 170% | 116116 |
| `PrefixDecomposition.bend` | 3.51 | 136% | 548140 |
| `Cardinality.bend` | 3.61 | 136% | 584616 |
| `IndexEquality.bend` | 0.67 | 173% | 236936 |
| `MembershipIteration.bend` | 3.78 | 134% | 769460 |
| `FrequencyIteration.bend` | 3.68 | 133% | 777684 |

Reproduce a check from the repository root by setting `BEND_MAIN` to the pinned
checker entry and running:

```bash
timeout 86400s taskset -c 0,1 env OMP_NUM_THREADS=2 bun --smol "$BEND_MAIN" \
  native/bend_engine/standalone/proofs/bitboard_membership/FrequencyIteration.bend \
  --check-only
```

The local checker directory contains no Git metadata, so its source identity is
recorded by the two hashes above rather than by a claimed commit ID.

Portable candidate source hashes:

| Source | SHA-256 |
| --- | --- |
| `ActualClearBit.bend` | `29be18e2cc0a0d5e14f283ead79e1676d872f5ca57acda7a441d940eb4780707` |
| `BitSquaresStep.bend` | `a68f6760b1205a3015d2e9d86fbfe6696ca292cbeae7908de3b0fbcbe900cc54` |
| `Cardinality.bend` | `ab68a000fbbd6d09822576d52d96f5fcd741df7b8a286421f387d8af3e05c71a` |
| `ClearModelBit.bend` | `dd6302364361b6d1c3952c229a5ffc0ddf9cddf4769df31bd818c0286fc957f8` |
| `FrequencyIteration.bend` | `f91c00b9a9298517ed0d5b1a4927e04670547fdfe30f9968ff378b0cbed7e7b8` |
| `IndexEquality.bend` | `888aec0806f6c3ecf5aa07b67aab30b8a8745813b2b8cbc8c9b0fca82957e293` |
| `MembershipIteration.bend` | `9a6a4f9b91f4a4fc362f8305f79174be222825770de7efb4e66cbb9d96c27088` |
| `PrefixDecomposition.bend` | `c6bedfa4ef8877eb4176cdaafe7fdb1075977d0ae679fd9048162c4ee54a108f` |
| `RemainingBudget.bend` | `a3aa031ced3003deda917fa343e151b3c29596c99f1a75e7305c98c9854182bf` |

## Negative controls

Both mutations were made only in disposable source copies and were rejected by the
same checker:

1. In a disposable source copy, replacing the head-equality contribution with
   `False{}` in the `PrefixDecomposition.occurrences_append` proof expansion caused
   that lemma to fail (`expected occurrences(append)`; observed a spurious added
   head count). This mutation targeted the proof's expansion of the counter, not
   the counter definition itself. As the independent reviewer pointed out,
   changing the counter definition globally to always return zero would preserve
   the append lemma and instead invalidate the final frequency claim for set bits.
2. Replacing `Chess.bit_squares(...)` in the actual-consumer theorem statement with
   `Nil{}` caused `actual_bit_squares_frequency` to fail because the actual output
   decomposition no longer matches the stated result.

The qualification entry itself has only repository-relative imports. This run used a
source archive because Git fetch stalled in this WSL session; the archive matches the
published head by request identity, but the checkout was not a local Git worktree.
Independent review found no logical or soundness issues and independently reran
`FrequencyIteration.bend` successfully. The reviewer confirmed the distinction
above for the first negative control. The sources remain an archive snapshot rather
than a local Git worktree because both WSL Git fetch attempts stalled.
