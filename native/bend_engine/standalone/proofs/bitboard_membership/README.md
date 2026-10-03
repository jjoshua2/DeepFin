# Bend multibit bit-membership qualification

## Result

`actual_bit_squares_frequency` proves the exact occurrence count in the actual optimized
`Chess.bit_squares` output. For every U64 bitboard and every U32 query with
`U32.to_nat(query) < 64`, the count equals `Bool.pick(Nat, U64.test_bit(bb,
U32.to_nat(query)), 1, 0)`. Thus each set square occurs exactly once and each
clear square occurs zero times, preserving list multiplicity.

The theorem uses the bounded prefix decomposition and checked actual
`U64.clear_lsb`, `U64.test_bit`, `U64.ctz`, and U32-to-Nat equality bridges. It
does not build move tables or evaluate attacks.

## Published source and checker identity

- The proof run is bound to the exact PR #1000 source snapshot at head
  `033774f6fecacbb079180705c16667ee609322b3` (the PR head when checks ran),
  based on PR #992 head `0908d932a0fa9c7d2690c9d33dacaea8cc8ff774`. The
  evidence/docs-only commits later advanced the draft branch to
  `0449d6e3538154de277507b858bea401bdff35e8` as an intermediate review head.
  At that commit, all 43 transitive source-closure Git blobs matched this
  receipt. Subsequent README/PR-description corrections also changed docs only;
  no proof module, imported proof dependency, or toolchain input changed. The
  current branch head is shown in the live PR metadata.
- The exact PR1000 codeload snapshot SHA-256 is
  `0014c96625bd471b66a9660841b3cf71d7fa9d8eead6d1c5cefca886beb61d25`
  (47,542,300 bytes). The nine proof hashes and transitive Bend import closure
  are in [the qualification receipt](evidence/2026-10-03/qualification.json).
- The repository pin is Bend `2.0.21 + U64`, revision
  `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 source files,
  fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
  The checker itself reports `bend 2.0.21`; runtime was Bun 1.4.2. The former
  `2.0.34` value was the cached update notice, not the executed checker version.
  The complete 84-file SHA-256 manifest is [here](evidence/2026-10-03/checker-tree.json).
- Git fetch stalled in WSL; the immutable codeload archive succeeded. Checks ran
  from that isolated extraction, not a live checkout.

The recovered scratch Cardinality source and portable source are compared in
[this exact diff](evidence/2026-10-03/Cardinality.scratch-to-portable.patch) with
a [hash receipt](evidence/2026-10-03/Cardinality.scratch-to-portable.json). The
diff contains only import-path changes; all other lines match. The recovered
scratch file hashes to `41422a1d4148f8bcbea9ada9917038e0a0ea5096f441e722a7da288ff372d654`,
which differs from the earlier summary's cited
`41422a1d4148f8bcbe9ada9917038e0a0ea5096f441e722a7da288ff372d654`. The
portable source checked below hashes to
`ab68a000fbbd6d09822576d52d96f5fcd741df7b8a286421f387d8af3e05c71a`; the
summary hash discrepancy is retained explicitly rather than treated as a source
identity match.

## Positive checks

All nine modules passed with `All terms check.` and exit code 0. Each check used
an 86,400-second timeout, CPU affinity 0,1, `OMP_NUM_THREADS=2`,
`BEND_NO_TELEMETRY=1`, and `--check-only`. The receipt records exact source
SHA-256, command, raw stdout/stderr, start/end UTC, elapsed time, peak RSS, and
the full `/usr/bin/time -v` output for each module.

| Module | Wall seconds | Peak RSS KiB |
| --- | ---: | ---: |
| `ActualClearBit.bend` | 3.590 | 321636 |
| `BitSquaresStep.bend` | 3.781 | 533724 |
| `Cardinality.bend` | 3.816 | 562484 |
| `ClearModelBit.bend` | 0.560 | 154772 |
| `FrequencyIteration.bend` | 4.007 | 802852 |
| `IndexEquality.bend` | 0.783 | 239640 |
| `MembershipIteration.bend` | 3.861 | 773136 |
| `PrefixDecomposition.bend` | 3.812 | 549408 |
| `RemainingBudget.bend` | 0.367 | 114416 |

## Negative controls

Both controls were made in disposable copies of the exact snapshot and rejected
by the same pinned checker:

1. The first control replaced the recursive `occurrences(tail,query)` contribution
   in the `PrefixDecomposition.occurrences` definition with `0n`. The checker
   rejected the additive append obligation at `Location: occurrences_append`.
   Its exact mutation is [recorded here](evidence/2026-10-03/occurrences-recursion.patch).
2. The second control replaced `Chess.bit_squares(...)` with `Nil{}` in the
   `actual_bit_squares_frequency` return statement. The checker rejected
   `actual_bit_squares_frequency`. Its exact mutation is
   [recorded here](evidence/2026-10-03/consumer-statement.patch).

Raw stdout/stderr, command, source and mutation hashes, exit codes, elapsed time,
and peak RSS for both controls are in the qualification receipt.
