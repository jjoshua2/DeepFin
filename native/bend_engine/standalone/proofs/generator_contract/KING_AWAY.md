# Off-home king castling exclusion

`KingAway.exact` proves this complete pair proposition for the actual optimized
generator, with its ordinary scan and both filtering paths retained:

```text
CastleSpec.result(Chess.legal_moves(table,b)) == (table,Nil{})
```

`result` retains the entire returned `Array<U64>` and projects exactly the
complete Ply values whose flag is 2. The empty list therefore has no castling
occurrences or duplicates. Ordinary generated moves are not claimed to be legal
by this theorem.

The inputs are an arbitrary array, arbitrary Board, Boolean moving side and Nat
square, with exactly these four premises:

- `Chess.get_turn(b) == Bool.to_u32(white)`
- `castle_kings/Spec.kings(b,white) == U64.bit(sq)`
- `Nat.is_lt(sq,64n) == True{}`
- `Nat.is_eq(sq,U32.to_nat(castle_kings/Spec.src(white))) == False{}`

There is no valid-Board, initialized-array, historical-rights or attack-correctness
premise. This excludes both castles even for fabricated rights. It does not close
the independent-history, initialized attack, home-king castling or public-builder
obligations in `docs/bend_legal_generator_contract.md`.

`KingAwayGuard` checks all 64 possible square values for each Boolean side; the
home-square cases are discharged by the contradictory off-home premise and the
remaining unbounded Nat case by the contradictory bound. These are exhaustive
checked source cases, not sampled native evidence. Side selection is transported
through the actual Board turn, so both actual wing guards reduce to false.

`KingAwayWings.cells` uses `castle_emission/Flow.rejected` for both actual producers.
The `castle_chain/Scan.initial` continuation supplies ordinary-tag provenance
from the actual ordinary scan. `castle_chain/Filter.prepare` retains that property
through both the checked and optimized branches. `CastlePrefix.ordinary_project`
then gives the empty projection; `table_preservation/Generation.legal` supplies
whole-array equality. The importing consumer loads the inherited proof bodies.
The existing proved `storage/Representation.reify` consumes each arbitrary affine
array once and supplies its complete duplicable Cells image internally. Equality
transport removes that internal witness from the public theorem. No caller-supplied
array certificate or initialized-array premise is introduced.

Reproduce from the repository root, using the unchanged compiler pinned in
`standalone/toolchain.json`:
Set `BUN` to your Bun 1.4.2 executable if it is installed elsewhere; the
default below uses the current user's standard Bun installation.

```bash
export BUN="${BUN:-$HOME/.bun/bin/bun}"
export BEND_NO_TELEMETRY=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
taskset -c 0,1 python3 -m native.bend_engine.standalone.proofs.generator_contract.qualify_king_away \
  /tmp/deepfin-king-away-checker-aaeb9bc --report /tmp/king-away-qualification.json
python3 -m unittest native.bend_engine.standalone.proofs.generator_contract.test_qualify_king_away
```

The gate starts with `NOT_COMPLETED` and zero credited candidate definitions. It
requires exact safe checker output, a verified unchanged compiler, transitive
source hashes, and two intended semantic rejections before crediting the complete
consumer. The controls remove the actual home-king guard and inject flag 2 into
the actual ordinary `put_move`. A timeout, warning, ownership/kind error, missing
definition, unrelated diagnostic or zero-output success cannot pass the gate.
The guard and ordinary helpers and each control are bounded at 60 seconds; the full importing consumer
is bounded at 900 seconds. Run output belongs outside the checkout.

The residual trusted boundary remains the pinned checker/normalizer and Base.
This theorem is not a native lowering/ABI check, complete legal-move theorem,
successor/history theorem or performance/playing-strength result.
