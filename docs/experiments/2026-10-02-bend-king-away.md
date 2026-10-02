# Off-home king castling exclusion, 2 October 2026

Outcome: **PASS for one new component theorem**, not complete legal-generator
correctness. Base: `dd187ac09c8e60061bfcd2f8f00d748a7543c9ca`.

`KingAway.exact` and its importing consumer prove

```text
CastleSpec.result(Chess.legal_moves(table,b)) == (table,Nil{})
```

The complete returned array is equal to the arbitrary input array and the actual
optimized generator has no flag-2 occurrences. Inputs are an arbitrary Array<U64>,
arbitrary Board, Boolean side and Nat square, with exactly four premises: turn is
Bool.to_u32(side), the selected moving king is U64.bit(sq), sq<64, and sq differs
from that side's home square. Neither board validity nor table initialization nor
attack correctness is assumed. The internal complete Cells witness is derived by
existing checked Representation.reify; it is not a caller premise.

The theorem guide is
[`KING_AWAY.md`](../../native/bend_engine/standalone/proofs/generator_contract/KING_AWAY.md).
The actual ordinary scan and both filter paths remain in the statement and proof.
Both actual wing producers are rejected via the derived home-square guard.
The public-builder bridge, independent history/representation derivation and full
five-part generator claim remain open.

## Reproducible source qualification

The unchanged compiler is `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21
plus the existing U64 support. Its 84 compiler/Base/effect inputs have fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Local host: Python 3.10.12 and Bun 1.4.2. Execution was in a separate josh-owned
Ubuntu WSL worktree, CPU affinity 0,1, numerical thread settings 2 and nice 10.
No training, arena, GPU, data, compiler or production source was changed.

From repository root, with a source checkout matching the existing pin:

```bash
BUN=/home/josh/.bun/bin/bun BEND_NO_TELEMETRY=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 10 taskset -c 0,1 python3 -m \
  native.bend_engine.standalone.proofs.generator_contract.qualify_king_away \
  /tmp/deepfin-king-away-checker-aaeb9bc --report /tmp/king-away-qualification.json
```

The [complete receipt](evidence/bend-king-away/source.json) has SHA-256
`6e20c442ca16b43a80803c3d2156bf6471da7414a6080d5507cccda6e432f800`. The gate records 249 transitive source identities, both helpers,
both control diagnostics and exact consumer output; final identities are checked
unchanged before PASS. The complete consumer returned exit 0, no timeout and exact
`All terms check.` in 591.629 seconds, inside the original
900-second bound. Output SHA-256:
`3155557f2fa6b6fe55b661347e56893dc0b52fa1977b6800a8d26dbff1d3db84`.

Guard and ordinary-provenance helpers passed in 15.593
and 15.826 seconds. Two actual-source semantic
controls passed within 60 seconds each: removing the home-king guard rejected at
`Flow.side`; forcing flag 2 in ordinary put_move rejected at `Scan.put`.
Timeouts, malformed sources, quantity/kind/backend failures and warnings receive
no semantic-control or positive acceptance credit.

## Host checks and review

[Host validation](evidence/bend-king-away/host-validation.json) records three wrapper
tests in each of normal, -O and -OO modes, all passing, and all 12 unchanged
compiler-pin tests passing. Final focused ruff, basedpyright and vulture pass.
Whole-repository lint reports 277 environment/type diagnostics in both the clean
base and candidate; the complete logs are byte-identical after worktree-path
normalization, SHA-256
`c60747b89cc1633a3002d6c7b465618f9bbabce0798f70172aa92cd5bf7d4a8e`.
The fresh code-only worktrees have no native extensions or locked development
environment. No rebuild or broad dependency installation was performed. The lint
shell returned zero despite these diagnostics; they are not credited as a passing
whole-repository lint result.

An initial consumer attempt rejected affine array reuse. The repaired proof uses
existing checked complete-array reification and preserves the public contract.
A subsequent unfinished consumer was stopped to fix a host test typing issue
before its source identity could drift; that receipt remains NOT_COMPLETED with
zero credit. The complete final receipt above is the sole acceptance evidence.

Independent review is recorded separately in
[`independent-review.md`](evidence/bend-king-away/independent-review.md), bound to
the actual final proof/gate source hashes and completed receipt. This is source
proof qualification only. Residual trust remains the pinned checker/normalizer
and Base; native lowering, ABI, successor/history and whole-generator semantics
are separate.


## Publication permission boundary

GitHub rejected the initial branch push because the existing OAuth credential lacks
workflow scope. No credential or account configuration was changed. The reviewed
bounded CI job is preserved byte-for-byte as
[`proposed-workflow.yml`](evidence/bend-king-away/proposed-workflow.yml), rather than
installed under .github/workflows. Its SHA-256 remains
`39455a1349394bb50aa60617322d52816d9a03ac73349b15a11c60bca021439d`.
The exact proof sources, importing consumer, complete local gate and receipts are
unchanged. This new gate is reproducible locally; the proposed job requires a
separately authorized workflow-capable publication before it becomes an automatic
CI check. Existing CI is not claimed to execute this new gate.
