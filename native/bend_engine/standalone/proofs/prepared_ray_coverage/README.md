# Actual prepared queen-path coverage

This typed suite is stacked on PR1029 at `f73c27bc3ab2cbe4c8fd3a037087ea80fc2355b0` (tree `18dda6ef1256344f51931a879ec25b0088bcdf9e`).

`Coverage` lifts bitwise subset certificates through the actual two-level OR tree. It covers each of eight `Path.attack(Rays.path(7n,q,dir),occ)` masks and transfers that result to the unmasked `Tables.slider` rook/bishop union using the existing typed slider refinement. Occupancy is arbitrary; the first occupied square is included by the actual traversal.

`Query` composes two separate **whole returned-pair** contracts for the supplied table, at the same square, side and occupancy: actual `Chess.attack(3)` equals the unmasked rook slider plus the exact input table; actual `Chess.attack(2)` equals the unmasked bishop slider plus that exact table. Queen dispatch uses the returned rook table for bishop, so preserving the table is a substantive contract. With the actual moving-king `ctz(kings & own)` equal to `U32.from_nat(q)`, this covers the mask actually queried by `filter_checked(False)`, observed as `accepted_castle.Spec.rays`.

`consumer.use_ray` derives both old-path coverage and the actual seven-step input route. It composes them with PR1029 actual pre-castling candidate membership and the actual fast-filter bypass guard to obtain old/new ordinary-move ray subset. Fuel is 7, both flags are False, and the same arbitrary accumulator is used on both sides. The move is exactly `Ply{src,dst,0,0}`; its source bounds and moving-color ownership come from membership, not additional caller premises. Generation table and supplied filtering table remain separate.

The caller supplies the two lookup contracts, square <64, direction <8, selected-king equality, actual candidate membership, and actual bypass. `Observe.Cells` is a complete constructor image of a supplied `Array<U64>`, including arbitrary shapes, not an initialization or validity predicate. The suite does **not** prove that the production builder satisfies the lookup contracts. A checked zero-leaf counterexample shows that an arbitrary supplied table can return zero queen rays while a king path is nonzero. No board-validity, target geometry, attack safety or full legal-move theorem is claimed.

## Qualification

Use the existing pinned Bend 2.0.21+U64 checkout at `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, and the verified 84-file checker manifest. The harness hashes the entire import/support closure, verifies reused dependencies against the exact base Git blobs, and on a clean committed head verifies every qualified file against HEAD. Prior evidence and compiler/source identities must remain unchanged.

```bash
taskset -c 1,3 env BUN="$HOME/.bun/bin/bun" PYTHONPATH=. \
  python3 -m native.bend_engine.standalone.proofs.prepared_ray_coverage.qualify_prepared_ray_coverage \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /path/to/verified/checker-tree.json \
  --report /external/new/qualification.json --evidence-dir /external/new/checks
```

Every positive and negative checker invocation has an 86400-second timeout, two-CPU affinity, 6 GiB address-space and measured RSS limits, and 16 MiB output limits. A control passes only with one typed mismatch at its expected obligation; parse/import/linearity/resource failures and timeouts are rejected as evidence. Each record distinguishes **contract-coupling** from **concrete-false-witness** controls. Omitted contracts, ranges, selected king, actual membership/bypass, changed query/table or post-move output, and the generic missing-diagonal proof test whether the proof body still establishes its stated contract; a rejection alone does not establish logical necessity or semantic falsity. Four concrete false witnesses claim coverage for the zero-leaf table, an empty actual pawn-producer output, wrapping test-bit behavior at index 64, or diagonal membership in the rook-only mask. These force direct reductions against actual total functions. Seventeen controls are qualified.

The stack also corrects PR1029's harness classification and README without changing Source.bend or its consumer. Its disconnected empty-scan provenance conclusion is true (Unit), so the old rejection is labeled contract coupling; a separate concrete actual-output-to-empty mutation supplies the sound discriminator. The README no longer claims independent necessity of both key premises. The existing typed AllEmittedSound.u64_bit_outside/u64_test_bit_outside results, imported by this checked consumer, establish that every Nat index >=64 has bit zero and test_bit False for any U64. The two corrected evidence files are explicitly included in HEAD/source hashes and excluded from reused-base dependency claims.
