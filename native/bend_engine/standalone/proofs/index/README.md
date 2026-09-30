# Compact-index source laws

This additive proof module uses the pinned standalone compiler from PR #805.
It does not change the compiler, `Subsets`, tables, the engine, old accepted laws,
or the original proof verifier. All application and inherited proof bytes remain
unchanged. These are P1 support laws, not the carry-rippler's ordering theorem.

## Exact contracts

`LAWS.bend` explicitly states nine universal laws and `PROOF.bend` discharges all
nine against **actual `U64.pext`, `U64.pdep`, `U64.popcount` and the imported
`proofs/LAWS.at` recurrence**. The recurrence invokes production `Subsets.next`.

For a mask with mathematical population `k`:

- Every extracted compact value is strictly less than `2^k`.
- For `0 <= compact < 2^k`, `pext(pdep(compact, mask), mask) = compact`.
- Without that bound, reverse composition equals the low-`k`-bit projection,
  not the original compact value. `compact=1, mask=0` is a counterexample to
  an unrestricted inverse; `compact=2, mask=bit63` refutes an inclusive bound.
- PDEP produces only mask bits, is injective on the bounded compact domain,
  and supplies a masked witness for every member of that domain.
- PEXT is injective on masked bitboards. Every imported recurrence state's
  extracted index is in range, but this alone does not establish its ordinal.

`Bits.capacity` uses unbounded `Nat`, not U32/U64 exponentiation. Its equality to
`Nat.pow(2n, U32.to_nat(U64.popcount(mask)))` is itself proved. In particular, a
population of 64 denotes mathematical `2^64`, not zero. Generic Word induction
proves the range and inverse; `Count.bend` bridges only the 65 possible population
counts to their actual U32 encoding. It does not enumerate possible bitboards.
`Bits.clip` is an independent specification of truncation, never an engine
replacement. It does not implement or assume the missing subtraction theorem.

## Gates

From the repository root, with a verified full compiler checkout:

```sh
bun native/bend_engine/standalone/proofs/index/verify.js /path/to/bend
bun native/bend_engine/standalone/proofs/index/verify_native.js /path/to/bend
```

The source gate first runs the **unchanged** previous gate: eight engine laws,
16 byte-pinned inherited U64 laws, and eleven previous negative controls. It then
requires exactly `All terms check.` and exit zero for this module and a separate
non-vacuous importing consumer. It rejects all fifteen new negative controls,
including weakened bounds, missing/omitted laws, unsafe/foreign proof dependencies,
and disposable Base mutations. Original compiler and proof files are hash-checked
before/after; no mutation occurs in a real compiler or application checkout.

The optional native gate compiles a small raw-operand probe in generic,
forced-portable U64, native-target and UBSan modes. Each mode compares the same
1,590 distinct operand pairs: 702 in-range and 888 out-of-range inputs, with
mask populations spanning 0 through 64. BigInt scatter/gather supplies an
independent reference and exact `2^64` bound. Four malformed/budget requests are
rejected per mode. These are bounded samples, not exhaustive arbitrary-mask tests.
No reference result is supplied to the candidate. No Python/model/perft or full
engine rebuild is involved. The probe is never imported by the engine or proofs.

Both gate commands accept an optional `--report FILE`. The commands are opt-in;
no new recurring workflow or ordinary-pytest traversal is installed.

## Remaining theorem and trust

This proves the **representation bijection**, not that successive calls to
`Subsets.next` enumerate it in ordinal order. Still required: connect the actual
two-U32 borrow implementation of `U64.sub` to the masked successor, prove compact
indices advance modulo `2^k`, then prove the Nat-indexed ordinal/coverage theorem.
No existing ordering obligation has been removed or weakened. Table-array
initialization/lookup refinement and chess-mask geometry are separate P2 work.

The pinned checker/normalizer and Base semantics remain trusted. Native lowering,
affine storage, ABI, C compiler, runtime and hardware are not proved by source
laws. Native results are reported separately. No engine/model performance,
trained-checkpoint, CUDA, new Python removal or complete migration claim follows.
