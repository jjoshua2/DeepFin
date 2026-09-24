# Uniform full-table lookup boundary

This suite consolidates the selected-block theorem into a table expression that
is independent of the queried key. It imports the existing ray theorem and the
exact archived full-domain certificate; it does not repeat the geometry proof.

```sh
bun native/bend_engine/standalone/proofs/builder/focused.js /path/to/pinned/bend --report /tmp/builder.json
bun native/bend_engine/standalone/proofs/builder/verify_native.js /path/to/pinned/bend --report /tmp/builder-native.json
# Expensive complete parent chain, not implied by modular receipts:
bun native/bend_engine/standalone/proofs/builder/verify.js /path/to/pinned/bend --report /tmp/builder-all.json
```

## Source guarantees

`full_table_lookup` and `u32_full_table_lookup` return independent blocker-ray
attacks and the complete final array for every valid Nat or actual engine U32
key. In both statements the same array expression serves every key:

```text
Tables.extras(extra, U32.from_nat(start),
  Tables.tables(n, 0, 512, Array.new(U64, depth, seed)))
```

The caller supplies `depth == 17`, `n == 128`, the valid-key bound, and
`extra + start <= 64`. Seed and occupancy remain arbitrary. There are no
before/after counts, selected-key equations, header, data, mask, or desired-value
certificates in this interface. `Binding.image` proves the conversion from the
previous selected-block pipeline; `Boundary` transports the complete returned
pair and supplies the original budget certificate for every key. The U32 law
also supplies its own exact key conversion. `Domain.bend` is byte-identical to
#875's supplementary `FullDomain.bend.txt`, now a regular checked dependency.

## Public-builder linkage: exact scope

`recipe.js` checks the complete signature/body tokens of actual `Tables.build`
and the symbolic `Spec.run`. It ignores whitespace/comments but rejects changes
to the calls, arguments, initial key/offset, allocation depth/seed, extras count
or starting square. It deliberately rejects some semantically harmless rewrites
as well: any changed recipe needs explicit review rather than silent acceptance.

This is a **source-link guard, not a discharged closed equality between
Tables.build() and the symbolic recipe**. A direct closed-equality draft reached
its 30-second normalization limit and is recorded separately. The source theorems
are genuinely checked for the symbolic full-table recipe; its specialization
agrees with the currently guarded public expression. No checker rule is changed,
no result is assumed, and the unfinished closed normalization is not hidden.

The new focused gate has four semantic controls, seven recipe-link controls,
eight manifest/import controls, and one synthetic exact-output wrapper unit.
Only the first four are actual checker rejections. Compiler errors, missing files,
crashes and timeouts never count as valid semantic rejection. Source success
requires zero status and exactly `All terms check.`. Controls-only mode explicitly
does not claim a consumer run. All existing source gates remain unchanged.

## Native behavior

The native wrapper reruns the unchanged four-mode real builder/Chess.slide test,
retaining every query, malformed request and existing shifted-lookup mutation.
It then shortens **only the actual public build call** to 127 blocks. That code
must compile/run and fail at lookup row 1016, the first query for key 127; the
recipe guard must independently reject it. This catches drift in the public
wrapper, rather than only corruption inside the already-proved table loop.

Selected query checks are not full-buffer comparisons or exhaustive occupancies.
Modes repeat the same fixtures. Source equality does not prove physical array
identity, allocation success, native lowering or ownership/lifetime safety.

No production runtime, earlier accepted law, compiler input, routine test budget,
model/GPU or training behavior changes. Self-review only. Independent review,
public closed-expression normalization, and the distinct board/move and later
P3-P7 obligations remain separate from these boundary contracts.
