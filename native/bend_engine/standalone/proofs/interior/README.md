# Bounded interior addresses and allocation normalization

Opt-in continuation of #826. Five contracts connect actual prefix endpoints to
arbitrary absolute interior addresses and remove masking from the actual source
read API under an explicit source-size certificate. No production code changes.

```bash
bun native/bend_engine/standalone/proofs/interior/verify.js /path/to/pinned/bend --report /tmp/interior-proofs.json
bun native/bend_engine/standalone/proofs/interior/verify_native.js /path/to/pinned/bend --report /tmp/interior-native.json
```

The aggregate first checks the new consumer and 21 controls, then runs the unchanged
74-law/126-control prefix gate. Acceptance means 79 laws and 147 controls, not
counts alone: all commands must succeed and the source CLI must print exactly
`All terms check.`. Nothing is added to ordinary CI/perft budgets.

For bounded development, `focused.js` checks the five new laws and controls only.
Its optional `--controls-only` explicitly returns `focused_gate: NOT RUN` and
`consumer: NOT RUN`; it cannot satisfy the aggregate. Negative controls alone
never establish a successful proof. The full gate accepts no such flag.

## Statements and premises

`allocation_mask_exact` proves `x & 131071 == x` for any U32 x<131072 by structural
Word induction. `allocation_mask_injective` derives injectivity on that domain.
`prefix_interior_certificate` takes k<128 and the ordinary endpoint inequalities
`prefix(k)<=x< prefix(k)+actual_size(k)`, and constructs x>=512, x<131072 and exact
normalization. Actual endpoint/size/order certificates come from the explicitly
imported, already discharged prefix and layout proofs, not caller assumptions.
`ordered_normalized_addresses_distinct` proves unequal normalized U32 addresses
for members of two ordered blocks. It does not claim unequal physical pointers.

`interior_read_uses_unmasked_address` quantifies over the actual affine array,
requires its source `Array.size` result to be 131072, and proves equality of the
complete public `Array.get` pair with the internal `Array.get.go` pair at the
unchanged address. Certified reification and the existing size-pair identity
transfer the lemma to real arrays. Source equality is not native lifetime safety.

The absolute membership premise is intentional: deriving it for every relative
PEXT offset or incrementing write remains work. Numeric normalization is also not
full tree-path separation. The consumer includes a nonuniform tree where distinct
in-range addresses select the same leaf, along with the out-of-range 131072/0
masked alias. A regular depth-17 shape certificate and route injectivity are still
needed to use the prior `separation` frame laws for the complete builder.

## Native comparison and observed compiler limitation

The native probe builds the real Tables buffer and, at all 131072 addresses,
compares public reads, public reads with an explicitly normalized argument, and
out-of-range aliases offset by 131072. The independent signed-coordinate/BigInt
reference is retained from the prefix verifier. Every returned U64 is compared;
proof predicates and proof representations never execute in the candidate.

An initial probe tried to compile a direct `Array.get.go` call. The pinned native
compiler rejected it with `an open Array element type`. That result is a retained
limitation, not a passing native check. The final native probe tests only the public
API; it does not native-qualify standalone internal-helper calls. Source contracts
are unchanged by this distinction, and the compiler is not modified.

Complete table-address coverage is not exhaustive arbitrary-array or arbitrary-
occupancy proof. Four modes repeat fixtures. Final initialized contents, complete
path separation, independent geometric lookup refinement, allocation and physical
ownership remain separate obligations. No model/GPU, strength or speedup claim.
