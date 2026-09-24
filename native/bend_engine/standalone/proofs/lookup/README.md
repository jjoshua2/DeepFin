# Public actual-lookup contracts

Promotes the four previously archived derived lookup statements from PR #872
without altering their helper bytes or any existing production/proof source.
The helpers' historical "supplementary" comments are retained for provenance;
`LAWS.bend`, `PROOF.bend`, and the importing consumer now register their contracts.

```sh
bun native/bend_engine/standalone/proofs/lookup/focused.js /path/to/pinned/bend --report /tmp/lookup-focused.json
bun native/bend_engine/standalone/proofs/lookup/verify_native.js /path/to/pinned/bend --report /tmp/lookup-native.json
# Full inherited chain: expensive and opt-in, not implied by a focused pass.
bun native/bend_engine/standalone/proofs/lookup/verify.js /path/to/pinned/bend --report /tmp/lookup-aggregate.json
```

The new consumer and controls may also be checked as separate commands:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/lookup/consumer.bend
bun native/bend_engine/standalone/proofs/lookup/focused.js /path/to/pinned/bend --controls-only --report /tmp/lookup-controls.json
```

A controls-only report explicitly sets `focused_gate` and `consumer` to `NOT_RUN`.
Accept a consumer only with exit zero and exactly `All terms check.`, without
unsafe/foreign warnings. The separate-command execution must not be described as
an execution of the complete 110-law wrapper.

## Contracts and assumptions

`certified_header_route` relates the actual `Chess.slide` to its actual final
array read, given explicit read certificates for its mask and prefix headers.
It retains the entire returned array/value pair. This low-level helper is
conditional on those certificates.

`selected_state_is_masked` identifies the actual compact index's recurrence state
with the occupancy restricted to the actual relevant mask, for all U32 keys whose
Nat value is below 128 and arbitrary U64 occupancies.

`initialized_indexed_lookup` and `initialized_masked_lookup` derive the header,
data, index-bound, and complete-shape certificates from the actual allocation,
table, and extras operations. Their callers supply only symbolic allocation
depth equal to 17 and the table/extras budgets, with arbitrary seed and occupancy.
The selected key is `before + start_key`; `before + 1 + after + start_key <= 128`
and `extras_count + extras_start <= 64`. Both return the complete final array,
not the original allocation. The last result refers directly to the production
slider computation on masked occupancy.

No caller assumes a desired stored header, computed data, or per-write separation
for these initialized contracts. The consumer invokes every law and checks
inhabited full/partial/suffix budgets. It avoids expanding a literal enormous
allocation; a separately closed normalization of `Tables.build()` is not claimed.

## Controls and native coverage

Seven semantic controls reject five real lookup-routing defects, an unmasked
state claim, and a public contract that returns the original rather than updated
array. Eight controls enforce manifests/imports and exclude holes, foreign/unsafe
proofs and symlinks. One synthetic unsafe-warning test exercises the output
wrapper, not another compiler invocation. Missing files, affine-use errors,
crashes and timeouts are not successful semantic rejections.

The native gate constructs the actual full table once per execution, then threads
it through 1,024 real `Chess.slide` queries over all 128 keys. There are 1,022
distinct key/occupancy pairs. Generic, forced-portable, native-target and UBSan
builds repeat those fixtures. Six malformed/budget requests are rejected per
mode. An actual shifted-prefix mutation must compile and execute before the
independent signed-coordinate ray reference rejects its wrong values.

The candidate receives only keys and occupancies. It receives no oracle table,
mask, offset or expected attack. This is selected-query coverage, not complete
buffer comparison or exhaustive arbitrary-occupancy testing. Source-to-source
lookup equality is not an independent blocker-ray theorem or native ownership,
allocation, lifetime, pointer-identity or compiler-correctness proof.

No production code, earlier accepted law, compiler input, permanent workflow,
routine perft budget, model/GPU/training or benchmark workload changes.
