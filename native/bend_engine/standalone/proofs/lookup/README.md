# Public actual-lookup contracts

This suite promotes #872's checked supplementary lookup composition. The archived
Route, Pipeline, Canonical, MaskedLookup and probe Bend bytes are unchanged.
LAWS/PROOF now register three obligations and the consumer invokes the main one.
No production engine, compiler, earlier proof or ordinary test budget changes.

```sh
# Full focused check, including consumer and all new controls:
bun native/bend_engine/standalone/proofs/lookup/focused.js /path/to/pinned/bend --report /tmp/lookup-proofs.json
# Actual Tables.build then Chess.slide in all four native modes:
python native/bend_engine/standalone/proofs/lookup/verify_native.py "$PWD" /path/to/pinned/bend \
  --bun /path/to/bun --cc clang --report /tmp/lookup-native.json
# Complete inherited aggregate, substantially more expensive:
bun native/bend_engine/standalone/proofs/lookup/verify.js /path/to/pinned/bend --report /tmp/lookup-full.json
```

`--controls-only` explicitly marks consumer/focused success NOT_RUN. Qualification
may execute that mode and the direct source consumer separately. A modular record
must retain exact-source parent receipts, not call it an executed full aggregate.

## Contracts and bounds

`lookup_from_headers` is the conditional routing theorem for actual `Chess.slide`:
correct full mask/offset read certificates imply the computed actual data read.
It is useful as a helper, not a whole-table theorem on its own.

`selected_state_is_masked` proves that the existing recurrence state at the
actual PEXT index equals occupancy AND the actual relevant mask, for every key
below128 and every U64 occupancy.

`initialized_lookup` composes actual allocation, bounded table construction,
extras and the real Chess lookup. Its result includes the complete final array
and the production slider value at masked occupancy. The only premises are
symbolic allocation depth=17, table budget `before+1+after+start_key <=128`, and
extras budget `count+start <=64`. Seed and occupancy remain arbitrary. It derives
shape, headers, data and index correctness; none is an extra caller assumption.

This is a source-to-source result, **not yet equality to independent ray geometry**.
It preserves every domain of the archived derivations and does not normalize a
separately closed literal Tables.build in the source checker. Native execution
uses actual Tables.build and retains that one array across query requests.

## Test discipline

Eight mutations require ordinary type/refinement failure at the intended lookup
or public-contract statement: wrong offset, wrong mask/offset slots, subtraction
instead of addition, ignored occupancy, unmasked state, zero result and excessive
table budget. Eight guards enforce named obligations/imports and reject holes,
foreign/unsafe dependencies and symlinks. One exact-output wrapper unit test
rejects synthetic status-zero unsafe warnings; it is not a compiler execution.
Crashes, timeouts and missing dependencies are not semantic rejection.

The native driver preserves all 1,024 archived query rows and six invalid cases,
adding portable and native-target modes alongside generic and UBSan. It computes
an independent signed-coordinate ray reference; the candidate receives only
keys and occupancies. It compares selected query values, not every buffer cell.
Rows repeat across modes and some inputs repeat within the original fixture list.
The report records distinct pairs separately. An actual shifted-offset mutation
must fail the native oracle and the source routing lemma. No proof model runs
inside the candidate executable.

The pinned checker/Base, lowering, native allocation/lifetime, ABI, toolchain and
hardware remain trust boundaries. No additional Python application responsibility
is moved by these proof/test changes. Self-review only unless separately recorded.
