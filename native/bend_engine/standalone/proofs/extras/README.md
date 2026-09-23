# Extras-stage preservation of headers and slider data

Two source laws connect the real `Tables.extras` loop to the previously proved
complete-tree, array-frame and returned-pair contracts. No production code changes.

```sh
# Bounded new-law feedback, explicitly not the inherited aggregate:
bun native/bend_engine/standalone/proofs/extras/focused.js /path/to/pinned/bend --report /tmp/extras-focused.json
# Four native modes, complete-buffer observations:
bun native/bend_engine/standalone/proofs/extras/verify_native.js /path/to/pinned/bend --report /tmp/extras-native.json
# Full qualification (not executed during local delivery):
bun native/bend_engine/standalone/proofs/extras/verify.js /path/to/pinned/bend --report /tmp/extras-aggregate.json
```

The aggregate retains the unchanged 98-law/212-control contents gate and then
adds these two laws and 14 controls. Its expected 100-law/226-control total is a
command contract, **not a claimed executed result**. `focused.js` uses a different
`focused_gate` field and explicitly reports `inherited_gate_run: false`.

## Accepted contracts

`bounded_extras_read` quantifies over actual complete depth-17 affine arrays,
Nat counts n and starting squares k, and U32 query addresses q. Preconditions are
n+k <= 64, q < 131072, and q < 256 or 512 <= q. Actual get after the entire extras
loop equals the **complete final updated array paired with the original query
value**. The protected value is not assumed to be zero, an attack, or a header.

`table_pipeline_extras_read` produces the required shape from actual `Array.new`
and `Tables.tables`, then establishes the same returned-pair equality. Depth d
is kept symbolic with an explicit equality d=17; the table-loop count/key/start,
seed, extras count/start, and query remain symbolic. The caller does not assume
array shape or the expected read value. Specializing to d=17, 128 table blocks,
key=0, start=512, seed=0, extras count=64 and extras start=0 yields the production
pipeline's preservation statement. A fully expanded closed `Tables.build` consumer
was not successfully checked; the universal symbolic law is the accepted result.

`Facts.square` discharges the four write-address bounds and successor identity
for all 64 squares inside the checker. It is not a host-generated table of attack
values. `Primitive` derives normalized-path separation from the numeric ranges
and complete shape. `Steps` connects the actual four writes through existing
certified storage helpers. `Actual` inducts over the real extras loop and returns
the entire updated pair. `Certified` supplies the actual pipeline shape.

The importing consumer applies both public contracts symbolically and checks
closed scalar witnesses: a full 64-square budget, one last-square iteration,
protected boundaries, and excluded overlap/overflow cases. Closed concrete-array
instantiations caused excessive expansion and are not claimed executed results.
No public statement that passed checking was weakened to resolve those costs.

## Checks and trust

The focused gate checks the consumer and 14 rejection controls. Five require
ordinary failures in the new range/returned-pair contract, two deliberately fail
in the existing implementation-linked `storage/Build.extras` proof, and seven
check manifests/imports/regular files/exact safe output. Wrong locations, missing
files, linearity/termination errors, crashes and timeouts do not count as semantic
rejection. A successful checker invocation must exit zero and print exactly
`All terms check.`. The protected compiler is unchanged.

The native probe executes actual extras on fresh complete allocations, with five
nonzero seeds and five distinct protected sentinels. It compares all 131072 cells
of each of five buffers, in generic, portable, native-target and UBSan builds.
The independent reference uses flat arrays and signed coordinate knight/king/pawn
steps. It supplies no candidate values. Seven malformed requests per mode reject.
Native tests do not execute proof predicates/models or rerun the table builder.
Modes repeat fixtures; counts are not disjoint or exhaustive native coverage.

This proves preservation, not correctness of the preexisting mask/prefix values,
computed slider geometry, or persistence through earlier table-header/other-block
writes. It also does not source-prove the values *inside* the extras region.
The independent native oracle tests those values, which is a separate evidence type.
Source equality does not establish pointer identity, native allocation success,
physical lifetime, or correctness of compiler/runtime/OS/hardware.
