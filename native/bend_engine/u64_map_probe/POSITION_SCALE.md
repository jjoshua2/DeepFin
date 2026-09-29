# Full-record PositionIndex regression

The original position-index entry point checks field equality, forced collisions
and chess-derived records, but its maximum-size case only allocates the new record
layout. `position_scale.bend` separately fills **32,768 complete identity records**
and checks them through the unchanged public `PositionIndex` API. The older
numeric-map capacity test does not substitute for this record-layout test.

## Population and collision cases

Seven scenarios run in each existing map build mode:

| Record capacity | Records sharing each supplied hash |
| ---: | ---: |
| 2 | 1 or up to 8 (two actual members) |
| 512 | 1, 8, or all 512 |
| 32,768 | 1 or 8 |

At maximum capacity the eight-member case uses only 4,096 distinct hash keys,
while still consuming all 32,768 record slots. This distinguishes identity-record
capacity from hash-head occupancy. The one-member case also reaches the numeric
map's admission limit. The deliberately long chain is bounded to 512 records;
there is no unbudgeted quadratic run with 32,768 records sharing one hash.

The native driver generates synthetic eleven-field identities and scalar hashes.
Every record's first two bitboards are independently injective over the tested
range. These field vectors are not asserted to be legal chess boards, generated
self-play, or a measurement of real collision frequencies. The BoardIndex/CBoard
legal-position suites remain separate and unchanged.

## Observable checks

Every case begins with an empty lookup, populates its entire record array,
reads every ID in reverse order, and interns every existing record again while
full. It rejects both a new identity under an existing hash and an identity with
a new hash. At the first and final stored records, it changes each of all eleven
identity fields in turn: those lookups must miss and admission must remain full.
It then reads every original record in an odd-multiplier permutation, proving
that rejected operations did not change existing bindings, and reports final size.

The independent oracle is a Python dictionary indexed by (hash, all fields).
It reproduces neither probing nor collision-chain traversal. All operation labels,
returned IDs/dispositions and sizes must match, including the exact row count and
final newline. No final checksum substitutes for per-operation comparison.

Each maximum-size case has **131,121 operations / 131,123 output rows**. Across
all seven scenarios there are **268,647 operations / 268,661 rows per build**.
Generic, portable-U64, explicit BMI2/POPCNT and UBSan repeat the same fixtures,
not independent datasets. This covers the advertised record count, not every
possible key distribution, operation ordering, allocator error or client.

Two deliberately broken programs must compile and exit normally before failing
the exact oracle: aliasing upper-half record writes onto lower records, and
silently populating only half the advertised record count. Six invalid scalar
configurations are separately rejected with the expected diagnostic in every
build. Crashes, unrelated failures and timeouts are not passing negative results.

## Reproduction and evidence

```sh
python -m pytest tests/test_bend_map_position_scale.py
python -m native.bend_engine.u64_map_probe.position_scale \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-position-scale
```

Use a fresh output directory. The harness reuses the checked ownership command
runner with its 120-second process-group timeout, strict compiler diagnostics and
retained command records. Every native invocation explicitly selects its scalar
configuration. Reports retain source/build hashes and complete raw stdout/stderr.
The source-only numeric-map workflow includes this gate and uploads its evidence;
no historical artifact is required and no timing threshold is used.

Local checks: 50 new Python cases passed with global conftest disabled. All 28
positive native executions, 24 rejected configurations and both executed mutations
passed separately using the verified Bend compiler and Clang 17. The initial new
driver needed parameter-based dispatch before a match; neither the index nor the
oracle changed. The full new harness, its shared command-runner integration, and
the locked environment/static gates are qualified separately by hosted CI, with
results recorded on PR #876. Local commands are not substituted for that result.

PositionIndex, BoardIndex, U64Map, the registry and prior benchmark results remain
unchanged. No performance, memory, playing-strength, formal-proof or independent
review claim is made. This does not integrate a production search/cache consumer
or validate full-scale board canonicalization; self-review only, no deployment.
