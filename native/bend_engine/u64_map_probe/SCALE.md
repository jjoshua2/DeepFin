# Full advertised-capacity map and registry regression

The earlier map/registry fixtures exercise boundary values and high addresses,
but do not populate the largest admitted table. `scale.bend` fills **32,768
entries in 65,536 buckets**, the API's advertised half-full entry limit, and
checks behavior throughout fill, saturation, deletion, refill and reuse. It
also runs at 2 and 512 entries. The underlying map and registry are unchanged.
This is correctness coverage, not a performance or memory experiment.

## Observable contract

The map is filled, rejects an extra new key, replaces every existing value at
capacity, and checks insert-once preserves those replacements. It deletes every
even-numbered key, looks up all old keys, fills the recovered capacity with new
keys, and reads both old and new domains. Finally it deletes all survivors,
checks that all prior keys are absent, and reuses the emptied map. Every result
and reported size is checked, not only the final size or an aggregate checksum.

Two fresh registries per size exercise the full entry limit. One starts IDs at
zero; the other starts so that its final admissible entry receives U32_MAX. Each
fills all but the last entry, reserves and aborts that last slot, confirms the
key is absent, and commits a different key without losing an ID. Missing-key
rejection, known-key reservations, and readback of every binding are checked
after saturation/exhaustion. Returned IDs and optional next-ID state are visible
on every line. Full means the public admission limit, not 100% occupied buckets.

The Python oracle uses a dictionary and an unbounded integer counter. It does
not mirror hashing, probe chains or native U32 wraparound. Operations are generated
inside the native driver, avoiding the earlier per-environment-string transport
budget. Its scalar exponent is bounded before allocation. Synthetic full-width
keys are deterministic and distinct over these workloads; this is not a new
CBoard corpus or a claim about production collision distributions.

For exponent 16, the complete run contains **524,305 operations / 524,311 exact
output rows** across the map and its two registry lifetimes. Across exponents
2/10/16 there are 532,563 operations / 532,581 rows per build mode. Generic,
portable-U64, explicit BMI2/POPCNT and UBSan builds reuse these fixtures, not
independent datasets. This covers the full advertised *entry count*, not every
possible key distribution, operation ordering, allocator behavior or client.

Two negative controls must compile and exit normally, then fail this oracle:
folding writes from high buckets onto lower buckets, and silently halving the
initial population while still claiming the original size. A timeout, compiler
failure or crash cannot stand in for a detected semantic mismatch. The original
small collision, mutation, ownership and chess-replay checks remain in CI.

## Reproduction

```sh
python -m pytest tests/test_bend_map_scale.py
python -m native.bend_engine.u64_map_probe.scale \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-map-scale
```

Use a fresh directory. Reports retain all per-operation stdout, diagnostics,
source hashes, build flags and command outcomes, including failures. Individual
commands are bounded to 120 seconds and their process group is killed on timeout.
The existing numeric-map workflow runs this source-built gate; no historical
artifact, model, GPU, throughput threshold or live configuration is required.

Local validation: all 31 new Python cases passed with global conftest disabled.
The pinned Bend compiler and Clang 17 completed all twelve positive executions
and both executed mutations. The enclosing local harness was interrupted by its
outer 180-second command budget during the last mutant's build; that build and
execution were completed separately and all traces rechecked. This is not called
an uninterrupted end-to-end local pass. One initial helper arrangement was rejected
by the Bend checker; parameter-based dispatch fixed the new test driver without
changing an API or expected trace. Hosted CI and whole-repository static results
are recorded separately on PR #876. Self-review only; no independent review,
formal proof, new speedup, engine integration, merge or deployment is claimed.
