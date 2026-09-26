# Reservation ownership and a caller-owned payload

The registry's [reserve/commit/abort protocol](ID_REGISTRY.md) relies on exclusive
ownership of the table while an insertion is staged. These checks exercise the
compiler boundary and a concrete caller, not another map implementation.

## What the checker rejects

`reservation_ownership.py` checks nine pairs of programs against the pinned Bend
compiler. Each invalid program differs from its valid control at one use site:
committing a reservation twice; committing and aborting it; aborting then committing;
using the old reservation after candidate() has returned its owner; retaining the
original registry alongside its reservation; reserving from that registry twice;
duplicating a stage that holds both reservation and array; capturing a reservation
in a closure and also consuming it outside; and marking a reservation copyable
with a `+` annotation.

A rejection counts only with exit 1, no stdout, and the expected ownership/kind
error at the intended `exercise` definition. Missing imports, parser errors,
wrong locations, crashes and timeouts are not passing negative results. The
corresponding valid program must check cleanly first. Three extra positive
controls cover mutually exclusive branches, using the owner returned by candidate,
and deliberately dropping a reservation. All generated cases and diagnostics
are retained, so changes in compiler behavior can be inspected directly.

**Bend is affine, not a must-resolve transaction system.** The checker prevents
these duplicate uses, but it permits dropping an owner. It does not force every
caller to commit or abort, automatically restore the registry on an exception,
or prevent leaking a copyable provisional U32 ID. Recovery and publication remain
explicit caller responsibilities. These tests qualify the named programs on the
pinned compiler; they are not a proof of compiler soundness or every client.

## Concrete owning-payload path

`reservation_payload.bend` places a reservation and an actual `Array<U64>` in one
owning Stage. The caller explicitly fails preparation, aborts, and gets back both
the unchanged registry and its array owner. It checks that the old key is absent,
modifies the returned array, reserves a DIFFERENT key, and commits with the same
previously unissued ID. Final lookup and array reads check publication, counter
state, preservation of the first slot and the update of the last slot.

The four starting IDs are zero, seven, U32_MAX-1 and U32_MAX. In the final case,
abort leaves the last ID available; commit exhausts it, and a later lookup/reserve
of the committed key still succeeds. Generic and UBSan builds each check 52 exact
output rows. A separate compiled driver with its payload update removed must run
normally and then fail the output comparison. A compilation failure or crash does
not count as observing the incorrect bytes.

This is an actual owning-array caller, not an external search-node store. The
explicit failure is not malloc/OOM injection. There is no crash-atomic publication,
allocation-address/RSS claim, automatic cleanup guarantee, or neural-cache identity
contract. The array's data is checked; allocator implementation details are not.

## Reproduction and evidence

```sh
python -m pytest tests/test_bend_map_ownership.py
python -m native.bend_engine.u64_map_probe.reservation_ownership \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-map-ownership
```

Use a fresh output directory. Summary and command records retain failures as well
as expected compiler rejections; generated examples and stdout/stderr are saved.
The command runner kills its own process group on timeout. Current sources and
compiler fingerprints are recorded. Continuous numeric-map CI includes this gate
without timing measurements, benchmark thresholds or historical artifact inputs.

Local checks before publication: 32 Python cases passed with global conftest
disabled; the complete ownership entry point passed with the pinned Bend sources,
Bun 1.4.2 and Clang 17. Hosted locked-environment and whole-repository static results
are reported separately on PR #876. No map/registry API, compiler, engine, earlier
benchmark, live setting or deployment is changed. Self-review only; no independent
review or formal proof is claimed.
