# Overlay qualification reuse

The dual-digest change reads each local file once to calculate the existing whole-tree and replacement-subtree SHA256 digests. It preserves their sorted relative names, little-endian length framing and bytes. Numerical validation, base identity checks, before/after overlay stamps and anchored operation checks still run. The code does not change receipt formats or reuse a hash from an earlier operation.

## Existing sound boundary

`qualified_paths()` performs full corpus validation and returns a `BaseSeal`/`BaseSeals` context. `GameAwareEpochBuffer` already passes that operation-local context through planning and loading. `_validated_overlay()` caches dense validation only inside that context; every reuse still checks local tree identity/membership, the base tree, receipt identity and operation roots/metadata. Closing a buffer ends its use of the context. A fresh constructor currently starts a fresh full qualification.

These checks are change detection inside one controlled operation, not a claim that mutable directories are immutable. They rely on the same local filesystem and writer assumptions as the current validator. In particular, timestamps alone are not authority, and a persistent success keyed by path/mtime is not safe. File identities include device, inode, size, mtime and ctime; full byte validation establishes the initial content.

## Small proposed API separation (not implemented here)

A caller that needs several consumers inside the same qualification operation can own a nonserializable `QualifiedOverlayOperation`. Creation calls the unchanged full validator once. The capability retains the exact qualification reference and ordered canonical shard paths, expected composed digests, validator revision and operation-local contexts. Opening a consumer accepts that capability rather than a free-form success flag.

Before opening, require the same process/operation, exact receipt reference and ordered paths, live context, and unchanged anchored identities. Planning must continue to call the existing guarded `_validated_overlay()` for every shard and compare its composed digest to the qualified value. Loading keeps its current per-shard checks. Reject changed paths, receipt replacement, membership, base/local storage, stale/closed context and cross-process use; never refresh a changed cache entry. Consumers can have separate schedules and decoded-array working sets while sharing the validated metadata context. Existing concurrency rules must be made explicit before sharing a mutable context across threads.

This API would reuse initial semantic checks within a single operation; it would not skip batch correspondence or numerical validation for a new bank version. New processes and independently reopened mutable banks still fully validate. Persistent cross-process reuse needs separately reviewed immutable snapshot/content authority binding all overlay and base bytes, membership and qualification/validator versions. This change neither introduces that authority nor changes OS security or mounts.

## Required tests before adopting that API

Compare plans, deterministic samples, objective census and composed digests with the original fresh-qualification path. Prove dense validation runs once within the capability while hot identity guards continue to execute. Reject target/base edits, same-byte inode replacement, receipt substitution, every membership change, changes during first validation, closed contexts and cross-process use. Do not weaken value dtype, finite/nonnegative checks, normalization tolerances or legal-mass rejection.

## Adoption

The running selected-lineage verifier uses its frozen source revision and remains unchanged. This main-based patch must be reviewed and repinned into a distinct runtime before production adoption; its isolated tests are not proof that an already-running process loaded it. The longer admitted unchanged verifier should proceed independently of this optional optimization. No speedup is budgeted from these source-only savings.
