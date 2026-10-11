# Exact-epoch construction reuse

`GameAwareEpochBuffer` accepts an opt-in construction artifact. A cold constructor
can set `startup_cache_write=Path(...)` and an explicit `startup_cache_recipe`
dictionary describing model-independent target and objective-counter settings.
It exposes the resulting `startup_cache_sha256`. A subsequent constructor uses
`startup_cache_read={"path": "...", "sha256": "..."}` and the identical recipe.
Read and write are mutually exclusive; a failed read never silently builds a new
plan or combines old and new state. The ordinary constructor remains the default.

The artifact contains every census and plan dataclass field, all three execution
arrays, the planner's complete returned shard order, and the three fresh RNG states.
The initial seeded shuffle can be moved by the planner's frontier rule; the exact
returned order is retained. The caller-pinned manifest digest authenticates those
bytes. Structural validation checks aggregate census, namespaced game keys,
objective totals, schedule coverage, array dtype/shape, resource configuration and
fresh RNG state without rerunning the planner. It does not independently prove an
arbitrarily rehashed alternative schedule correct. Zero-row reservations are hash
bound in the source roster and intentionally absent from the scheduled census.

Bindings include resolved staging/source roster, every shard content hash, Python
and NumPy versions, sampler/storage/encoding source hashes, counter source and
qualified-overlay reference, as well as seed, batch, input, workers, budget,
mirror/overlap and recipe settings. The explicit recipe must describe callback
configuration not represented by its source code. A cache reference is trusted
only when its digest came from the cold qualified construction.

Mutable bytes are fully hashed on every cache read. Ordinary census decoding,
objective counting and schedule planning are reused; full byte verification and
fresh qualified-overlay validation remain. `_load_one` retains its before/after
decode hashes and input, target and objective checks. Timestamps do not establish
immutability, and this change does not authorize omitting validation.

Publication writes state and then the manifest commit marker into a private sibling
directory, fsyncs both files and the directory, uses Linux `renameat2` with
`RENAME_NOREPLACE`, and fsyncs the parent. Existing destinations are never replaced,
including a concurrently created empty directory. Unsupported publication hosts
fail closed. Failed temporary directories are retained. Loading rejects incomplete,
extra-file, symlink, wrong-digest and incompatible artifacts. JSON has a closed
tagged schema, duplicate keys are rejected, numeric arrays have explicit dtype and
shape, and serialized file reads are capped at 256 MiB. This format uses no pickle.

This increment starts from main `cba7beafaa882359aef293d3533948e2e56a5781`.
Main supports qualified overlays but lacks the selected-row census and exact
recovery cursor methods of the pinned fresh mixed runtime
`/home/josh/chess-artifacts/operations/deepfin-fresh-mixed-count-runtime-20261010-v1`
whose sampler SHA256 is
`5d6d53876f1bccef8095fdaab425ddb46322187154098aa5a64e8a874d86b543`.
This artifact rejects that different source version. It does not implement indexed
runtime reuse, exact-bundle restoration, or remove consumed-prefix replay. Cached
construction starts at batch zero; public fresh-pass continuation remains a new
sampling pass. No runner/CLI adopts the optional route in this increment.

CPU tests exercise the real constructor, mirrored tensor delivery and RNG parity,
overlap delivery, frontier movement, objective populations, qualified overlays,
manual prefix reconstruction, stale mutable bytes with preserved size/timestamp,
changed bindings, malformed/rehashed state, incomplete publication and no-replace
races. Simulated fsync failures establish failure handling, not a power-loss study.
These tiny fixtures establish no full-corpus speedup, capacity or 104M-row runtime
qualification. Live training, checkpoints, corpus data and source pins are untouched.
