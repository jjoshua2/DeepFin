# Immutable policy and value overlays

Use overlays for frozen offline experiments that share inputs, legal masks and
row/history identities. Mutable replay rejects them. They require retention of the
complete sealed bases for the lifetime of every dependent run.

Schema 1 remains policy-only. Schema 2 permits `policy_target`, `search_wdl`, or
both, inheriting every other array and every group attribute unchanged. Target
shape and dtype must match the base; policy has no illegal mass and both targets
must contain normalized nonnegative finite probabilities. No overlay chains or
links inside a shard are accepted.

```python
from pathlib import Path
import zarr
from scripts.target_overlay_storage import seal_base
from chess_anti_engine.replay.target_overlay import sha
from chess_anti_engine.replay.target_overlay_v2 import (
    begin_target_shard, finish_target_shard, qualify_target_roots,
)

# Base sealing reads and validates the corpus once. Put receipts outside corpora.
seal_path = Path('/data/receipts/base.json')
seal_base(Path('/data/base'), seal_path)
ref = {'path': str(seal_path), 'sha256': sha(seal_path)}
base = Path('/data/base/shard_000000.zarr')
out = Path('/data/arm/shard_000000.zarr')  # parent exists
begin_target_shard(base, out, ref, replacements=('policy_target', 'search_wdl'))
group = zarr.open_group(str(out), mode='a')
# Producer writes every row, checks teacher alignment, and records its recipe.
group['policy_target'][:] = policy_probabilities
group['search_wdl'][:] = wdl_probabilities
finish_target_shard(base, out, ref, recipe=recipe_manifest)
# Repeat for every shard, then freeze the ordered roots used by --shards.
qualify_target_roots([Path('/data/arm')], Path('/data/receipts/arm.json'))
```

The exact-epoch training CLI consumes schema 2 using its existing
`--overlay-storage-qualification` and
`--expected-overlay-storage-qualification-sha256` arguments. Multiple ordered
`--shards` roots can depend on separate base seals. Qualification pins root files,
shard membership, base identities, row/history declarations, replacement bytes
and the recipe dictionary. Staging order must match the qualified order.

Storage qualification does **not** establish teacher provenance, mixture weights,
calibration semantics or scientific validity. An experiment producer must verify
those separately before producing targets; changing a recipe description without
changing targets is not an experiment. In particular, when mixing from stored
float16 targets, record inherited rounding and avoid applying teacher temperature
a second time to already sharpened probabilities.
