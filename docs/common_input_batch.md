# Bounded common-input batches

`scripts/common_input_batch.py` prepares a frozen selection from one or two
original rooted-corpus sources. Each source runs derive → storage snapshot →
adapt already-completed raw BT4 → phase-zero d9 ranks → common-input qualification.
It performs no labeling inference, policy mixing, training or automatic retry.

Use a fresh state/output layout and an immutable JSON manifest. The default action
validates it; execution additionally requires its exact SHA256:

```bash
PYTHONPATH=. python scripts/common_input_batch.py \
  --manifest /absolute/batch.json --expected-manifest-sha256 SHA256
# After review, bound the full invocation, including preflight. Example: four hours.
batch_deadline=$(python -c 'import time; print(time.time() + 14400)')
timeout --signal=TERM --kill-after=30s 14370s \
  python scripts/common_input_batch.py \
  --manifest /absolute/batch.json --expected-manifest-sha256 SHA256 \
  --action execute --deadline "$batch_deadline"
```

The manifest must contain these fields; unknown top-level, derivation and limit
fields are rejected. Metadata paths use `{ "path": "/absolute/file", "sha256": "…" }`.
No example below constitutes an executable registration.

| Field | Contract |
| --- | --- |
| `schema` | `1` |
| `state` | Canonical absolute directory; existing metadata allowed, no prior attempt/output |
| `checkout`, `commit`, `python` | Exact clean frozen runtime checkout, 40-character commit and interpreter |
| `runtime_qualification` | Pinned receipt with `status: qualified`, matching checkout/commit/python, transitive `pins`, and true `features.closed_shard_selection` / `features.support_exclusion_requires_result` |
| `preregistration` | Pinned scientific/resource registration; separate from this executable manifest |
| `pins` | Absolute-path→SHA256 map including all qualified runtime pins, resolved interpreter, runner, three consumer scripts, `/usr/bin/time` and `/usr/bin/timeout` |
| `max_concurrent_sources` | `1` for sequential preparation or `2` for concurrent lanes |
| `derive_options` | `scheme: uniform-d9`, `policy_observation: phase0`, `value_observation: latest-phase`, `value_scheme: search`, `temp: 0.0005`, `floor: 0`, `workers: 2`, `row_provenance: true`; explicit nonnegative integer `seed` and positive integer `rows_per_shard` |
| `limits` | Positive integer `wall_seconds_including_kill` (>30), `new_output_cache_bytes`, `minimum_free_bytes`, `adapter_index_cache_bytes`, `rank_index_cache_bytes`; fixed `numeric_threads: 2`, `nice: 19`, `ionice_class: 3`, `CUDA_VISIBLE_DEVICES: ""` |
| `sources` | One or two objects described below; no fixed row count or shard window |

Each source object has a unique safe `source_id`, original canonical `source_dir`
and `sidecar_dir`, `cpu_affinity` with two available CPU IDs, and these input pins:
`source_manifest` (the original `source_dir/manifest.json`), `selection`,
`closed_bt4_receipts` and `source_metadata`. Concurrent lanes must use disjoint
CPU pairs. Sequential lanes may reuse a pair. Additional source fields may retain
informational inventory facts, but do not change execution.

The selection uses the [shared closed-shard format](corpus_derivation.md): exact
original source/config/manifest identity and selected shard names, closed row
counts and raw SHA256. `physical_rows` must equal their sum. `source_metadata` is
a list covering exactly those files, with `source_path`, `device`, `inode`,
`bytes`, `mtime_ns`, `ctime_ns`, `sidecar_path` and a pinned
`sidecar_attrs_snapshot`. Unselected live source growth is allowed. Raw payload
checks occur in the actual consumers, not in a redundant preliminary census.
The runner accepts reversed selection JSON; consumers retain original corpus order.

`support_drop_ceiling` and `missing_result_count_ceiling` are nonnegative integer
caps below `physical_rows`. The latter must equal
`floor(physical_rows * missing_result_fraction_ceiling)`, with a finite fraction
in `[0,1)`. Envelope drops remain forbidden. Support exclusions require the
source-qualified actual ledger and independent rank validation; neither count
caps nor an earlier census substitute for current row eligibility.

`derived_output`, `adapted_output` and `rank_output` must be distinct direct children
of `state/source_id`, separate from all inputs. Existing outputs, `.writing`
partials, stage logs or attempt receipts are refused. Keep different original
sources in separate physical parents so the training loader's existing namespaces
remain meaningful; this runner does not flatten or merge them.

## Ownership and evidence

The operational launcher must pin and use the outer GNU timeout shown above; it
bounds manifest/runtime preflight as well as execution. Use the registered total
wall limit minus 30 seconds for its TERM timer and 30 seconds for kill grace. Compute
the absolute deadline before that outer launch and pass it via `--deadline`; the
runner refuses a missing or expanded deadline. The plan itself contains a duration,
not a moving timestamp. Invoking the Python API directly does not supply an outer
surviving preflight timeout.

Each lane has its own GNU timeout process group, with that same absolute deadline
and 30-second kill grace. Its two-worker deriver and later stages inherit the
lane's CPU affinity, low priority, hidden GPU and two numeric/compression threads.
A failed lane cancels all remaining owned lanes; unrelated jobs are untouched.
The timeout survives coordinator death. Each live lane also checks the resource
limits while waiting for its stages. All failures preserve partial outputs and
receipts for reconciliation; rerunning requires a new state and decision.

STOP markers in the state or its parent and free space are checked at least once
per five-second stage wait (the coordinator also polls every second). Aggregate
state/output/cache bytes are sampled every sixty seconds plus stage/final
boundaries. Atomic `.writing` renames may remove a file during a live sample;
those disappearances are skipped, while symlinks and permission errors still
fail. The final sample, after all owned lanes stop, requires a stable traversal.
This is a sampled apparent-byte limit, not an allocated-byte quota: brief peaks
between samples may be missed. Receipts label observed peak bytes/minimum free
space accordingly. Stage GNU-time receipts record wall/CPU/RSS; process-tree
observations are sampled and can miss short-lived children.

Qualification retains the existing physical-row bitmap, source/history keys,
BT4/rank admission, normalized legal BT4 payload hashes and unchanged derived
storage proof. Independent rank missing-result/support counts, injective eligible
row references and exact universe cardinality establish the complete survivor
complement. Omitted IDs remain reconstructible from the pinned universe minus
emitted references. A completed batch means qualified common inputs for later
recipes, not a training schedule, recipe selection, throughput speedup or strength
result. Compare timings across different batches only with their actual data,
CPU concurrency and shared-host load disclosed.
