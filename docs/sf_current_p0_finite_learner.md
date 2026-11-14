# Finite current-P0 SF main-WDL capability

At the 2026-11-14 16:17:24 UTC readout cutoff, 22 consumer and 21 legacy CPU
cases passed, with zero skips; direct CLI,
original-G1 model use and GPU behavior remain unqualified. This is a
separate finite learning family, not a production YAML option or an admitted
training recipe. Policy anchoring is deferred. Alpha, native-versus-stored value,
learner run UUID, LR, seed, dose and scientific initialization remain unselected.

```bash
python -m scripts.sf_current_p0_finite_learner \
  --contract /external/learning.json --contract-sha256 SHA256 \
  --stage /external/segment.json --stage-sha256 SHA256
```

All artifacts must be external, hash-pinned and read-only as in the original
[finite learner](core_finite_learner.md). Root owns resource admission and launch.
The old `core_finite_learning_v1` CLI and its exact closed-parent command/schema
guards remain strict. Neither family imports a saved state from the other.

## Explicit contracts and fixed data

The learning schema is `sf_p0_main_wdl_learning_v1`. It has the existing finite
contract fields, replacing `updates_per_generation` with an explicit finite
`updates_per_segment` list and adding `sf_p0`. All existing architecture, donor,
runtime, source/file, optimizer, memory, mirror and LR validation still applies.
The source manifest must also include the new consumer and its carrier,
augmentation and collate dependencies. No unknown field is ignored.

The exact `sf_p0` profile fields are:

| Field | Required meaning |
| --- | --- |
| `schema` | `sf_p0_main_wdl_profile_v1` |
| `scope` | `FIXED_ORIGINAL_G1`, or bounded `SYNTHETIC_ONLY` qualification fiction |
| `alpha` | Explicit finite number in [0,1]; no default scientific weight |
| `value_variant` | Explicit `native` or `stored` |
| `data_identity` | Original producing `run_uuid` and `source_revision`, separate from learner identity |
| `generation_manifest` | Exact root-admitted original G1 path/SHA reference |
| `limits` | Positive `metadata_bytes` (at most 128 MiB), `game_decoded_bytes` (at most 64 MiB), and `resident_bytes` |

Real scope fixes original run `b0db4b6b-2bb3-4753-80d6-96825decbbee`, source
`090b5143315a0faef7ab0ac1a90e2d5cebf3b929`, generation manifest
`35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878`,
all 256 TRAIN games and 24,827 retained rows. No new cohort, HOLD or relabeled
producer identity is supported. Synthetic scope requires an explicit
`SYNTHETIC_NOT_REAL_AUTHORITY` generation marker and one game with at most four
rows; it cannot satisfy the real-population identity checks.

The segment schema is `sf_p0_main_wdl_segment_v1` with exactly `schema`,
`contract_sha256`, `segment`, `generations`, `parent`, `output`, `sf_p0_sidecars`.
Every segment's `generations` is the same single original-G1 manifest reference.
The segment number counts learner intervals; original `DrawnRow.generation` stays
1. Segment 1 supports fresh donor bootstrap only. Later segments require a
successfully closed same-family boundary and the identical contract, source,
profile, original data and sidecar references. Full-state resume uses the shared
existing helper and restores RNG after construction/data reconstruction. There
is no legacy-cc17 transition, in-window recovery or new-collection route.

## Acyclic carrier admission before model construction

For nonzero alpha, `sf_p0_sidecars` contains exactly one object:
`{root_acceptance: {path,sha256}, independent_review: {path,sha256}}`.
Candidate completion alone is insufficient. The future root acceptance is:

- Status `ROOT_ACCEPTS_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY`.
- Exact `scope`, `generation_manifest`, `candidate`, `validation`, `source`,
  `counts`, `rows`, `physical_games`, `post_closure`, `training_admitted: false`.
- `produce` and `validate`, each binding `protocol`, `admission`, exact full child
  `command` and closed `terminal`. Protocol owner SHA and cap must match its terminal.

The separate actual review requires status
`ACCEPT_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY_NO_FINDINGS`, `findings: []`,
`scope`, and exact `root_acceptance`, `candidate`, `validation` references.
The root receipt precedes the independent review; the learning stage pins both,
avoiding a circular hash. The post-closure reference authenticates root's census
document without inventing its internal field semantics. Both producing CPU
owners are independently checked for exact PASS/command/admission/owned closure.
Synthetic root receipts must explicitly say `SYNTHETIC_NOT_REAL_AUTHORITY`.

The consumer rechecks the complete immutable carrier inventory and each game's
original shard/proof/order/row IDs before model construction. Selected numeric
files are loaded read-only with the carrier's safe NPY header checks and an
explicit decoded/resident cap. Carrier and root acceptance custody are checked
again before publication. Original TRAIN evidence and input shards still pass
the shared finite learner's proof and inventory gates.

## Draw, overlay and loss semantics

There is exactly one original uniform physical-row draw and one unchanged draw
journal callback. A single-thread replay view captures that same callback tuple;
it never samples again, reweights rows or advances RNG. Every selected row joins
the original shard-inventory SHA and row index, game/ply, input F16 SHA and legal
map. Unknown, duplicated or unconsumed callback identities are refused.

Only a sampled batch dictionary and affected SF arrays are copied, before
`Trainer._prepare_host_arrays`. The original `x`, `policy_target`, integer
`wdl_target`, input/source arrays and sampled row order remain unchanged. Native
W,D,L permille become F32 probabilities; stored query-STM F16 WDL widens to F32.
The existing temperature-1 normalization handles F16 dequantization. Masked
invalid/unavailable rows store finite outcome fallback with `has_sf_wdl=0`;
these values are not teacher observations.

The selected SF share is the existing `sf_wdl_frac`; search share is zero.
The unchanged loss therefore forms, per row,
`(1-alpha)*onehot + alpha*(mask*SF_P0 + (1-mask)*onehot)`.
Both primary coefficients stay 1 with their existing denominator masks. All
other CORE auxiliary coefficients are zero; no adjustment, taper, dampening,
target rebuilding or policy anchoring is enabled.

The same predecessor's raw meta/PV diagnostics and availability flags replace
old P1 diagnostics coherently. Old SF reply-policy/move fields are disarmed,
not used for a main-policy blend. The existing host health check runs before
raw-PV pruning and the single existing mirror. Horizontal mirroring leaves WDL
in current STM; it does not flip W/L. Missing diagnostics remain unmeasured;
health metrics are not a teacher-quality gate or configured weight.

At alpha zero, sidecar bindings must be null: no sidecar acceptance/review,
inventory or numeric payload is opened, and replay is unwrapped. The original
trainer kwargs/loss path and draw/augmentation RNG are used without an overlay
or extra batch copies. Source-code/data-provenance checks still run normally.

Boundary status is `COMPLETED_SF_P0_MAIN_WDL_SEGMENT_V1`, with `segment`, original
generation references, exact profile/carrier lineage, sampled status/qualified
counts and explicit teacher/outcome mass. Positive coverage is reconciled with
actual loss availability. The child cannot attest its own eventual owner closure;
later continuation requires the external exact closed owner binding. No boundary
changes a champion or constitutes a strength/adoption decision.

See the [engineering preregistration and completed CPU readout](experiments/2026-11-14-sf-current-p0-finite-consumer-qualification.md).
