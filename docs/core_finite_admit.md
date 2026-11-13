# Closed-generation CPU admission

`python -m scripts.core_finite_admit --protocol PATH --protocol-sha256 SHA`
audits one complete, closed finite generation. It creates no NN model, performs
no forwards or updates, and queries no engine. It **does** decode the declared
game archives/shards, reconstruct native inputs, and probe its separately owned
strict tablebase handle. Run it only under the existing finite CPU owner.

This is a root-authorized data audit, not a collector or a training controller.
The collector, auditor and learner must use the **same final integrated source**
and qualified native images before fresh generation. An engineering run from
another source head cannot be silently relabeled as that generation.

The hash-pinned audit protocol has exactly these fields:

| Field | Contract |
| --- | --- |
| `schema` | `CORE_FINITE_ADMISSION_V1` |
| `learner_contract` | Ref to the static learning contract, including this auditor's source hash |
| `producer_protocol` | Ref to the actual `CORE_FINITE_GENERATION_V1` protocol |
| `producer_readout` | Ref to the actual complete producer `readout.json` |
| `producer_inventory` | Ref to `{path,files}` for the **entire** sealed producer output tree; files use complete sorted relative path/bytes/SHA256 records |
| `producer_owner` | `{terminal: ref, owner_sha256: SHA, command: [exact child argv]}` |
| `gates` | `{attempts, min_resolved_games, min_total_rows}`; positive integers |
| `runtime` | `{owner: ref, seconds, inner_seconds, cpu_affinity: [CORE,CORE], python: PATH}` |
| `output_root` | New absolute path outside all checkouts |
| `limits` | The six explicit positive caps below |

Refs are `{path,sha256}`. Ordinary JSON remains bounded at 4 MiB. Only the full
producer readout and full-tree inventory use the supplied `max_metadata_bytes`,
which cannot exceed 128 MiB. This accommodates per-row metadata and duplicate
inventory descriptions for a fixed 256-game, 450-ply generation without making
ordinary evidence unbounded. Other required limits are `max_inventory_files`,
`max_inventory_bytes`, `max_raw_decoded_bytes_per_game`,
`max_shard_decoded_bytes_per_game`, and `max_total_decoded_bytes`.

The raw NPZ entry sizes are checked before loading with `allow_pickle=False`.
Zarr shape/dtype/chunk metadata bounds physical decoded arrays before eager load;
the loaded arrays are bounded again. Total decoded accounting counts raw and
loaded arrays, not all temporary mirror/encoding copies or imported libraries.
The finite owner separately controls wall time and available host resources.

The producer terminal must be hash-pinned `PASS`, integer return code zero,
`owned_processes_closed: true`, and `custody: OWNED_PROCESSES_CLOSED`. Its owner
SHA, GPU resource, seconds and exact child argv must match. The command must
invoke the exact collector protocol and separately hash-pinned root admission.
The owner's own checkout revision is not treated as the NN package revision.
Failed markers, pending attempts, failed callback records, changed files,
incomplete fixed ordinals and foreign game directories refuse admission.

The auditor checks the current original actor file hash/stat, all 494 served
state metadata, separate file/tensor-state identities, original protocol/profile,
native/source/assets receipts and engine closure. It imports the exact qualified
native images and applies the repetition fix **before** creating any CBoard.
The encoding adapter holds only explicit history/feature attributes; it is not
a model and does not allocate model parameters.

Every fixed attempted game is retained in the audit, including resolved empty
and censored games. For each original provenance it replays the complete legal
opening and played action trace, verifies native final state and clocks, and
calls the real `VerifiedOutcomeVerifier.verify` with a fresh owned strict handle.
Natural, strict TB and unresolved classes remain distinct. All eligible raw
policy witnesses are re-probed, including their applied/abstained classifications.

For retained rows it reconstructs player-to-move outcomes, chronological physical
IDs, full-history F16 inputs, legal masks and raw/production-mirror fingerprints.
The actual raw F32 archives must reproduce the public raw adapter's bytes,
simplex/mask/POV controls, and the stored F16 targets must match their exact
quantization. Original producing proof/provenance/validation/raw metadata hashes
and the entire sealed inventories are authenticated before and after checks.
This verifies the banked policy/search observations and strict TB overrides; it
does **not** independently rerun Gumbel or infer a search distribution from a
serialized outcome. No stronger policy provenance is claimed.

The public shard writer drops both payload and availability flag when an optional
target is universally inactive; the loader leaves that pair absent. For the
search/categorical diagnostic checks only, a temporary validation view supplies
the public `zeros_for_storage_field` mask **only** when both stored members are
absent and the independently verified raw archive mask is all zero. It never
changes stored dictionaries/inventories, supplies a target, repairs a missing
flag beside a present payload, or changes mandatory game/ply/NET/SP/legal identity
checks. Active or mixed raw masks still require the actual stored payload/flag.

The playing/exposure gate is recomputed on **all** retained attempts. The supplied
gate attempts must equal the fixed producer attempts and learner maximum;
`min_resolved_games` must equal the learner minimum. Every resolved nonempty
verified game is included, in fixed ordinal order. No exact-size subset is chosen,
no censored game is renamed a draw, and no failed attempt is replaced. Fewer
eligible games/total rows than the supplied gates produces no admission.

After all checks and verifier closure, output contains one
`VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1` wrapper per eligible game, `generation.json`
with the learner's `ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1` schema, and a readout
accounting for **all** attempts. These are accepted-schema **candidates** authorized
by the root audit protocol. The child cannot attest its own future owner closure.
Root supplies learner refs only after the actual audit owner closes PASS and a
separate immutable promotion receipt links the exact command and output hashes.

All outputs are sealed. Any publication error attempts an external sibling
`OUTPUT.failed.json`; such a marker, failed owner, or incomplete hash review
forbids promotion even if a candidate generation file exists. No existing output
is overwritten and there is no automatic retry. Literal zero side-effect fields
describe this source call graph; they are not fabricated runtime hook telemetry.
