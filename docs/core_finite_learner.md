# Finite core learner

`scripts/core_finite_learner.py` performs **one** externally admitted generation of
learning. It has no collector, scheduler of jobs, champion promotion, or research
authorization. It never changes production YAML. Run it through the existing
finite resource owner, with two CPU threads and explicitly selected CUDA visibility.

```bash
python -m scripts.core_finite_learner \
  --contract /outside/checkouts/learning.json --contract-sha256 SHA256 \
  --stage /outside/checkouts/generation-1.json --stage-sha256 SHA256
```

Inputs must be absolute, immutable files outside checkouts. JSON refs are exactly
`{"path": "/absolute/path", "sha256": "..."}`. Duplicate JSON keys, changed
hashes, links, writable payloads, reused outputs, HOLD, and unresolved games refuse.
Use a new output directory for each generation; it must not already exist.
Ordinary metadata is capped at 4 MiB. Only the inventory-heavy accepted generation
manifest has a 32 MiB cap, sufficient for a declared 256-game Zarr roster. Both
caps are checked before decoding; neither permits an unbounded metadata input.

The static contract has these fields, with **no** implicit scientific defaults:

| Field | Meaning |
| --- | --- |
| `schema` | `core_finite_learning_v1` |
| `run_uuid`, `source_revision` | Canonical UUID and exact clean checkout commit |
| `source_files` | Repository-relative file → SHA256; must include `SOURCE_FILES` from the helper |
| `architecture` | Complete donor architecture including `_schema_version`; no field dropping or migration |
| `donor` | Original model checkpoint ref; model-only fresh loading occurs once |
| `batch_size` | Positive fixed batch size |
| `games_per_generation`, `max_games_per_generation` | Positive minimum/maximum admitted resolved nonempty game counts; minimum ≤ maximum |
| `updates_per_generation` | Nonempty list of positive applied-update counts; defines global horizon |
| `replay_generations` | Exactly `4`, FIFO of complete generations |
| `seed`, `mirror_prob` | Initial uint32 seed and explicit augmentation probability |
| `runtime` | Exact `device` (`cpu`/`cuda:0`), `use_amp`, Torch/NumPy version strings, `torch_threads: 2`, `cuda_devices: 0/1` |
| `lr` | Explicit `peak`, `floor`, `warmup_start`, `warmup_steps`, `first_period` |
| `limits` | Positive `max_rows` and `max_decoded_window_bytes` |

No research LR constants are provided. The first cosine period must exceed the
largest post-warmup scheduler epoch within the entire horizon. Trainer's `cosine`
means `CosineAnnealingWarmRestarts`; this bound prevents a restart. Its LR update
sets the **next** update's LR. There is one warmup and one global step across all
generation calls, not a release schedule restarted for each call.

The code fixes main policy CE and main outcome WDL CE at weight 1, all twelve
other loss weights at 0, policy target temperature 1, SF/search value shares 0,
adjusted/terminal-transfer targets off, feature/group/residual dropout and channel
balance off, target rebuilds off, eager BF16 on CUDA when AMP is selected, no
prefetch, no clipping, and SWA disabled. The main policy is the supplied qualified
Gumbel improved target; no PUCT replacement is implied. A39's coupled categorical
topology remains: disabling categorical CE leaves main-WDL gradients through its
shared value embedding intact.

NadamW is fresh only at generation 1, using the real Trainer's β=(.9,.98), ε=1e-7
and decoupled decay. Its actual two-group mask applies `1e-4` to parameters with
dimension >1 except names ending `.bias`, and zero otherwise. Matrix/aux optimizer
knobs do not configure this branch. This is a declared local adaptation, not an
assertion of numerical equivalence to another framework's NadamW.

A stage has exactly `schema: core_finite_stage_v1`, `contract_sha256`, `generation`,
`generations` (ordered manifest refs for 1 through this generation), `parent`, and
`output`. At generation 1 `parent` is null; later it is this root-supplied envelope,
created **after** the preceding actual resource owner has successfully closed:

```json
{
  "boundary": {"path": "/previous/output/boundary.json", "sha256": "..."},
  "owner_terminal": {"path": "/previous/owner/terminal.json", "sha256": "..."},
  "owner_sha256": "SHA256_OF_EXISTING_RESOURCE_OWNER",
  "command": ["/qualified/python", "-B", "-m", "scripts.core_finite_learner",
              "--contract", "/static/learning.json", "--contract-sha256", "...",
              "--stage", "/previous/stage.json", "--stage-sha256", "..."]
}
```

The separately hash-pinned terminal must report `status: PASS`, integer
`returncode: 0`, `owned_processes_closed: true`, and
`custody: OWNED_PROCESSES_CLOSED`, matching `supervisor_sha256` and the **exact**
command. The command's module/contract/stage suffix joins the actual original
hash-pinned stage recorded in the boundary. Its output, generation, parent and
data history must match too. Wrapper `source_revision` is deliberately not treated
as the child's NN source identity: an existing owner can launch from a different
checkout. Original owner terminals can remain 0644; their pinned bytes and stable
stat/read are authenticated without modifying their permissions. Root's admission
owns the independent actual custody review; this child cannot issue that terminal.

The preceding boundary's manifest prefix must be byte-identity-equivalent to
the parent's recorded refs. The current generation's actor checkpoint SHA must
equal that parent's learner checkpoint SHA (generation 1 uses the original donor).

Each generation manifest is issued **by the external data admission/audit**, with
`schema: ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1`, `run_uuid`, `source_revision`,
consecutive `generation`, `actor_sha256`, `actor_serving` ref, and all the actual
eligible `games` within the declared minimum/maximum. A fixed attempted roster may
contain censored or empty games; the independently audited admission accounts for
every attempt and retains **all** resolved nonempty games, without choosing an
exact-size subset or replacing failures. Qualification may set minimum=maximum.
The [closed-generation CPU auditor](core_finite_admit.md) provides the reusable
actual verification path and accepted-schema candidates. Root must promote its
hashes only after that auditor's own finite owner has successfully closed.

Each admitted game contains:

```json
{
  "physical_game_id": "COLLECTION_UUID:ORDINAL",
  "game_id": 0,
  "bank": "train",
  "label_class": "NATURAL",
  "rows": 1,
  "producer_proof": {"path": "/absolute/admission.json", "sha256": "..."},
  "shard": {
    "path": "/absolute/game/shard.zarr",
    "files": [{"path": ".zattrs", "bytes": 123, "sha256": "..."}]
  }
}
```

The inventory is the **complete** sorted relative list, including metadata and
seals; a single NPZ file uses relative name `.`. Directories and files are read-only.
`producer_proof` is a trusted audit wrapper with
`status: VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1`, the same physical/game/row/source/
actor/class fields, `bank: train`, `resolved`, `full_history_verified`, and
`raw_targets_verified` all true, and `shard_inventory_sha256` equal to SHA256 of
canonical inventory JSON (sorted keys, compact separators, UTF-8, no newline).

Its mandatory `evidence` refs are `outcome_proof`, `validation`, and `provenance`.
These are the **original producing documents**, not truth issued by this learner.
All carry matching `run_uuid`, `source_revision`, `generation`, `actor_sha256`,
`served_state_sha256`, `physical_game_id`, `game_id`, and `rows`. The outcome doc's
schema is `CORE_PRODUCING_OUTCOME_PROOF_V1`; `proof_sha256` binds canonical
`verification`, which retains actual resolved/retained-row/source/game/actor/state
identities. Original `STRICT_TB_ADJUDICATED` maps explicitly to wrapper
`TB_ADJUDICATED`; NATURAL remains NATURAL. Both must be resolved and nonempty for
TRAIN. Root's independent audit qualifies that proof and full history; this learner
authenticates its bytes/joins and **does not** query tablebases or infer truth from
filenames or these booleans. Producer collection alone may not issue root acceptance.

`actor_serving` binds checkpoint file SHA separately from the full tensor-state
digest: `pre_state_sha256 == post_state_sha256 == checkpoint_model_state_sha256`.
Every producing evidence doc's `served_state_sha256` joins that digest, not the
checkpoint file SHA. This is reuse of externally qualified serving evidence.

Before updating, the caller eagerly loads and validates all declared game arrays,
checking row count, F16 input/policy storage, input/policy/repetition encodings,
network/selfplay/policy/legal masks, game IDs and distinct ply indices. It reconstructs
generations consecutively through `GenerationReplay`, releasing evicted generations
rather than retaining an all-history array list. Decoded-window limits exclude
temporary copies, optimizer/model allocation and compressed input bytes; the outer
owner must separately bound whole-process memory, time and I/O.

Every pre-augmentation draw is durably journaled by generation, source-inventory
SHA and local row index. Draws are not automatically applied views. The thin
Trainer subclass preserves successful update math but makes RuntimeErrors terminal
before Trainer's CUDA retry branch can replace a batch, rejects skipped nonfinite
steps, and checks finite model/optimizer state. Any failure makes that output
unqualified, preserves its journal, and permits **no automatic retry or in-window
resume**. Partially mutated optimizer state is never advertised as usable.

At a successful quiescent boundary, `learner.pt` preserves model/optimizer/global
step/scheduler and `boundary.json` binds its full SHA, static contract, parent, data
history, membership, drawn/applied counts and RNG. RNG capture/restore reuses the
pure functions in `offline_epoch_resume.py`: its `mask` field represents NumPy's
global RandomState here, alongside the replay/mirror Generator, Python, Torch CPU
and all visible CUDA RNG states. No fixed-corpus epoch cursor is reused.
Loaded project modules must originate in the pinned checkout. Actual precision,
determinism and thread backend settings are recorded at the first boundary and
must match on continuation; this check does not establish cross-device determinism.
Completion writes/fsyncs a pending boundary, freezes payloads, then exclusively
links the final boundary name and seals/fsyncs the directory. Any exception after
output creation always attempts a read-only **external** sibling marker
`OUTPUT.failed.json`, even if final publication already exposed `boundary.json`.
It preserves a terminal failure if the output itself is already read-only. Parent
consumption refuses that marker or a pending boundary and requires the independently
successful closed owner above; a child's `COMPLETED` file alone is insufficient.
If the filesystem refuses even the external marker, the error remains nonzero and
the missing successful owner terminal still prevents continuation.

Continuation uses `resume_bootstrap`'s strict trained-state/name/architecture checks,
then restores boundary RNG **after** construction/load/replay reconstruction,
superseding that helper's temporary Torch reseed. The same contract persists across
generations; this does not enable strict Worker/Tune restoration. Only a sealed
completed boundary can be consumed. The new learner checkpoint is available for
the next private actor; the preserved champion is unchanged. None of this establishes
strength or authorizes an experiment.
