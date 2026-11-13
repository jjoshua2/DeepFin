# Finite core generation collector

`scripts/core_finite_collect.py` collects one fixed generation through the real
finite `play_batch` API. It supplies reusable producer evidence for the
[finite learner](core_finite_learner.md), never a root admission or promotion.
The caller supplies the roster, exploration, horizon, resources and qualified
source/assets. No scientific data count, LR, evaluation threshold or champion
selection is chosen here. Gumbel remains explicit; UCI `Threads` is irrelevant.

```bash
python -m scripts.core_finite_collect --check \
  --protocol /outside/checkouts/generation.json --protocol-sha256 SHA256 \
  --report /outside/checkouts/descriptor.json

# Run only through the separately admitted finite resource owner:
python -m scripts.core_finite_collect --execute \
  --protocol /outside/checkouts/generation.json --protocol-sha256 SHA256 \
  --admission /outside/checkouts/root-admission.json --admission-sha256 SHA256
```

`--check` authenticates source/native/asset receipts, the roster/starts/exclusions,
actual dataclass consumers and profile. It imports the project but constructs no
model, evaluator, engine or tablebase handle. `--execute` additionally requires
`ROOT_ADMITS_CORE_GENERATION_V1`, matching the protocol SHA, study UUID,
generation, source revision and actor SHA. It checks the direct parent owner,
GPU resource, whole-stage cap, two declared CPU cores, CUDA visibility and thread
limits. The outer owner remains responsible for whole-process resources and
complete descendant closure. A descriptor check does not admit execution.

## Protocol

JSON references are exactly `{path, sha256}`. All paths are explicit. The input
protocol has schema `CORE_FINITE_GENERATION_V1` and these required groups:

| Field | Contract |
| --- | --- |
| `run_uuid`, `collection_uuid` | Canonical study UUID and unique predeclared physical generation UUID |
| `generation`, `attempts`, `concurrent_slots`, `seed` | Positive generation/counts, slots ≤ attempts, explicit nonnegative seed |
| `roster` | Ref to the entire ordered roster; ordinals `0..attempts-1`, bank `train`, unique `physical_game_id=collection_uuid:ordinal`, canonical `root_fen` and ordered legal `opening_moves` |
| `starts` | Ref to exact roster lines `root_fen | ordered UCI moves`; legal replay must agree with the actual opening consumer |
| `actor` | `kind`, immutable file `path`/`sha256`, complete schema-20 `architecture`, `expected_checkpoint_step`, and `predecessor` |
| `source` | Exact clean `checkout`/`revision`, repository-relative `sha256` map including `REQUIRED_SOURCE_FILES`, and `native_qualification` ref |
| `runtime` | Actual `python`, `torch_version`, `numpy_version`, `device: cuda:0`, two `cpu_affinity` cores, `threads: 2`, integer `inner_seconds` below the owner cap |
| `owner` | Exact `path`/`sha256` and positive `seconds` ≤ 7200 |
| `tablebase` | Full `path`, metadata `asset_inventory_sha256`, accepted consumer `qualification` ref |
| `stockfish` | Binary `path`/`sha256`, accepted `qualification` ref, explicit positive `nodes`, `multipv`, `hash_mb`, `read_timeout_seconds`, and one or two `workers` |
| `configs` | Explicit direct dataclass kwargs for `game`, `search`, `temp`, `opening`, `opponent`, `diff_focus` |
| `resolved_profile_sha256` | SHA256 of compact sorted-key JSON of all six realized dataclasses |
| `serving` | DirectGPUEvaluator kwargs: CUDA0, AMP true, `amp_dtype: bfloat16`; buffer sizes/routes must be declared |
| `exclusions` | Hash-bound `prior` and `playing_starts`, exact `playing_start_count`, `prior_overlap_action: report/refuse`, `playing_overlap_action: refuse` |
| `output_root` | Fresh absolute output directory outside the checkout |

The native receipt must accept this checkout/revision, all declared native build
inputs, five distinct module/image paths and their SHA256s, and a hash-bound
closed PASS owner terminal. Runtime imports are joined to those exact images.
The TB receipt binds source revision, `tablebase_path` and the asset metadata
inventory; SF qualification binds source revision, `binary_sha256` and full
`syzygy_path`. These are supplied qualifications, not invented by the collector.
Metadata inventory identity is not a claim that TB files were byte-hashed.

The prior-index document contains `history_sha256`,
`stored_input_float16_sha256`, `coverage_scope` and
`donor_pretraining_exposure: UNKNOWN`. Playing-start entries are exactly
`{id,input_sha256,mirror_sha256}`. Retained inputs and actual production mirrors
are compared after the entire fixed generation. A collision preserves the
attempts and returns an unqualified gate; no starts or games are replaced.
No finite known index establishes universal novelty or unknown donor exposure.

## Actual actor and finite selfplay

Generation 1 requires immutable original A39 file SHA
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
Its historical checkpoint step is preserved; fresh learner step zero is a
separate initialization. Later actors must be the preceding generation's final
learner checkpoint. `actor.predecessor` then has
`{boundary,owner_terminal,owner_sha256,command}`: two refs plus the exact owner
hash and child argv. The terminal must be PASS/return 0 with owned closure,
matching command and owner SHA; contract/stage/boundary joins authenticate the
previous final checkpoint. Pending-boundary or external failure markers refuse. A child `COMPLETED` label alone is insufficient.
No intermediate checkpoint or champion score selects this actor.

One call to the public strict embedded-architecture loader constructs one model.
The observer checks the full 494-tensor checkpoint state against the actual CUDA
model before serving. It records file SHA separately from the full tensor-state
SHA, exact embedded architecture and actual file counter. Pre/post model digests
must agree, and the actor file's SHA/stat/readonly identity must remain unchanged.
Real forwards must observe CUDA175/no-grad/BF16. This does not infer that CUDA was
used merely from memory allocation or a filename.

The realized route requires SP fraction 1, Gumbel64 at every recorded ply,
full-simulation fraction 1, `policy_row_retention: all_eligible`, diff-focus off,
fast value rows off, history repetition fix on, and strict rule50 TB search,
adjudication and policy correction on the same full path. Exploration and other
caller settings are authenticated in the profile. Diagnostic SF capture can
remain; no SF-loss or no-query claim follows from this tool. Initial and replacement
SF handles receive explicit 50MoveRule=true/ProbeLimit=6 and readiness checks.
All completed/censored attempts retain their original identities; an exception
preserves completed attempts and the failing audit, never fabricates a draw.
There is no Worker, Tune, continuous mode or interrupted-generation resume.

## Evidence and learner admission

`attempt.json` is written/fsynced before callback count/proof checks. Each game
then gets original `provenance.json`, `outcome-proof.json`, raw F32 targets and
metadata, `validation.json`, and a complete sorted shard-file inventory. Failure
hashes are bounded and explicitly unqualified. All successful game files and
final output directories are sealed read-only.

The reusable `VerifiedOutcomeVerifier` owns a second safe strict handle, legally
replays full history, rechecks native result/clocks and fresh rule50 resolution,
re-probes **all** TB-eligible raw policy records and retained policy effects, and
issues an in-process proof. Serialized truth flags do not issue a proof. The
classes remain `NATURAL`, `STRICT_TB_ADJUDICATED`, and `UNRESOLVED`; positive-clock
unresolved outcomes play on, and censored outcomes have zero retained rows.
The raw adapter checks actual F32 policy/value provenance and player-to-move
outcome, then joins F16 serialization, IDs/plies, availability and NET/SP/legal
flags. It does not choose a learner value mixture.

All original evidence documents carry `run_uuid`, `collection_uuid`,
`source_revision`, `generation`, `actor_sha256`, `served_state_sha256`,
`physical_game_id`, `game_id`, `bank`, `rows`, and `scientific_class`.
`outcome-proof.json` schema is `CORE_PRODUCING_OUTCOME_PROOF_V1`; its
`proof_sha256` authenticates canonical `verification`. That payload includes the
same identities, boolean `resolved`, `retained_rows`, actual witness/history/asset
identity and complete policy classification. `validation.json` reports actual
raw/history/row joins and `shard_inventory_sha256`. Inventories cover every shard
file, including metadata/seals; a single-file payload uses relative name `.`.

`actor-serving.json` provides `actor_sha256`,
`checkpoint_model_state_sha256`, `pre_state_sha256` and `post_state_sha256`.
Every game's `served_state_sha256` joins that tensor digest, not the actor file
SHA. The final `COLLECTED_CORE_GENERATION_V1` readout accounts for all attempts,
classes, exclusions, serving and observed SF closure. It explicitly admits no
learning and issues no root acceptance.

An independent closed-stage CPU audit must validate this evidence and create the
learner's `ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1` admission and
`VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1` wrappers. The explicit source-class mapping
is `STRICT_TB_ADJUDICATED → TB_ADJUDICATED`; natural stays natural. Only resolved,
nonempty TRAIN games can enter that learner admission. The collector neither
chooses a minimum count nor manufactures replacements if that gate fails.
