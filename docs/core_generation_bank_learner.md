# Finite learner for accepted generation banks

`python -m scripts.core_generation_bank_learner --contract PATH
--contract-sha256 SHA --stage PATH --stage-sha256 SHA` is a distinct finite
family. It consumes caller-trusted postclosure bank references through
`validate_accepted_generation`, verifies and decodes actual payloads, draws from
`GenerationReplay`, runs the actual finite `NoRetryTrainer`, and publishes a
checkpoint and closed-segment boundary. It does not collect, promote candidates,
select actors, select a scientific recipe, or authorize its own resource owner.

No candidate status is changed. In particular there is no generated legacy
`ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1` envelope. The old CORE CLI, metadata
schema and default generation1 replay path remain unchanged.

## Contract and stage

The contract has exactly:

```text
schema = core_generation_bank_learning_v1
run_uuid, source_revision, source_files, architecture
initial_model = {path, sha256}
initial_actor_step, first_generation
batch_size, games_per_generation, max_games_per_generation
updates_per_segment, replay_generations, seed, runtime, lr, mirror_prob, limits
bank_limits = {max_file_bytes, max_total_bytes, max_games}
```

`run_uuid` and source pins describe this learner. They do not replace the
historical producer/auditor/owner identities in accepted bank records.
`source_files` must include all existing `core_finite_learner.SOURCE_FILES`
plus this entry and `scripts/core_generation_bank_acceptance.py`. The strict
existing training-setting validator checks the explicit runtime/LR/horizon,
architecture, initial-model file custody, source pins and limits. Its internal
settings projection has no data manifest and supplies no admission alias.

The caller reuses existing `CORE_KWARGS` training math, including its fixed
optimizer and objective definitions. No new loss, teacher overlay or KL path
is introduced. The remaining numeric settings must be supplied by a future
caller; this implementation does not choose a real model, bank or configuration.
Replay remains four complete generations. Metadata limits are explicit positive
integers; file bound is at most128MiB. These are bounded reader policies, not
claims that all candidate graphs fit a universal small-file limit.

Stage fields are exactly:

```text
schema = core_generation_bank_stage_v1
contract_sha256
segment
banks = [{root_acceptance: ref, independent_review: ref}, ...]
parent = null | {boundary, owner_terminal, owner_sha256, command}
output
```

Both receipt pins for every bank must be authenticated by the trusted caller.
Status strings do not authenticate issuers. There must be `segment` banks in
consecutive logical order starting at `first_generation`. All eligible games
are kept, with no repeated physical game IDs across banks. Architectures must
match exactly under canonical JSON; original raw/history/TB qualification is
inherited from the closed data audit and trusted independent acceptance.

For example, `first_generation=2` and `segment=1` mean a genuine G2-only bank
and the first learner segment. They do not imply a hidden G1 or optimizer step2.
`updates_per_segment[segment-1]` determines only this segment's requested dose.

## Payload and sampling path

The accepted view is recursively copied from mapping proxies/tuples to ordinary
dict/list metadata for existing inventory helpers. The copy preserves every
status, source identity and descriptor; it never changes candidate files.
A shallow `dict(game)` would leave tuple inventory entries and is insufficient.

The real existing `verify_inventory` and `load_shard_arrays(validate=True)` path
checks custody before/after eager decode, input/target shape and F16 storage,
all active TRAIN flags, legal policy support, integer game/ply/outcome fields,
game IDs, unique retained plies, history/policy/repetition encodings and replay
row/decoded-window budgets. Payload admission completes before Trainer/model
construction. Source/import custody is checked separately against the learner
checkout, without comparing historical bank source pins to that checkout.

The shared loader's new keyword is `first_generation=1`; old callers keep that
default. This caller passes its authenticated first logical bank explicitly.
The sampler preserves consecutive admission, lifetime source-duplicate refusal,
transactional refill, four-bank FIFO eviction, uniform row draws, and the same
Generator used for augmentation. The actual `on_draw` callback fsyncs
`draws.jsonl` before augmentation; no additional sampler or journal exists.

## Initial weights and adjacent continuation

The first segment requires `parent=null`. Its first bank actor SHA/step must
match `initial_model`/`initial_actor_step`. The existing strict complete donor
loader checks model keys/shapes/dtypes, finite values, shared aliases and exact
architecture. It loads model weights only into a new Trainer. Optimizer step0,
empty optimizer state, unchanged fresh scheduler/zclip state and no SWA are
required. The historical actor checkpoint step is recorded independently; its
optimizer state and RNG are not imported.

Later segments require the exact adjacent closed new-family parent, same
contract/run/learner source, unchanged prior accepted-bank history, expected
cumulative optimizer step, and the next bank's actor SHA/step equal to that
parent checkpoint. The external owner terminal must prove PASS/return0,
owned-process closure and exact new-entry command/owner SHA. Its wrapper revision
is not treated as the learner revision. The trusted caller must pin the terminal
after independently observing full owner closure; the child cannot attest it.

The checkpoint must carry this family's contract/run/source/segment/logical-bank
tag. The existing strict resume helper verifies complete model, named optimizer
ownership, scheduler, step, peak LR and zclip state; no weights-only fallback is
allowed. All five boundary RNG components are then restored and compared **after**
model/load/replay reconstruction: replay Generator, global NumPy RandomState,
Torch CPU, Torch CUDA and Python. Backend settings must match separately.

Applied optimizer post-hook counts, Trainer step delta, completed steps, zero
skip/retry rates, drawn samples and batch-size view totals must reconcile. The
boundary records segment/logical generation, initial actor step, optimizer start
and end steps, accepted bank references and historical identities, checkpoint
SHA/tag, membership, metrics, draw SHA, five RNG states and backend settings.
`Trainer.save()` does not itself contain RNG/backend/membership; the boundary
does. Publication uses the existing sealed pending-file/hardlink/fsync boundary
protocol. Failed or interrupted outputs are terminal, cannot be reused, and a
failure marker invalidates any exposed boundary.

## Limits and tests

This is a caller capability, not a qualified on-policy loop. The unchanged
collector still accepts later actors only through its old CORE predecessor
command/status/source chain. This new family is deliberately not relabelled to
pass that chain. A separately reviewed collector lineage bridge is still needed
before a new-family checkpoint can produce a genuine next generation.

Synthetic tests use fictional trusted receipts, tiny actual model/Trainer state
and real serialized replay payloads. They do not qualify the original producer's
full494/175 architecture or any real bank. Tests cover a real G2 update and journal,
model-only bootstrap, deep metadata conversion, payload refusal before model
construction, adjacent resume against a direct checkpoint continuation reference, five RNG
states, and contract/bank/owner failures. Existing legacy and primitive replay
regressions remain relevant. Author-side execution, tests and lint are unrun;
all numerical qualification belongs to the root's later owned CPU phase.
