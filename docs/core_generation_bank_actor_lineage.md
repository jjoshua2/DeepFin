# Adjacent bank-learner actors in the finite collector

The existing `scripts.core_finite_collect` entry and `CORE_FINITE_GENERATION_V1`
protocol support one additional actor kind, `bank_learner`. It is a strict
versioned lineage route for an adjacent, separately closed checkpoint from
`scripts.core_generation_bank_learner`. It is not an arbitrary checkpoint,
historical-bootstrap collection mode, deployment instruction or strength claim.
The old `donor` and `learner` branches keep their original checks.

## Descriptor

For this kind, the actor has exactly:

```text
kind = bank_learner
path, sha256, architecture, expected_checkpoint_step
predecessor = {boundary, owner_terminal, owner_sha256, command}
lineage = {
  schema: core_generation_bank_actor_lineage_v1,
  run_uuid, source_revision, contract_sha256, segment, logical_generation
}
```

The lineage fields identify the LEARNER. The producing protocol's own run UUID,
collection UUID, logical generation and source still identify the PRODUCER.
The predecessor owner wrapper may have a third source revision; it is not
compared to either child source. Historical learner pins are checked as pinned
metadata, never rehashed against the current producer checkout.

The named new branch requires a canonical learner UUID, lowercase40/64-hex
source/contract/checkpoint hashes, strict positive integer segment/logical
generation/completed optimizer step, exact descriptor keys, and an external
checkpoint path. Unknown fields or compatibility flags are refused. New-family
parents presented as old `learner` actors still fail the original old CORE path.

## Closed predecessor and counter joins

`closed_bank_predecessor` verifies the actual new-family contract, original
stage, boundary and exact closed owner command. It requires:

- `core_generation_bank_learning_v1` with its complete versioned field roster;
  `core_generation_bank_stage_v1`; and
  `COMPLETED_CORE_GENERATION_BANK_SEGMENT_V1`.
- PASS/return0, owned-process closure, supervisor digest, and exact command
  suffix `-m scripts.core_generation_bank_learner --contract ...
  --contract-sha256 ... --stage ... --stage-sha256 ...`.
- Matching learner run/source/contract/segment, unchanged parent and accepted-bank
  history, consecutive logical bank identities, canonical full architecture,
  initial-model/actor-step provenance and cumulative update/view accounting.
- `parent.logical_generation == producer.generation - 1`, independently of
  learner segment; the actor checkpoint step equals the parent's actual
  cumulative optimizer step. The actor file reference is exactly the parent's
  final `learner.pt`, with no failure marker or pending boundary.

For example, a first learner segment consuming logical G2 can end at optimizer
step2 despite bootstrapping historical weights at step9000. Its resulting actor
feeds logical G3. It is not relabelled as a G1 parent to claim a G2 collection.
Old G1 donor and old CORE adjacent-learner behavior remain unchanged; no new
historical bootstrap fallback is enabled.

New-family descriptor/protocol, contract, stage and boundary metadata references
must be canonical absolute paths outside every checkout and without symlinks.
The new metadata seam checks regular readonly files, size before reads (32MiB
per JSON file), SHA256 and stable dev/inode/size/mtime/ctime/mode. Duplicate keys
and nonfinite JSON are rejected; atime is excluded. The supervisor terminal may
retain its existing0644 mode, but its bytes and full non-atime stamp must still
be stable and hash-pinned. All four predecessor metadata objects are rechecked
before the join returns. Accepted-bank history refs remain exact tokens; the
collector does not reopen historical banks or source payloads.

As with the original collector, trusted callers must obtain the predecessor
pins after observing complete owner closure. Status strings are not issuer
signatures. A child boundary cannot attest its supervisor's later exit.

## Actual execution and serving witness

There is no new CLI or producer schema. `--check` and `--execute` use the actual
public validator; the latter retains the existing separate
`ROOT_ADMITS_CORE_GENERATION_V1` GPU-owner admission and full protocol hash.
Source/native/asset qualifications, fixed roster, exclusions, complete input
history, resource limits, strict TB, all attempts and cleanup remain unchanged.

`load_actor` rechecks the lineage before actor work. Its existing `capture_load`
requires the actual loaded payload's `core_generation_bank_learner_v1` tag to
match the descriptor's learner run/source/contract/segment/logical generation.
The check occurs before model construction, inside the existing single
weights-only decode. No second checkpoint decode is added. The original embedded
step, full494 state, complete architecture, one constructor, exact served tensor
state and actor file pre/post custody guards still apply.

The serving record additionally banks `bank_learner_tag`, `actor_lineage` and
`predecessor_boundary` for this kind only. The actual observed tag is retained;
no serialized Boolean substitutes for observing the loaded payload. An old
boundary's `collector_lineage_qualified=false` is not rewritten or treated as a
permission flag. This bridge and a separately admitted collector establish their
own facts; no past record gains retrospective qualification.

## Data-only audit and downstream bank view

The recipe-independent entry `scripts.core_generation_data_admit` uses the same
collector validator and real shared audit body. For the new kind it additionally
joins the serving actor kind, exact lineage, observed tag and predecessor boundary.
The legacy recipe-bound auditor refuses this kind explicitly.

The existing collector command, producer/readout schemas, root producer admission,
data protocol/spec and candidate manifest/wrapper statuses do not change.
`candidate_only=true`, `learning_admitted=false` and separate root/independent
postclosure acceptance remain required. The data spec continues to describe
producer identities and expected actor SHA/step, not a learning recipe. Parent
learner identities remain in the pinned producer protocol and serving evidence.
The bank-acceptance API requires no schema relaxation, and the new learner can
check a later bank's actor SHA/step against its own adjacent checkpoint.

## Scope and validation

Synthetic tests exercise the real public validator, actual loader capture with
inert loader/model resources, and the actual recipe-free audit with existing
native/raw/Zarr/FakeTB fixtures. They cover counter/source separation, strict
lineage and metadata custody failures, corrupt/missing loaded tags before any
constructor, legacy refusal, candidate preservation and bank-view handoff. A CLI
spy test establishes dispatch only. None of these fixtures are real accepted
bank/owner authorities or full GPU serving qualification.

No actual actor, parent, recipe, population or collection is selected by this
source increment. No cancelled study is resumed. Author-side project imports,
tests, lint, builds and runtime/payload access remain unrun; root owns subsequent
bounded qualification and publication.

## Subsequent full-model software qualification

The [November15 bounded handoff readout](experiments/2026-11-15-full494-bank-handoff-software.md)
records one genuine public learner update on a synthetic fixture, followed by
actual full494 actor loading and CUDA BF16 inference from its separately closed
checkpoint. Both loaded-tag fault controls refused before construction. Its
initial producer/auditor/root/independent authority graph is explicitly fictional;
the learner execution and closure are real. This follow-up qualifies that finite
software route, with separate root/independent actual acceptance. Real collection,
sustained-loop readiness, recipe/adoption and playing strength remain unqualified.
