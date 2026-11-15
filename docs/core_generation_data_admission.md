# Recipe-independent finite-generation data audit

`python -m scripts.core_generation_data_admit --protocol PATH --protocol-sha256 SHA256`
audits one separately closed finite producer against explicit data assertions. It
does not select an actor, collect data, promote a bank, or admit model learning.
There is no `--data-only` switch on the legacy auditor.

The new entry requires exactly these protocol keys:

```text
schema = CORE_FINITE_DATA_ADMISSION_V1
data_spec = {path, sha256}
producer_protocol, producer_readout, producer_inventory = {path, sha256}
producer_owner = {terminal, owner_sha256, command}
gates, runtime, output_root, limits
```

The producer envelope, finite CPU runtime, limits and gates have the same strict
fields and checks as `scripts.core_finite_admit`. Gates are `attempts`,
`min_resolved_games` and `min_total_rows`: every fixed attempt is audited and all
eligible games are retained, rather than selecting exactly the minimum count.
The CPU owner, source/native qualification, assets and actual process closure
must independently satisfy the existing audit path.

The referenced specification has **exactly** these fields:

| Field | Required meaning |
|---|---|
| `schema` | `core_generation_data_spec_v1` |
| `run_uuid`, `collection_uuid` | Canonical UUID strings matching the producer |
| `generation` | Positive Python integer, the actual logical collection generation |
| `source_revision` | Lowercase40-hex producing revision, in the same integrated checkout |
| `source_files` | Canonical in-checkout relative filenames and actual lowercase SHA256 digests |
| `architecture` | Complete producer actor architecture; must survive the existing config reconstruction |
| `actor_sha256` | Lowercase64-hex checkpoint identity matching the producer |
| `actor_checkpoint_step` | Nonnegative Python integer matching producer and actual serving evidence |
| `games_per_generation` | Positive minimum resolved nonempty game count |
| `max_games_per_generation` | Positive fixed attempt count, at least the minimum |

Booleans and floats are not integer identities. Source pins must contain every
`core_finite_learner.SOURCE_FILES` member plus `scripts/core_finite_admit.py` and
`scripts/core_generation_data_admit.py`. Every supplied pin is checked, including
extra source files. Unknown specification fields are refused: no donor, optimizer,
LR, losses, batch size, updates, seed, mirror or replay recipe belongs here.

The new route never calls `learning.validate_contract`, fabricates a learning
contract or constructs/restores a model. It shares the actual legacy audit body:
hash-pinned producer protocol/admission and closed owner command; exact roster,
source/native and realized configuration; full inventory and serving-state
identity; every original game proof/history/physical ID; raw-versus-stored targets,
native175-plane encoding, legal1858 masks and strict rule50/TB verification;
whole-attempt accounting, exposure exclusions, min/max/row gates and final custody.
The separately pinned actor checkpoint is still hashed for file identity, never
decoded. A data spec's expected actor step is an assertion against this chain,
not an actor-selection decision. The original producer admission pins the entire
protocol, including collection UUID and actor step, while its explicit identity
fields and the readout/serving evidence are also joined by the shared auditor.

## Candidate output and promotion boundary

The output paths remain `generation.json`, `readout.json` and one game wrapper per
eligible physical game. New identities are deliberately distinct:

- Generation schema: `CORE_TRAIN_GENERATION_DATA_CANDIDATE_V1`.
- Game status: `VERIFIED_RESOLVED_CORE_DATA_GAME_CANDIDATE_V1`.
- Readout status: `AUDITED_CORE_DATA_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1`.
- Failure marker status: `FAILED_CORE_DATA_ADMISSION_V1`.

The generation, readout and game wrappers carry the `data_spec` reference,
`collection_uuid`, `logical_generation`, `actor_checkpoint_step`,
`candidate_only=true` and `learning_admitted=false`. Their `generation` identity
is not renumbered. Postclosure promotion is explicitly required; the child cannot
attest the future closure of its own owner.

The current learner rejects the new generation schema and the new game statuses.
Changing an envelope label or a promotion Boolean alone is not admission. A trusted
root/independent postclosure data-promotion policy and a compatible new-family
learner integration remain separate, unimplemented work.

`scripts.core_finite_admit.execute` and its CLI still accept only
`CORE_FINITE_ADMISSION_V1` with `learner_contract`, run the original strict learner
validator and emit their original schemas/statuses and fields. The new entry
rejects that protocol; the legacy entry rejects the new one.

## Scope and validation limits

This increment does not change the collector, learning engine, profiles, contracts
or production callers. The existing collection path still requires its full
494-entry/175-plane architecture and serving evidence. Its generation1 actor is
the pinned A39 donor; later generations still require its original closed OLD CORE
learner predecessor. The same integrated producer/auditor source restriction is
unchanged. This is not arbitrary new-family actor support, refreshed-donor G2
admission, optimizer continuation, a working on-policy loop, or authorization to
resume any stopped study. It selects no labels, target transform or training recipe.

Synthetic tests exercise the actual new public entry and shared audit body using
real native encoding, raw-target serialization, Zarr and full-history checks with
a fake owned tablebase. Only external runtime/native qualification and synthetic
actor resources are substituted; no real bank, checkpoint or engine is involved.
They cover identity/source/gate failures, producer closure/failure markers, serving
step, exclusions, raw corruption, schema crossing, legacy outputs/recipe validation
and rejection by the current learner. These tests do not qualify an installed
runtime, actual producer closure, real tablebase truth or any real data/model.
