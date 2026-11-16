# Prospective whole-game HOLDOUT family

This is a software interface, not a study allocation, metric rule, data acceptance,
actor choice or execution admission. A held-out cohort must be assigned before
collection, with its own immutable collection UUID, ordered whole-game roster,
opening histories, exclusions and finite root-owned protocols. This route does
not relabel an old TRAIN shard or revive a stopped collection.

The existing TRAIN collector, recipe-bound auditor, recipe-free TRAIN auditor,
accepted-generation API and bank learners remain strict. They refuse HOLDOUT
schemas and tokens. There is no `--bank`, `--data-only` override on a legacy entry,
promotion boolean or user-supplied validation callback.

## Collection

The dedicated entry is:

```text
python -B -m scripts.core_finite_holdout_collect --execute \
  --protocol <sealed-json> --protocol-sha256 <sha> \
  --admission <root-admission-json> --admission-sha256 <sha>
```

Its producer schema is `CORE_FINITE_HOLDOUT_GENERATION_V1`; its separate producer
admission status is `ROOT_ADMITS_CORE_HOLDOUT_GENERATION_V1`. The descriptor-only
`--check` route retains the existing no-model/no-engine behavior and emits
`CHECKED_CORE_HOLDOUT_GENERATION_DESCRIPTOR_V1`. Both modes validate the real
actor, profile, source, native, asset, full-history roster and exclusions. The new
entry file must be source-pinned alongside the original collector closure.

Every planned and actual nonempty game has literal `bank: "holdout"` in its
provenance, outcome-proof common metadata, raw-target metadata, validation and
completed receipt. The shared executor retains real raw/stored-target checking,
full-history and integer game/ply identities, strict tablebase resolution,
original policy retention, actual served-state and process-closure observations.
No new search, label, retention, engine or training math is introduced.

The readout is `COLLECTED_CORE_HOLDOUT_GENERATION_V1`; failures, including final
seal failures, remain `FAILED_CORE_HOLDOUT_GENERATION_V1`. Every fixed attempt is
accounted for; no replacement or outcome selection is added. The child still
cannot attest its future owner closure or issue data acceptance.

G1 still requires the exact immutable original donor. Later actors still require
the same explicit adjacent legacy or bank-learner parent checks. A producer's
logical generation/source, historical learner generation/segment/source and
optimizer step are distinct. The HOLDOUT path does not permit an arbitrary G2
bootstrap, fake an old parent, or change a reference checkpoint during serving.

## Recipe-free CPU audit

The dedicated entry is:

```text
python -B -m scripts.core_generation_holdout_admit \
  --protocol <sealed-json> --protocol-sha256 <sha>
```

The protocol schema is `CORE_FINITE_HOLDOUT_DATA_ADMISSION_V1`. It has the same
exact recipe-free fields as the TRAIN data auditor: `data_spec`, producer protocol,
readout, inventory and exact closed GPU-owner command/terminal references, gates,
finite CPU runtime, output and byte limits. The data specification schema is
`core_generation_holdout_data_spec_v1`, with the same exact data-identity fields:
run UUID, collection UUID, generation, source revision/files, full architecture,
actor SHA/embedded step and minimum/maximum game counts. Learning recipes remain
unknown fields and are refused. Required source pins include both new public
entries in addition to the original shared audit/data closure.

The real shared CPU audit verifies the new producer schema, exact new CLI and
producer-admission status, serving lineage, complete attempts, native origins,
raw vectors, actual full-history input reconstruction, integer IDs, legal support,
strict tablebase/outcome proofs, all inventories, exclusions and postguards. No
model, optimizer, checkpoint decode or forward is added. This is the same audit
body; fixed private family branches select only the public schema/status/bank.

Outputs retain immutable candidate bytes and `learning_admitted: false`:

| Output | Identity |
| --- | --- |
| game wrapper | `VERIFIED_RESOLVED_CORE_HOLDOUT_GAME_CANDIDATE_V1` |
| generation manifest | `CORE_HOLDOUT_GENERATION_DATA_CANDIDATE_V1` |
| readout | `AUDITED_CORE_HOLDOUT_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1` |
| failed audit marker | `FAILED_CORE_HOLDOUT_DATA_ADMISSION_V1` |

Successful HOLDOUT wrappers/manifest/readout additionally require
`evaluation_only: true`. Original candidate-only/postclosure-required fields are
kept; they do not become permission to train. Resolved natural and strict-TB games
remain distinct; unresolved/empty attempts remain in full accounting and never
become eligible shards.

## Read-only postclosure evaluation view

`scripts.core_generation_holdout_acceptance.validate_accepted_holdout` takes
caller-trusted root and independent receipt hash pins plus explicit per-file,
total-metadata and game limits. It reads bounded stable JSON only, not arrays,
models, original bulk evidence or current consumer source. The receipts have the
existing exact graph shape plus `evaluation_only: true`, with distinct statuses:

- root: `ROOT_ACCEPTS_CLOSED_CORE_GENERATION_HOLDOUT_V1`
- independent: `ACCEPT_CLOSED_CORE_GENERATION_HOLDOUT_NO_FINDINGS_V1`
- CPU audit admission: `ROOT_ADMITS_CORE_GENERATION_HOLDOUT_AUDIT_V1`

The root binds audit protocol/spec/readout/candidate/source, closed CPU owner,
normalized actual command, terminal and closure assertions; independent acceptance
binds that exact root and graph with empty findings. Owner-wrapper revision is not
mistaken for producer/auditor revision. As with TRAIN, receipt strings do not
prove issuer identity: trusted pins must come from the caller's authority.

The returned `AcceptedHoldout` is a distinct frozen type with deeply read-only
identity/games/evidence. Eligible members must have literal HOLDOUT bank, exact
integer game IDs, complete physical membership, wrappers, inventories and gates.
It is never an `AcceptedGeneration`; neither the existing TRAIN validator nor
bank learner accepts these references, even when supplied through `stage.banks`.
The function does not rewrite a candidate, issue an accepted learner-manifest
alias or set learning authority.

A deterministic scoring consumer and its metrics remain a separate source step
after scientific preregistration. This increment does not use sampled training
replay as a holdout ruler, create a cohort, score a model or authorize collection.
Synthetic regression fixtures exercise the actual raw/Zarr/FakeTB audit and
metadata graph with fictional external owners/assets. Such fixture receipts are
software controls, never real bank, study or closure evidence.
