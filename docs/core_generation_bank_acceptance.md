# Postclosure generation bank metadata validation

`validate_accepted_generation` in `scripts.core_generation_bank_acceptance` joins
trusted postclosure receipts to an existing recipe-independent data candidate.
It returns an immutable `AcceptedGeneration` view. It writes nothing, issues no
receipt, and does not change candidate schemas or `learning_admitted=false`.
The existing learner continues to reject these candidate manifests and wrappers.
A compatible consumer integration remains separate work.

```python
from scripts.core_generation_bank_acceptance import validate_accepted_generation

bank = validate_accepted_generation(
    trusted_root_ref, trusted_independent_ref,
    max_file_bytes=metadata_file_bound,
    max_total_bytes=metadata_graph_bound,
    max_games=fixed_attempt_bound,
)
```

Every reference is exactly `{path, sha256}`. Both top-level pins must come from
caller-authenticated root and independent authorities. Status text or finding a
file on disk does not authenticate its author. This API does not provide issuer
signatures or an unforgeable capability; consumers must call it with trusted pins.
It also does not establish an actor's utility or choose a learning recipe.

## Receipt interfaces

The root record has exactly these keys:

```text
status = ROOT_ACCEPTS_CLOSED_CORE_GENERATION_DATA_BANK_V1
learning_admitted = false
audit_protocol, data_spec, readout, candidate = references
audit_source = {revision, checkout, source_files}
audit_owner = {terminal, admission, command, source_revision, cwd, closure}
```

`audit_source` records the historical integrated producer/auditor revision and
checkout, plus the data spec's source-file pins. Producer protocol source pins
remain independently hash-bound by that protocol; overlapping pins must agree.
No historical source file is reopened against a later consumer checkout.

`audit_owner.command` is the complete executed child argv, ending exactly in
`-m scripts.core_generation_data_admit --protocol PATH --protocol-sha256 SHA`.
It includes the protocol's interpreter once. The owner revision and cwd describe
the supervisor checkout, which can differ from the audited child source.
`terminal` must identify a readonly CPU PASS/return0 terminal with the exact
command, supervisor digest, cap, owner source/cwd and nonempty owned-birth map.

The root's `closure` has exactly `supported_eof=true`, `exit_code=0`,
`supervisor_pid`, `supervisor_closed=true`, `owned_births_closed=true` and
`known_processes`. The PID and full birth map must match the terminal. This is
an external root assertion, independently accepted below: the terminal written
by a supervisor cannot itself establish that supervisor's later exit. The API
makes no process queries and does not independently observe closure.

`admission` references the separately issued CPU authorization. Its required
fields are `status=ROOT_ADMITS_CORE_GENERATION_DATA_AUDIT_V1`, `audit_protocol`,
`audit_source`, `command`, `owner` (the runtime's owner reference), `resource=cpu`
and `seconds`. These fields must match the audited protocol and owner. Other
admission evidence may be present; it is hash-bound but not interpreted here.
No new execution is authorized by reading this historical admission.

The independent record has exactly:

```text
status = ACCEPT_CLOSED_CORE_GENERATION_DATA_BANK_NO_FINDINGS_V1
root_acceptance = trusted_root_ref
audit_protocol, data_spec, readout, candidate = exact root references
findings = []
```

These interfaces describe future caller-issued metadata, not an assertion that
any real generation has already received these receipts.

## What is joined

The API checks the strict recipe-free protocol/spec, canonical UUIDs, actual
logical generation, source pins, architecture, actor hash and checkpoint step
against the producing protocol. It checks the saved serving-state metadata and
requires the candidate/readout under the audit output root, with no sibling
failure marker. All fixed physical attempts must occur once in ordinal order.
Every eligible game is retained in that order, with unique game IDs, train bank,
matching rows, scientific/label class, shard descriptor and game wrapper. Each
wrapper must retain the candidate-only status and identity, verified history/raw
flags, original evidence references, inventory digest and serving-state identity.
It reconciles class counts, all eligible rows, min/max game and row gates, decoded
byte ceiling, and the audit's source-scoped zero side-effect declarations.
Those declarations remain source-scoped rather than instrumented counters.

The returned identity, games and evidence are recursively readonly snapshots.
`evidence` includes the root/independent/candidate pins, distinct historical
source and owner bindings, eligible row and class totals, and
`learning_admitted=false`. It does not output a legacy accepted-generation alias,
select a future consumer revision, restore optimizer state, or change replay.

## Read and validation limits

Only the root/review, audit protocol/spec, producer protocol, CPU admission and
terminal, audit readout/candidate, actor-serving metadata and eligible wrappers
are opened. Shard descriptors and original game-evidence refs are compared but
not followed. Producer readout/inventory refs are bound without opening those
potentially large inputs. Actor checkpoints, native images, source files,
arrays, tablebases and other original audit payloads are never opened.
The closed auditor and trusted root/independent review remain responsible for
those actual payload checks, exclusions, native verification and producer closure.
Existing full494/175 architecture and collector lineage restrictions are unchanged.
This does not qualify arbitrary actors or a new on-policy loop.

Inputs must be canonical absolute, regular readonly JSON files. Size is checked
before bounded reads; duplicate keys and nonfinite JSON numbers are refused.
The caller supplies positive strict integer file/total/game bounds; per-file
bound is at most128MiB. Unique metadata bytes count against the total, followed
by one re-read of the same bounded graph to check final SHA/stat custody (atime
is excluded). Missing/malformed inputs or joins fail rather than returning a
partial bank. No caller callback, project numeric import or runtime launcher is
involved.

New tests use synthetic JSON graphs with deliberately nonexistent payload paths.
They cover positive generation2 membership, all eligible retention, distinct
owner/child revisions, immutable output and unchanged candidate bytes; rehashed
identity/closure/command/schema/gate/membership faults; reference, path, size and
final-custody failures. Existing public-auditor tests separately cover actual
legacy learner refusal of candidate schemas/wrappers. Source implementation is
not a real-bank audit or runtime adoption; author-side tests/lint remain unrun.
