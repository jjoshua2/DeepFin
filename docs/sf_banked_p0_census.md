# Banked Stockfish P0 evidence census

`scripts.sf_banked_p0_census` audits a fixed authenticated TRAIN generation. It
writes evidence and missingness reports, never labels, training admissions or
model promotions. It makes no engine query and constructs no model, optimizer,
Torch/ORT session or tablebase handle. Actual execution needs a separately bounded
CPU owner; publishing this tool does not authorize a corpus census.

```sh
python -m scripts.sf_banked_p0_census \
  --protocol /shared/audit/protocol.json --protocol-sha256 EXACT_SHA256
```

The caller supplies a readonly, hash-pinned JSON object with schema
`SF_BANKED_P0_CENSUS_V1` and these fields:

- `audit_uuid`: canonical UUID; `output_root`: an absent absolute external directory.
- `generation_manifest`, `producer_protocol`: exact `{path, sha256}` references.
  The manifest uses `ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1`, including original
  authenticated per-game proof wrappers, inventories, provenance and validation.
- `selection`: the root's hash-pinned fixed TRAIN data-qualification decision,
  joined to this audit UUID, generation manifest, producing source and all row/game
  counts. Selection is not an execution admission.
- `input_authority`: a separate readonly root record with schema
  `SF_BANKED_P0_INPUT_AUTHORITY_V1`, status `ROOT_AUTHENTICATED_INPUT_AUTHORITY`,
  `audit_uuid`, exact `selection`, `generation_manifest`, and `population` equal to
  `expected`. Its `source_audit` contains `protocol` and `readout` refs plus
  `owner: {terminal, owner_sha256, command}`. The original CPU audit must be closed
  PASS with its exact `scripts.core_finite_admit` command, no live owned births or
  failure marker. Protocol/readout/generation/producer/owner/count identities join
  before any numeric payload load or output allocation. A candidate's
  `ROOT_ACCEPTED` schema or pending-root child readout is not root acceptance:
  the audit writes those before its final seal and owner closure. Root may
  authenticate an already accepted legacy generation through this narrow record;
  the census neither invents that authority nor repeats TB verification.
- `expected`: exact `run_uuid`, producing `source_revision`, `generation`,
  `physical_games` and retained `rows`. Every game must be TRAIN and nonempty;
  identity corruption, duplicates and denominator mismatches refuse. No sampling
  or replacement is supported. This is not a new interpretation of an old trial.
- `producer_owner`: exact closed collector `{terminal, owner_sha256, command}`.
  The terminal and producing core admission are hash authenticated. Closed flags
  alone do not replace the fixed protocol and original proof references.
- `auditor_source`: `revision` and `files` (relative path to SHA256 map) for the
  clean executing checkout. Include this script, policy mapping and WDL helpers
  and their consumed source dependencies. Caller pins producing-source evidence
  separately; executing auditor revision and historical producer revision differ.
- `limits`: positive `metadata_bytes`, `game_decoded_bytes`, `inventory_files`,
  `report_bytes` bounds. Metadata references and padded Zarr allocations are
  bounded before decoding; rows stream one game at a time. Only ordinary numeric
  Zarr directories with safe byte codecs and no filters are accepted, not packed
  archives or object codecs. The report byte cap applies to `rows.jsonl`.
- `runtime`: `cpu_affinity`, `inner_seconds`, `owner_seconds` (at most900), and
  `owner: {path, sha256}`. The direct parent must be that finite CPU owner.
  CUDA is hidden, OMP/MKL threads are2, Python optimization is off. Final source
  and runtime guards run before publishing the complete report.

The caller must fix the population and identities before opening arrays. Original
producing outcome proofs are authenticated and joined to prior accepted metadata;
this tool neither reissues those proofs nor repeats their TB/target validation.

For each game the auditor replays the full opening and played action history,
checks raw record and retained row ply/POV/STM/clock context, legal compact masks,
serialized game/NET/selfplay identities, and stored F16 fingerprints against the
accepted original encoding proof. It does not freshly encode the position or
claim widened F16 equals original F32. All shard files are authenticated before
and after use; inputs are never written. Symlinks, unsafe codecs, changing or
writable payloads fail technical qualification.

Two evidence routes remain separate. Existing `sf_p0_policy_target` and
`sf_p0_regret` were shifted by actual ply before producer thinning. A retained
predecessor's P1 label can describe the next retained P0 only at actual ply+1 in
the same full history. Adjacent retained rows across a gap do not qualify. Regret
is not a distribution; neither explicit P0 field supplies native P0 WDL.
The regret transform itself is not rebuilt by this census; its stored range,
source identity and shift context are reported without claiming numerical parity.
An absent optional payload/flag pair is unavailable. A surviving payload without
its presence flag is malformed/ambiguous, even if other SF fields are valid. It
keeps its row but invalidates that reduced-label route. Ordinary flag-zero rows
remain absent separately. Both P1 and explicit-P0 reports preserve these field facts.

The census preserves exact storage semantics: F16 vectors, U8 masks/flags,48×5
int16 sparse candidates and int32 record metadata. Native permille WDL stays
separate from record-POV flipped or CP-logistic F16 WDL. It checks only the
producer-declared conversion, with source representation bounds; it selects no
new CP transform. CP/mate boundary values are *possibly* clipped. Summary
nodes/depth are observations, not per-PV effort. Partial support, padding,
missing native WDL and possible fallback are explicit; well-formed probabilities
and availability flags do not prove a complete or wire-authenticated teacher.
Query IDs/history, raw bounds/full PV, escalation and requested/actual candidate
budgets unavailable from CORE remain UNKNOWN. Full wire is not a prerequisite
for the narrower source-reconstructed reduced-score tier.
The public Python move helper can import its optional native module. This tool
invokes no native mapping/search/encoding kernel; that does not qualify an existing
ignored image. Root must still bind/qualify the actual CPU import environment.

`rows.jsonl` preserves every retained-row denominator and references predecessor
rows. `readout.json` records every game, all structural raw records and overlapping
reason/distribution counts. Invalid/unavailable labels remain reported; technical
failures after output allocation preserve the partial attempt and an external
`.failed.json` marker. Earlier input-gate refusals leave no output and remain in
the owning stage's logs. A complete report has no minimum coverage threshold, confidence interval,
teacher/training acceptance or playing-strength interpretation. Independent review
and the closed owner remain separate from the child report.
