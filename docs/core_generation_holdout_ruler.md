# Deterministic evaluation-only HOLD ruler

The two public entries consume a separately selected prospective study. They do
not collect games, select a model, accept a TRAIN bank, train, issue delivery or
playing authority, or change the existing evaluation ruler. Numerical software
qualification, preregistration, actual data/model delivery and phase admissions
are separate requirements. Synthetic test receipts are explicitly fictional.

`python -B -m scripts.core_generation_holdout_score --protocol PATH
--protocol-sha256 SHA --admission PATH --admission-sha256 SHA` is the GPU entry.
The corresponding CPU entry is `scripts.core_generation_holdout_read` with the
same four flags. Their protocols are `CORE_GENERATION_HOLDOUT_SCORE_V1` and
`CORE_GENERATION_HOLDOUT_READ_V1`. Each has exactly:

- `schema`, `study_uuid`, `design`, `holdout`, `parents`, `models`, `source`,
  `runtime`, `owner`, `stage`, `outer_seconds`, `inner_seconds`, `output`,
  `limits`, `source_review`, `native_basis`;
- the reader additionally has `scoring`.

Every reference is `{path,sha256}`, canonical, bounded, stable and sealed JSON.
Historical owner terminals alone may retain their genuine writable mode; their
hash and non-atime identity are bracketed. `holdout` and `parents` each contain
`root` and `independent`. `source` is `{checkout,revision,files}`: current clean
Git source and complete required Python/native input pins, separate from every
historical producer, auditor, learner and owner-wrapper revision. `runtime` has
`python,prefix,python_version,managed_site,torch_version,numpy_version`. The
single managed `.pth`, interpreter, dependency versions/origins, two-CPU
`20,22` affinity, thread environment, CUDA visibility and direct existing owner
command are checked. GPU outer cap is1200 seconds, reader cap900; root freezes a
positive smaller `inner_seconds` before admission. An alarm covers the actual
body without resetting its deadline. Root also performs actual resource and
complete-main-preregistration admission; the child does not issue either.

## SCORE/READ phase admission V2

Both entries require `status="ROOT_ADMITS_CORE_HOLDOUT_RULER_PHASE_V2"`.
The admission has `normalized_command`, the entire actual child argv with only
its unique `--admission-sha256` value replaced by `SELF_ADMISSION_SHA256`.
It must not contain the legacy `command` field. Seal this admission first,
then substitute its actual SHA into the launched child command. The actual
child must end with the exact public module and four path/SHA flags above;
all four flags must occur exactly once. Every other token, including the
interpreter/bootstrap prefix, is compared unchanged. The direct owner must
have one child separator and the selected stage, resource, cap and owner hash.

V1 required a sealed admission to contain its own actual hash in `command`,
which could not be constructed by ordinary hash-and-seal authoring. There is no
V1 fallback. Protocol schemas remain V1; source and phase admissions must be
bound explicitly to this successor before execution. READ also verifies the
closed SCORE owner's V2 normalized command, exact terminal child argv and
full existing closure evidence. Other historical owner routes (legacy CPU
saved-state audits and the explicit same-GPU learner proof) are unchanged.
This source correction grants no execution, model-delivery or study authority.

`limits` has positive integer `metadata_file_bytes` (at most128MiB),
`metadata_total_bytes`, `game_decoded_bytes`, `total_decoded_bytes`, `max_rows`,
`row_log_bytes` per model, and `reference_cache_bytes`. These are root-bound
operational limits before execution. No capacity failure changes batch size,
precision, rows, or cap. The dense internal legal-policy cache is bounded by
`max_rows*1858*4` and never written. The GPU phase checks six current readonly
native images, exact bounded records, all31 current input hashes, freshness,
interpreter extension suffix and actual import origins before/final. Both brackets
invoke actual `ext_check.extension_problems()` (required symbols and numeric
`_mcts_tree` ABI), before any model/data work in the initial bracket. This is a
named-file bracket, not a claim about continuously mapped bytes. The CPU reader
never opens those files, checkpoints, or shards and never imports Torch/native.

## Prospective authorities, not existing receipts

The parent-roster root status is
`ROOT_ACCEPTS_PROSPECTIVE_WHOLE_GAME_PARENT_ROSTER_V1`; exact fields are
`status,study_uuid,design,producer_protocol,producer_roster,parent_map`.
Independent status is
`ACCEPT_PROSPECTIVE_WHOLE_GAME_PARENT_ROSTER_NO_FINDINGS_V1`, with the same
identity fields plus `root_acceptance,findings:[]`. The parent-map object has
`study_uuid,collection_uuid,games`; each of all64 planned attempts has exactly
`physical_game_id,parent_id,parent_member_ids`. Complete nonempty parent-member
sets are pairwise disjoint. The actual collector's **list** roster is joined in
its original order, including attempts that later become censored/empty. The
root selector/binding phase must authenticate complete-parent provenance and
issue this authority **before outcomes**; a scoring result cannot establish
preassignment or exclusions. Each scored game must be in that fixed map.

`models` has exactly `A39,A,B,C`. A/B/C bindings each have `root,independent`.
The root status is `ROOT_ACCEPTS_CLOSED_CORE_BANK_LEARNER_DELIVERY_V1`, with
exact fields `status,contract,stage,boundary,checkpoint,source_revision,
source_checkout,learner_owner,payload_audit`. Independent status is
`ACCEPT_CLOSED_CORE_BANK_LEARNER_DELIVERY_NO_FINDINGS_V1`, with exact fields
`status,root_acceptance,contract,stage,boundary,checkpoint,payload_audit,findings`.
`learner_owner` has `terminal,admission,command,source_revision,cwd,closure`;
its wrapper revision is distinct from the child contract revision. Full closure
joins the exact public anchored-family CLI, return0, PASS, resource/cap,
command, all birth identities and root's supported-EOF closure assertions.

`payload_audit` is `{readout,owner}`. The proposed closed CPU audit readout has
status `CORE_BANK_LEARNER_SAVED_STATE_AUDIT_PENDING_ROOT_CLOSURE_V1`, references
`contract,stage,boundary,checkpoint`, `source_revision,architecture,step`, actual
`model_state_sha256`, an exact `command_suffix`, and `reconciled` containing
exactly nine true entries: `model,optimizer,scheduler,zclip,counters,rng,
journal,membership,policy_anchor`. This is an explicit **future integration
interface**. The learner boundary does not emit a model-state digest or perform
this CPU audit. An actual saved-state audit and its closed owner/root+independent
delivery must exist before real scoring; this module does not invent them or
count a delivery status alone as a saved-payload check.

Alternatively, `payload_audit` can have exactly `schema,readout,owner`, with
`schema="core_bank_learner_same_gpu_payload_proof_v1"`. This explicit variant
accepts the separately reviewed finite passive prover, identified by the pinned
`GPU_PROOF_HELPER_SHA256`; it does not reinterpret the CPU status. Its readout is
`CORE_BANK_LEARNER_GPU_OWN_LIVE_SAVED_PROOF_PENDING_ROOT_CLOSURE_V1` and its complete
six-field owner must equal `learner_owner`. The same closed GPU owner has an exact
900-second cap and the real anchored CLI suffix. No extra CPU phase is implied.
The outer delivery root and independent review still bind the entire proof.

Only this variant accepts the prover's normalized admission. It requires the
exact `ROOT_ADMITS_SCIENTIFIC_LEARNER_PASSIVE_PROOF_GPU_V1` status and execution
flag, actual admission path/SHA and observer-plan path/SHA in unique command
flags, and replacement of only that admission SHA with `SELF_ADMISSION_SHA256`.
This avoids a self-hash cycle without fabricating an `admission.command` field.
The default legacy/CPU admission checks, all owner source/CWD/terminal/cap/wall/
birth/closure checks, and all existing delivery gates remain mandatory.

The proof must join its original plan, accepted source review, historical source
map, contract, stage, boundary and checkpoint. Full494/175 counts, own-live/final
save equality, one final CPU-only checkpoint decode, named optimizer evidence,
reference immutability, exact five-stream RNG and backend, membership and journal
hashes, and selected N/K/views must reconcile. RNG is held in the boundary, not
in `learner.pt`. After `training_population` validates the actual accepted TRAIN,
its full physical-game roster is joined to every per-game coverage count and
fraction, including zero-draw games. With-replacement sampling does not imply
full coverage; no coverage or efficacy threshold is added.

This is metadata validation only: the consumer never opens checkpoints, raw
journals, shards or historical source files. The learner, wrapper and current
consumer revisions remain separate. Score and CPU reader share the adopted
validator; general arena loading does not automatically gain this gate. Real
use requires reviewed publication, explicit consumer adoption and genuine
postclosure root/independent deliveries. Fictional graph controls establish no
numerical saved-state, study, playing or strength qualification.

All three delivered models are the actual anchored family, fresh segment1,
with selected constant-LR contracts and the exact selected dose derived from
complete TRAIN replay membership. A/B have zero coefficient/null reference;
C has0.1/original A39. Source/data/horizon/stream, replay membership, draw SHA,
final RNG and backend are matched. HOLD collection identity is refused in the
training bank identities; no HOLD view is accepted as a learner bank. Complete
preplanned TRAIN/HOLD parent exclusions remain the selector's responsibility.

A39 binding fields are `checkpoint,architecture,step,model_state_sha256,root,
independent,asset_source`. The latter has `revision,checkout,source_pins` for the
selected current-H qualification. They join the genuine current-H closed asset-readiness root, independent
receipt and model-only readout, original A39 hash and embedded680276 step.
Actual public loader calls decode once on CPU with explicit complete embedded
architecture and strict weights. Family/reference tags are checked before the
real constructor; no ancestral `params.json` fallback, reference reopen,
optimizer construction or restoration occurs. Temporary constructor RNG is
restored. Complete494/175/full-state counts are enforced before GPU transfer.

## Observations and estimands

The scorer reuses actual inventory and shard decoding and rechecks active
policy/network/selfplay/legal flags, integer game/ply/genuine outcomes,
encoding/history, nonempty legal support and zero illegal target mass. All
qualified resolved rows are visited in physical-attempt ordinal/serialized-row
order once. There is no replay sampler, mirror, feature dropout, relabel,
filtering or imputed draw for `'*'`. Fixed32 batches have a deterministic final
remainder **per game**; this is not a batch-size fallback. Actual relation fields
are carried through the public collator if enabled. Scoring is eval/no-grad,
eager CUDA BF16 autocast, with FP32 attention-policy logits and observed BF16
WDL/policy-q linear output. This does not claim every operator is BF16.

WDL win0/draw1/loss2 uses FP32 log-softmax CE and FP32 softmax probabilities.
Unhalved three-class Brier and `(pWin+.5*pDraw-outcomeScore)^2` are descriptive.
Main policy CE calls the existing CORE finite-sentinel legal-mask and F32
F16-target renormalization helpers. Forward KL is the distinct exact legal
conditional `A39||candidate` convention, not an anchor penalty or teacher blend.
All model state digests/checkpoint custody and forward RNG are checked unchanged.
Setup uses the selected common1673 seed (full integer2026111673); initialization
is disclosed, not described as preserved pre-setup global RNG.

Readonly bounded `A39.rows.jsonl,A.rows.jsonl,B.rows.jsonl,C.rows.jsonl` retain
source-qualified game/parent/row/ply/outcome, shard-inventory and producer-proof
identities, three WDL log-probabilities, and five metric scalars. No policy vector
or legal cache is emitted. A compact <=4MiB `readout.json` includes exact per-game
sums/counts, row identity hashes, full coverage/class/censored/empty counts,
model/RNG/state/precision evidence and file SHA/size/count descriptors. The child
status is `CORE_HOLDOUT_OBSERVATIONS_PENDING_ROOT_CLOSURE_V1`.

The reader's `scoring:{root,independent}` uses prospective root status
`ROOT_ACCEPTS_CLOSED_CORE_HOLDOUT_SCORING_OBSERVATIONS_V1`, exact
`status,protocol,readout,owner`; independent status
`ACCEPT_CLOSED_CORE_HOLDOUT_SCORING_OBSERVATIONS_NO_FINDINGS_V1`, exact
`status,root_acceptance,protocol,readout,findings`. It checks actual closed GPU
owner before streaming any observation file, then joins all four logs row by
row, checks source/identity/outcome/finite-logprob/CE consistency, counts and
checksums, and independently reconstructs every game and aggregate numerator.
It does not reconstruct dense policy CE/KL from omitted1858-vectors: those are
source-qualified GPU observations. No reader is allowed to substitute a
successful partial model/game panel.

Per-game mean WDL CE is averaged equally within parent, then equally across
parents. The selected one-game-per-parent design makes the parent/game counts
equal. Descriptive row-weighted metrics remain separately labeled. Min48 parents
and2000 rows are required. The CPU reader uses a local NumPy2.2.6 PCG64,
10000 paired complete-parent resamples, seed2026111680, linear2.5/97.5 percentiles
for C-minus-A39 equal-parent CE. No global NumPy RNG changes. Degenerate constant
parent differences produce a labeled constant interval. Empty, nonfinite or
incomplete evidence is UNREAD, never zero-imputed.

The sole health veto is point>0.20 nats **and** lower95>0. Nontriggering does not
prove safety/noninferiority, and neither result chooses a best arm or authorizes
playing. Reader status `CORE_HOLDOUT_HEALTH_PENDING_ROOT_CLOSURE_V1` still needs
its own actual owner closure and root+independent disposition.
