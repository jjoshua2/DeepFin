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


## Versioned current-integrated A39 asset descriptor

The strict seven-key A39 descriptor above remains the legacy current-H route.
Current-integrated receipts require an explicit eighth field:
`schema="core_a39_current_integrated_asset_v1"`. Unknown versions and extra fields
are refused. Omitting this field does not adapt current-I receipts to legacy H.

This route requires the exact root, independent and child readout statuses
`ROOT_ACCEPTS_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_V1`,
`ACCEPT_CLOSED_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_READINESS_NO_FINDINGS_V1`
and `COMPLETE_CURRENT_INTEGRATED_COLLECTOR_CPU_ASSET_QUALIFICATION_CHILD_ROOT_CLOSURE_REQUIRED_V1`.
It joins the root/independent/readout protocol, terminal reference, source revision,
checkout and source-pin reference, plus the complete model-only observation.
The fixed original A39 checkpoint and state hashes, actual embedded step 680276,
complete 494-state/175-input counts and architecture must match the descriptor;
optimizer restoration must remain unqualified. This validates trusted metadata,
not another checkpoint load or a new model/optimizer qualification.

`asset_source` continues to identify the historically qualified asset loader and
its receipt graph. A later scoring/reader consumer has its own source revision.
The consumer neither relabels historical source as its current checkout nor
reopens historical source or model bytes in this metadata join. SCORE's actual
public loader retains its independent full checkpoint/state/architecture checks.
Current-integrated qualification does not imply PLAY adoption or playing strength.


## Exact thread-2 same-owner GPU prover version

The existing GPU payload schema/status and original `7b45088d...` prover route
remain supported. The consumer additionally accepts only the reviewed prover
fingerprint `721cf5d141843ef0db61adb5169e9424e4e7fcbca69758b8dadb8c5321a66b26`.
That private successor sets and asserts Torch intra-op and inter-op threads at
2 before project/native/NN imports and asserts both at final custody. It emits
the existing actual `Torch_intra` and `Torch_interop` fields, not a new CPU status.

For that exact fingerprint, both final fields must be strict integers equal to
2; initial backend `torch_threads` and `interop_threads` must also be strict
integers equal to 2. The existing equality with the saved boundary and actual
backend remains required. Original prover receipts do not acquire this new
requirement retrospectively. Unknown fingerprints, crossed source reviews and
missing, boolean, floating or wrong thread evidence are refused. All existing
same-owner GPU 900 / 870-second, root/independent delivery, helper/source/plan,
checkpoint/state/reference, membership/RNG/journal and coverage joins remain.
This metadata support does not execute or qualify the prover, alter any learner
math, or grant model delivery, scientific continuation or playing authority.


## Finite D1 development-first consumer

The additive protocols `CORE_GENERATION_HOLDOUT_SCORE_D1_V1` and
`CORE_GENERATION_HOLDOUT_READ_D1_V1` require the caller-trusted D1 design,
`selected_arm`, and a `d1` root/independent selection binding. The original
four-model protocols and defaults retain their original contract and metric
conventions. This source interface issues no admission or scientific acceptance.

D1 authenticates all four closed learner deliveries as metadata before accepting
selection: `K0`, `K001`, `K01`, and `K1` have coefficients 0, .01, .1, and 1,
respectively. Each uses the same accepted 17,568-row TRAIN bank, seed, LR 3e-6,
batch 32, mirror probability .5, 4,096 updates, and first period 4,096. Thus each
journal contains 131,072 sampled views with replacement; this is not proof that
every row was sampled. Complete accepted per-game membership and actual draw
coverage remain independently reconciled.

The delivery root retains its nine keys and the independent review its eight.
Only `payload_audit.schema` selects the new route:
`core_bank_learner_same_gpu_payload_proof_D1_v1`. The proof and learner must have
identical complete owner objects. The new route requires the exact reviewed D1
prover fingerprint, D1 admission/source/pending statuses, GPU 3,600 / inner
3,570 seconds, a 64 MiB journal cap, and actual initial/final intra/inter-op
thread counts of two. The older CPU audit and two GPU 900-second fingerprints
remain separate. Historical learner/source pins describe the actual producer;
they are not rewritten to the current consumer revision.

Selection requires all seven complete 80-pair DEV edges: four arms against A39
and three positive coefficients against K0. It joins their actual closed GPU
900-second and CPU 300-second owners, admitted protocols, independently reviewed
source, delivered checkpoint identities, pre/post served-state fingerprints,
160 complete reciprocal-colour games, exact pair/history order digests, WDL and
five-bin paired counts, and recomputed point scores. The exact fixed eligibility
rule is own score versus A39 at least .55 and, for a positive coefficient,
score versus K0 at least .55. The winner has the largest eligible A39 score;
exact ties choose the lower coefficient. No eligible arm refuses HOLD.

The role-use binding consumes the actual D1 prefix producer, not a second role
phase. Its protocol is `D1_PREFIX_METADATA_CPU_PROTOCOL_V1`; its CPU 300-second
admission and actual `--protocol` / `--protocol-sha256` flags are joined directly.
The root has nine keys and its independent review adds root acceptance and
findings. All six original 400-pair books remain frozen. Two actual first-80
books and typed complete-parent projections must preserve original ordinals,
history, order, and complete parent members. The eight genuine public book-read
observations and actual six-domain count dictionary must reconcile. Group,
history, terminal EPD and history-SHA counts are each 2,656; raw/mirror inputs
are 5,312. The source-parent count is a strict integer at least 2,656, because a
complete group can retain multiple distinct source parents; no exact parent
count is invented. Boolean, floating, missing, extra or inconsistent counts
refuse. Books and parent
payloads are never opened by this metadata consumer. The prefix parent digest
is SHA256 of canonical `games`; its history digest is SHA256 of the canonical
ordered list of 80 histories, without a newline. Canonical JSON uses compact
separators, sorted keys, UTF-8, and the default ASCII escaping.

The DEV profile pins the selected 109-field source descriptor and the explicit
root selection authority. `d1_playing_projection` derives the active Gumbel,
runner, input, WDL/TB, history, move-limit and serving settings; both actually
observed sides must match it. In particular a flat .75 noise scale is explicit:
the arena's ordinary training-shape default of 1.0 is insufficient. The full
109 fields remain source provenance, not a claim that an arena observed every
collector recording, teacher, opening scheduler, or label-only field. Such
collector-only fields do not become new DEV knobs. The future DEV implementation
must check actual constructed search objects and runner arguments, including
both sides, before this interface can receive honest closed evidence.

Only the selected arm, K0 and A39 are loaded or scored; a K0 winner uses only K0
and A39. Metadata checks on unselected deliveries never load those checkpoints.
All qualified HOLD rows are scored once with the existing F32 metrics and
source-qualified identities. The CPU reader remains Torch/model/native-free.
The equal-parent selected-minus-A39 CE contrast uses the same local PCG64 seed,
10,000 paired resamples, linear quantiles and severe-damage veto as the original
ruler. No anchor penalty enters the evaluation metric. An unread report or veto
cannot authorize final matches. This consumer does not implement DEV matches,
issue selection receipts, adopt a runtime, or authorize PLAY; final playing
acceptance remains a separately reviewed, closed evidence path.

The added synthetic tests use explicitly fictional authorities and trust-anchor
hashes, real versioned consumer predicates and real tiny model loading/scoring.
Only the previously qualified upstream bank authority is replaced in selection
fixtures; the D1 saved-proof, per-game coverage, owner, profile, prefix and winner
predicates run directly. All new cases are source-declared and unrun until a
root-owned qualification. No actual learner, DEV, HOLD, or strength result is
implied by this source change.


The D1 successor also exercises the public `bindings` dispatch with a sealed,
acyclic fictional D1 design, real parent-map metadata and real D1 delivery,
coverage, prefix/profile, all-seven-edge and winner predicates. The established
upstream HOLD-token and donor-asset APIs are the only additional inert seams.
Both two-model and three-model selected sets are accepted; any extra unselected
model refuses before payload work. Future root-owned qualification additionally
requires two genuine accepted-prefix metadata API controls, without reading
books or role payloads, in the same fixed 400-second engineering phase.


D1 profile classification correction: `categorical_blend_frac`,
`categorical_search_blend_frac`, and `syzygy_rescore_policy` are training-label
construction controls in selfplay finalization. They are absent from the arena
and match APIs. The exact selected descriptor values remain authenticated in
`profile.source_only_not_applicable`, with `NOT_APPLICABLE_LABEL_ONLY`; they
are excluded from both sides' required observed active playing projection.
A profile copy is never evidence of an observed runtime parameter.

The remaining grouped projection describes real semantic checks, not a claim
that `GameConfig` is an arena argument. `history_rep_fix` requires the actual
model attribute and delegated `rep_fix.apply(True, boards_discarded=True)` before
search. `tree_reuse=False` requires actual compiled-runner bound arguments
`tree=None`, `root_node_ids=None`, and `cboards=None` on the cold per-ply path.
`syzygy_adjudicate_fraction=1` means no fractional/random filter: the same strict
fullboard handle is used and the reader must reject a move beyond the earliest
natural or rule50-aware result across all 160 games. `net_raw` is the actual
unblended WDL-to-Q source path, not an invented argument. Gumbel fields, vloss
and target batch remain observations of actual constructed objects and actual
positive calls on each side. These are required contracts for the future DEV
helper/reader; this source change has not executed or qualified them.

### D1-T1: one selected K0 saved-endpoint recovery

The additive `core_bank_learner_K0_failed_gpu_closed_cpu_recovery_v1` family is
restricted to D1 K0 and the exact failed GPU terminal, public boundary and saved
checkpoint in source selection `116d422c…`. It uses a distinct Root10/ind12
receipt and a separate closed CPU 600 / 570-second audit. The original GPU
terminal remains FAILED. It is never an old Root9/ind8 same-GPU PASS receipt.

The Root10 adds `recovery_authority` to the existing nine fields. The independent
receipt copies all ten facts, substitutes its distinct status, and adds
`root_acceptance` and `findings`. The authority is a caller-trusted strictref at
the canonical Root-owned T1 path. It binds the exact source selection, its review,
a Root-derived publication projection of the actual mechanical amendment, the
accepted auditor, and the selected trigger. The actual CPU admission must name
the same authority. Source selection, publication or status alone grants no
execution or model-delivery authority.

The saved-only proof verifies complete saved metadata/state, active optimizer
slots, scheduler, ZClip, provenance, recorded five-stream boundary, full replay
membership and all original journal draws. It carries no reconstructed own-live
fingerprint or historical final GPU/source/native/runtime/RNG bracket. The
architecture-matched historical synthetic full494 shape/alias map is explicitly
`MODEL_LAYOUT_BASIS_ONLY`; it provides no K0 live weights or training credit.
Current CPU runtime/source/native custody is separate. The scheduler endpoint
is index 4,095 because the real Trainer updates its scheduler before incrementing
the completed-update counter to 4,096.

The ordinary CPU and same-GPU families retain their original gates. The corrected
production D1 prover fingerprint `139ec1c4…` additionally requires its exact
accepted source review; the original `cceb3d1d…` route remains unchanged. The
engineering-only two-update prover is never a scientific delivery. Recovery for
other arms, other failed owners, arbitrary saved survivors, or an unissued Root
T1 authority is refused. The later 400 / 370-second consumer qualification still
requires four real model metadata controls (recovered K0 plus three complete GPU
arms), alongside bank, asset and prefix controls, before DEV/HOLD use. It is not
a prerequisite that would cyclically block the three remaining training arms.
