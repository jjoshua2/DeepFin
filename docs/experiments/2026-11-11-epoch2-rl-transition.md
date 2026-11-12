# Epoch-two transition to reinforcement learning

## Objective and limits

Continue the verified supervised epoch-two network through the real distributed
selfplay → replay → update → checkpoint path, then test playing strength against
the preserved donor. The user requested ongoing work for up to 48 hours. Compute
stops by **2026-11-13 22:06:18 UTC**. An unsuccessful or inconclusive experiment is
reported as such; the deadline does not establish success.

The starting development revision is `c730c3dab`. The deployment checkout remains
at `a38988814` with its unrelated dirty files preserved. Its committed history is
already contained in main. Development uses a separate checkout.

## Baseline and evidence

The donor is the permanent epoch-two checkpoint under
`~/chess-artifacts/operations/all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery654213-a4349851abddea4a/training/epoch-2/checkpoint.pt`.
Its SHA-256 is
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
The payload was rehashed and read with `weights_only=True`: step 680276, 494 state
tensors, 16 blocks, width 512, 16 heads, and 61,444,448 unique parameters. The
checkpoint supplies the complete architecture and encoding contract, including
175 planes, corrected legacy-root history, coupled categorical head, linear
policy embedding and 128-dimensional auxiliary policy projections.

The older source-only RL assembly is evidence, not an installed runtime. Its 25
source hashes were checked against its composition manifest. The corrected 15
toy CPU tensor cases now pass. A separate direct real-donor CPU helper check
preserved every model tensor and deterministic output, fresh Aurora optimizer and
scheduler objects/state, and SWA. These do not prove worker execution or CUDA
qualification. Raw receipts are under
`~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/`.

## Qualification before candidate training

1. Integrate the policy-preserving strict donor/startup path with current main.
   Keep its newer configuration and retention safeguards. Refuse wrong hashes,
   model contracts, partial loads, policy resets, salvage conflicts and unqualified
   exact RL restore. Discard a trainer after a load that failed during mutation.
2. Exercise YAML → CLI conversion → TrialConfig → ModelConfig → Trainer → actual
   first publication. Verify complete model/SWA preservation and fresh optimizer
   and scheduler state. Include incompatible and malformed inputs.
3. Independently review the resulting source revision. Run the Stockfish-backed
   end-to-end smoke with an explicitly pinned engine; skipped tests do not qualify
   that path. Establish package and native-module origins.
4. Run bounded local Ray and CUDA runtime smokes before adopting the separate
   Python 3.13 / locked Torch 2.14 environment. These use synthetic models, not
   donor training, and each has at most 600 seconds plus owned cleanup.
5. Qualify actual donor publication and worker/broker outputs with the new runtime.
   Advertising a model hash is not independent proof of its served weights.

## First pilot: deciding questions and resource rule

The first scientific experiment will ask whether search-improved targets from
newly generated positions can improve the donor. The proposed SF value component
is a new intervention for this donor, not preservation of its supervised recipe;
see the lineage correction below. This is not outcome-only AlphaZero. Settings, sample/update
budget and exact launch commands will be appended **before** its launch, after
measuring the qualified generator's throughput. No production template is adopted
wholesale: it still names an older donor and enables policy-head reset.

Use a new server, replay and output directory, one local worker, a fixed donor
for the first collection, fixed search/SF difficulty, and no shared/sibling replay,
external FEN feeds, PID changes or simultaneous arenas. Report actual termination
and SF adjudication counts; a max-ply draw is not necessarily a factual game draw.
Existing mixed-model inflight resumes must not enter the initial collection.

Generation and learning are sequential. The existing advisory pause flag is not
an acknowledged drain. The local opt-in phase boundary must stop and verify owned
workers and the owned broker before learner work; remote/shared brokers cannot
claim this exclusion. Long-lived production work and controls are untouched.

Every scarce-resource stage holds the canonical absolute GPU lease
`~/projects/chess/scratchpad/gpu0_experiment.lock` through owned-process
cleanup, observes relevant STOP/PAUSE controls, has a finite wall bound and unique
outputs, and checks available RAM/disk. A newly observed live job or lease owner
causes this experiment to yield. Do not clear user controls or kill unrelated work.
The initial candidate training stage has a 30-minute wall ceiling. Check finite
losses/gradients, legal normalized targets, entropy/policy drift and realized
schedule/clipping before expanding it.

## Playing-strength decision

Bank a candidate before evaluation. Use paired openings with reversed colors,
identical recorded code/search/tablebase settings, and the donor as reference.
The deciding metric is the paired score against the donor, with uncertainty
computed over opening pairs. Lower loss and teacher agreement do not decide Elo.

Before observing the first candidate's playing results, record an exact bounded
screening command and sample size, plus a disjoint confirmation opening bank and
budget. A screening result may reject a clear regression; it cannot establish
improvement. Claim a reproducible gain only when the preregistered independent
confirmation's 95% paired-score interval lies above 0.5. No repeated favorable
stopping or silently pooled banks. If the budget is insufficient, report an
inconclusive result and preserve the evidence.

## Recovery and outcomes

Preserve the donor unchanged and retain separate baseline/candidate copies outside
Ray-managed storage. Bank source/config/checkpoint hashes, raw selfplay shards,
per-game evaluation records and all runtime failures. No interrupted exact RL
resume is claimed: an interrupted disposable pilot retains its completed files
but does not automatically resume uncommitted updates. Do not restart epoch three,
the user-stopped THREE evaluation, or a closed publisher.

After a failed startup, preserve the actual error and process outcome, fix only
the established cause and independently qualify the changed path before retrying.
After timeout/resource abort, reconcile owned descendants and custody before any
new conflicting stage. After a clean pilot, evaluate before expanding training;
Further bounded adjustments within the authorized 48-hour window require a
recorded readout and a new preregistered hypothesis; the final independent
confirmation remains limited to one selected candidate. Record the readout against its rule in this
file, including negative/inconclusive verdicts and returned resource custody.

## November 11 qualification readout and CPU wiring preregistration

Source `bd1ae0cbe7d863dbb3dd124034b0bbe7090a4a99` composes the strict startup
and sequential local phases. Separate component and final-composition independent
reviews found no remaining runtime defect. The line-accounting test finding was
corrected. The composed focused suite passed 337 tests with no skips, including
all 18 Stockfish-backed end-to-end cases; the composition reviewer independently
passed 67 relevant checks. Local whole lint's three `_posixshmem` resolution
failures reproduce on unchanged source; CI lint passes without a suppression.

Bounded CUDA and Ray qualification both passed with owned process closure and
lease release verified. CUDA tested BF16 forward/backward, finite gradients,
model/optimizer checkpoint roundtrip and compiled/eager parity under Torch
`2.14.0+cu130` on the RTX 5090 (33.991 seconds inside the child, peak allocated
285.38 MiB). Ray `2.58.0` tested local startup, persisted Tuner restoration and
checkpoint continuation (10.952 seconds inside the child). Both are synthetic
runtime checks, not donor training or strength evidence.

The installed CPU qualification at
`rl-transition-preflight-20261111-local-v1/installed-cpu-qualification-v2/result.json`
exercised real argparse/driver conversion, actual donor loading and first export,
manifest reconstruction and the installed worker loading method. All 494 tensors
and the policy/value forward outputs match exactly, with fresh Aurora and
scheduler objects/state, sqrt-release schedule and clipping cap 1.0. It did not
start an HTTP worker process. The first diagnostic's reporting-only ZClip field
typo is preserved in the failed v1 receipt and corrected in v2.

After the owned CUDA job closed, external GPU use remained around 50% with no
Linux compute process reported. Further GPU work is deferred while it is busy.
The next independent gate is a **CPU-only two-iteration wiring run**, whose short
game horizon supports no playing-strength claim.

- Config: `~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/cpu-wiring-config-v1/config.yaml`
  SHA-256 `9fcca3f0786cb9363e5130feb2cd46280d96fe40412830dd1f3956680ca4f56b`.
- New output root: `~/chess-artifacts/operations/rl-cpu-wiring-20261111-48546328`.
- Two iterations, two completed-game targets each, Gumbel 16 simulations, maximum
  48 plies, batch 8 and one-update cap per iteration. Both learner and worker use
  CPU and Ray allocates zero GPUs. The readonly baseline copy is hash-verified;
  all model settings and the strict startup contract are retained. The two
  iterations intentionally exercise changed-weight publication and worker recreation.
  The supervised command begins `/usr/bin/env CUDA_VISIBLE_DEVICES=` **inside**
  the stage owner, so its child override survives the owner's environment setup.
  This is required because worker capability telemetry probes every visible GPU
  even when its model device is CPU; zero Ray GPU resources alone is insufficient.
- Fixed SF 1,000 nodes, PID off (so curriculum uses unhandicapped best-SF moves),
  half selfplay/half curriculum, SF/search/outcome WDL shares 0.5/0.2/0.3, and
  fresh Aurora LR 3e-6 with matrix multiplier 20 and eight-update warmup. SF may
  adjudicate at the ply cap; these are wiring samples, not factual-outcome research.
- Finite 600-second stage, then verified owned cleanup, under the canonical leases
  and existing controls. New local-only credentials provision only the new loopback
  server. No existing account, run, checkpoint or control is changed.
- Success: two actual updates, finite reported losses and all final tensors, plus
  publication of changed weights for the next collection and closed owned processes.
  Deciding readout command:

  ```bash
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ~/chess-artifacts/venvs/rl-transition-managed-20261111-cu130/bin/python \
    ~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/verify_cpu_wiring.py
  ```

- Kill on timeout, control request, resource floor or runtime exception. A failure
  is a wiring failure with retained receipts, not a negative RL-strength result.
  Do not resume it as an exact RL recovery. After success, qualify the real CUDA
  handoff when the GPU is available, then preregister the scientific pilot and its
  fixed paired evaluation budgets before observing playing results.

## CPU wiring verdict

**Pass for wiring only.** The supervised stage returned zero and closed all exact
owned PID/birth identities in 455.27 seconds. Independent review verified four
completed games and 128 stored positions: two completed games per cycle, one
selfplay and one curriculum. All four were SF-adjudicated at 48 plies; four
additional in-flight games were abandoned with resume disabled.

The actual learner reached steps 1 and 2, with finite losses 5.117904 and 5.009705.
All 494 final model tensors remain finite with donor shapes/dtypes; 486 changed on
each update. Both local producer-stop barriers ran. The initial cached worker
weights match all donor tensors. The step-1 publication
`865a8a4dc47ca19e4dde90c0552791c87d560104f4af369f0f50bee4ab552739`
matches the step-1 checkpoint exactly and supported the next collection. **Final
step 2 was checkpointed, not published or served.** Its checkpoint SHA-256 is
`dc01bf952de9aa8d9c2e152c2228cb8cdc7e41ced5e6b5180c38ec29afd809c4`.
A readonly diagnostic candidate copy is retained under
`rl-transition-preflight-20261111-local-v1/wiring-candidate/`, outside Ray storage.

CPU-only execution is supported by the explicit child CUDA mask, CPU worker and
learner settings, and zero Ray GPU allocation; this is not independent GPU
allocator telemetry. The source's local tests, synthetic CUDA checks, and real
CPU wiring establish different gates. No strength comparison has run, and the
loss decrease is not evidence of Elo improvement.

## Donor lineage correction before scientific training

The preliminary `first-pilot.yaml` proposal imports several old online defaults
and **is not qualified for strength training**. The retained supervised training
YAML is
`/mnt/c/Users/jjosh/Documents/Codex/2026-10-13/task-5/all-accumulated-raw-training-launch-v1/ALL-ACCUMULATED-THREE-EPOCH-CONFIG-v1.yaml`,
SHA-256 `d5b00c177e0d454f82053eac6f22d0aba6f6a2e9b146fe43fc59634135c231d8`.
Its WDL recipe is SF fraction 0/search fraction 1, consuming stored neural-teacher
values, rather than the proposed live SF/search/outcome blend 0.5/0.2/0.3. The
historical online SF-blend result is conditional on its older donor and data; it
does not establish the best blend for epoch two.

| Setting | Retained supervised recipe | Preliminary proposal |
| --- | ---: | ---: |
| Soft policy weight | 1.0 | 0.5 |
| Future policy weight | 0.01 | 0.15 |
| Opponent policy weight | 0.05 | 0.15 |
| P0 policy floor weight | 0.8 | 0 |
| SF-value auxiliary weight | 0 | 0.15 |
| Categorical weight | 1.0 | 0.1 |
| Volatility weight | 0.1 | 0.05 |
| Hard clipping cap | 6.5 | 1.0 |

The donor also used Aurora `polar_express`, eight polar steps, three PP iterations,
PP beta 0.25 and FP16 polar arithmetic. These differ from the preliminary
proposal's inherited optimizer defaults. Preserve unaffected settings where
possible. The policy floor needs actual P0 labels; retaining its numeric weight
without compatible labels would silently remove its effect. Any necessary target,
loss or teacher-source change must be explicit in the scientific preregistration.

Next, audit the exact stored wiring inputs and target-information fields, including
legal policy mass, entropy, SF/search-value availability and disagreement, and
adjudication/provenance limits. Use the frozen observations when correcting a
readout. A same-root 64/256-search mechanism screen can justify a larger search
budget only if its quality improvement and cost warrant it; target KL alone is
insufficient. The old proposal's maximum 8,192 sample views would also weakly test
the RL idea after 348,107,734 delivered supervised rows. A bounded dose increase
needs its own precommitted limit and realized-step check. Freeze disjoint paired
screening and confirmation opening panels before inspecting candidate play.


### Sequential restart audit and corrective qualification

The original CPU bank has 128 rows but only 64 byte-identical unique inputs.
Exact producer joins show the same two trajectories in both cycles, caused by
reusing the trial/worker seed at every restart. This is a pipeline defect.
Audit: `~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/cpu-shard-audit-v1/readout.json`.

Independently accepted correction `2eba488ab6e9624aaf4e1b493b1233e7f0bcc973`
varies seeds by global collection iteration and publishes current weights before
starting sequential workers; health revival retains the collection seed.
Default streaming behavior is preserved.

Successor preregistration: `~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/cpu-wiring-config-v2/manifest.json`,
config SHA256 `38900afe4ee63165a7b3271e8aba59905e09296fbdfa7760c7f0d1acdd9e43d4`, source `2da5c6054ae54a6daee024eef79d2cc0413fb1b7`.
Two CPU-only collections, one selfplay game each, 24-ply cap, 8 simulations,
batch 4, one update cap each, 360-second ceiling. Require actual completion,
distinct observed worker seeds, finite updates, step-1 publication consumed next
and owned closure. Compare trajectories; shared opening positions are legitimate
and changed seeds alone do not prove independence. This uses smoke loss settings
for wiring only, with no Elo claim. Failure/timeout preserves evidence for
diagnosis and reconciles owned closure before any successor. Existing controls
and CUDA-hidden child command remain required.


### Frozen conditional search-mechanism screen

Before scientific training or candidate play, freeze 64 roots sampled with seed
20261111 from 334 unique nonterminal roots with at least two legal moves and
complete direct nonmate SF centipawn labels. Source audit SHA256:
`d8e26efa0b010450abf9374693afc45027db6d146571785ab897af5061144df2`;
selected panel SHA256:
`87fd7a5c0f324e661f86f0bb7290f27a0734634be059ff4eb1885614aea22598`.
Manifest: `~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/search-mechanism-config-v1/manifest.json`.
Use readonly epoch-2 donor, main linear selfplay shape (topk16, c_scale0.1,
policy_temp1, vloss1), no root noise, 64 and256 simulations on the same roots,
batch16, CPU2threads, 1200-second owned-stage ceiling, CUDA hidden.

Deciding metric is expected capped-at-1000cp SF regret(prior) minus regret(actual
native improved-policy target), over all legal moves. Use 10000 root-paired
bootstrap resamples, seed20261112, percentile95% intervals. A rung's positive
lower bound supports its conditional mechanism; select256 over64 only when its
paired incremental lower bound is positive, otherwise retain a healthy cheaper
rung. If neither passes, report inconclusive and inspect the bank before further
compute. Require exactly64 complete paired roots, actual native policies, full
labels, finite normalized probabilities, matching prior vectors and the independent
UCI-index control. Counterfactual reconstructed policies are not scored.

This is a FEN-only frozen historical-SF subset, potentially biased toward endgames;
it establishes neither historical-input fidelity nor Elo. Timeout/incomplete
outputs are partial, not a passed screen. Preserve raw root observations and
reconcile owned closure before any successor.

Disjoint evaluation panels were frozen before candidate play from the production
weighted opening book: 128 screen pairs (256games), SHA256
`ae09af5d6fd6cf5dce16e802248b817d82099c6b6049b5418eef1516d6f5c2f6`,
and500 confirmation pairs (1000games), SHA256
`fe1d02ec6e7af621753209ca5bcc96a435d71b1ad103e64a765b53af3a1a3ad7`.
Actual legal move histories survive the arena reload. Panel manifest:
`~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/evaluation-openings-v1/manifest.json`.
A fixed screen score above0.5 permits at most one preselected candidate into the
unseen confirmation; only a confirmation95% pair-level score interval above0.5
can establish gain. Partial or null-crossing results stay inconclusive. Exact
arena runtime/search commands remain to be frozen before scientific training.


### Corrected sequential runtime readout

Successor completed in203.041seconds with owned processes closed. Observed worker
seeds were3576761918 and3458919155 with CUDA hidden. Two completed24-ply
adjudicated games produced41retainedrows (19/22),38byte-unique inputs,3shared
opening inputs, different content-derived game IDs and distinct final trajectories.
Two additional inflight games were abandoned. All41rows reached six replay
shards. Default distributed diff-focus thinning remained active despite the smoke
YAML's false flag: those original fields are intentionally pinned to worker defaults
and not forwarded. The scientific config must describe the actual sampler as true.

Two actual updates had finite losses5.620818138 and5.283257961,494finite model
tensors and486changedtensors. Step1 SHA256
`fbb69c37c6f5950b5cec48db1eb71662fa2ae862b645b3c722c2a97dffff00ac`
was published and is declared by the next producer. Step2 was checkpointed, not
published, SHA256
`b78f5d6876927f94d8aec3cf020bcdf666b7a2dcd2489b18e5aa90ee38befc2e`.
Independent review checked logs/CSV/producer metadata and exact PID/birth closure;
it did not repeat the parent tensor comparison or prove statistical independence.
Readout and raw bank: `~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/cpu-wiring-diversity-v2/result.json`
and `cpu-wiring-readout-v2.json` under the same artifact root. No strength claim.


### CI contract integration

The first remote ordinary CPU suite on8252 failed15cases: strict startup identity
fields lacked updated atomic-group fixtures/reader classifications, an isolated
AST harness lacked the new real validator bindings, and public documentation
contained local absolute home paths. Independently accepted correction
`731417da05d625fee9f8bab2389af5be35b6fe67` changes tests/docs only, preserves
negative consumer-detection controls, and passes211focusedchecks with zero skips.
Exact original failures and raw CI job log remain in
`~/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/ci-failure-8252/`;
corrective evidence is in sibling `ci-contracts-2da5-20261111-v1/`.
This is not a claim that the whole remote suite has passed; rerun CI on the final
composed branch. Local whole lint retains only three established standalone-Python
`_posixshmem` resolution errors; ruff and vulture pass. Runtime source remains
unchanged during the fixed CPU mechanism stage.


### Corrected scientific pilot and exact evaluation preregistration

Before scientific training, freeze proposal
`scientific-pilot-config-v2/0b08fba3-fcc4-407e-adf2-00fd460c0bc9/manifest.json`
under the artifact root, manifest SHA256
`ac764f35fea027f1cff5eefb7ddfcbad163da308c27572c30edbd21113dfc689`.
Config and separate experimental evaluation YAML both have SHA256
`71b4cb1be3d4249887f36b640f48c6987f0ca13437f759657b97e160b474eae3`.
Actual CLI/TrialConfig/trainer-kwargs/recommended-worker/worker-dataclass checks
pass without model or CUDA execution;258 explicitly chosen effective settings are
strict-bound. This remains a proposal until CUDA/optimizer/producer qualification.

Restore supported donor auxiliary weights and Aurora polar_express recipe,
FP16 polar arithmetic,8 polar steps,3PPiterations,beta0.25 and ZClip cap6.5.
Retain w_sf_policy_floor0.8 with record_sf_p0_regret=true (a separate field from
the disabled w_sf_own distribution). Hypothesis: online search policy plus
5000-node SF-assisted value/floor targets can improve this donor at low LR.
The main and categorical0.5SF/0.2search/0.3outcome mixtures are explicit new teacher
sources; historical stored BT4/Ceres targets are not preserved. Policy-floor
coverage is limited to eligible consecutive full selfplay plies. Require actual
coverage, finite labels/gradients/weights, realized LR/clipping and loss readouts.

Provisional finite pilot:64 completed games,450ply cap,64generation simulations,
selfplay batch16 and fraction0.5, SF5000/MultiPV8/PIDoff; one iteration, batch256,
at most128updates, two-window budget and views-per-ingested-position0, window50000,
no imports/sharing, sequential stop barrier, broker/async/prefetch off, one GPU.
1800-second supervisor ceiling with1500-second generation wait. This cap does
not guarantee128updates or completion. Fresh optimizer and eight warmup updates
are deliberate. Search256 needs a preregistered successor justified by the fixed
mechanism and throughput budget, not an in-place edit.

The original distributed sampler remains enabled with normalization off. Its
retained rows carry priority for additional surprise-weighted replay sampling;
the Bernoulli keep probabilities are not inverse-probability loss corrections.
Report generated/retained rows and realized priorities. This explicitly corrects
the proposal notes' loose use of 'importance weights'.

Exact arena argv is frozen in `evaluation-protocol-v1/manifest.json`, SHA256
`fcc31a747917725d95b7878797683304089ea3496899faf48215bf4e9d884559`.
Both sides use64 matched simulations, the frozen experimental training search
shape, constant default root noise1 (unscheduled), move temperature0.1,450 additional
plies after the16-ply opening,
no tablebases, eager inference,32concurrentgames and evaluator cap1024. The actual
no-relations architecture requires at most1024leaf rows, so that cap does not
change search by absorbing excess leaves. Existing frozen128/500opening-pair
panels and seeds20261112/20261113 are used. Candidate is banked and hash-bound
before screen; the same file is selected before any confirmation play.

Each arena has6900seconds internal and7200seconds external wall ceilings within
the global48-hour deadline. All fixed pairs must complete. Screen score>0.5 only
permits confirmation; final gain requires an available95% normal-approximation
score interval from the existing pentanomial pair variance, with lower bound>0.5
on all500confirmationpairs. Zero empirical variance, partial pairs, inconsistent
logs, or an interval crossing0.5 are inconclusive; no favorable extension.
Source/config/model hashes, exact per-game outcomes and histories are retained
for independent acceptance. No scientific pilot or candidate play has run yet.


Before viewing any scored mechanism outcomes, prepare a conditional hardware
successor for a CPU completion failure: `search-mechanism-cuda-successor-prereg-v1/manifest.json`
under the artifact root. If the1200-second CPU stage times out/incompletely banks
its exact panel, retain that incomplete bank unread for deciding metrics and
rerun the identical64roots/both rungs on CUDA, with the same1200-second bound,
only after GPU availability and prior owned closure. No CPU/GPU pooling, missing
root imputation, resampling, or hardware choice by favorable scores is permitted.
The complete GPU bank then becomes the sole conditional mechanism result, with
its actual arithmetic disclosed. The original CPU cap is never extended.


### Completed conditional mechanism and CI readout

Both rungs completed in1140.806seconds, within1200, with all128native rows
and owned processes closed. The prespecified10000-resample paired-root reader
scored actual native policy probabilities against full legal-move SF labels,
with the same1000cp regret cap and exact matching prior vectors.

| Search simulations | Mean expected-regret reduction versus prior | Root-paired95% interval |
| --- | --- | --- |
|64|12.744cp|[2.741,23.293]cp|
|256|3.831cp|[-12.610,17.493]cp|
|256 minus64|-8.913cp|[-30.484,7.069]cp|

The registered rule selects64. Larger search has no supported incremental
benefit on this panel. Population:44endgame,17middlegame,3opening roots, with
FEN-only history and historical SF labels; this remains conditional target-quality
evidence, not production fidelity or Elo. The conditional CUDA mechanism retry
is ineligible because the original CPU panel completed; no rerun is needed.

Native bank SHA256
`f63ed8b74338fea40c2218386e49da8faee511e58613af4acca41a502dd41e44`;
readout: `search-mechanism-config-v1/readout.json`. Independent reader review
corrected a genuine control gap before execution: finite normalized policies alone
can hide nonfinite native child Q through softmax fallback. The accepted reader
now rejects nonfinite visitedQ/counts and malformed trace shapes;15synthetic
controls pass. Reader SHA256
`3fdf7bf5b3c9543db662945c8f0b23b63b368e0bfd2f150ad4a3061644facf9c`,
review in `mechanism-reader-review-v1/acceptance.json`. Independent math.fsum-based scoring and bootstrap of the actual bank also
accept64, matching all means and intervals within1e-10, and verify PID/birth
closure. Acceptance manifest:
`mechanism-independent-acceptance-v1/51d3dee7-2713-41e0-bb22-14d566b2c777/manifest.json`,
SHA256 `76f80df4d9a8cfbec78ba07bdba5733d9f0ae0e4ca7a9a01d464503771bd9a5d`.

All remote checks on5250bc205 are now green: full ordinaryCPU suite15m54s,
lint,pext-sliders andbacklog-handoff. Exact check-runsAPI receipt is banked as
`ci-pass-5250-check-runs.json`. Scientific pilot config source/consumer review
also passed independently, including39retained donor fields and the actual
128-step ceiling; review in `scientific-pilot-independent-review-v2-0b08fba3/`.
CUDA batch256/Aurora qualification remains unrun while the user's Windows GPU
workload is active; the prepared disposable qualification has its own UUID,
never adopts its10-step repeated-row fixture trajectory, and leaves online target
and P0-floor coverage explicitly unqualified.


Independent static CUDA-qualifier review accepts the frozen script and its actual
APIs/observer ordering/LR checks/fixture hashes, without executing it. Review:
`scientific-cuda-static-review-v1-214f0e10/review.json`, SHA256
`76543acedb8fdfe0b2e2f44487bd30fe1977774a2418d99ade132f2e78def58a`.
The qualifier checks concurrent trainer/worker inference residency then frees
the worker before backprop, matching the real sequential lifetime. It does not
qualify production leaf-batch memory or online P0/teacher target availability.

Arena readout precision: recompute the registered interval from integer
pentanomial counts through the existing summarize_pentanomial helper, not rounded
score/SE fields. Require every fixed pair and validate per-game logs/history/net
hashes/settings even when the arena exits0. The manifest's raw carried shape
includes four inert PUCT fields omitted from the active result projection; only
the active shared Gumbel fields are consumer settings.


Independent arena source/protocol review accepts the exact frozen manifest with
zero blockers, while retaining the inert-field metadata note. Review:
`arena-review-sequential-selfplay-v1/review.json`, SHA256
`f2fa072ccd50f1b94bc5f4e96b0ed06bada8def27ee6ab141b9fe31b211a815e`.
The450-ply arena ceiling counts additional plies after each16-ply opening, and
max-ply draw outcomes must be stratified in the readout. This has always been the
command's semantics; it is a prose clarification, not a changed match. No candidate
file or match output exists yet.


### Conditional SF teacher-effort screen preregistration

While the GPU remains occupied, reuse the exact64-root frozen panel to compare
Stockfish5000 versus50000 requested nodes, MultiPV8, one engine/one thread,
Hash16MiB, no handicap or TBs, cold TT before every query.180-second existing
owned-stage ceiling,170-second helper alarm, CUDA hidden. Preregistration and
script were independently source-reviewed before launch; no model forwards.
Manifest: `sf-label-quality-screen-v1/88151623-3571-4d8c-92bd-c97d3296d337/manifest.json`,
SHA256 `01c130ef5767148eb115c29a70de37164e294ce502553ef999a96c1cf0321b00`.

Deciding event: native current SF best move is within20cp of the complete fixed
old deep-SF reference (regret cap1000cp). Require Wilson two-sided95% lower bound
at least0.8, corresponding to at least58/64good roots; choose the cheapest tested
passing effort, otherwise inconclusive. Full native/UCI observations are banked.
Secondary teacher-window comparisons use the actual producer sparse-PV/regret
helpers, report float16 boundary effects and unknown unreported moves separately,
and do not qualify policy-dependent floor binding or WDL calibration. No fresh
population, online-history fidelity, donor-teacher equivalence or Elo is claimed.
No automatic pilot edit follows: with sf_move_nodes0, raising sf_nodes also raises
PID-off opponent effort, requiring an explicit successor and coupling readout.
Timeout/invalid/incomplete data stays unqualified; retain raw files and prove
owned closure before another stage.


The first SF screen completed128cold queries in8.144seconds with owned closure.
5000nodes:49/64good, Wilson95%[0.6487,0.8525],mean capped regret36.375cp.
50000nodes:51/64good,[0.6829,0.8773],mean28.266cp. Neither lower bound reaches0.8;
no effort was selected. Storage-realized teacher windows contain29/119 and26/113
false-positive moves against the fixed old reference;2deep-good moves per rung
are unmeasured and remain unknown. Independent bank recomputation verifies the
primary, window estimators, cold protocol, hashes and exact closure.

Preregister one bounded successor before new queries:1000000nodes on the exact
same64roots, preserving MultiPV8/threads1/Hash16/TBoff/coldTT and the same58/64
Wilson criterion, with170/180-second alarms. No further escalation is automatic.
Manifest: `sf-label-quality-screen-v2/d0c41114-a1d5-4fbc-b038-c1161b7844ed/manifest.json`,
SHA256 `47ed8104f09bcf81f780f63a81fed92c901ace4ba69c0f2870d4663209e3c779`.
The historical reference used roughly1Mnodes; low tested efforts were unsupported.
V1's actual3.522Mnative nodes took3.409queryseconds, making this64Mnode diagnostic
plausible inside180seconds without guaranteeing completion. V1 negatives remain
unchanged; no pooling, root replacement, threshold relaxation or pilot edit.


V2's first wrapper (`sf-label-quality-execution-v2`) failed1.138seconds into an
output-path assertion before output creation/importing engine code. The parent
passed the wrong path; no engine queries occurred and owned closure was proven.
The failure and exact error remain intact. A corrected wrapper
`sf-label-quality-execution-v2b` consumes the hash-checked manifest argv directly,
with its output field checked against preregistration before launch. This is a
preflight argument correction, not a failed scientific result or data-dependent
reroll.

Main advanced during this session to72a3c3f3 with upstream checkpoint-retention
fixes. A read-only integration audit finds only harness.py changes among230
package files, preserving RL validation/sequential function ASTs and all model/
optimizer/worker/Stockfish/native sources. The imported harness change is an
explicit exception to byte-identical CUDA-source claims. Finish the current
5250-bound finite stage first, then integrate main and run composed startup/
retention/source-contract checks. Prepared CUDA artifacts remain unrun and must
be refreshed against the final reviewed source before admission. Audit:
`main-integration-audit-v1-72a3c3f3/review.json`, SHA256
`e41c2dd721e8f9da73129664f703612af951823c52f3083c02ba64db62a3b3c7`.


The corrected1M-node successor completed64cold queries in47.288seconds with
owned closure.55/64bestmoves are within20cp, Wilson95%[0.75384,0.92421];mean
capped regret11.078cp. The lower bound still misses0.8, so the registered
verdict is inconclusive and no teacher effort is selected. Stored near-best
windows include108true positives and19false positives out of127members;
2deep-good moves remain unmeasured. Native bank SHA256
`241c62bfb29fc664c47ff91277195ed85f33b28bf6a3a3e3d09aa3732e7d6149`.
No automatic further node escalation or threshold relaxation follows. This does
not prove those teachers are below80%accuracy or that SF-assistedRL cannot work;
the old selected reference and interval leave uncertainty. The SF-assisted pilot
proposal remains unrun. A policy-focused control supported by the actual64sim
search-target result is being reviewed as the conservative next hypothesis.
