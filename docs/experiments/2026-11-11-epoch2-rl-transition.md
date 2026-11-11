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
