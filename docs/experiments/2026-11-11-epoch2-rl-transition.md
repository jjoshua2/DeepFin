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
`/home/josh/chess-artifacts/operations/all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery654213-a4349851abddea4a/training/epoch-2/checkpoint.pt`.
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
`/home/josh/chess-artifacts/operations/rl-transition-preflight-20261111-local-v1/`.

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

The first experiment asks whether search-improved targets from newly generated
positions can improve the donor while retaining SF value support. This is
Stockfish-assisted RL. It is not outcome-only AlphaZero. Settings, sample/update
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
`/home/josh/projects/chess/scratchpad/gpu0_experiment.lock` through owned-process
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
at most one further adjustment is justified by the first pilot's evidence unless
the user changes the budget or scope. Record the readout against its rule in this
file, including negative/inconclusive verdicts and returned resource custody.
