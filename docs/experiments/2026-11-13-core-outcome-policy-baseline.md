# Core policy and resolved-outcome baseline

Status: **DRAFT_NOT_FROZEN_OR_ADMITTED**. This records prospective v5, not an
execution authorization. The first milestone is **two successive supported
playing gains**, each independently confirmed on fresh games. Roughly 50 Elo per
leg is illustrative; smaller reliable gains count. Preserve the first improved
full learner boundary and continue its state. Rofchade, Stockfish and Lc0 strength
milestones remain unachieved. Earlier [E4](2026-11-13-rl-larger-outcome-regime.md)
and other negative/unresolved studies remain closed and are not pooled here.

## Fixed learning question

Can a sustained, minimal policy/outcome loop adapt immutable original A39 and
then improve again from its first confirmed endpoint? Keep the full schema-20,
175-input, 494-tensor architecture, including inherited coupled value topology.
Use finite selfplay with explicit Gumbel64, full simulations, strict full-history
rule50 six-piece TB search/adjudication/policy correction, neutral retention of
all eligible policy rows and no diff-focus thinning. Natural and qualified TB
outcomes retain distinct provenance; unresolved caps are censored, never draws.
This is a local LC0-style baseline, not a replica of a competitive LC0 run.

Train main dense Gumbel policy CE and verified outcome WDL CE at weight 1; all
other twelve objectives are zero. SF/search WDL shares, target rebuilds,
dropout/channel balance and SWA are off. Use eager BF16, mirror probability 0.5,
explicit no clipping with finite-gradient refusal, and fresh local NadamW once
(β .9/.98, ε 1e-7, existing ndim/bias decay mask 1e-4/0). Subsequent calls preserve
model, optimizer, scheduler, global step, replay lineage and boundary RNG. An
immutable champion and the latest private training actor are separate: every
next actor is the preplanned previous-generation final learner, not a
score-selected checkpoint. Diagnostic SF fields imply no adopted SF loss.

The static horizon is two conditional eight-generation legs: 512 updates of
batch 256 per generation, 4,096 per leg and at most 8,192 total. Uniform physical
rows are sampled with replacement from the last four complete generations.
Every generation attempts all 256 preassigned games, admits **all** resolved,
nonempty eligible games only after audit, and requires 192–256 such games and
at least 10,000 eligible rows. No subset is chosen to force exactly 192; failed
minimums are unread, with no replacement games. Proposed admission limits are
160,000 replay rows and 8 GiB decoded FIFO. Overflow is unread; rows are not
dropped to fit. Maximum FIFO, steady throughput and total process memory still
need qualification.

One global 256-update warmup starts at 1e-6, followed by the fixed cosine schedule
with peak 3e-5, floor 6e-6 and first period 16,384 (multiplier 1). No restart or
warmup reset is introduced at a leg boundary. The actual CPU LR-consumer trace
used a placeholder model and no forwards/optimizer steps: applied update 257 is
3e-5, update 4,096 is 2.6894503325964055e-5 and update 8,192 is
1.859340844764102e-5. Stored next-step rates differ; the horizon does not reach
the floor. These constants are experimental, not an optimal-LR claim.

## Starts and fixed decisions

Candidate study UUID is `b0db4b6b-2bb3-4753-80d6-96825decbbee`. The fixed hash
assignment proposes 512 TRAIN histories in alternating 256 halves, one shared
128-pair diagnostic roster, separate 256-pair terminal rosters per leg, fresh
500-pair confirmation rosters per leg, and a 256-pair C2-versus-A39 context panel:
2,408 distinct roles, with 108 unused identities never serving as replacements.
Full root plus 16 legal UCI plies, parent IDs and raw/mirror identities are
preserved. Training-start reuse and correlated diagnostic reuse are intentional.
Known exposure checks are bounded; shared book provenance and unknown donor
pretraining prevent universal novelty or opening-family independence claims.
Retained raw/mirror inputs must remain disjoint from all playing starts.

After local generations 2/4/6, run 256-game diagnostics against the leg reference.
They never select an intermediate endpoint. Continue through local generation 4
unless correctness, nonfinite, control or resource failures require closure;
only the generation-4 diagnostic's interval upper below .35 triggers the declared
severe-harm stop. A failed midpoint promotion is not a horizon cutoff.

Only local generation 8 is eligible: its fixed 512-game/256-pair screen must have
point score ≥ .53 and interval lower > .5. An eligible endpoint gets at most one
fresh 1,000-game/500-pair confirmation with the same fixed model hash and deciding
rule. Technical failure, unresolved or negative confirmation admits no fallback,
reroll, pooling or other checkpoint. Leg 2 runs only after C1 confirms, using its
exact continued learner state and unchanged contract; C2 is tested against C1.
The separate C2-versus-original-A39 context comparison is not obtained by adding
adjacent Elo estimates.

The proposed reusable [calculator](../../chess_anti_engine/paired_score.py) accepts
only complete color-pair means in {0,.25,.5,.75,1}. It uses unbiased sample variance
and z=1.96 normal 95% approximate bounds when variance is positive. At zero
variance or n=1 it uses radius sqrt(log(40)/(2n)); bounds are clamped to [0,1].
This new estimator does not change E4's estimator or results. Each two-sided
lower threshold has nominal one-sided α .025; the two preplanned confirmations
have a nominal .05 family bound **conditional on calibration assumptions**.
The mixed estimator, book/family dependence and conditional C1 selection do not
establish exact or global error control. Final estimator/command binding is
still pending; no Elo or promotion decision is implemented by the calculator.

## Cadence, resources and readiness

Aim for roughly one informative complete leg per day; cadence does not truncate
the fixed horizon. After establishing the baseline, compare full strength versus
end-to-end wall-time curves, time to a supported gain, uncertainty and censored
crossings. Include collection, preprocessing, learning, evaluation, checkpoint-
ready/evidence-ready times and operator pauses. Allow crossing curves; do not
force monotonicity or select a noisy best checkpoint.

The proposed maximum is 124,200 GPU-owner seconds if both legs and confirmations
run, within the unchanged 172,800-second week envelope and deadline
**2026-11-19 16:09:20 UTC**. Each owner remains ≤ 7,200 seconds. These are unreserved
ceilings, not an ETA: shared qualification 1,800; per leg collection 28,800,
learning 14,400, diagnostics 5,400, terminal 3,600, confirmation 7,200; leg-2 donor
context 3,600. Each stage still needs actual remaining resources and admission.

PRs [1266](https://github.com/jjoshua2/DeepFin/pull/1266) and
[1267](https://github.com/jjoshua2/DeepFin/pull/1267) published reusable finite
collector/learner tooling. Qualified runtime `090b5143315a0faef7ab0ac1a90e2d5cebf3b929`
has the same whole source tree as merged main
`aa60da65e6e48c7890bc721a0e8f4f9c6f78c228`. Its actual two-generation engineering
qualification delivered four games, 438 rows and four updates, including actor
feedback and continued state. It did not exercise a full four-generation FIFO,
maximum memory, sustained research dose or strength. The 2,408-start CPU consumer
stage returned PASS; final independent acceptance/protocol linkage is separate.
Maximum-FIFO/steady qualification, current arena/profile qualification, final
roster/seed/estimator/source bindings and the frozen scientific protocol remain
prerequisites. Engineering success establishes neither an optimizer recipe nor gain.

Content-hash references below identify external operation artifacts, not new
research admission:

| Artifact | SHA256 |
| --- | --- |
| `lc0-style-baseline-design-draft-v5/proposal.json` | `6917c71ff916fd36e93814a0d8023a54444ef7369ab08e831933ccac76f88055` |
| `core-baseline-v4-candidate-roster-v1/manifest.json` | `8c2f311d3fd7fd13977ea533e4f61bc438a8197a90d8dcb630b85130882c56d1` |
| `core-pipeline-L-two-generation-readout-v2/readout.json` | `73d67cfdc84ba656a3d1fb2f28de62daf14709775b6114059b6c30b64a720e2a` |
| `core-baseline-LR-consumer-results-v1/readout.json` | `b8f0ea746570d93b9431b36a87f96a0e642d74cd1decd746e32b069233961016` |
| `core-baseline-LR-consumer-results-v1/actual-update-LRs.csv` | `c8f7d15a315f1ca07051f723b32cd18621ad4e0d0fa31f0ea9c62e4dcbaa5fcd` |
| `core-baseline-roster-consumer-results-v1-b0db4b6b-2bb3-4753-80d6-96825decbbee/readout.json` | `f137fe768057b22e9c1a114138507ae9563ddd26b63a21023e56edb9ad4ddbb3` |
