# Core policy and resolved-outcome baseline

Status: **FROZEN_SCIENTIFIC_PROTOCOL_SEPARATE_ROOT_EXECUTION_ADMISSION_REQUIRED**.
The local protocol froze at **2026-11-13 16:23:56 UTC**, before the first research
collection. Its SHA256 is
`dd16774a4cf1cd9f0765501d135355fcac82cb8be7ef09bb27a050d34dbeae49`.
Per-stage independent review and fresh root execution admission remain separate.
The original v5 record entered main in PR 1268 as
**DRAFT_NOT_FROZEN_OR_ADMITTED**; its provisional capacity is superseded below.
This document preserves that chronology and does not itself launch a stage.

The first milestone is **two successive supported playing gains**, each independently confirmed on fresh games. Roughly 50 Elo per
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
minimums are unread, with no replacement games. The measured capacity envelope
is **460,800 replay rows** (4 × 256 × 450) and **24 GiB decoded original-window
arrays**, replacing the original draft's 160,000 / 8 GiB proposal. Overflow is
unread; rows are never dropped to fit. This covers the tested field layout and
NNZ density, not every future sparse/dense layout, transient allocation or RSS.
Root requires fresh host MemAvailable ≥ 64 GiB before each learner owner and
monitors during the stage, stopping below 16 GiB or on control/resource danger.
These are external operational checks; research learner code has no per-copy
64 GiB guard. The engineering helper's pre/post-copy guard is a separate fact.

One global 256-update warmup starts at 1e-6, followed by the fixed cosine schedule
with peak 3e-5, floor 6e-6 and first period 16,384 (multiplier 1). No restart or
warmup reset is introduced at a leg boundary. The actual CPU LR-consumer trace
used a placeholder model and no forwards/optimizer steps: applied update 257 is
3e-5, update 4,096 is 2.6894503325964055e-5 and update 8,192 is
1.859340844764102e-5. Stored next-step rates differ; the horizon does not reach
the floor. These constants are experimental, not an optimal-LR claim.

## Starts and fixed decisions

Study UUID is `b0db4b6b-2bb3-4753-80d6-96825decbbee`. The fixed hash
assignment assigns 512 TRAIN histories in alternating 256 halves, one shared
128-pair diagnostic roster, separate 256-pair terminal rosters per leg, fresh
500-pair confirmation rosters per leg, and a 256-pair C2-versus-A39 context panel:
2,408 distinct roles, with 108 unused identities never serving as replacements.
Full root plus 16 legal UCI plies, parent IDs and raw/mirror identities are
preserved. Training-start reuse and correlated diagnostic reuse are intentional.
Known exposure checks are bounded; shared book provenance and unknown donor
pretraining prevent universal novelty or opening-family independence claims.
Retained raw/mirror inputs must remain disjoint from all 1,896 playing starts.
The current 2,408-start CPU consumer and 1,896-entry playing guard passed
independent review. The final named prior union covers 304,620 known raw/mirror
hashes and 29,795 histories with zero known collisions across these roles. Arena PGNs contribute
conservative possible full-history prefixes, not forward-call telemetry. Arena
175-plane raw/mirror hashes and unretained search-leaf inputs were not recorded
and remain unknown; no retrospective encoding or universal novelty claim follows.

All sixteen fresh collection identities are preassigned. Initial learner seed
is `2026111520`, once; its actual shared replay/mirror Generator and global RNG
states then continue through boundaries. Collection seed and physical identity
change by generation, while the two TRAIN halves alternate intentionally:

| Generation | Collection UUID | Seed | TRAIN half | Final global step |
| --- | --- | ---: | ---: | ---: |
| 1 | `feed4977-74dc-4065-8c04-8c1572012979` | 2026111501 | 1 | 512 |
| 2 | `7d794f5a-11de-4979-ab2a-7a5636e2ff48` | 2026111502 | 2 | 1024 |
| 3 | `8ec37239-de51-4865-a385-4b8ff8f03720` | 2026111503 | 1 | 1536 |
| 4 | `60327148-0853-47df-b630-0b2cb6f8893d` | 2026111504 | 2 | 2048 |
| 5 | `f243047c-223a-423c-b217-4da37e07d1aa` | 2026111505 | 1 | 2560 |
| 6 | `ded47872-4656-4416-a3e2-eddfd3b6134e` | 2026111506 | 2 | 3072 |
| 7 | `0cfd3e29-cf6a-4c0b-8f03-aa6bc1f49edc` | 2026111507 | 1 | 3584 |
| 8 | `a528cd6b-e9b1-4bee-a5ef-305bbf110bac` | 2026111508 | 2 | 4096 |
| 9 | `058f066e-e199-4730-bd2c-57727493a86b` | 2026111509 | 1 | 4608 |
| 10 | `537a7498-2d3b-4bef-a446-007b8d03d37a` | 2026111510 | 2 | 5120 |
| 11 | `ddc92968-5f45-49ed-8e43-02af1de2ea5c` | 2026111511 | 1 | 5632 |
| 12 | `6aacba58-ce5c-46b7-8b43-3063a47f5b8c` | 2026111512 | 2 | 6144 |
| 13 | `12f314f2-1b3c-4243-bce0-201fffd2e508` | 2026111513 | 1 | 6656 |
| 14 | `76804de5-ad46-402f-b4fa-6c072d8bf99e` | 2026111514 | 2 | 7168 |
| 15 | `c2c2d594-2516-47ab-993e-71d055c74570` | 2026111515 | 1 | 7680 |
| 16 | `01ae0d3c-80f5-4002-9aa8-e5e2787d246a` | 2026111516 | 2 | 8192 |

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

The reusable [calculator](../../chess_anti_engine/paired_score.py) accepts
only complete color-pair means in {0,.25,.5,.75,1}. It uses unbiased sample variance
and z=1.96 normal 95% approximate bounds when variance is positive. At zero
variance or n=1 it uses radius sqrt(log(40)/(2n)); bounds are clamped to [0,1].
This new estimator does not change E4's estimator or results. Each two-sided
lower threshold has nominal one-sided α .025; the two preplanned confirmations
have a nominal .05 family bound **conditional on calibration assumptions**.
The mixed estimator, book/family dependence and conditional C1 selection do not
establish exact or global error control. Its 29 focused CPU cases and scoped
lint passed. Exact endpoint reader/command/custody binding belongs to the final
protocol; no Elo or promotion decision is implemented by the calculator.

## Cadence, resources and readiness

Aim for roughly one informative complete leg per day; cadence does not truncate
the fixed horizon. After establishing the baseline, compare full strength versus
end-to-end wall-time curves, time to a supported gain, uncertainty and censored
crossings. Include collection, preprocessing, learning, evaluation, checkpoint-
ready/evidence-ready times and operator pauses. Allow crossing curves; do not
force monotonicity or select a noisy best checkpoint.

The frozen maximum is 124,200 GPU-owner seconds if both legs and confirmations
run, within the unchanged 172,800-second week envelope and deadline
**2026-11-19 16:09:20 UTC**. Each owner remains ≤ 7,200 seconds. These are unreserved
ceilings, not an ETA: shared qualification 1,800; per leg collection 28,800,
learning 14,400, diagnostics 5,400, terminal 3,600, confirmation 7,200; leg-2 donor
context 3,600. Each stage still needs actual remaining resources and admission.

PRs [1266](https://github.com/jjoshua2/DeepFin/pull/1266) and
[1267](https://github.com/jjoshua2/DeepFin/pull/1267) published reusable finite
collector/learner tooling. Runtime `090b5143315a0faef7ab0ac1a90e2d5cebf3b929`
remains pinned. Its whole source tree matched historical main
`aa60da65e6e48c7890bc721a0e8f4f9c6f78c228`; PR 1268/main
`0cc0a5df47e6960588018ef9aeff1bee2a35d18a` added statistics/docs with unchanged
training/arena projection. There is no latest-whole-tree-equality claim.

Actual two-generation engineering delivered four games, 438 rows and four
updates, including actor feedback and continued state. The separate full-capacity
measurement constructed five synthetic generations of 115,200 rows, exercised
four-generation eviction and 4 warmup + 64 timed batch-256 updates. Its GPU owner
used **233.578347 seconds**; helper total **229.275951 seconds** includes cold
assembly and updates. The **26.764375 seconds** for 64 timed updates is not a
whole-generation ETA. The source cold-reconstructs all accepted history on each
learner call before eviction; that cost remains part of end-to-end learning.
Synthetic repeats of four games at constant engineering LR 1e-6 are not fresh
research data, an all-layout memory guarantee or evidence of gain.

The corrected public-arena instrument completed four full-history engineering
games with qualified serving/log/TB/custody checks: **50.440523 seconds GPU-owner
wall** plus **5.256057 seconds CPU readback**. Its earlier WdlTable fd-guard
failure occurred before models and remains preserved. This small instrument
qualification establishes no playing strength, large-panel throughput guarantee
or research result. No optimization recipe is justified by engineering alone.

The immutable local scientific freeze fixes the recipe, horizon, all sixteen
identities, thresholds, references, resource caps and deadline before collection.
The root learning contract is bound at SHA256
`c53eac5896c3465c55ce019f2ace24b039dec85740904c43d4b83493d49476e3`.
Independent CPU admission audits account for all attempts with 128 MiB per
metadata file and bounded 64 MiB per-game raw/shard decoding; no smaller eligible
subset is substituted. The operational arena observer/strict-reader adaptation
must be reviewed and pinned before the first generation-2 evaluation; it may not
change the fixed science and does not block generation-1 collection.

Each actual stage still requires its observed identity/custody bindings and
fresh resource admission. Existing main code/record publication suffices; this
doc-only update may publish concurrently with the first admitted collection,
with its CI reported separately. It introduces no new main-publication launch
gate and changes no loaded runtime. No research collection had started at the
scientific-freeze checkpoint; subsequent operational readouts are separate.

Content hashes below identify the immutable scientific artifact and external
operation evidence. The scientific protocol contains the full binding list;
engineering receipts alone do not admit research:

| Artifact | SHA256 |
| --- | --- |
| `lc0-style-baseline-design-draft-v5/proposal.json` | `6917c71ff916fd36e93814a0d8023a54444ef7369ab08e831933ccac76f88055` |
| `core-baseline-v4-candidate-roster-v1/manifest.json` | `8c2f311d3fd7fd13977ea533e4f61bc438a8197a90d8dcb630b85130882c56d1` |
| `core-pipeline-L-two-generation-readout-v2/readout.json` | `73d67cfdc84ba656a3d1fb2f28de62daf14709775b6114059b6c30b64a720e2a` |
| `core-baseline-LR-consumer-results-v1/readout.json` | `b8f0ea746570d93b9431b36a87f96a0e642d74cd1decd746e32b069233961016` |
| `core-baseline-LR-consumer-results-v1/actual-update-LRs.csv` | `c8f7d15a315f1ca07051f723b32cd18621ad4e0d0fa31f0ea9c62e4dcbaa5fcd` |
| `core-baseline-roster-consumer-results-v1-b0db4b6b-2bb3-4753-80d6-96825decbbee/readout.json` | `f137fe768057b22e9c1a114138507ae9563ddd26b63a21023e56edb9ad4ddbb3` |
| `core-baseline-FIFO-full-capacity-results-v1-b3e7298f-2b16-4190-b937-277beb1f11ad/readout.json` | `a64635b513b120684aa089bef456408b751ea27e94a4b1718fa0e50c0c82bd04` |
| `core-baseline-arena-qualification-readout-results-v2-c038b10a-ed3a-471c-a568-067b9700b07c/readout.json` | `e46edd305cfb2fd5878cb91f3be2b4649d4101ee13ff7b4721a8ca83b6432111` |
| `core-baseline-known-prior-arena-union-root-v2/index.json` | `11e98b44c59f9e8cffb03440c93ebf4e5a9e0e14f6b3dca97a7bddd7ea1e69e8` |
| `lc0-style-baseline-design-draft-v6/manifest.json` | `d92a3aec12aa649204a85e54e8c7065d8a0a83151f153e0b315b2d3af4bfdc93` |
| `core-baseline-scientific-freeze-v1/protocol.json` | `dd16774a4cf1cd9f0765501d135355fcac82cb8be7ef09bb27a050d34dbeae49` |
| `core-baseline-scientific-freeze-v1/binding-proof.json` | `15dc4298932e5bc3788ec91a6487698911832f73633a7dbf039d6253fee19b01` |
