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


## 2026-11-13 18:05 UTC — G2 interim readout; experiment ongoing

**ONGOING: clear conditional early regression, no promotion or harm stop.** The
first registered diagnostic compared the generation-2, step-1,024 checkpoint
with immutable original A39. All 256 games / 128 color pairs qualified. The
candidate scored **53 wins, 81 draws and 122 losses: 36.5234375%**, with paired
normal approximate 95% bounds **[31.97931425985534%, 41.06756074014466%]**.
The interval is below 50% under this fixed book and profile. This is not a full
eight-generation verdict, a general recipe or mechanism verdict, or external
Elo. No confirmation is eligible, and the champion remains unchanged.

G1 and G2 together delivered all **512 planned games**, **51,332 audited eligible
rows** (24,827 + 26,505), **1,024 updates** and **262,144 sampled training views**.
These are training-window observations, not a fixed held-out strength ruler.
Unweighted auxiliary loss metrics and label-availability fractions do not mean
those objectives or SF/search value shares were active; the source-qualified
recipe still has only policy and resolved-outcome WDL weights enabled.

| Checkpoint / endpoint | Global updates | Audited rows accumulated | Result / readiness from first G1 owner |
| --- | ---: | ---: | --- |
| G1 final | 512 | 24,827 | checkpoint ready at 1,906.168774 s |
| G2 final | 1,024 | 51,332 | checkpoint ready at 4,163.230231 s |
| G2 diagnostic | 1,024 | 51,332 | score .365234375; CPU-verified evidence ready at 5,138.042137 s (85.6 min) |

The diagnostic itself cost **788.936835 GPU-owner seconds** and **15.467622
CPU-owner seconds**; these are stage wall times, not GPU-active time or the
whole pipeline cost. The banked curve point separates checkpoint-ready and
CPU-evidence-ready times. It does not turn training throughput or loss into a
strength proxy.

Runtime source remains L `090b5143315a0faef7ab0ac1a90e2d5cebf3b929`, with the
pinned R7 owner source `ba9d6de1987902e1759d984532a4c63f091b9269`. The matched
arena used Gumbel64 with noise on, the 175-input BF16/strict-TB profile, an
observed 32-game pool and 1,024 uncapped leaf rows. The reader checked full
history, legality, endings, counts and model custody; independent metadata and
pair arithmetic review accepted the result. Neither numerical identity across
GPU runs nor universal concurrency safety is implied.

The frozen science `dd16774a…` is unchanged. G2 has no promotion or severe-harm
stop rule. At this snapshot, fixed G3 collection was already live from the same
step-1,024 private actor. Continue the predeclared minimum adaptation through
local G4: only **G4 interval upper < .35** triggers the severe-harm stop, aside
from correctness, nonfinite, control or resource failures. Otherwise retain the
fixed G8 endpoint and conditional fresh confirmation. No endpoint, data,
optimizer, recipe or actor selection changed in response to this interim result.
Earlier E4 and donor results remain separate and unpooled; the long-term
Rofchade/Stockfish/Lc0 milestones remain unachieved.

The following external references are relative to the shared
`rl-week-one-20261112-v1` operations artifact root. The trained candidate is
`1ae760db745aebf9129f966b08310a32519e558e628c09c4d16fcb9f3b1b6371`;
original A39 remains
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-research-g01-audit-results-v1/readout.json` | `f5a36590bf16efd3001db06d7f8c0048251c3bd893d5a13bb82a944e9923e7b9` |
| `core-baseline-research-g02-audit-results-v1/readout.json` | `1bead1274fe918376cc540e9d353832ac25f8a8faf3121d2816116a0c0f06c6a` |
| `core-baseline-research-g01-learner-results-v1/boundary.json` | `2c6670d4388e99514f6cdcd412576cdfc45263718747826cd3b2c84317482a8b` |
| `core-baseline-research-g02-learner-results-v1/boundary.json` | `25bc02fa476324eabe540bc2d4aa586016aa1ef7e3100dd2806d30889637a693` |
| `core-baseline-research-g01-delivery-review-v1/review.json` | `d2786886c8690663373e9da6d6743891797c91d45360017feda0a339461bc2a9` |
| `core-baseline-research-arena-leg1-diagnostic-g2-readout-v1/readout.json` | `517cbe31a8ea4dfb60fc6160488845b2642272a54cd43c423ccff0b7b4f6d3cc` |
| `core-baseline-research-arena-diagnostic-g2-independent-actual-review-v1/review.json` | `0a290c63fdba9170eb6687062bfff2d00bd60bd7e3651fe7335379c7c7d15658` |
| `core-baseline-research-arena-bindings-v1/leg1-diagnostic-g2/root-readout-acceptance.json` | `ed5ccaf4bf6a66a8a51aa6517d4de350c8b12d63b1f689a6678e6ee415867cc1` |
| `core-baseline-research-arena-bindings-v1/leg1-diagnostic-g2/curve-point.csv` | `04b1cada968c1f74cda25076f75984770a6c8e6c9c23cfd57dab67ae6a16adbb` |


## 2026-11-13 19:32 UTC — G4 interim readout; no declared harm stop

**ONGOING: technically qualified negative interim result.** The fixed G4
step-2,048 candidate scored **40 wins, 77 draws and 139 losses** against original
A39 over all **256 games / 128 color pairs**: mean **.306640625**, paired normal
approximate 95% interval **[.262414192176604, .350867057823396]**. Endings were
126 rules resolutions and 130 qualified TB resolutions. The conditional score
is clearly below 50% in this matched profile. This is not external Elo, a full
G8 verdict or evidence identifying a failure mechanism.

The exact upper bound **.350867057823396 > .35** means the predeclared strict
G4 severe-harm test is **false**. Rounding the bound to .35 must not change that
decision. Continue fixed G5–G8 unchanged, with G6 descriptive only and G8 the
terminal eligibility endpoint. No promotion, confirmation, checkpoint selection
or LR-probe release follows from this result. The original scientific protocol
`dd16774a…`, learning contract `c53eac58…` and runtime source pins remain unchanged.

G1–G4 delivered all **1,024 planned games**, **102,126 audited eligible rows**,
**2,048 updates** and **524,288 sampled views**. G4 added 25,582 rows and 512
updates. Training-window losses and auxiliary diagnostics are not fixed held-out
strength measures or active objective weights. G2 and G4 intentionally reuse
the diagnostic book; their intervals are conditional on that book and retain
the frozen opening-family/exposure and approximate-calibration limits. These
results are not independent replications or pooled with earlier E4 studies.

| G4 stage / readiness | Seconds |
| --- | ---: |
| Collection GPU-owner wall | 1,522.189310 |
| Audit CPU-owner wall | 66.336517 |
| Learner GPU-owner wall | 431.697826 |
| Learner internal training time, included above | 295.566334 |
| Diagnostic GPU-owner wall | 820.396214 |
| Diagnostic CPU-owner wall | 15.455622 |
| First G1 owner → G4 checkpoint ready | 9,822.157859 |
| First G1 owner → CPU-verified G4 diagnostic | 10,750.592649 |

Owner wall includes its stage overhead and is not CUDA-active time. Internal
training time must not be added to its owner wall; readiness intervals include
pipeline gaps and are not the sum of GPU stages. Neither clock is an ETA or an
Elo-time gain claim.

The arena retained matched Gumbel64/noise-on/strict-TB settings, two independently
loaded 494-tensor models, 175-plane float32 inputs under BF16 autocast and
float32 policy/WDL transport. Its observed pool was 32 games with 1,024 uncapped
leaf rows. Served-state/file custody checks and the strict CPU reader qualified
all games. Runtime L `090b5143315a0faef7ab0ac1a90e2d5cebf3b929` and the R7 owner
source `ba9d6de1987902e1759d984532a4c63f091b9269` remain pinned; no deterministic
GPU identity or universal throughput claim follows. Independent actual metadata
and pair-statistics review accepted these identities, counts and exact harm-rule
arithmetic; it did not rerun game physics or inspect checkpoint tensors.

G4 candidate SHA256 is
`852816fe6dde4b055e226a434ed2580e048cbb87491df65fe1ce87ff33731e54`;
original A39 is unchanged at `a39f900c…`. External references below use the same
operations artifact root as the preceding readout. The full-generation receipt
was banked before the arena and correctly records its then-pending status; the
later strict reader and acceptance supply the completed diagnostic result.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-research-g04-bound-v1/full-generation-readout.json` | `e91ae624da84a2c58f920171778f073571ee73cec55877b2bfd3c813aa7d3323` |
| `core-baseline-research-arena-leg1-diagnostic-g4-readout-v1/readout.json` | `5225848ab2c900341e72a956184e250732a6da2f8e2bed759671683bccb69ecc` |
| `core-baseline-research-arena-bindings-v1/leg1-diagnostic-g4/root-readout-acceptance.json` | `25025e6b00d11c4200956e5383e2fe5b93276a97d430a11c4aca0c996afdcc4b` |
| `core-baseline-research-arena-diagnostic-g4-independent-actual-review-v1/review.json` | `9e07f81530f77a2e053d82a556d6c6dc575c179e4f7f976a0b559a8ef2ca7893` |


## 2026-11-13 21:06 UTC — G6 descriptive readout and actual FIFO eviction

**ONGOING: technically qualified negative G6 interim result.** The fixed
step-3,072 candidate scored **41 wins, 76 draws and 139 losses** against original
A39 over **256 games / 128 color pairs**: mean **.30859375**, paired normal
approximate 95% interval **[.26168684033942846, .35550065966057154]**. Endings were
149 rules resolutions and 107 qualified TB resolutions. The conditional score
is below 50% in this matched profile. G6 is **descriptive only**: it has no
harm-stop rule, LR-probe release, confirmation eligibility or promotion decision.
Continue fixed G7/G8; G8 remains the sole terminal eligibility endpoint before
any same-hash fresh confirmation. No recipe or actor selection changed.

The G4 and G6 points do not establish recovery: these are correlated checkpoints
on the intentionally reused diagnostic roster, not independent replications.
No pooled Elo, external-strength claim, mechanism explanation or full-horizon
verdict follows. The original scientific protocol `dd16774a…`, learning contract
`c53eac58…`, L source `090b5143…` and R7 owner source `ba9d6de1…` remain unchanged.
G1–G6 delivered all **1,536 planned games**, **153,771 cumulative audited rows**,
**3,072 updates** and **786,432 sampled views**. Training-window loss and auxiliary
metrics remain distinct from fixed held-out or playing-strength measurements.

G5 supplied the **first actual FIFO eviction evidence**. Saved membership held
only generations **2–5**, totaling **103,319 rows**; all **131,072** recorded
preaugmentation physical draws resolved to that membership, with **zero G1
draws**. Immutable generation-1 ancestry remained recorded rather than deleted.
G6's source/root-verified window then held generations **3–6**, **102,439 rows**,
with no recorded draw from generations 1 or 2. The cumulative 153,771 rows and
active 102,439-row window therefore measure different things. Journal readback
checks observed membership and physical IDs; it does not independently decode
postmirror tensors, establish statistical uniformity, replay the optimizer or
prove GPU bitidentity. The G5 reader authored the learner, so its journal review
is not a new independent source qualification.

| G6 stage / readiness | Seconds |
| --- | ---: |
| Collection GPU-owner wall | 1,512.690103 |
| Audit CPU-owner wall | 67.401385 |
| Learner GPU-owner wall | 519.114131 |
| Learner internal training time, included above | 296.325541 |
| Diagnostic GPU-owner wall | 808.440529 |
| Diagnostic CPU-owner wall | 15.514475 |
| First G1 owner → G6 checkpoint ready | 15,312.535208 |
| First G1 owner → CPU-verified G6 diagnostic | 16,250.774148 |

Owner wall includes startup/data/seal overhead, not just CUDA activity. Internal
training time is already inside learner wall; readiness intervals include gaps
and do not equal summed GPU time. These are realized costs, not an ETA or an
Elo-per-time improvement. Matched Gumbel64/noise-on/strict-TB serving retained
two models, float32 175-plane inputs under BF16 autocast, float32 policy/WDL
transport, an observed pool of 32 and 1,024 uncapped leaf rows. Source, asset,
served-state and closure checks remain bound; no deterministic-GPU or universal
throughput claim follows. Independent actual metadata and complete pair
arithmetic review accepted the identities, counts and descriptive decision;
it did not rerun game physics or inspect checkpoint tensors.

G6 candidate SHA256 is
`568ea0b702940fb81779442e68e0e62fe5ff8bfbfefe887fceefac182b70c113`;
original A39 remains `a39f900c…`. External references use the same operations
artifact root as above. The generation receipt predates the arena and preserves
its then-pending arena status; the subsequent reader/acceptance provide the
completed diagnostic. Earlier negative E4 and donor results remain separate.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-research-g05-FIFO-independent-actual-review-v1/review.json` | `22681a2c959b03381269a1941b47cda68d2943bb7698247acc2ae580ff1b2e9c` |
| `core-baseline-research-g06-bound-v1/FIFO-journal-readback.json` | `e022be2103901799e1c2cfddddd9516781667aa80bc9bed91d4896bfa1bee5b1` |
| `core-baseline-research-g06-bound-v1/full-generation-readout.json` | `f97d94d7836f673c78a02329978b709bbd67fa4e9f92a0dfbf088cb1f4ec685f` |
| `core-baseline-research-arena-leg1-diagnostic-g6-readout-v1/readout.json` | `7610432a8a22da75365791607fe35b1f62bb53d35f59e8cb11336ef6abc47876` |
| `core-baseline-research-arena-bindings-v1/leg1-diagnostic-g6/root-readout-acceptance.json` | `1783b838533c92b28e9d148d53269bfc6a3894d70d2788624b639e1d5919e6ae` |
| `core-baseline-research-arena-diagnostic-g6-independent-actual-review-v1/review.json` | `1c5b46768cbc124dad7cb42e6476f8141732ebca926bb3d9a74ad025cfca2ebb` |


## 2026-11-13 22:57 UTC — G8 terminal nonpass; no confirmation or second leg

**LEG 1 COMPLETED: technically qualified terminal negative.** The fixed G8
step-4,096 candidate scored **52 wins, 120 draws and 340 losses** against
immutable original A39 over all **512 games / 256 color pairs**: mean **.21875**,
paired normal approximate 95% interval
**[.19030425270421453, .24719574729578547]**. Pair-total counts for
{0, .5, 1, 1.5, 2} were {111, 82, 48, 14, 1}. Endings were 324 rules resolutions
and 188 qualified TB resolutions. The strict technical reader passed; the
registered strength endpoint **did not pass** either gain threshold.

No confirmation was eligible or consumed, no C1 was established, and the
conditional second leg did not run. There is no alternate-checkpoint fallback,
promotion, adoption or external-engine win claim. The long-term strength goal
and two-successive-gains milestone remain unmet. This is a negative result for
the frozen eight-generation recipe, data, horizon and playing profile; it does
not establish an optimal recipe, identify a causal LR failure or condemn all
policy/outcome RL. Correlated G2/G4/G6 diagnostics are not pooled with this
terminal panel or treated as independent replications. Earlier E4/donor records
remain separate.

G7 and G8 each completed all 256 planned collection attempts, audit and 512
learner updates with the preplanned previous-generation actor. G7 had no
scheduled arena; G8 alone supplied terminal eligibility. The final leg totals
were **2,048 games, 203,794 audited eligible rows, 4,096 updates and 1,048,576
sampled views**. Actual active FIFO membership continued to advance:

| Generation | New eligible rows | Active generations | Active rows | Final global step |
| --- | ---: | --- | ---: | ---: |
| G7 | 24,982 | 4–7 | 102,209 | 3,584 |
| G8 | 25,041 | 5–8 | 101,668 | 4,096 |

The G8 journal checked all 512 × 256 recorded preaugmentation physical draws
against the exact saved 5–8 membership; none came from evicted generations 1–4.
Independent pipeline/FIFO metadata review accepted closure and lineage. It did
not decode postmirror tensors, replay optimizer arithmetic, establish statistical
uniformity or re-read checkpoint payloads. Training-window losses remain
training observations, not a held-out strength proxy.

The terminal arena used **1,504.941275 GPU-owner seconds** and **23.660658
CPU-owner seconds**. From the first G1 owner, G8 checkpoint readiness took
**21,429.064381 seconds**, and qualified terminal evidence took
**23,030.740921 seconds**. These distinguish stage wall, checkpoint readiness and
evidence readiness; none is CUDA-active time or an ETA. Matched Gumbel64/noise-on,
175-plane float32 input/BF16-autocast, float32 policy/WDL transport and strict-TB
serving retained the observed 32-game pool / 1,024 uncapped leaf-row profile.
The original `dd16774a…` science, `c53eac58…` contract, L `090b5143…` and R7 owner
`ba9d6de1…` remain unchanged. No GPU determinism or universal throughput claim
follows. Independent actual metadata and complete 512-game/256-pair arithmetic
review accepted the terminal nonpass and custody joins, reusing the qualified
strict CPU physics readback; it ran no new probes or checkpoint/array reads.

At this readout, the separately source-accepted ten-times-lower-LR probe on the
same G1 data/seed for 512 updates plus a direct 256-game diagnostic remained
**unrun and not freshly admitted**. It is a separate exploratory question with
its own closure/admission dependencies, not a completed result or a change to
this frozen baseline.

G8 candidate SHA256 is
`47bed4f13e32340266d597d4fd4b7c40d2f13c60c2461d1b466fd2c476ad1689`;
reference A39 remains `a39f900c…`. External references use the same operations
artifact root. Pipeline receipts preserve their original pre-arena statuses;
the subsequent terminal reader and acceptance supply the completed verdict.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-research-g07-bound-v1/full-generation-readout.json` | `9cb8133b5b269fb1e98f8b889c4f818d1ed81053550bd5e54e00c2c3adf111f6` |
| `core-baseline-research-g08-bound-v1/full-generation-readout.json` | `886679646b699a0d01ec20af5e317aaca30454c97c1348f68b9336b52b0c2171` |
| `core-baseline-research-g08-bound-v1/FIFO-journal-readback.json` | `138e0a4544cc384322cad83b4c999b1fa8617d7fe499369aaf3c2860bae59de9` |
| `core-baseline-research-g08-pipeline-FIFO-independent-actual-review-v1/review.json` | `81a5f0aaf69985fbaedd4e4643cd4b8042a2caf8650a3fafc1671d843cb6df3f` |
| `core-baseline-research-arena-leg1-terminal-readout-v1/readout.json` | `3b88d5f4bff1d2c569015cd139c94f1e3c357a3a4da047fd3699cc4e410987f9` |
| `core-baseline-research-arena-bindings-v1/leg1-terminal/root-readout-acceptance.json` | `d7dfefc02bba8e785c92dfc4171db707ba6b83c13704694382c5ca003cbc0173` |
| `core-baseline-research-arena-leg1-terminal-independent-actual-review-v1/review.json` | `b048e12250521abea3857c7b7544e99615ceb6b2e655fd4297796b082e5e46b6` |
