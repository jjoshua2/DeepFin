# Full-state lower-rate on-policy continuation

Current status at **2026-11-14 08:20:47 UTC: ONGOING through the fixed G8 horizon**.
G5/G6 data and learner continuation completed through step 3,072. The qualified
G6 donor diagnostic scored **43.359375% over 256 games, paired 95% interval
38.4232053599–48.2955446401%**: negative descriptive underperformance on this
conditional reused DIAG128 panel. It is not a clean/general regression or causal
LR verdict. G6 has no efficacy or severe-harm gate; fixed G7/G8 remain unchanged.
No gain, confirmation, model promotion or adoption is claimed. Root is preparing
G7; its actual data/models are unknown here. See the [G5/G6 readout](#2026-11-14-completed-g5-and-g6-negative-descriptive-g6-diagnostic).

## Earlier status snapshot — 2026-11-14 06:22:53 UTC

Current status at **2026-11-14 06:22:53 UTC: ONGOING through the fixed G8 horizon**.
G3 and G4 collection, independent data audits and full-state learner continuation
completed through step 2,048. The qualified G4 donor diagnostic scored
**52.9296875% over 256 games, paired 95% interval 48.1105005605–57.7488744395%**.
It is descriptive: no supported gain, equivalence, clean regression, promotion
or confirmation. The registered G4 severe-harm condition is false. Root reports
fixed G5 collection running from that same step-2,048 actor; no G5 data or model
result is claimed. The strength goal remains unmet. See the [G3/G4 readout](#2026-11-14-completed-g3-and-g4-descriptive-g4-diagnostic).

## Earlier status snapshot — 2026-11-14 04:04 UTC

Current status at **2026-11-14 04:04 UTC: ONGOING**. G2 collection, independent
CPU data audit and full-state learner continuation completed. The fixed G2
A39 diagnostic was technically qualified: **82W/88D/86L over 256 games,
score 49.21875%, paired 95% interval 44.2989710828–54.1385289172%**. It is
descriptive, with no gain, equivalence, clean regression, promotion or
confirmation claim. G3 collection is running from the same step-1,024 actor;
that status was supplied by the owning run, without inspecting G3 data here.
The strength goal remains unmet. See the [dated G2 readout](#2026-11-14-g2-completion-and-descriptive-diagnostic).

## Preregistration snapshot — before adoption and new data

The following prospective text and evidence retain their original publication
status. The dated readout below records later adoption and execution; it changes
no scientific rule, source pin, role, count, seed or horizon.

Status: **PROSPECTIVE REVIEWED PROPOSAL — NOT FROZEN, ADOPTED, ADMITTED OR
EXECUTED.** V2 source and the actual prospective role proposal have independent
acceptance. The scoped recorded-use audit is evidence, not role release.
Publication before root adoption/new data and fresh per-stage execution
admissions remain separate. No new training, predictions or actual endpoint
bindings exist yet.

Study UUID: `0803b7c9-a68d-4fa1-8ce3-4ccf6d8c0b4b`. Can the unchanged
lower-rate regime improve internal playing strength at a fixed longer horizon
with genuinely new current-actor games? Start from complete saved LOW512 state,
SHA256 `5d5c93cb669e24e3e97848960ea9bcf78c6f7880327a896aa7d2a30693641b53`,
including model, NadamW optimizer, scheduler, global step, clipping state and
boundary RNG. Original immutable A39 is the first-leg reference.

The [existing LOW512/A39 screen](2026-11-14-low512-donor-screen.md) remains
qualified **NONPASS**: 147W/207D/158L, score .4892578125, paired 95% interval
[.45818409026139667, .5203315347386034], no allocated confirmation or adoption.
This study does not rescreen that unchanged checkpoint on another book/seed.
The [original eight-generation baseline](2026-11-13-core-outcome-policy-baseline.md)
remains negative. The [fixed-G1 LR/WDL observations](2026-11-13-fixed-g1-lr-g2-wdl.md)
motivate, but do not establish donor-relative gain from, this **joint on-policy
data-and-dose continuation**. It is not an isolated causal LR/data comparison
or an optimum search. No old C1 or leg-2 history is rewritten.

## Fixed learner and new collection lineage

Retain the existing contract at its original external location (shown relative
to the operations artifact root): `core-baseline-fixed-G1-LR-probe-preparation-v1/learning-lowLR.json`,
SHA256 `cc17ac07c143eec18a42f28182ea21909ad8eb8a2a53fa844a3e201ba30da0ca`.
The actual absolute path is part of the closed-parent command identity: copying
identical bytes to another path fails that check. Keep learner/data provenance
`b0db4b6b-2bb3-4753-80d6-96825decbbee` and original G1 manifest `35279e2e…`;
the new scientific UUID does not rename that lineage. Preserve the existing
sixteen-entry scheduler definition, even when only the first leg is admitted.

Restore the complete step-512 parent, reconstruct admitted replay in fixed
source order, then restore boundary sampling/mirror, global NumPy, Python and
Torch CPU/CUDA RNG. Do not reset or re-warm the effective schedule. Replay is
reconstructed, not serialized; this is quiescent generation-boundary recovery,
not interrupted-epoch replay or a GPU bitidentity claim.

Collect seven new generations 2–8, each from its exact preceding low-rate final
checkpoint; the sole first-leg candidate is generation 8 / global step 4,096.
Every generation keeps all 256 fixed attempts, needs 192–256 resolved nonempty
games and at least 10,000 eligible rows, and applies 512 updates of batch 256
(131,072 views). There are no replacement games. Never relabel original
high-actor G2 as low on-policy data. Original G1 is inherited, not regenerated.

Only fresh confirmation of that exact step-4,096 model permits continued full
state through generations 9–16 to step 8,192, with no checkpoint selection.
Retain schema-20/175-plane architecture, policy CE 1 and genuine-outcome WDL CE 1,
all other twelve objectives 0, SF/search WDL shares 0, target rebuilds/dropout/
SWA off, no clipping with finite-gradient refusal, and the same NadamW identities,
mask, decay, backend and strict full-history rule50/TB/search/audit profile.
Diagnostic SF capture is unchanged; zero SF loss does not mean no SF queries.
Uniform physical rows are drawn with replacement from the last four whole
generations, sharing the mirror-.5 RNG. LR peak 3e-6, floor 6e-7, warmup start
1e-7, original 256-update warmup and period 16,384 remain fixed, without restart.
LR also scales decoupled decay exposure; it does not isolate overshoot.

### Preassigned collection identities

The inherited G1 used collection UUID `feed4977-74dc-4065-8c04-8c1572012979`,
seed `2026111501`, TRAIN half 1 and the once-initialized learner seed
`2026111520`. The following fifteen identities are new. G9–G16 are conditional
on the first fresh confirmed gain; configured seed reuse does not imply shared
whole-trajectory RNG or independent replicate evidence.

| Generation | New collection UUID | Seed | TRAIN half | Final global step |
| --- | --- | ---: | ---: | ---: |
| 2 | `a2e38b8c-78c8-5ca4-81a5-e45ef5fd8969` | 2026111502 | 2 | 1024 |
| 3 | `4ba80f6e-f67d-5e56-a50b-7fe9a771f405` | 2026111503 | 1 | 1536 |
| 4 | `7a43256b-8451-546a-a5e8-bcc70866f83e` | 2026111504 | 2 | 2048 |
| 5 | `5c0e801b-3ea3-5a7b-b246-d314051b6110` | 2026111505 | 1 | 2560 |
| 6 | `56b1a606-758b-5676-89bb-bfa302922c4a` | 2026111506 | 2 | 3072 |
| 7 | `02982ca7-9ddb-5931-804b-d525c89acd57` | 2026111507 | 1 | 3584 |
| 8 | `8f4ff646-2969-5c27-8347-465996c1e088` | 2026111508 | 2 | 4096 |
| 9 | `d4b34d87-e1ec-5cd0-a771-e5f5bb013519` | 2026111509 | 1 | 4608 |
| 10 | `fd21d97b-ea32-529d-819e-e8a5152f3399` | 2026111510 | 2 | 5120 |
| 11 | `953c5379-60d9-51df-b1fe-95c65ba5cb15` | 2026111511 | 1 | 5632 |
| 12 | `71839aa3-a445-501c-9f6b-62bc2f734950` | 2026111512 | 2 | 6144 |
| 13 | `02e2cf1f-54c5-57bd-a480-4314096cc53d` | 2026111513 | 1 | 6656 |
| 14 | `b221e99d-4db8-5396-b59a-bc5f6bbaad98` | 2026111514 | 2 | 7168 |
| 15 | `5a3a2a53-5f66-58eb-b278-f0472fbb2c50` | 2026111515 | 1 | 7680 |
| 16 | `5d7c699d-8f03-5cc6-88dc-6341c1123fd7` | 2026111516 | 2 | 8192 |

## Fixed playing endpoints

All endpoints use the pinned matched Gumbel64/noise-on/temperature-.1/eager-BF16,
float32 transport, 32-game rolling / 1,024-leaf, 450-ply strict 3–6-piece TB
profile. The eleven endpoints below are all fixed before new data. Candidate
is the corresponding diagnostic step, fixed final 4,096 or fixed final 8,192;
`C1` means the exact freshly confirmed step-4,096 model, not original dd167 C1.
Caps below are GPU and CPU **outer / inner** seconds, not runtime forecasts.

| Endpoint | Reference | Bank | Games / pairs | Seed | GPU cap | CPU cap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `leg1-diagnostic-g2` | A39 | DIAGNOSTIC | 256 / 128 | 2026111532 | 1800 / 1770 | 1800 / 1740 |
| `leg1-diagnostic-g4` | A39 | DIAGNOSTIC | 256 / 128 | 2026111534 | 1800 / 1770 | 1800 / 1740 |
| `leg1-diagnostic-g6` | A39 | DIAGNOSTIC | 256 / 128 | 2026111536 | 1800 / 1770 | 1800 / 1740 |
| `leg1-terminal` | A39 | CONTEXT256 | 512 / 256 | 2026111551 | 3600 / 3570 | 1800 / 1740 |
| `leg1-confirmation` | A39 | CONFIRM1 | 1000 / 500 | 2026111561 | 7200 / 7170 | 3600 / 3540 |
| `leg2-diagnostic-g2` | confirmed C1 | DIAGNOSTIC | 256 / 128 | 2026111542 | 1800 / 1770 | 1800 / 1740 |
| `leg2-diagnostic-g4` | confirmed C1 | DIAGNOSTIC | 256 / 128 | 2026111544 | 1800 / 1770 | 1800 / 1740 |
| `leg2-diagnostic-g6` | confirmed C1 | DIAGNOSTIC | 256 / 128 | 2026111546 | 1800 / 1770 | 1800 / 1740 |
| `leg2-terminal` | confirmed C1 | CONTEXT256 | 512 / 256 | 2026111552 | 3600 / 3570 | 1800 / 1740 |
| `leg2-confirmation` | confirmed C1 | CONFIRM2 | 1000 / 500 | 2026111562 | 7200 / 7170 | 3600 / 3540 |
| `post-confirmed-C2-A39-context` | A39 | CONTEXT256 | 512 / 256 | 2026111570 | 3600 / 3570 | 1800 / 1740 |

Every primary screen and confirmation separately requires **point ≥ .53 AND
paired 95% lower > .5**, using complete color-pair means and unchanged `42eb7f46…`
statistics: unbiased n−1 variance, z=1.96 normal approximate bounds for positive
variance; zero-variance radius sqrt(log(40)/(2n)), with bounds clamped to [0,1].
Approximate calibration and opening-family dependence remain; no exact/global
error guarantee is asserted. The first screen uses initially unused-in-recorded-
coverage CONTEXT256; the second uses that same bank as explicitly **nonfresh
eligibility only**, not a fresh independent replication.

Each eligible, technically qualified screen permits at most one fresh same-hash
1,000-game confirmation, on CONFIRM1 or CONFIRM2 respectively, against its fixed
leg reference. Allocation consumes the attempt even on preflight/technical
failure. No pooling, favorable early stop, checkpoint/seed choice, adaptive
extension, fallback, technical reroll or automatic adoption is allowed. Fault,
cap or incomplete data is unread, not a clean playing verdict.

Diagnostics at global G2/G4/G6 compare with A39; conditional G10/G12/G14 compare
with confirmed C1. Only **G4 and G12 upper < .35** against their leg reference
have the predeclared four-generation severe-harm stop. All other diagnostic use
is descriptive, with no favorable early promotion. This deliberately permissive
recipe-specific continuity/safety rule preserves minimum adaptation; it is not
a universal efficacy gate. Technical, provenance, nonfinite, skip/retry, cap,
control, resource and deadline failures remain stopping conditions throughout.

The post-C2/A39 context is **fixed and conditional**, not optional: it runs only
after fresh confirmation of exact C2/8,192, uses that same model against original
A39 on reused CONTEXT256, and is descriptive with no gain/selection gate. It
provides a direct common-reference point rather than adding predecessor Elo.
Even two future internal confirmations would not establish unhandicapped
multicore Rofchade/SF/Lc0 strength, LR optimality or additive/transitive Elo.
Those long-term milestones remain unmet.

## Roles, scoped evidence and recovery

The actual prospective disposition proposes intentional **nonfresh TRAIN and
DIAGNOSTIC reuse** and retirement of only the original never-eligible
CONTEXT/CONFIRM1 conditional claims after reviewed root adoption. CONFIRM2 is assigned but unplayed under the
completed cda successor; its unallocated closed conditional claim must be
retired before reassignment here. CONTEXT's first scoped-fresh use and later
precommitted nonfresh screen/context uses must be distinct. TERMINAL2 is played
and stays closed; all other roles remain reserved. The bound proposal
has independent review acceptance but is not adoption; root adoption and each
actual phase admission are separate from this publication.

The immutable updated audit covers **190 current + 32 preflight recorded owners**,
all recorded closed, and the two new TERMINAL2 screen/readback uses. TRAIN,
DIAGNOSTIC and TERMINAL1 were also used. No declared recorded predictions use
CONTEXT256/CONFIRM1/CONFIRM2 within that coverage; current metadata intersections
are zero and all 1,896 nonTRAIN starts match the retained training-exclusion
start guard. This does not establish universal novelty/non-use or independent
opening families. External/unbanked use, unretained search, donor pretraining
and unrecorded human/tool use remain **UNKNOWN**. The lightweight census is a
point-in-time observation, not resource ownership or a release of roles.

| Planned resource ceilings | GPU-owner seconds | CPU-owner seconds |
| --- | ---: | ---: |
| First leg: seven new collection/audit/learner sequences plus endpoints | 54,000 | 23,400 |
| Conditional second leg | 59,400 | 25,200 |
| Fixed post-confirmed-C2/A39 context | 3,600 | 1,800 |
| Maximum new total | 117,000 | 50,400 |

At the banked closed 190-current-owner cost census, charged GPU time was
**35,858.514620649104 seconds**, leaving **136,941.4853793509** of the unchanged
172,800-second week cap; full new ceilings leave 19,941.4853793509. Deadline is
**2026-11-19 16:09:20 UTC**. These are not ETAs, entitlements or guarantees that
all sixteen generations fit. Cold reconstruction traverses prior admitted
history before eviction, and future NNZ/RSS/seal I/O may fail closed.

Use one owner at a time, affinity 20,22 and two Torch/OMP/MKL threads. Require
fresh MemAvailable ≥ 64 GiB before each learner and stop below 16 GiB; retain
460,800 rows / 24 GiB decoded-window limits without dropping rows. Each phase
needs current controls, custody, resources and separate root admission.
Preserve all partial/failed artifacts and exact closed-boundary recovery; never
retry a consumed scientific attempt or silently widen caps. Reviewed timestamped
ready-PR publication may precede predictions, with honest main/CI status; no new
mandatory main-merge gate is invented. Publication alone launches nothing.

## Readiness and external identities

Final V2 source has independent full-source acceptance and root static review.
V1 remains preserved **UNRUN** with two metadata blockers: duplicate wrapper/core
`--admission` parsing and a PID-only historical-supervisor absence check. V2
limits flag parsing to the wrapper prefix while comparing the exact full command,
and accepts a present supervisor PID only when its birth is strictly later than
the earliest authenticated child birth, retaining exact child checks. These
source-only corrections change no numerical/scientific/UUID/seed/role/cap rule;
they are not runtime qualification or executed controls.

The actual role proposal and disposition are bound to V2 and the unchanged
6438 audit; independent role acceptance covers the scoped proposal only, not
release or adoption. Root adoption and actual phase bindings remain future
actions. This prospective record precedes adoption and new data; it cannot
include a future adoption hash. No models, arrays, project imports, tests, jobs or role writes were used
to author this record. References are relative to `rl-week-one-20261112-v1`
operations artifacts; source/audit/proposal evidence creates no execution authority.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-lowLR-on-policy-continuation-selection-v1/decision.json` | `3499f4c25efb8191449178937a35cb7ba1694fd940964e9099ec3a9a7da4d4bf` |
| `core-baseline-lowLR-continuation-preparation-v1/preparation.json` | `5bd2579f5667fb9b82685d95fa0e332ccfcbe2e489dd6a2fe7bc126735241bc8` |
| `core-baseline-lowLR-on-policy-continuation-independent-feasibility-review-v1/review.json` | `1abbc4aeb7e727e4d4ae8aca9120f4b671e033f465fb37cedff6cc679d59593a` |
| `core-baseline-lowLR-continuation-recorded-use-audit-v1/readout.json` | `6438c835f7bc2ea9adc4609b7961ef26d78db8ac20dacb2563f594803400b5ff` |
| `core-baseline-lowLR-continuation-preparation-v2/role-bank-identities.json` | `e43ee07e969e6f3aa81cf601e38bb7b8852a3073c1a8adbc83d028a61c7c1502` |
| `core-baseline-fixed-G1-LR-probe-preparation-v1/learning-lowLR.json` | `cc17ac07c143eec18a42f28182ea21909ad8eb8a2a53fa844a3e201ba30da0ca` |
| `core-baseline-fixed-G1-LR0p1-learner-results-v1/boundary.json` | `533a7ca081f34bd6daa732caa6c0b3b17f17ceb22c586abc6f60b01e82513b21` |
| `core-baseline-low512-screen-closed-cost-census-v1/readout.json` | `7b53a0344c3ec5a3edfc3a930cdb72255421f5ad7d4285edcb448df635a615f0` |
| `core-baseline-lowLR-continuation-preparation-v2/preparation.json` | `1b11b6380be046265485871e35cafac6257a1c99e531cda90559db3e5167e278` |
| `core-baseline-lowLR-continuation-preparation-v2/manifest.json` | `b10f5f5becd47a422359523ada96283b1d6805b68543035d8522a5a258038dd4` |
| `core-baseline-lowLR-continuation-preparation-v2/specification.json` | `12ad1fc72714a3eb75c1adc519b7e56a8c80e40e9a8f0e45708479bbb8bc226e` |
| `core-baseline-lowLR-continuation-independent-source-review-v2/review.json` | `29540596be696b0c81e14a68e848931826e7f68594ea5acb03e7aa86fb59cae3` |
| `core-baseline-lowLR-continuation-root-preparation-review-v1/final-v2-source-review.json` | `3012162c991cb1475ff29b0dc24edd8a85290b37db70cef11a3f94729d6162d7` |
| `core-baseline-lowLR-continuation-bindings-v1/role-proposal.json` | `1c518701cef61fdec56d547b1a846a8e55c074d4b247ae3141f31fe2d6783a78` |
| `core-baseline-lowLR-continuation-bindings-v1/role-disposition.json` | `fb9fccf3615ebd6e8aa1fdc3ec3054cd23d83375e69ec315fe490933a7fe4023` |
| `core-baseline-lowLR-continuation-bindings-v1/role-disposition-review.json` | `43be8bb2268e3fe4a0a2935ca60b2bf948107286543b517c00e614a762002044` |
| `core-baseline-lowLR-continuation-role-disposition-independent-review-v1/review.json` | `6971a0e797b10601f4cfa7e66f6109c5e5aa96876d28567c59d2ae24598d6cce` |


## 2026-11-14: G2 completion and descriptive diagnostic

**Cutoff: 2026-11-14 04:04 UTC.** The reviewed ready PR was published and its
remote bytes verified at 02:20:42 UTC, before root role adoption at 02:22:39
and new data. PR [#1289](https://github.com/jjoshua2/DeepFin/pull/1289) subsequently
merged at `bf0afdb410126a1ffaf7878b3cd691b0ec8bfa9e`, with all four CI jobs
passing and complete feedback checks empty. That main merge occurred **after
G2 collection started**; it is not retroactively described as pre-launch main
publication. The V2 specification `12ad1fc7…`, manifest `b10f5f5b…`, original
`cc17ac07…` contract path, L `090b5143…` and owner R7 `ba9d6de1…` remain pinned.
Later changes to main are not runtime adoption.

| Completed stage | Actual delivery | Owner wall seconds |
| --- | --- | ---: |
| New low-parent G2 collection | All 256 fixed attempts; no replacement | GPU 1,589.507718 |
| Independent CPU data audit | All 256 eligible; 26,010 rows; 107 natural / 149 strict-TB / 0 unresolved | CPU 66.892230 |
| Full-state learner continuation | Step 512 → 1,024; 512 updates × batch 256 = 131,072 views | GPU 358.975789 |
| Fixed G2 versus A39 diagnostic | All 256 games / 128 color pairs; 117 rules / 139 strict-TB endings | GPU 659.474557; CPU readback 14.453168 |

The data promotion `c90f281e…` admits the closed audited generation only; it
promotes no model or champion. Original G1 plus new low-parent G2 total
**512 games / 50,837 audited rows / 1,024 updates / 262,144 views**, including
the inherited first 512 updates. Original high-actor G2 was not relabelled or
used. The new model is `learner.pt` SHA256
`89e9804d23384dd8833609fd59f277200e24ff1167d249dda63977a99057e1c0`,
boundary SHA256 `46d7ec5eb04f32bb9fc5da48357613499a8b92540f261cca0ba07e4e2f232a44`.

The pinned learner restores complete saved LOW512 model/optimizer/scheduler/
step/peak-LR/ZClip state, reconstructs replay, then restores boundary RNG before
sampling. Its closed boundary retains the exact parent, backend and contract.
Independent metadata review reconstructed all **512 shard chunks / 50,837 rows**
and checked every journal draw: 64,538 inherited-G1 and 66,534 new-G2 draws,
all 512 complete batches, no out-of-window source or row. Root separately
verified the checkpoint's full-file hash; the independent review did not reload
tensors or assert GPU bitidentity/interrupted-epoch recovery.

Learner metrics recorded zero nonfinite skips, retries and clips; mean gradient
norm 4.484922 and mean LR 2.993775905e-6. Mean total loss 1.567210148 agrees
with policy plus outcome-WDL loss within 4.8e-9. Nonzero auxiliary loss or
SF/search availability metrics do not make those objectives active: the pinned
consumer retains policy 1 / genuine-outcome WDL 1, twelve other weights 0 and
SF/search WDL shares 0. These training-window metrics are not heldout or strength
proxies and do not establish an LR failure mechanism.

The diagnostic independently reconciled all 257 JSONL records, 256 PGNs and
128 opening/color pairs. **82W/88D/86L gives .4921875**, unbiased paired variance
.08064714566929133 and normal-approximate 95% bounds
[.44298971082813166, .5413852891718683]. Pentanomial counts in ascending pair-total
order 0/.5/1/1.5/2 are **[13, 31, 45, 25, 14]**. Both owners closed PASS. Actual
serving observations bind the unchanged step-1,024 candidate and immutable A39,
494 states / 61,444,448 parameters each, strict embedded architecture,
175-plane BF16 inference with float32 transport, pool 32 and leaf capacity 1,024.
The admitted CPU reader verified full-history legality and rule50-aware TB endings;
the independent metadata review reused that physics without new probes.

DIAG128 is intentionally reused and the interval contains parity. This supplies
**no supported gain, equivalence or clean regression**, no allocated/consumed
confirmation, no promotion, and no causal LR or general-recipe verdict. Do not
pool it with prior correlated diagnostics, the original negative G8, or the
completed LOW512/A39 nonpass. G2 has no severe-harm or promotion decision.
At this cutoff G3 was collecting with the same `89e9804d…` actor under separate
root admission; the predeclared G4 severe-harm exception, fixed G8 endpoint,
conditional confirmation and all technical/resource/deadline stops remain.
No recipe change or external Rofchade/Stockfish/Lc0 milestone follows this readout.

The references below are relative to the same operations artifact root as the
preregistration table. Root receipts that were sealed before independent review
retain their historical pending labels; the corresponding completed independent
receipts are separate. Reviewers share source history with the collector, as
disclosed in their receipts. No model/array payloads or project imports were used
for this documentation update.

| Completed evidence | SHA256 |
| --- | --- |
| `core-baseline-lowLR-continuation-prereg-main-publication-v1/publication.json` | `6c6eaa3f226d10f242cb8e6dec7e56be4532c3dad55fc29961f9be1afc191d0a` |
| `core-baseline-lowLR-continuation-bindings-v1/role-adoption.json` | `dc1ff297fb4edf0ad84b965d11a18217f1b154e775edde158c11c4b448653be1` |
| `core-baseline-lowLR-g02-collection-GPU-execution-v1/terminal.json` | `a7a3e67a952555d8b0876707047d3f217c06b23e288dde8f0072653f5680ea4c` |
| `core-baseline-lowLR-g02-audit-results-v1/readout.json` | `9f2d582c716a708de9f1545b7ae5e822d15a98c904a6ba6ad9e201bee4aa1d25` |
| `core-baseline-lowLR-g02-audit-independent-actual-review-v1/review.json` | `5e32a39974ccfc88b9b143514f69bfa2f20330164fa8d030265d603935e1b546` |
| `core-baseline-lowLR-continuation-bindings-v1/g02/audit-root-promotion.json` | `c90f281e0826019b9fc38c4cd62a46bc428a218fae3d3b0ac9e2dd34ad9177e8` |
| `core-baseline-lowLR-g02-learner-results-v1/boundary.json` | `46d7ec5eb04f32bb9fc5da48357613499a8b92540f261cca0ba07e4e2f232a44` |
| `core-baseline-lowLR-continuation-bindings-v1/g02/learner-root-metadata-acceptance.json` | `83d5f2c7f3eaf2acb323730424aebed1fb8b7866963f1b412049ecdb715d2027` |
| `core-baseline-lowLR-g02-learner-independent-actual-review-v1/review.json` | `b2dd20155c621a56df354e8147d3277fb4cf425c6a1419e37175c34ad5eaf77b` |
| `core-baseline-lowLR-leg1-diagnostic-g2-readout-v1/readout.json` | `b0d1e48b777c384e9a4f9b18efedcfd4ab045f9cc66974522b016e2fa6111e16` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g2/root-readout-acceptance.json` | `00b9e05443840628ebf300efe72c00485b44ccf1fbd29532fc7a7932efe49b82` |
| `core-baseline-lowLR-leg1-diagnostic-g2-independent-actual-review-v1/review.json` | `12c2399e2d209f4a3badc1f18b04378cc1b47a21061e1a2623c245bc18d0c374` |


## 2026-11-14: Completed G3 and G4; descriptive G4 diagnostic

**Cutoff: 2026-11-14 06:22:53 UTC.** G3 and G4 completed on the unchanged V2
specification `12ad1fc7…`, original `cc17ac07…` contract path, L `090b5143…` and
owner R7 `ba9d6de1…`. Independent reviews accepted the closed data and learner
metadata, followed by the complete fixed G4 diagnostic. No scientific rule,
source, role, seed, checkpoint-selection rule or horizon changed.

| Completed stage | Actual delivery | Owner wall seconds |
| --- | --- | ---: |
| G3 low-parent collection | All 256 fixed attempts; 104 natural / 152 strict-TB / 0 unresolved | GPU 1,632.392473 |
| G3 independent CPU data audit | 256 eligible games / 25,757 rows; data-only root promotion `7bc8ba9c…` | CPU 70.882020 |
| G3 full-state learner | Step 1,024 → 1,536; 512 updates / 131,072 views | GPU 437.560464 |
| G4 low-parent collection | All 256 fixed attempts; 111 natural / 145 strict-TB / 0 unresolved | GPU 1,885.560283 |
| G4 independent CPU data audit | 256 eligible games / 26,303 rows; data-only root promotion `e80d8600…` | CPU 76.088461 |
| G4 full-state learner | Step 1,536 → 2,048; 512 updates / 131,072 views | GPU 464.058021 |
| Fixed G4 versus A39 diagnostic | All 256 games / 128 color pairs; 113 rules / 143 strict-TB endings | GPU 767.041080; CPU readback 15.532970 |

G3's checkpoint is SHA256
`8eb3aa69fde8da409c0a9c211e251d85c52f7c18aa403061ba5d750807290def`,
boundary `514c95df1520bb66910cdedff256ba9053d6ba0b93f4111200b074a4b23ba81f`.
Inherited G1 plus low-parent G2/G3 totalled **768 physical games / 76,594 audited
rows / 1,536 updates / 393,216 views**. Its 512 complete batch-256 draws comprised
41,980 G1, 44,980 G2 and 44,112 G3 rows, covering all 768 chunks and 62,752
distinct sampled physical rows, with no out-of-window source or row.

G4's checkpoint is SHA256
`8fe45bf71be92cbd2ba06fc1acd0104f5be397b24411b7b15d532c76fed28aff`,
boundary `b8d41968dbb3e1099abdce48236913d7471703ac37e0766ffc1ff397f10e6f35`.
Cumulative inherited G1 plus low-parent G2–G4 are **1,024 physical games /
102,897 audited rows / 2,048 updates / 524,288 views**. All four generations
remain in FIFO; no generation has yet been evicted. The G4 journal reconciled
all 512 batches / 131,072 draws: **31,891 G1, 33,318 G2, 32,422 G3 and 33,441 G4**,
all 1,024 chunks and 74,129 distinct sampled physical rows. No high-actor G2
was relabelled or used. The closed G4 CPU audit reports 1,446,741,895 decoded
bytes; policy-TB eligible/applied/abstained retained-record counts were all zero.
The 145 strict-TB final outcomes were verified, without a new retained-row
TB policy-correction exercise.

Both continuations retain full saved state, exact parent/contract/backend and
boundary RNG metadata under the qualified source. Root verified complete
checkpoint bytes; independent metadata/journal review did not reload tensors,
recompute GPU/RNG transitions or establish GPU bitidentity. G4's 512-step metrics
were finite, with skips/retries/clips all zero, mean gradient norm 5.245727 and
mean LR 2.947967400e-6. They describe a changing training window, not fixed-heldout
or playing-strength evidence. Nonzero auxiliary diagnostics remain separate
from the unchanged policy-1/outcome-WDL-1 objective and twelve zero weights.

### Fixed G4 diagnostic and continuation decision

The scheduled step-2,048 model faced immutable A39 on intentionally reused
DIAG128, seed **2026111534**. Complete paired readback and independent review
reconciled **88W/95D/73L over 256 games / 128 color pairs**, score **.529296875**,
unbiased paired variance **.07738296628937008**, and approximate 95% bounds
**[.4811050056052596, .5774887443947404]**. Pentanomial counts in ascending
pair-total order 0/.5/1/1.5/2 are **[12, 24, 41, 39, 12]**. All 256 PGNs and
24,237 additional SAN plies joined the fixed histories; endings were 113 rules
and 143 strict-TB. Both GPU and CPU owners closed PASS. Recorded serving
remained 494 states / 61,444,448 parameters per model, strict embedded
architecture, 175-plane BF16 inference with float32 transport, pool 32 and leaf
capacity 1,024. No model load or new TB probe was needed for independent review.

The interval contains parity. This establishes **no supported gain, equivalence
or clean regression**, causal LR/optimality conclusion, general-recipe verdict
or external Rofchade/Stockfish/Lc0 milestone. It is descriptive, with no pooling,
model/champion adoption or eligible/allocated/consumed confirmation. The registered
severe-harm test is **false** because the upper bound is not below .35. Continuing
fixed G5 follows that predeclared rule, rather than favorable selection of this
point estimate. No G8 threshold, horizon or recipe changed, and the correlated
G2/G4 snapshots are not independent replications or a pooled recovery claim.

At the root-verified cutoff, G5 collection was running from the same `8fe45bf7…`
step-2,048 actor, fixed UUID `5c0e801b-3ea3-5a7b-b246-d314051b6110`, seed
**2026111505**, TRAIN half 1 and 256 fixed attempts. This is a root-reported stage
status only; no G5 payload was inspected and no G5 data, learner or model result
is asserted. Fixed G6 descriptive, G8 screen and conditional confirmation, plus
all technical/resource/deadline stops, remain unchanged.

### Preserved review history and evidence

The G3 root review initially treated a legitimate read-only `tb/` directory as
a file; `b29e56ff…` preserves that reviewer-only predicate correction. Two G4
diagnostic root-review predicates also required correction: model identity has
three fields and excludes the PGN label (`4b759571…`); the reader file is 0444
while its qualified parent directory is 0755 (`e5ddb79a…`). These were review
schema errors, not runtime faults. Source, data and readout stayed unchanged;
no GPU/CPU rerun or numerical result reroll repaired them.

The independent G4 diagnostic receipt's `root_acceptance_at_review: null` retains
its historical meaning; root `d71e74c0…` is a separate acceptance receipt. Root
receipts sealed before independent completion retain their original pending
labels. References below use the same operations artifact root. No checkpoint/
array payloads, project imports, probes or jobs were used to author this update.

| G3/G4 completed evidence | SHA256 |
| --- | --- |
| `core-baseline-lowLR-g03-collection-GPU-execution-v1/terminal.json` | `c44f1a15172d0beeb44c0909d3a2ecaa480030ef8fb412e6ad13851fc8f35bde` |
| `core-baseline-lowLR-g03-collection-results-v1/readout.json` | `f205350994ca69fbd905cd586b765af66222d22ede18550704f50e7138cedcfd` |
| `core-baseline-lowLR-g03-audit-results-v1/readout.json` | `60b257fa2fe380b9cd29c3b23603cdc09d801c91bf16524e2d8bbf54a5238c4e` |
| `core-baseline-lowLR-g03-audit-independent-actual-review-v1/review.json` | `e380fe09338f58cfcf5fa3def467696707e016a0eb3c100ca45f47437daea414` |
| `core-baseline-lowLR-continuation-bindings-v1/g03/audit-root-promotion.json` | `7bc8ba9c9a82539cdae99a04f8d87cf312bf951aa06e174e4461770a37dd7e71` |
| `core-baseline-lowLR-g03-learner-results-v1/boundary.json` | `514c95df1520bb66910cdedff256ba9053d6ba0b93f4111200b074a4b23ba81f` |
| `core-baseline-lowLR-continuation-bindings-v1/g03/learner-root-metadata-acceptance.json` | `c5064166920846cccba572fc77ad31cc097e6259814f1185fdb1de8d48051a0d` |
| `core-baseline-lowLR-g03-learner-independent-actual-review-v1/review.json` | `2aaf4a5f98d590d80119df4a4b5365593eb1c1710b714a87d052d8ed055ed0e1` |
| `core-baseline-lowLR-g03-root-learner-review-attempt1-v1/reviewer-predicate-correction.json` | `b29e56ff600f3e7936946095d4cf7d1d1dc51cf274bdbd94536e6df4e9bcf72a` |
| `core-baseline-lowLR-g04-collection-GPU-execution-v1/terminal.json` | `9364b7e15ef773c4a2f686b9d84c77f2f7c2238d2a7e174a91afdcbca751744d` |
| `core-baseline-lowLR-g04-collection-results-v1/readout.json` | `8116aeb16d163ae092035ab59a95c13b11cd836e0b594b1c4957b7d557056a2d` |
| `core-baseline-lowLR-g04-audit-CPU-execution-v1/terminal.json` | `87358e3c8dba0a8185f7d0afe084ac1d6845d860e21b1227d02dd5f4646e524a` |
| `core-baseline-lowLR-g04-audit-results-v1/readout.json` | `a4eaca2d0926c3570b8d442963ea2613416684c8e550ce3418850965d8299ede` |
| `core-baseline-lowLR-g04-audit-results-v1/generation.json` | `ff7a2341cced51fa02f617b0b9038df5a1f812226284e6088ddda9ace4a957d1` |
| `core-baseline-lowLR-continuation-bindings-v1/g04/audit-root-metadata-acceptance.json` | `e8558a631e4563d715505ee12b8d2ed695b0984c3626f925ba21f24ea65e5a68` |
| `core-baseline-lowLR-g04-audit-independent-actual-review-v1/review.json` | `acae7d78188ed28fc37625b952ec845c8ada9ce3b47888223c161a9e6b1f5939` |
| `core-baseline-lowLR-continuation-bindings-v1/g04/audit-root-promotion.json` | `e80d8600cff6dbc55f220d3ca1718394b30786403b8f5b78f438a8be1788f156` |
| `core-baseline-lowLR-continuation-bindings-v1/g04/learning-stage.json` | `e820dea4884050c9d5e215719c386c0d55d6b7bd0044402ddd2aebb816e20153` |
| `core-baseline-lowLR-continuation-bindings-v1/g04/learner-descriptor.json` | `ab7ba7c3ce3bc807c96816694c7d794794406a92dac7debd08ed3f0bbac02799` |
| `core-baseline-lowLR-g04-learner-GPU-execution-v1/terminal.json` | `eddbe439220ac1f208d79e34bf101dbdf891e1c77eed9b26469a432cabf1b85b` |
| `core-baseline-lowLR-g04-learner-results-v1/boundary.json` | `b8d41968dbb3e1099abdce48236913d7471703ac37e0766ffc1ff397f10e6f35` |
| `core-baseline-lowLR-continuation-bindings-v1/g04/learner-root-metadata-acceptance.json` | `c7f05d85f79a6bccda521383b57bf56cfd2f7358b5d7203b759fa626831501bc` |
| `core-baseline-lowLR-g04-learner-independent-actual-review-v1/review.json` | `fe710e0e996772cf8d809fc3d26f5f9c8bccc255df500ac21a96be42f4c9dd46` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g4/protocol.json` | `7b6ba2afc95c8097411b10a1c63767151b7223a860fad2c81b843976c11b1bec` |
| `core-baseline-lowLR-leg1-diagnostic-g4-GPU-execution-v1/terminal.json` | `ca2e017066385a490805170c00c5e934cb5a337af6bb4e7c8c5f66e72e976914` |
| `core-baseline-lowLR-leg1-diagnostic-g4-CPU-execution-v1/terminal.json` | `af5c97986de544aef713058ab898ee0b05dcfeaa425bb23f1f24489c3b9c5700` |
| `core-baseline-lowLR-leg1-diagnostic-g4-results-v1/serving.json` | `7207d19637ff1475239e43ef08adc9627bf489bd84dd8e0ecb616f77ec87fd45` |
| `core-baseline-lowLR-leg1-diagnostic-g4-readout-v1/readout.json` | `53c2fb5248698d8b4591d5f633e3cb13226928c0eeb611a625cea1d2f3811a4b` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g4/root-readout-acceptance.json` | `d71e74c05387eb33b5a508ec3590b71e27261dde1cd0cd1772e0ed6f58b6feee` |
| `core-baseline-lowLR-leg1-diagnostic-g4-independent-actual-review-v1/review.json` | `cae841d44f98ff8dcd2ce8380fc7ce912a59ed5a90cff4cba3ecbdce6f4f79aa` |
| `core-baseline-lowLR-leg1-diagnostic-g4-root-review-attempt1-v1/reviewer-predicate-correction.json` | `4b75957169238d35864cba4019a139fd1d918e762dc13dbbcda1bf47cfe4e1d5` |
| `core-baseline-lowLR-leg1-diagnostic-g4-root-review-attempt2-v1/reviewer-predicate-correction.json` | `e5ddb79a5c40a66f74b80c5b9d971e2872a690b5ce987221a28c5103294cab4e` |


## 2026-11-14: Completed G5 and G6; negative descriptive G6 diagnostic

**Evidence cutoff: 2026-11-14 08:20:47 UTC**, the accepted independent G6
readout review. G5 and G6 data and learner delivery, followed by the fixed G6
strict CPU readback and root/independent actual acceptance, completed under the
unchanged V2 specification `12ad1fc7…`, manifest `b10f5f5b…`, original absolute
`cc17ac07…` contract, L `090b5143…` and owner R7 `ba9d6de1…`.
G4's record was published by PR #1298, merged as `1cc80bc9…` with proof
`623bedb1…`; publication did not adopt runtime code or change the recipe.

| Completed stage | Actual accepted delivery | Owner wall seconds |
| --- | --- | ---: |
| G5 low-parent collection | All 256 attempts; 105 natural / 151 strict-TB / 0 unresolved | GPU 1,508.099043 |
| G5 qualified CPU data audit plus independent metadata review | 256 eligible games / 25,732 rows; data-only promotion `f39eeea9…` | CPU 68.636316 |
| G5 full-state learner | Step 2,048 → 2,560; 512 updates / 131,072 views | GPU 486.996459 |
| G6 low-parent collection | All 256 attempts; 114 natural / 142 strict-TB / 0 unresolved | GPU 1,448.541396 |
| G6 qualified CPU data audit plus independent metadata review | 256 eligible games / 26,035 rows; data-only promotion `ec02395f…` | CPU 81.949602 |
| G6 full-state learner | Step 2,560 → 3,072; 512 updates / 131,072 views | GPU 588.061360 |
| Fixed G6 versus A39 diagnostic | All 256 games / 128 color pairs; 124 rules / 132 strict-TB endings | GPU 793.708849; CPU readback 16.592178 |

G5 checkpoint SHA256 is
`e11e592b112fa99af30c4fe162808012b371bd603f1b1caa3bc0b6ed903cf1d5`,
boundary `1a32a6b7…`, root acceptance `ee3aeec1…` and independent metadata
review `63ab54e1…`. Five-generation ancestry retained 1,280 physical games /
128,629 audited rows. Replay itself was only G2–G5: 1,024 chunks / 103,802 rows,
with original G1 evicted from active FIFO but retained in ancestry. Its 512
complete batch-256 draws were 32,928 G2 / 32,367 G3 / 33,301 G4 / 32,476 G5;
all 1,024 chunks were drawn, with 74,595 distinct sampled physical rows.

G6 checkpoint SHA256 is
`61f10e48e298f771a22c55f8544400a03c5083cec1a6aa59d8e1d97cceeffa87`,
boundary `b15d0f3d…`, terminal `fb176290…`, later root acceptance `615349d2…`,
independent metadata review `7b7c5210…` and supplement `5baf0c95…`.
Full six-generation ancestry is **1,536 physical games / 154,664 audited rows**.
The active FIFO is only **G3–G6: 1,024 chunks / 103,827 rows**; G1/G2 are no longer
active replay. The full 512 batch-256 journal reconciled 131,072 draws:
**32,669 G3 / 33,248 G4 / 32,482 G5 / 32,673 G6**, all 1,024 chunks and 74,439
distinct sampled physical rows. No original high-actor G2 bank was substituted.
Cumulative training through G6 is **3,072 applied updates / 786,432 views**,
including the initial LOW512; G5 and G6 each applied 512 updates / 131,072 views.

Both continuations preserve the exact closed low-parent state and contract,
reconstruct prescribed replay before restoring boundary RNG, and retain
backend/parent/RNG metadata continuity under the qualified source. Independent
reviews reconcile saved metadata and draw identities; they do not deserialize
model/optimizer tensors, replay exact RNG transitions or establish GPU
bitidentity. G5/G6 scalar metrics were finite, with zero skipped/retried/clipped
updates. G6 mean LR was 2.858043151e-6 and mean gradient norm 5.178045;
these changing-window metrics are neither fixed-heldout fit nor playing strength.
Nonzero auxiliary/availability diagnostics do not change policy-1/outcome-WDL-1
and twelve zero loss weights. G6 had zero eligible/applied/abstained retained
TB-policy records; its 142 strict-TB final outcomes do not establish a new
nonvacuous retained-row policy-correction exercise.

### Preserved G6 root checkpoint-review correction

`3057e219…` preserves a reviewer-only composite assertion failure after a full
checkpoint stream. Whole `os.stat_result` equality includes access time, so it
is inappropriate as an immutable-custody predicate. However, the exact failed
subconjunct and first computed hash/before-after stats were not banked and remain
**unknown**; no specific access-time change is asserted as observed. The later
root receipt `615349d2…` banked a fresh second complete hash matching the producer
and stable device/inode/size/mtime/ctime/mode. There were explicitly **two complete
root streams**, not one. Numerical execution, model/data, source and observations
were unchanged; no retraining, new checkpoint, reseed, resample or experiment
rerun repaired this metadata-review error.

The earlier independent learner receipt retains
`root_metadata_acceptance_at_review: null` as a historical snapshot.
Supplement `5baf0c95…` binds the later root evidence without rewriting that
receipt or implying that its author independently rehashed checkpoint bytes.
The documentation uses existing root/independent metadata and stat evidence;
no model or checkpoint payload was opened for this update.

### G6 result and unchanged continuation rule

Actual protocol `2ccabed2…` binds the exact G6 step-3,072 checkpoint versus
immutable A39 on intentionally reused DIAG128, seed **2026111536**, 256 games /
128 color pairs and the unchanged matched-Gumbel64 profile. The strict CPU
reader and independent actual review accepted **69W/84D/103L**, score
**.43359375**, unbiased paired variance **.08118540846456693**, radius
**.04936169640057779** and approximate paired 95% interval
**[.3842320535994222, .4829554464005778]**. Ascending pentanomial counts for
pair totals 0/.5/1/1.5/2 are **[17, 43, 35, 23, 10]**. All 256 games / 128 pairs
completed; endings were 124 rules / 132 strict-TB, with 25,427 additional SAN
plies. Both owners closed PASS. Recorded serving retained 494 states /
61,444,448 parameters per model, strict embedded architecture, 175-plane
float32 inputs under CUDA BF16 autocast, float32 policy/WDL transport,
32-game rolling pool and 1,024 leaf rows. Actual source-qualified reader
physics and serving evidence are reused; documentation did not reimplement
legality/TB verification, reload models or rerun inference.

This is **negative descriptive underperformance on this conditional reused
DIAG128 panel**: the interval is below parity. It does not establish a clean or
general training regression, recovery, causal LR effect, optimum, external
Rofchade/Stockfish/Lc0 milestone, fresh confirmation, gain or equivalence.
No model/champion adoption, confirmation eligibility/allocation/consumption,
checkpoint selection, pooling or favorable early promotion follows. Previous
negative screens and G2/G4 snapshots remain separate, correlated observations.

**G6 has no efficacy or severe-harm gate**, regardless of score or interval.
The registered upper < .35 safety exception applies only local G4 (and
conditional second-leg G12). The fixed G7 → G8 horizon therefore remains
unchanged, subject to technical/resource/deadline stops and the accepted G6
readout dependency. The first fresh screen remains G8 / step 4,096, with the
same point ≥ .53 and paired lower > .5 rule and conditional sole fresh CONFIRM1.
Root reports preparing G7; no G7 launch, data or model result is asserted here.
The strength goal remains unmet and A39 remains this study’s unchanged donor/incumbent.

Root readout `af5483e2…` retains its pending-independent label as a historical
receipt; separate independent acceptance `0d317d14…` was available at this cutoff
and explicitly binds that root receipt. This is distinct from the earlier
learner review's historically null root field. No receipt was rewritten to
imply a different review or sealing order.

### Closed cost snapshot and evidence

Census `a30647d0…`, timestamped 2026-11-14 08:19:45.276221 UTC, records **211
closed week-root owners**, including failed attempts: **48,478.49211211607
GPU-owner seconds / 2,161.538633844757 CPU-owner seconds**, including G6's arena
and CPU readback. It is a point-in-time closed-owner census, not an ETA or budget
entitlement. The 172,800 GPU-second week cap and 2026-11-19 16:09:20 UTC deadline
remain unchanged. It makes no claim about later stages or unrelated processes.

All earlier published preregistration, G2 and G4 text remains historical and
byte-exact beneath the current-status wrapper. Paths below use the existing
operations artifact root. This update reads only bounded JSON metadata and
published docs, not raw games, nested payloads, arrays or checkpoint bytes.
The author shares authorship of continuation metadata/arena wrappers; this is
not a new independent numerical, tensor-restoration or physics qualification.

| G5/G6 completed metadata evidence | SHA256 |
| --- | --- |
| `core-baseline-lowLR-continuation-preparation-v2/specification.json` | `12ad1fc72714a3eb75c1adc519b7e56a8c80e40e9a8f0e45708479bbb8bc226e` |
| `core-baseline-lowLR-continuation-preparation-v2/manifest.json` | `b10f5f5becd47a422359523ada96283b1d6805b68543035d8522a5a258038dd4` |
| `core-baseline-fixed-G1-LR-probe-preparation-v1/learning-lowLR.json` | `cc17ac07c143eec18a42f28182ea21909ad8eb8a2a53fa844a3e201ba30da0ca` |
| `core-baseline-lowLR-g04-readout-main-publication-v1/publication.json` | `623bedb19cde73daefd665947d74a062f88588dbb8c2a5ce0b3d5c8a7da4443f` |
| `core-baseline-lowLR-g05-collection-GPU-execution-v1/terminal.json` | `bdd41bdc11b468d197eb7d0f364ae521b5d1a5edad99b65968c673cc856462d1` |
| `core-baseline-lowLR-g05-audit-CPU-execution-v1/terminal.json` | `ea902e48d9cc554ce6606a04d8d89dd245f99e1be98d3f5923ffc42dd66b13cd` |
| `core-baseline-lowLR-g05-audit-independent-actual-review-v1/review.json` | `08cdb25890124359bb2920e7c7115d051737be4a806380ef2c26a600a68a977d` |
| `core-baseline-lowLR-continuation-bindings-v1/g05/audit-root-metadata-acceptance.json` | `cb0f9ba62cc23a788268c12b606056b335a4e4d89b340a10e7836f345ad4d180` |
| `core-baseline-lowLR-continuation-bindings-v1/g05/audit-root-promotion.json` | `f39eeea9255d58f11ef813dc81bced0c18fd352810459f15444265697840cde5` |
| `core-baseline-lowLR-g05-learner-GPU-execution-v1/terminal.json` | `17371df9ea675d43881de84a56481e57107fdce8ec2237ae40d77e105d69c8e0` |
| `core-baseline-lowLR-g05-learner-results-v1/boundary.json` | `1a32a6b73c4aa1ab649b2aa06b420b94ed84ac0a9ea3635671d3f381d4e59430` |
| `core-baseline-lowLR-continuation-bindings-v1/g05/learner-root-metadata-acceptance.json` | `ee3aeec1ea2134ab962820133e26d4afa2f739f3b42be902fe8bcb7bc5715458` |
| `core-baseline-lowLR-g05-learner-independent-actual-review-v1/review.json` | `63ab54e19b2cab3b66f8f452842dcfbc935edf287380742ccd5befd2f48db05f` |
| `core-baseline-lowLR-g06-collection-GPU-execution-v1/terminal.json` | `1baf1ebdb95e4908ed2ce428305103e2b216bf0a5592989e64d04cd67032db75` |
| `core-baseline-lowLR-g06-audit-CPU-execution-v1/terminal.json` | `f798f7e655380f3d8ae381ca67b9d33d959807cc13b2fc9097f5729ea917b7dd` |
| `core-baseline-lowLR-g06-audit-independent-actual-review-v1/review.json` | `dc4a363964693c3028820598322ae13c3148778d6c7e0187de898761a1f31c61` |
| `core-baseline-lowLR-continuation-bindings-v1/g06/audit-root-metadata-acceptance.json` | `f05b1b8f976de71b884eb20428f0b82c79fc3c051efeadf40621bf0cbd42ddb2` |
| `core-baseline-lowLR-continuation-bindings-v1/g06/audit-root-promotion.json` | `ec02395fb0c8ed1acd98249b9198b14d87ae80b6ec61c6506ce29d2dd62b81c2` |
| `core-baseline-lowLR-g06-learner-GPU-execution-v1/terminal.json` | `fb17629045e3a303c4292ad724067fee3b7e4f8c45f11fd83d29a1d763a5e727` |
| `core-baseline-lowLR-g06-learner-results-v1/boundary.json` | `b15d0f3d196ac4957959549f3e6020c88a4ccf524bdaf6b5cca39c2cee8f1c8c` |
| `core-baseline-lowLR-continuation-bindings-v1/g06/learner-root-metadata-acceptance.json` | `615349d21c882f40758130afc56715c8a7d6b1e69c43ca23c81607df541eca90` |
| `core-baseline-lowLR-g06-learner-independent-actual-review-v1/review.json` | `7b7c52105f28fde54ba2f1f395bcb401640194a4304edf9297128635abba61dc` |
| `core-baseline-lowLR-g06-learner-root-verification-supplement-v1/review.json` | `5baf0c95111467840f1f3a2f515c18721019b82613d3fe5d9ceff10f09a9e5d7` |
| `core-baseline-lowLR-g06-root-learner-review-attempt1-v1/reviewer-predicate-correction.json` | `3057e2192dcbd9691a847a4634029a331706c254895c36c26a22b03c496e9b79` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g6/protocol.json` | `2ccabed2065a48fc9788fc6b388b75292aadf6d81ff41ac84bea249d21ee0444` |
| `core-baseline-lowLR-leg1-diagnostic-g6-GPU-execution-v1/terminal.json` | `25acaff13441d1a1be6643cab5dda58bb910c4e244436ee144491f3a3bb1730a` |
| `core-baseline-lowLR-leg1-diagnostic-g6-CPU-execution-v1/terminal.json` | `21ee7a84f4f34c36cd03a6b051be95800adda7738c0b56641dc2be9fb7c12ae2` |
| `core-baseline-lowLR-leg1-diagnostic-g6-results-v1/serving.json` | `a080abd471a27b6d0f44e17f9cf70644e820b9c46945b54e00da9c3ba6bc108d` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g6/cpu-admission.json` | `a1f4303d2615573f28af5e24ce8e39f54d5dbb20e7fd33f75ab68aca1b71fceb` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g6/reader-bindings.json` | `b1e575454b39108938a05826601f4ca94d6c7137d122ce39580812ae306a80a9` |
| `core-baseline-lowLR-leg1-diagnostic-g6-readout-v1/readout.json` | `5a20845581c5e55f086b91f655f8157a903b77f29646474d67dcccd627a71f4c` |
| `core-baseline-lowLR-continuation-bindings-v1/leg1-diagnostic-g6/root-readout-acceptance.json` | `af5483e2b92f4d3d50deb0038f63fe3084fa4c1b4309be0398c51ca45aae54a0` |
| `core-baseline-lowLR-leg1-diagnostic-g6-independent-actual-review-v1/review.json` | `0d317d14489aaec8527955fe7a4066be60676d7be496cc955ab94a021435fb09` |
| `core-baseline-lowLR-g06-diagnostic-closed-cost-census-v1/readout.json` | `a30647d08ff832d456fa8aa0caa1eaa613af17256eee71ff0a5fb2a395f58393` |
