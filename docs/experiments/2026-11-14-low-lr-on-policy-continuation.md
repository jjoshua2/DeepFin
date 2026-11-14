# Full-state lower-rate on-policy continuation

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
