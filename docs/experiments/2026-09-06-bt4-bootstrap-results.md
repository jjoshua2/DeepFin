# BT4 bootstrap: completed global and SF-close screens

Recorded September 6, 2026. The bootstrap research supports restarting RL in roughly
one to two months, after selecting and confirming a useful data/target/loss recipe
and scaling toward approximately 100M positions. The present development corpus
contains 18,910,484 positions. No recipe has been declared optimal or promoted.

## Evaluation correction and completed results

The [global mixing experiment](2026-09-05-bt4-global-search-scaling.md) inherited
search-prior temperature 1.5 from an older network. Before reading either global
mixture's playing result, evaluation was changed to **1.0 on both sides**. Training
recipes, value labels and the frozen training runtime were preserved. Old 1.5
arenas remain separate historical evidence.

Completed same-checkpoint calibration at 100 simulations favored prior 1.0 over
1.5 for both S0 and E0: respectively +55.71 Elo [40.20, 71.45] and +85.78
[67.20, 104.86], each from 1,000 games / 500 color-swapped opening pairs. This
supports the correction for these checkpoints; it does not locate their optimum.

The corrected screen has nine completed cells, each with 1,000 games / 500 pairs
and no orphan rows. All candidates face the same S0 checkpoint, at matched
25, 100 or 400 simulations. Search uses the registered training shape, prior 1.0,
move temperature 0.1, book seed 42, 16 opening plies, maximum 300 plies, compilation,
128 concurrent games and evaluation batch 4096. The 100-simulation comparison is
primary; 400 simulations is also relevant to deployment.

| Training recipe | 25 sims: Elo [95% CI] | 100 sims | 400 sims |
| --- | --- | --- | --- |
| E0: raw BT4 redistribution among exact SF maxima | +98.07 [80.94, 115.68] | +91.71 [74.09, 109.81] | +93.20 [77.60, 109.18] |
| G20T1: 80% SF + 20% global raw BT4 | +62.15 [46.59, 77.97] | +58.21 [42.69, 73.97] | +76.61 [61.16, 92.36] |
| G20T05: 80% SF + 20% global BT4 at T0.5 | +103.73 [87.77, 120.13] | +109.07 [91.63, 127.06] | +114.07 [98.28, 130.32] |

These are direct comparisons against S0, not head-to-head Elo between the recipes.
S0 trains on the stored sharp SF targets. Global mixtures normalize each legal
teacher distribution before mixing; T0.5 squares BT4 probabilities before
normalization. Teacher temperature and search-prior temperature are distinct.

The registered raw-minus-sharpened common-S0 score contrasts favor sharpening:
at 25/100/400 simulations, differences are -5.65/-6.90/-5.00 percentage points,
with paired intervals [-8.70, -2.55]/[-10.10, -3.80]/[-8.00, -2.00].
The relative score changes from 25 to 400 simulations are +2.00 points for raw
global BT4 [-1.25, +5.25], +1.35 for sharpened global [-1.70, +4.30], and -0.65
for E0 [-3.85, +2.60]. Gains persist at deeper search; increasing relative search
advantage is not established.

An additional **post-hoc exploratory** comparison keeps E0 in contention:

| Simulations | G20T05 minus E0 score against common S0 | Paired 95% interval |
| --- | --- | --- |
| 25 | +0.75 percentage points | [-2.40, +3.95] |
| 100 | +2.30 percentage points | [-1.00, +5.55] |
| 400 | +2.75 percentage points | [-0.30, +5.70] |

Their difference in 400-minus-25 score changes is +2.00 points [-2.50, +6.30].
Neither superiority, equivalence nor better relative scaling follows. All cross-cell
intervals resample aligned opening pairs, 10,000 PCG64 replicates, seed 20260903.
They are nominal intervals for an exploratory one-training-seed screen, not
uncertainty over independently retrained models.

## Completed SF-close comparison

C20T05 uses the union of all stored SF maxima and d9 top-three moves within 20
effective cp of the best move. It redistributes that set's existing SF mass using
conditional BT4 at T0.5, leaving mass outside the set unchanged. Alpha 1.0 applies
inside the selected set; it is not pure global BT4 supervision.

The completed epoch covered 18,910,484 rows in 36,935 batches, with zero same-game
repeats, using the frozen wise-cloud trainer, seed 0, batch 512 and game_epoch
sampling. Independent review verified the completed source-normalized schedule,
actual staging and recorded launch identities. Checkpoint SHA256:
`8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3`.

All three direct C20T05-versus-E0 matches completed with 1,000 games / 500
color-swapped opening pairs each, zero orphans and search-prior temperature 1.0
on both sides. The frozen settings and opening sequences match across budgets.

| Simulations | C20T05 score against E0 | Elo [paired 95% CI] |
| --- | --- | --- |
| 25 | 55.35% | +37.32 [18.93, 55.92] |
| 100, primary | 57.25% | +50.74 [31.74, 70.04] |
| 400, deployment-relevant | 55.60% | +39.08 [21.98, 56.36] |

The primary interval meets the registered **promising** criterion, and the gain
persists at 400 simulations. The registered paired 400-minus-25 score contrast is
+0.25 percentage points [-3.35, +3.75], using 10,000 aligned-pair bootstrap
replicates with PCG64 seed 20260903. Increasing relative advantage over E0 with
search is not established. Training and all three arena charges closed at
**5.376 GPU hours**, below the separate 12-hour cap.

E0 uses teacher T1, so this comparison combines widening and sharpening. A
sharpened-exact-tie model remains a substantive challenger that could change the
winning recipe. E0 also retains an incompletely stamped historical runtime;
confirmation needs fresh matched training. The schedule proof covers game metadata
and batch choices, with row-order equality inferred from the pinned code; it is not
a full feature/target payload check. The historical `valid_control:false` limitations
remain: no held-out purity receipt, comparison to a committed configuration pin,
and game-epoch sampling differing from the old replacement-sampled control.
These nominal one-training-seed results do not promote a recipe. Direct-E0 Elo
cannot be subtracted from the global screen's direct-S0 Elo to rank C against G.

## Next decisions and evidence

The selected next development screen is a **direct C20T05-versus-G20T05 match at
100 simulations**, reusing the existing checkpoints: 1,000 games / 500 pairs,
prior 1.0 on both sides, with a hard **5,400-second** arena/supervisor cap including
termination allowance. It has not launched at this record update. C's paired Elo
interval wholly above zero favors C, wholly below zero favors G, and crossing zero
is inconclusive; an inconclusive result does not require more games or establish
equivalence. This one-budget screen selects a useful next challenger, not a
search-scaling slope or a deployment winner.

If the close family remains competitive, the already materialized sharpened
exact-tie recipe is a possible winner-changing challenger, not merely an
attribution control. Its bounded 8,192-row check verified target arithmetic,
raw-BT4 lineage and non-policy preservation; prospective and realized training
schedules still need qualification if selected. If the global family clearly
leads, the published and independently reviewed
[50% sharpened global BT4 mixture](2026-09-06-bt4-global-dose50-preparation.md)
tests a substantively larger teacher contribution. Corpus availability alone does
not justify a training grid.

Training-objective interactions remain relevant to bootstrap selection. SF agreement
and local gradient diagnostics cannot rank playing strength. PR
[#517](https://github.com/jjoshua2/DeepFin/pull/517) contains the separate graded-objective
investigation; no TailRL training is included in these screens. Reserve fresh
matched training-seed and fresh-opening confirmation, including deployment-relevant
search, before recommending a recipe for the larger bootstrap and RL restart.

Before transferring recipes to the growing G10 corpus, explicitly choose which SF
observation supplies policy, value and rank/quality labels. Legacy `uniform-d9`
uses later-phase d9 scores where available, while the SF rank sidecar uses phase 0.
The current 20M source has only one phase and is unaffected. One inspected G10 row
shows changed scores and exact ties; it establishes existence, not prevalence or
strength. Even phase-0 selection would not reproduce the old generator's search
state: G10's additional deeper, narrowed searches change the transposition-table
work inherited from earlier positions. Both generators retain tables within games.

Runtime evidence lives under `scratchpad/bt4_joint20`, outside git. Read actual
processes and completion artifacts when continuing; status here is a dated snapshot.

| Artifact | SHA256 |
| --- | --- |
| `global_run03/preregistration.md` (prior correction) | `4457c4f8ba407a2b16ad31f26dcebebc41e97760a82bbc5fe27a7cbc78c884e8` |
| `global_run03/completed_full_screen_readout.json` | `07e59cb0803e077379f6946e4bae7176681f4ff20d424aaf0a6c552971375f24` |
| `global_run03/completed_full_screen_independent_review.json` | `c7a84c318c269312ade96f623cd4b5a9486f95c388207c86e2035d1185b5e35c` |
| `global_run03/global_vs_exact_ties_exploratory.json` | `36c05e5806db8c29cf9a24193640b5284a0038da43c65f4671a2df6f92de21b4` |
| `global_run03/global_vs_exact_ties_independent_review.json` | `6a429b5a31742d0e36b0ae8b7eeab9c50e6da711e83b366be982ee429f9d049b` |
| `sf_close_run02/preregistration.md` | `c4c5a0b861efce888a62c9fc085d61509da0ab805790e108f0f293ea33d029b4` |
| `sf_close_run02/matched_schedule_completed_C20T05.json` | `1f1529fa2247d56ad4cf457a8cb5bae0929c4ea917e2aab0c09d81fbaeedccbd` |
| `sf_close_run02/C20T05.completed_epoch_independent_review.json` | `7ed7bd8863db4fb7bf2fa98016b7cdb269b751dc696c4f8827acc58f8cae5d5a` |
| `sf_close_run02/completed_full_screen_readout.json` | `2dde2ce5b46bc69b662be129b3752ffe37f93aa383f2d3bd06dbeb5d581a7257` |
| `sf_close_run02/completed_full_screen_independent_review.json` | `fc230b5db0ef7eecb686ae58220fb7b4a69ae358d9ed4ec48c7a419c6a6cbc76` |
| `sf_close_run02/optional_sharpened_ties_data_qualification.json` | `48897206e4af6d37bb4fa351cebe1e1d4b493e07b63126fd8a65332bd85af8c1` |
| `global_dose50_v1/data_transfer_notes.json` | `4b1d74ddafdf4e9443b23d51e0119bf991860525ad007fb79bae93b4cfc4f9da` |

The global full-screen independent reviewer recomputed all nine game banks and registered
contrasts, checked model/protocol identities, and reused the completed training
provenance. All global GPU charges closed at 22.3373 hours, below its 30-hour cap.
The post-hoc contrasts were separately recomputed from six raw banks. The SF-close
review independently rescored all three banks and its registered interaction,
checked protocol and completion identities, and verified the closed GPU charges. Confirmation
openings and launch/readout tooling have been prepared but no confirmation run is
selected or launched. Preserve active generation throughout.
