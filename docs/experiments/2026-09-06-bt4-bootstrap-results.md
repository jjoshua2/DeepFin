# BT4 bootstrap: completed global screen and next comparisons

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

## SF-close comparison now running

C20T05 uses the union of all stored SF maxima and d9 top-three moves within 20
effective cp of the best move. It redistributes that set's existing SF mass using
conditional BT4 at T0.5, leaving mass outside the set unchanged. Alpha 1.0 applies
inside the selected set; it is not pure global BT4 supervision.

Its 2,309-shard target corpus was published and its prospective ordered schedule
validated before training. Training started September 6 at 16:27:52 UTC, after the
global screen completed. It uses the frozen wise-cloud trainer, seed 0, batch 512,
game_epoch sampling and the full 36,935-step epoch. The driver subsequently runs
C20T05 directly against E0 at all three budgets, prior 1.0, with 1,000 games each.
Its separate cumulative cap is 12 GPU hours. At this record, training is in progress;
no completed C checkpoint or playing verdict is claimed.

The primary 100-simulation paired Elo interval is promising if wholly above zero,
a loss if wholly below zero, and otherwise inconclusive. Report 400 simulations
separately and run all three cells regardless of the shallow outcome. The planned
paired 400-minus-25 score contrast measures relative advantage against E0.

E0 uses teacher T1, so this comparison combines widening and sharpening. A
sharpened-exact-tie control would isolate widening if attribution changes the next
recipe decision. E0 also retains an incompletely stamped historical runtime;
confirmation needs fresh matched training. Successful launch verification does not
replace verification of the completed epoch's realized schedule and staging.

## Next decisions and evidence

Prepare a [50% sharpened global BT4 mixture](2026-09-06-bt4-global-dose50-preparation.md)
to test a substantively larger teacher contribution. After the SF-close results,
choose competitive direct recipe comparisons and any training-objective interaction
that could change bootstrap selection. SF agreement and local gradient diagnostics
cannot rank playing strength. PR [#517](https://github.com/jjoshua2/DeepFin/pull/517)
contains the separate graded-objective investigation; no TailRL training is included
in this screen. Reserve independent training-seed and fresh-opening confirmation
before recommending a recipe for the larger bootstrap and RL restart.

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

The full-screen independent reviewer recomputed all nine game banks and registered
contrasts, checked model/protocol identities, and reused the completed training
provenance. All global GPU charges closed at 22.3373 hours, below its 30-hour cap.
The post-hoc contrasts were separately recomputed from six raw banks. Confirmation
openings and launch/readout tooling have been prepared but no confirmation run is
selected or launched. Preserve active generation throughout.
