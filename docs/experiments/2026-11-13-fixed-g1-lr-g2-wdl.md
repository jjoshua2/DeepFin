# Fixed G1 lower-LR probe and G2 WDL adjunct

Status at **2026-11-13 23:46 UTC**: **LR probe and original WDL adjunct
completed, technically qualified; relative playing comparison and finite-bank
fit only.**
It changes neither the [completed eight-generation baseline](2026-11-13-core-outcome-policy-baseline.md)
nor its terminal nonpass: G8 scored 21.875% versus original A39, with no
confirmation, C1, conditional second leg or adoption.

## Prospective question and release

At the same accepted G1 data and first 512 updates, does a ten-times-lower LR
curve outperform banked original-LR G1 step 512? The sealed probe specification
and source review preceded the G8 resolution. They allowed release only after
a qualified original-leg nonpass (including the specified G4 harm or consumed
confirmation branches), not a failed interim, a technical stop, or an eligible
endpoint awaiting confirmation. Actual release used the qualified G8 terminal
nonpass; it did not interrupt or alter the original baseline.

Probe UUID is `76554bb7-c0b7-4bdc-86dc-8ebb06735ea5`. Start fresh from immutable
A39 with the same initial seed `2026111520`, whole accepted G1 bank (**256 games,
24,827 rows**), shared replay/mirror RNG, mirror probability .5 and otherwise
unchanged policy/outcome recipe. The only learning-contract changes are:

| LR scalar | Original G1 | LOW512 |
| --- | ---: | ---: |
| Peak | 3e-5 | 3e-6 |
| Floor | 6e-6 | 6e-7 |
| Warmup start | 1e-6 | 1e-7 |

Warmup stays 256 updates and first period 16,384. Learner batch is **256**, dose
**512 updates / 131,072 views**. This is a new fresh learner, not donor-optimizer
restoration or a continuation of the eight-generation actor. LR scaling also
scales decoupled weight-decay exposure, so the contrast cannot isolate an
overshoot mechanism or establish an optimal LR.

## Completed learner and direct playing readout

LOW512 completed under its finite owner in **252.231051 seconds**. Its full
physical draw journal SHA256 exactly matches original G1 (`72d99144…`), with
identical saved membership, backend and RNG metadata. This supports common
selected-row order and saved state, not GPU bitidentity or archived postmirror
tensor equality; original per-step posthost tensors are unavailable.

All **18 stdlib guard controls passed**, with zero forbidden calls and an
explicit predecessor stub. They exercised selected guard ASTs, not a real
producer, model, data or future authority. Actual learner/control metadata
review separately accepted the closed learner and lineage. The reviewer
coauthored the reusable learner; this is scoped actual metadata/journal review,
not an independent implementation of training physics.

One fixed direct comparison used **256 games / 128 color pairs** on the already
observed DIAGNOSTIC128 roster, arena seed `2026111601`, LOW512 versus banked
HIGH512. It retained matched Gumbel64/noise-on, strict-TB, BF16-autocast/float32
transport settings and an observed 32-game pool / 1,024 leaf-row profile.
LOW512 scored **101 wins, 90 draws and 65 losses**: mean **.5703125**, paired
normal approximate 95% interval **[.5205179000537302, .6201070999462698]**,
unbiased pair variance **.08261564960629922**. Pair-total counts for
{0, .5, 1, 1.5, 2} were {12, 17, 40, 41, 18}; endings were 120 rules and 136 TB
resolutions. The arena used **876.202572 GPU-owner seconds** and strict CPU
readback **17.668015 seconds**. These are owner wall times, not CUDA-active time.

The predeclared exploratory classification is **relative improvement over
HIGH512**. It establishes no A39 gain, online-RL recovery, C1, confirmation,
promotion, optimal LR or causal overshoot. No A39 playing comparison was added.
The reused book, opening-family dependence and approximate interval limits
remain explicit; no pooling with original G2/G4/G6/G8 results follows. The
qualified closed reader, independent actual pair arithmetic and root acceptance
agree on this limited result. No recipe or actor was automatically adopted.

Candidate LOW512 is
`5d5c93cb669e24e3e97848960ea9bcf78c6f7880327a896aa7d2a30693641b53`;
reference HIGH512 is
`29b704a75703456988d76fcb0e36d4924aa1789b0f3901f49a766c8bfe05856f`.
Original A39 remains `a39f900c…`. Runtime L `090b5143…`, R7 owner `ba9d6de1…`
and original baseline science `dd16774a…` remain unchanged. Study-specific
external adapters are source-reviewed artifacts, not a new general framework.

## Completed G2 outcome-WDL adjunct

The separately preregistered adjunct UUID is
`dd2457e3-5987-43d2-9d1c-704abf0eda9c`. Its release required the parent learner,
direct arena and strict CPU reader to close qualified **for every relative
direction**, not only a winning parent. The measurement received its own actual
bindings and execution admission after that closure; it was not winner-selected.

All **26,505 retained G2 rows / 256 games** were scored for each of **A39,
HIGH512 and LOW512**, including 90 natural and 166 qualified strict-TB games.
One model was resident at a time, with contiguous inference batches of **64**.
The measurement used stored single-orientation F16 input widened to F32 and
actual BF16-head logits transported as F32: no mirror, augmentation, search,
SF/policy/categorical scoring or new teacher/TB queries. Primary metric was
whole-corpus row-weighted genuine-outcome NLL using stable float64 CPU
log-sum-exp; secondary was the sum-of-three-classes WDL Brier score.

| Fixed model | Outcome NLL, row-weighted mean | WDL Brier sum3, row-weighted mean |
| --- | ---: | ---: |
| A39 | .7745946257994324 | .44390599697865757 |
| HIGH512 | 1.3052469370593807 | .5833544676154793 |
| LOW512 | .726214227287593 | .42862393539172366 |

Both descriptive metrics order LOW512, A39, HIGH512 on this complete finite
bank. There are **no intervals, thresholds, arm selection, promotion, Elo or
A39-strength claims**. The ordering does not identify causal overshoot, an
optimal LR or online-RL recovery. G2 is physically held out only from these
G1-only 512 updates, not universally unseen or held out for later-generation
learners. HIGH512 produced its path/search-conditioned inputs and outcomes;
input/mirror overlap with G1 and A39 pretraining exposure remain unknown.
Single-orientation scoring also differs from the training mirror expectation.

The GPU owner closed PASS after **204.262320 seconds**, within its fixed
900-second / inner-870 ceiling. All three models scored the same complete
bank: **79,515 views / 1,623 forwards** total, with **494 state tensors and
61,444,448 unique parameters per model**. Full data/model custody and pre/post
served state joined; weak model references closed. There was no training,
search, fresh TB probe or Trainer/optimizer construction/restoration. The trusted
checkpoint container can materialize embedded optimizer tensors on CPU during
loading; that is not optimizer construction or restoration. Source/counters
supply the stated side-effect scope, not a universal dynamic I/O monitor.
Root recomputed aggregates from all per-game sums. Independent actual review
streamed all 79,515 saved JSON observations and recomputed NLL/Brier, F32
row/game digests and input/history/POV/label joins. All matched (maximum row
residual 1.78e-15, game residual 5.68e-14); 1,024 original game/proof/evidence
JSON hashes and the actual V3 control closure also checked. This was a stdlib
readback of banked observations, with no new model, inference, probe or job.

### Preserved harness failures and scoped qualification

V1 closed after **1.207868 CPU-owner seconds**, before any case or output,
because host Python 3.10 AST-dump pins differed from managed Python 3.13. V2
corrected those pins but its artificial shared builtins import denier blocked
P1; it closed after **1.208706 seconds** with one failed case. The exact requested
module/slot was not retained and remains **UNKNOWN**. Neither attempted a real
WDL measurement.

A separately reviewed V3 harness retained the same nine cases, tolerances and
actual WDL metric/aggregation source. Its **nine synthetic NumPy controls
passed**, with zero forbidden attempts and closed PASS owner time **1.194278
seconds**. It allows trusted NumPy/stdlib imports while a static pinned-AST
check rejects explicit imports, direct I/O and dynamic code; scoped sentinels
and pre/post project/Torch absence checks remain. It is not a universal I/O
monitor or a qualification of data truth, GPU numerics or real model inference.
These were harness repairs before the **one original actual WDL measurement**,
not numerical-result rerolls. The adjunct source and preregistered math were
unchanged.

### Next scientific question

A fresh donor-strength playing test or another reviewed successor remains
**unrun**. It needs its own prospective question and execution admission; neither
better fit here nor the relative LOW512-versus-HIGH512 result automatically
selects, continues or adopts a model. The completed original baseline negative
and its immutable A39 comparison remain unchanged.

## External identities

References are relative to the shared `rl-week-one-20261112-v1` operations
artifact root. Original baseline evidence remains in its own record. Sealed
preparation statuses are historical source-only facts; actual receipts below
supply the completed LR and WDL observations, not fabricated retroactive execution.

| Artifact | SHA256 |
| --- | --- |
| `core-baseline-fixed-G1-LR-probe-preparation-v1/probe-spec.json` | `52ec0a2f6aecfb27d7983420cd6c63b52389f723b7a7e159d8fb60fd7a9d3411` |
| `core-baseline-fixed-G1-LR-probe-preparation-v1/manifest.json` | `80912459c9c67f100f08b3007c55e10dce49bdf9a7b35660f34d13f274a7c136` |
| `core-baseline-fixed-G1-LR-probe-independent-source-review-v1/review.json` | `b0fb0594e7090f9570071b0a392c32114c2e75f799735a19004b8d0dc72eb446` |
| `core-baseline-fixed-G1-LR0p1-learner-results-v1/boundary.json` | `533a7ca081f34bd6daa732caa6c0b3b17f17ceb22c586abc6f60b01e82513b21` |
| `core-baseline-fixed-G1-LR-learner-controls-independent-actual-review-v1/review.json` | `7aa78ea64009c80f8ba5f86d6dc3ba5255da8d460b09118abc4b8b8eb46984e9` |
| `core-baseline-fixed-G1-LR0p1-arena-readout-v1/readout.json` | `e695317e658d4f9976660c21407469acd12201da19e8ff18f9caa42d54ac4276` |
| `core-baseline-fixed-G1-LR-probe-bindings-v1/fixed-g1-lr0p1-vs-banked-g1/root-readout-acceptance.json` | `7ee0e7016bbebb5dc743a45ed2087cc2d6b3fa52f305adc1ec37cd05d518702c` |
| `core-baseline-fixed-G1-LR-arena-independent-actual-review-v1/review.json` | `184d0c3f4fece0a08e968cf8f7af39bd4acfa66ec11f0cd30e0cb4e19c26452c` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-preparation-v2/manifest.json` | `3abccc5d78bd84d5ea8fb65400ac6e247b4836a951c9da971c7961237ed34382` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-preparation-v2/preregistration.json` | `eb10e0ca041627cd74691c60e5a2c5eda8ff9462643eb39e9cd44ab8d273d270` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-independent-source-review-v2/review.json` | `77dceeeabda3e893ac946c2c3d155e165b35797b650eb10309f5bf5b6300d1f7` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-CPU-execution-v1/terminal.json` | `cf5e6621a47d4b1fd47c0cd232a20403aee21cef680818ebe3c45d233549485d` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-CPU-execution-v2/terminal.json` | `a04f39c193cb8c5938199ef6a3548a6df29e68b16b901084fa8150bcb0e28eed` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-preparation-v3/manifest.json` | `9fbd88b61b1736337db8e801efeba100490a20216941979b708c97490653b70e` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-independent-source-review-v3/review.json` | `a893eb59f471659abe0dc9fcbed380cc6af1f9dd0102a6cf6ce772d4622c19e1` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-results-v3/readout.json` | `c0bc3acfd127e562ad42cd2fa142ebccecdcbd3ba0612cf6a2e9541a2ad16545` |
| `core-baseline-fixed-G1-G2-WDL-math-controls-CPU-execution-v3/terminal.json` | `8f3dce04eff4dc273693d8238ad0349761f9b36492ae295847fa104c4fed12b7` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-results-v1/readout.json` | `b20b66e92e5faf1318c3bf2f4ee58054922ee112b6bc201b4e1fbc82b3afaaf8` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-bindings-v1/root-readout-acceptance.json` | `4549d7501f5fd2cf90190b34b0ebae267832b5d6ed771b1da8967631710d266f` |
| `core-baseline-fixed-G1-G2-WDL-adjunct-independent-actual-review-v1/review.json` | `f9c581897d8657b2e551d927adece02e942cf84a059753d41ef1501e52bd3cfe` |
