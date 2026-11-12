# First E2 RL comparison: search targets versus genuine outcome targets

Status: **PREREGISTERED. Execution admission remains separate.**

Scientific freeze: **2026-11-12 21:33:31 UTC**. The authoritative
[frozen scientific JSON](evidence/rl-week-one-20261112/first-E2-preregistration.json)
is copied byte-for-byte from the external owner packet, SHA256
`7d47cb547d8f1b0b96299dbcc097a31e8d9eab7cfd8522d1904ad88ff8f251d6`.
This registration precedes research collection. Every stage still needs its
qualified actual source, input, owner, resource and control bindings; publication
alone launches nothing.

This is one controlled first-week experiment in the [RL strength program](2026-11-12-rl-week-one.md). Its result cannot identify an optimum RL formula or establish the multicore Rofchade, Stockfish or Leela milestones. Preserve the epoch-two donor and all earlier negative and inconclusive results. No candidate is adopted by this protocol.

## Question and intervention

At a fixed fresh donor, new pure-selfplay data and 40 applied updates, does adding genuine completed-game outcomes to both value targets improve playing performance over the search-only recipe? Search draw targets partly retain the producer's own draw prediction, which motivates testing an external terminal-outcome anchor. That motivation is a hypothesis, not a causal explanation of the earlier regression.

| Arm | Main WDL target | Categorical target |
| --- | --- | --- |
| FULLA | Search 1; genuine outcome 0; SF 0 | Original search-only target |
| FULLB | Search 0.8; genuine outcome 0.2; SF 0 | HL-Gauss of `0.8 * raw_search_Q + 0.2 * genuine_outcome_Q` |

Categorical Q is signed, in `[-1,1]`; outcome Q is win `+1`, draw `0`, loss `-1` from the recorded position's side to move. Preserve the original raw F32 search and categorical targets before production F16 serialization. Reconstruct every FULLA categorical target with the actual producer functions and require exact original F32 equality and stored F16 equality. FULLB uses the same production F16 cast after its declared scalar blend. This is a scalar-target blend before HL-Gauss, not a 20% categorical-loss or gradient mixture. Both arms retain the original stored F16 main-WDL search input.

This is a **WDL plus categorical target bundle**. It does not isolate the WDL head, SF supervision or the categorical objective. Keep all original policy and auxiliary loss weights, full trainable trunk, optimizer routing, clipping, learning rates, warmup and reference-state handling common. Categorical and SF host rebuilding stay off. Do not change SF-opponent share, learning rate, architecture or search within this comparison.

## Data and exposure

Collect exactly 96 attempted physical games with one strictly loaded donor, 16 recycled runtime slots, Gumbel 64-simulation search and a 450-additional-ply cap after each complete 16-ply opening history. Collection is pure selfplay with every tablebase path, search/adjudication/rescore option and SF tablebase path explicitly off. A slot is not a physical-game identity. Assign namespaced ordinals to 32 TRAIN and 64 HOLDOUT games before generation.

The candidate roster contains 96 primary starts and 32 structural reserves. Before any collection, a primary start that fails the frozen structural or bounded-exposure check may be replaced by the earliest passing reserve, preserving its preassigned bank. Exhaustion makes the collection unready. No outcome, zero-row game, censoring, slow game or failed attempt permits replacement or a new seed. Retain every attempt and termination, including explicit zero-row archives and no-shard markers.

The exposure index covers the declared 803 known shard paths and 138,822 primary input rows, plus their actual production file mirrors. Metadata proves that this scope selects primary `x` in the donor's layout; recorded alternate-history arrays are unnecessary for these paths. Hash C-contiguous little-endian `175 x 8 x 8` F16 inputs. Missing chunks, changed sources or incomplete coverage make the index unqualified. This is a bounded known-RL-data check. Epoch-two pretraining, unknown archives and historical feature-dropout exposure remain unknown.

All 32 TRAIN games must have usable, full-history, genuine natural outcomes, correct policy/mask/POV joins and the raw-value control proof; otherwise no E2 learning is admitted. Known trajectory-input matches are counted and disclosed. They do not automatically invalidate a TRAIN game or an unselected heldout row.

Every row and sidecar from the 64 whole HOLDOUT physical games stays outside replay, sampling, priorities and normalization updates. After all TRAIN inputs are banked, filter heldout NET-policy rows at offsets 8 through 80 against the bounded old raw/mirror index and all new TRAIN raw/mirror inputs. Select one eligible root per heldout game by the fixed physical-row hash rule, before branch predictions or teacher scoring. All 64 games must have genuine natural outcomes and an eligible root for the complete offline endpoint; otherwise that endpoint is inconclusive. Do not substitute other games. The playing tests can proceed if TRAIN qualifies, even when this diagnostic endpoint does not.

Final playing books use the same original weighted SHA selection seed and fixed best-4096 catalogue cap, existing history/EPD/source-parent/input exclusions, all 128 cohort/reserve candidates, and the old raw/mirror index. Greedily take the first 128 passing distinct pairs for the shared screen, then the next 500 for confirmation. Fewer than 628 passing candidates is inconclusive. Freeze final identities before research collection. After all 96 games complete, compare the 628 playing-start input hashes against every RETAINED input row's raw/mirror hashes in all 96 games' `validation.rows`, before branch learning or scoring. A match makes this planned playing endpoint unqualified and stops E2 learning admission; do not replace books or games after collection. Discarded, unretained and search positions are outside this check; it does not establish whole-trajectory novelty. This specific playing-start check differs from the permitted old-input matches elsewhere in TRAIN or unselected heldout trajectories. The earlier 628 candidate observations remain preserved; structural filtering does not authorize outcome-based book changes.

## Matched learning and technical qualification

Use the preserved donor SHA256 `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. Each arm starts with all 494 state tensors exactly matching it, fresh optimizer/scheduler/clip state, no resume, no head reset and no SWA. Run arms sequentially with no generation or evaluation resident. Previous model, Trainer, optimizer and selected-parameter weak references must be dead before constructing the next arm.

Run exactly 40 applied updates of 256 views per arm: 10,240 views each, 80 updates total. Use the same uniform physical-row index plan with replacement, row multiplicities, augmentation stream and feature-dropout RNG. This sampler and pure-selfplay cohort are controlled study settings, not a claim of production replay equivalence. Save steps 0, 1, 8 and 40; only the preselected FULLB step-40 endpoint can qualify for confirmation. Intermediate losses, snapshots and offline diagnostics cannot choose an arm, step, seed or learning rate.

Before research learning, qualify the added observations on the excluded pilot's one 144-row TRAIN game with two fresh four-update search-only controls. Common batch/RNG capture is present in both; extra selected gradient/update copies are present in one. Require exact final hashes for all 494 tensors, paired batches, RNG, math settings, optimizer ownership, learning rates and counters. This does not independently compare every capture hook against a wholly uninstrumented Trainer. Require real caller integration, unchanged numerical operations and exact paired observations for both research arms.

Observe the complete optimizer parameter map and a positive, predeclared set of at most eight parameters within the copy budget. Selected gradient/update norms and cosines describe that subset; they do not establish full-trunk per-head interference. Record actual host and device targets, availability, loss coefficients, used learning rates and applied updates. Any undeclared input, RNG, optimizer or dose difference invalidates the paired interpretation. Nonfinite state, retries, skipped updates, extra draws or wrong checkpoint counters stop the comparison and preserve partial evidence.

## Playing decisions

Use the new common indexed three-through-six-piece tablebase profile only after actual root, leaf, history, DTZ and rule-50 consumers qualify. Both sides use matched simulations: training search shape, 64 simulations, temperature 0.1, root Gumbel noise on, 32 concurrent games, evaluator batch cap 1024, Torch compilation off and 450 additional plies. This differs from old TB-off arenas and the external wall-clock Rofchade baseline. Do not silently downgrade the profile if qualification fails.

Run all three fixed 256-game screens, using the same 128 opening pairs, unless a global stop or technical fault prevents completion:

1. FULLB40 versus FULLA40: primary target-bundle comparison.
2. FULLB40 versus unchanged donor: hierarchical eligibility for confirmation.
3. FULLA40 versus donor: descriptive context; no automatic FULLA adoption.

For each deciding comparison require score at least 0.53 **and** the existing paired empirical normal 95% lower score bound strictly above 0.50. The opening pair is the analysis unit. Zero empirical pair variance has unavailable uncertainty and cannot pass by inventing a zero-width interval. These sizes do not guarantee sensitivity to a three-point effect, nor establish independence of opening families or multiple seeds.

Only if both FULLB comparisons pass, play exactly one fresh 1,000-game, 500-pair confirmation of unchanged FULLB40 versus donor. Use the disjoint preselected confirmation book and the same profile. Require the same point-score and lower-bound rule. No best-arm fallback, extension, reroll or pooled engineering games. Require all planned games/pairs, legal full histories and closed process custody. Max-ply endings, faults or incomplete counts prevent a qualified strength verdict; explicitly qualified rule-50 tablebase adjudications are allowed and counted separately.

A passing confirmation supports an improvement under this internal TB6/64-simulation protocol. It does not establish external wall-clock strength or repeatability across training realizations. External nonregression and adoption need separate prospective evidence and review. The months-long strength objective stays open.

## Offline diagnostics

On the fixed one-root-per-game cohort, compare donor and both arms at steps 1, 8 and 40. Report expected-score squared error, draw Brier, three-class WDL Brier and legal-policy total-variation drift relative to donor. Expected score is `W + 0.5D`; these reporting units differ from signed categorical Q. Use 10,000 paired game-bootstrap samples with the frozen seed. Report incomplete outcome eligibility and exact overlap scope. These measurements are diagnostic only and are not a screen, confirmation, selection or adoption gate.

## Seeds, cost and stopping

Freeze generation seed `20261113`, split salt `E2FULL-whole-game-bank-v1:20261112`, root-selection salt `E2FULL-root-v1-20261112`, common draw seed `2026111201`, mirror seed `2026111202`, and common Python/NumPy/Torch CPU/CUDA seed `2026111203` after fresh construction and before prefetch. Restore the common state once for the second arm, without forced per-step RNG resets. All three screens use arena seed `2026111204`; confirmation uses `2026111205`; offline bootstrap uses `2026111206`. These define one realization, not independent replications.

| Stage | Maximum GPU-intended stage seconds |
| --- | ---: |
| Added-observation A/A qualification | 180 |
| 96-game collection | 1,200 |
| Both learning arms together | 1,200 |
| Offline evaluation | 300 |
| Internal arena engineering qualification | 180 |
| Three fixed screens | 3,600 each |
| One conditional confirmation | 7,200 |
| Maximum total | 21,060 |

The exposure index has a separate 300-second CPU stage wall ceiling; internal tablebase qualification has a 120-second CPU stage wall ceiling. Record actual elapsed time, failed stages, VRAM and data-inclusive costs. Previous arena costs are feasibility anchors, not forecasts for this new profile. Keep the week's 172,800 GPU-intended-second ceiling and November 19, 16:09:20 UTC deadline, one GPU experiment at a time, canonical leases, two conservative CPU threads and no paid compute. Honor all stop/pause/yield and intentional-stop controls; never clear them. Preserve the donor and stopped neighboring runs.

Any missing source, native, asset, data, owner, control or preregistration binding refuses the affected stage. A timeout or partial run is inconclusive, not a clean negative. Do not automatically repeat an interrupted or user-stopped job. A negative/inconclusive scientific result leads to a separately registered next decision, without ending the research program.

## Final binding before research compute

The frozen scientific JSON binds the source, configurations, donor, index, roster, books, helpers and completed A/A qualification. Publish this record on main before research collection. The collector's final actual profile/owner packet and resource checks remain a separate collection gate. Exact cohort, targets and playing-start exclusion gate learning. Actual TB6 and arena engineering qualifications gate deciding playing. The excluded A/A operation is not research evidence, and this registration does not admit execution.

## Frozen binding register

Artifact paths below are relative to the week-one external evidence bundle
`~/chess-artifacts/operations/rl-week-one-20261112-v1`. They describe prepared
inputs; qualified collection, target delivery, learning and playing require their
separate admitted owners and actual readouts.

| Identity | Artifact | SHA256 |
| --- | --- | --- |
| Accepted bounded old-input index | `e2full-exposure-index-results-v1-162f4f8c/index.json` | `5483fc9361a6e3a8840a5e27f2e1c2d34172c28593b56b88dce9850eee421e2b` |
| Index independent acceptance | `e2full-exposure-index-root-readout-v1/readout.json` | `777186b263e9ac1e3c995b37058ac546b201aefd71723a596d4846fce3dec2c2` |
| Final 96-game roster | `e2full-cohort-final-roster-v1/manifest.json` | `24011e8260f76d06252239395de5116dfeca7764eae1bd8ec53cb26ac666525f` |
| Final playing books | `e2-playing-books-final-v1-162f4f8c/manifest.json` | `38956f04ca2ec38cfed699222a91af9fddcdca28a8d9cf06bcad9303cdc5e66e` |
| Shared 128-pair screen | `e2-playing-books-final-v1-162f4f8c/screen.epd` | `974e1c7569727a97bf5d90d356623189795082deb2ca76041d4da4a81045bb67` |
| Disjoint 500-pair confirmation | `e2-playing-books-final-v1-162f4f8c/confirmation.epd` | `5eb478420f952692ed69762bf840a479241636cd1f1f8d8a096ff1cc626d15b1` |
| Roster/book independent acceptance | `e2full-roster-books-root-review-v1/review.json` | `301a074b70b212995879bbe929123b4e4987d068679c154d65260aa56a474505` |
| Frozen scientific JSON | Repository evidence linked above; external `first-E2-frozen-preregistration-v1/preregistration.json` | `7d47cb547d8f1b0b96299dbcc097a31e8d9eab7cfd8522d1904ad88ff8f251d6` |
| Four-update A/A actual independent acceptance | `e2-aa4-actual-independent-readout-v1/readout.json` | `fc114dcb839e834f8c194368f3429db7b9057484ebdea97fd5e677116cd2c9d9` |
| Subsequent fixed TB6 CPU consumer acceptance | `internal-tb6-consumer-qualification-root-readout-v3/readout.json` | `085bafd4120d78dce95fdc7fc7e125efd1be088358363eb64e6653462345bdc5` |
| Full neural arena engineering qualification | **PENDING actual complete readout** | **PENDING staged execution gate** |

The source revision is `6d7f2e6d1b1fc80aa9df2ee5b922eaa3fabf0814`,
using managed Python 3.13.15 and Torch 2.14.0+cu130. Model/Trainer/search/encoding
and native identities remain pinned to the actual measured runtime, separately
from this documentation commit. Native/source authority is
`fresh-game-pilot-preparation-v4/manifest.json`, SHA256
`b3a775d7bdf53e67d090a55f566adb93226a6c5e743b653908feb830e79ea821`.

The inherited configuration and playing-shape files are under the previous
operation's external artifact root, in
`cap40-config-preparation-v1/fe23025d-a55d-4e00-8eae-a001036c0755/`.
Their immutable hashes bind nominal settings; each stage must qualify its actual
explicit overrides, including pure-selfplay/TB-off collection and the declared
target bundle. These are not instructions to launch the old cap-40 pipeline.

| Component | Immutable SHA256 |
| --- | --- |
| Configuration `config.yaml` | `683addc8519057a2f5089a76e43934847c1cd0ba2e860b6e4a2e86cd6dc03faa` |
| Playing `evaluation-shape.yaml` | `71b4cb1be3d4249887f36b640f48c6987f0ca13437f759657b97e160b474eae3` |
| Owner `run_stage_game_provenance_v1.py` | `4507e6268e04bf096fd2add0dc7a4fc45bb720bc20a7a3ad95e8f1b8f758cbfa` |
| `collector` (`collect96.py`) | `2fc1516ab38d9c5c0598f01066675dee3ee5fb061b3cb2c7d3938db12dcea75d` |
| `raw_value_factory` (`value_bundle.py`) | `4a7a8cbae7f779b00e9ba046b3c83b0da98f4c8a557bf87555b53aba214f684c` |
| `paired_caller` (`run_paired.py`) | `c59ea8327546f76c807e2c7b2b221d9e777fb0b2c17cd8cfc1cf5c25e38adfbf` |
| `paired_observer` (`paired_observer.py`) | `92cd4e8f7125acb8dd5d7a3079803f9846efddbbebe4820533e3d9380b78116a` |
| `TRAIN_loader` (`train_inputs.py`) | `2bd8acc93e4a3e70c019bbf853ddbb655b96bc47a1f442b9bb1dd175b5487614` |
| `selected8` (`selected-parameters.json`) | `fbb738aaf19b30520ef92a55c93a77eed9971c3904abbb2e18b2479f2a4e79ff` |
| `postcollection_retained_playing_start_check` (`check_retained_playing_overlap.py`) | `a610f7fde662081a5bbbb6dad9140907cae15876fd47f6f2de1a8578c083aff1` |

The JSON retains exact external source paths as frozen provenance; portable prose
uses artifact-relative paths. Its future-stage requirements do not claim stage
completion at the scientific freeze. The subsequent bounded TB acceptance above
is distinct from the pending full neural arena qualification. No research data
or learned endpoint has been selected. The scientific registration
was independently reviewed against the prose, actual source and artifact hashes;
execution remains separately controlled.
