# First E2 RL comparison: search targets versus genuine outcome targets

Status: **PREREGISTERED. Execution admission remains separate.**

Scientific freeze: **2026-11-12 21:33:31 UTC**. The authoritative
[frozen scientific JSON](evidence/rl-week-one-20261112/first-E2-preregistration.json)
is copied byte-for-byte from the external owner packet, SHA256
`7d47cb547d8f1b0b96299dbcc097a31e8d9eab7cfd8522d1904ad88ff8f251d6`.
A prospective [command/classification addendum](evidence/rl-week-one-20261112/first-E2-precollection-addendum.json)
was frozen at **2026-11-12 22:17:15 UTC**, after the independent PR review and
before any research game or update. Parent bytes remain unchanged. The
[execution preregistration](evidence/rl-week-one-20261112/first-E2-execution-preregistration.json)
combines them, SHA256
`1e2ed5b098dea2e3b5829a08fa023f9a736b49cee618d1fc544ca6c0fd65a4a9`;
its additive record is SHA256
`8c66d3965f490c4cffd8a99520ab5c2bbe7e31ffb6ecb671bf9da8f88d95343e`.

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

## Exact arena commands and fixed-endpoint classification

The addendum's four `arena_commands[].argv_template` arrays are the exact
commands, including the source checkout, owner, environment, output paths and
`python -B scripts/arena_standard.py`. They use frozen source
`6d7f2e6d1b1fc80aa9df2ee5b922eaa3fabf0814`, owner `4507e626…`, the exact
`evaluation-shape.yaml` hash `71b4cb1b…`, CPU affinity 16/18, two OMP/MKL threads
and CUDA device 0. Compile is explicitly off; default root noise and rolling
pool remain on, with SPRT and resume off. Every profile argument is included in
the immutable arrays rather than left to an unspecified future command.

| Role | Candidate/reference | Games/pairs | Opening file | Seed | Inner/owner seconds |
| --- | --- | --- | --- | --- | --- |
| Primary | FULLB40/FULLA40 | 256/128 | `screen.epd` | 2026111204 | 3500/3600 |
| Eligibility | FULLB40/A39 | 256/128 | `screen.epd` | 2026111204 | 3500/3600 |
| Context | FULLA40/A39 | 256/128 | `screen.epd` | 2026111204 | 3500/3600 |
| Conditional confirmation | Same FULLB40/A39 | 1000/500 | `confirmation.epd` | 2026111205 | 6900/7200 |

Only `{FULLA40_CHECKPOINT_PATH}` and `{FULLB40_CHECKPOINT_PATH}` may be
substituted, with the qualified final step-40 files. Freeze both actual SHAs,
arm/donor/40-update receipts and all instantiated argv before the first deciding
arena. The same FULLB40 bytes must enter confirmation. Hash and legally load
all 128/500 full-history roots before launch, assert the exact requested pair
count, and require all 256/1000 games after closure. The reader's ability to
shrink/subsample a mismatched file is not an allowed fallback. Inner time limits
do not authorize a partial-score verdict, favorable extension or new candidate.

At the fixed complete endpoint, a primary FULLB40/FULLA40 paired-normal 95%
**upper** score bound strictly below 0.50 is evidence of a harmful bundle at this
40-update dose/profile: no confirmation or adoption. A corresponding complete
FULLB40/A39 upper bound below 0.50 is endpoint regression, not isolated proof of
the outcome-target mechanism. All three screens still run to their fixed counts,
except a global stop or technical failure; these are endpoint labels, not interim
looks or an early statistical stopping rule.

Without qualified harm, a non-pass is labelled
`UNREAD_FOR_REGISTERED_GAIN_AT_40_UPDATE_DOSE`. This does not establish
ineffectiveness, equivalence or an optimum. Actual observed scores and intervals
are still reported. Power is unmeasured; the dose remains 40 and cannot be raised
after observations. Incomplete/invalid runs are unqualified/inconclusive, and
zero-variance uncertainty remains unavailable for both gain and harm. Existing
point-score/lower-bound success and the sole conditional confirmation remain
unchanged. Intervals are conditional testwise summaries, not a familywise claim.

## Offline diagnostics

On the fixed one-root-per-game cohort, compare donor and both arms at steps 1, 8 and 40. Report expected-score squared error, draw Brier, three-class WDL Brier and legal-policy total-variation drift relative to donor. Expected score is `W + 0.5D`; these reporting units differ from signed categorical Q. Use 10,000 paired game-bootstrap samples with the frozen seed. Report incomplete outcome eligibility and exact overlap scope. These measurements are diagnostic only and are not a screen, confirmation, selection or adoption gate.

## Confounds and recovery

**Confounds:** this is a WDL-plus-categorical bundle, not head-isolated evidence.
It uses one new pure-selfplay cohort, one shared stochastic realization and a
uniform sampler rather than production's curriculum/prioritized replay. The
comparison holds those settings common between arms; it is not a causal contrast
with earlier cap-40 or eight-window runs. Known-input exclusion does not certify
epoch-two pretraining, unknown historical inputs or independent opening families.
Forty updates and these panels have no demonstrated power for a three-point gain;
low power is a possibility, not a measured result or a reason to change the dose
after observations. The offline endpoint cannot select the playing candidate.

**Recovery:** A39 stays immutable and neither branch is deployed. Preserve every
partial archive, failed stage, checkpoint, RNG/draw record and diagnostic. A
technical failure, timeout or explicit stop blocks the affected stage; it does
not authorize a retry, new game, replacement panel, seed change or extension.
Close owned processes and release existing leases through the admitted owner.
Any subsequent scientific decision needs its separate prospective record; do not
resume a stopped training process or replace the donor with either endpoint.

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
| Full neural arena engineering qualification | Pending at the 21:36 cutoff; subsequent accepted readout in the addendum below | `f504034609da9e7853439d250326274d9295e20700b6643ae8c53217a9817822` |

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

The collector, target factory, paired caller and observers are study-specific
external adapters, SHA-pinned to the source above. They are not supported tools
on main. Their code-review and qualification coverage is explicit:

| External review/qualification | Artifact | SHA256 |
| --- | --- | --- |
| Initial component/delta review, with required successor fixes | `e2-value-bundle-root-component-review-v1/review.json` | `388165cafaa67d4e79bced7b91a3325cca1c0490d9af226754be675f33271538` |
| Actual caller/loader and observer-delta review for excluded A/A | `e2-aa4-qualification-root-admission-v1/admission.json` | `539ddca7da947740c49f29345ad90cdd323212bf7e775c3b4b17da1a2bd765d2` |
| Independent actual A/A result, with scoped limitations | `e2-aa4-actual-independent-readout-v1/readout.json` | `fc114dcb839e834f8c194368f3429db7b9057484ebdea97fd5e677116cd2c9d9` |
| Final collector/profile and retained-input guard source review, not execution admission | `e2full-collection-inputs-root-review-v1/review.json` | `9c2172ebf7e3862c321849470be6b72ab85fe58972ad3fbd0eefd21841d661c1` |

These reviews and excluded-pilot results do not establish the future research
targets, 40-update delivery or learned quality. Actual stage guards enforce their
frozen file identities; the main scientific JSON is checked against its expected
SHA and exact Git blob before collection admission. A separate unit test that
merely repeats the same embedded hash would add no scientific qualification.

The user's week-one goal authorizes work within its stated scope and resource
ceiling. Root's per-stage admission enforces the existing controls and readiness;
it is not an added requirement for another human permission prompt.

The JSON retains exact external source paths as frozen provenance; portable prose
uses artifact-relative paths. Its future-stage requirements do not claim stage
completion at the scientific freeze. The subsequent bounded TB acceptance above
is distinct from the full neural arena qualification, which was pending at the
original scientific freeze and passed subsequently as recorded below. No research data
or learned endpoint has been selected. The scientific registration
was independently reviewed against the prose, actual source and artifact hashes;
execution remains separately controlled.

## Subsequent engineering snapshot and review resolution

At the addendum freeze, the separately admitted same-donor neural arena had
completed all 32 games: 22 qualified Syzygy adjudications, seven checkmates and
three next-move-only threefold claims. Its independent readout is
`internal-tb6-arena-engineering-actual-readout-v2/readout.json`, SHA256
`f504034609da9e7853439d250326274d9295e20700b6643ae8c53217a9817822`.
It qualifies the sampled instrument; it is neither a target-bundle result nor a
strength comparison. The earlier 21:36 week-one snapshot remains unchanged.

The independent [PR review](https://github.com/jjoshua2/DeepFin/pull/1241#issuecomment-6837102507)
requested exact deciding commands, harm/non-pass classification, explicit
confounds/recovery and direct baseline provenance. The prospective addendum and
this record address them before collection, preserving dose, targets, data,
sampling, seeds, books, success thresholds and compute ceilings. Its exact-source
parser/defaults and composite preservation were independently checked in
`first-E2-addendum-independent-review-v1/review.json`, SHA256
`e04ba8a6725f6bb8444747eb282851a2c940b58bb4a36f884486a1d73a839927`.
Neither registration is an automatic launch or adoption decision.
