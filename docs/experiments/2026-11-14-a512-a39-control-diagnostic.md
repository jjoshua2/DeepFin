# Outcome-only A512 versus A39 control diagnostic

Status: **COMPLETE PROSPECTIVE PREREGISTRATION; NEURAL PREDICTIONS UNRUN AT REGISTRATION.** Source, guard and opening-data qualification are complete. Main publication and a fresh phase/resource admission are required before play. This separate diagnostic does not reopen the closed native-SF .2 pilot.

## Question and decision

The [native-SF .2 pilot](2026-11-14-sf-current-p0-native020-pilot.md) closed with both mandatory screens qualified NONPASS. B512 scored 50.1953% against its matched outcome-only A512 control and 51.1719% against unchanged A39, on different panels. Those measurements do not determine A512's score against A39. This diagnostic measures that missing contrast to inform the next training experiment.

Candidate A512 is the already defined alpha-zero control, selected for its control role. Its checkpoint SHA is `b346d95fe976e6ab1c35524faee52f9e1e77cc7d6e721ee798ff539b719a4a18`, with actual step 512. Reference A39 is unchanged donor SHA `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`, actual historical step 680276. No checkpoint search or further training is included. A512 used fresh NadamW and the fixed original-G1 bank, with search-target and all twelve auxiliary-loss coefficients zero; this was not an exact continuation of A39's historical training recipe.

One fixed endpoint uses **500 opening color pairs / 1000 games**, seed **2026111851**. Each pair contributes the mean candidate score of its two color-reversed games. The estimator retains the existing unbiased pair variance, normal radius `1.96 * sqrt(variance / pairs)` and interval clipped to [0,1]. Its preexisting zero-variance/insufficient-pair Hoeffding fallback is unchanged. This is an approximate conditional-profile interval, without an exact coverage or global-error claim.

| Diagnostic class | Fixed condition |
| --- | --- |
| UPWARD_CONTROL_CONTRAST | Candidate mean ≥ .53 and paired 95% lower bound > .5 |
| DOWNWARD_CONTROL_CONTRAST | Candidate mean ≤ .47 and paired 95% upper bound < .5 |
| UNRESOLVED_CONTROL_CONTRAST | Every other technically qualified result |

All classes are diagnostic only. An upward contrast does not promote A512, rescue B512, establish teacher benefit or qualify an external-engine strength milestone. A downward contrast motivates investigation of the shared continued-training recipe; it cannot identify the responsible loss, optimizer, schedule, data coverage or target by itself. An unresolved contrast retains that uncertainty. No class allocates confirmation, extends the panel, selects another checkpoint or changes the closed pilot's verdict.

## Playing instrument

Use the same qualified internal training-shape profile: 64 Gumbel simulations, root noise on, temperature .1, eager BF16 autocast with F32 input/transport, 175 input planes, complete 494-entry model state and 61,444,448 unique parameters. The 32-game pool uses the two-slot DirectGPUEvaluator with maximum leaf batch 1024. Preserve each opening's full 16-ply history, 450 additional plies, and strict 3–6-piece WDL plus DTZ/rule-50 adjudication. Unresolved games are not imputed as draws.

The actual child is frozen L revision `090b5143315a0faef7ab0ac1a90e2d5cebf3b929`; R7 revision `ba9d6de1987902e1759d984532a4c63f091b9269` supervises the owner. The producer learner was D6 `d6b9b978b36faaeeb4bcfcc6ce3f6333bef7323a`. These distinct roles do not assert whole-runtime equivalence. Authenticate actual checkpoint bytes and complete loaded-versus-served state before/after this phase, actual input/autocast settings, immutable assets and final model closure. The separate CPU reader authenticates all games, histories, color pairs, PGNs and strict results without loading a model or initializing CUDA.

## Opening custody and scope

The new role is `A512_A39_CONTROL_DIAGNOSTIC`. Reuse the accepted full 16,384-entry encoded ranked pool in its original immutable order. Reapply its authenticated retained-use exclusions and protect all older 4096/retired/reserved starts plus **all 884 roles from the closed native-.2 pilot**, including its unused 500-pair confirmation reserve. Exclude protected source groups, parents, histories/EPDs and raw/mirror identities. Select the first 500 eligible distinct source groups and parents with 1000 distinct raw/mirror identities; do not generate another ranking, reencode the corpus, choose by outcomes/model scores or substitute a different corpus.

The bounded CPU selector has closed successfully and received root and different-author independent actual acceptance. It validated all 16,384 cached records and selected original ranks 884–1383: 500 source groups, 500 complete parent identities and 1000 raw/mirror identities. All six selected-versus-protected collision counters were zero. The fixed book SHA is `0384e1295bcbf91ad2bf026273351d88bbb87e2c11cc9443b490b7fe6d6f5897`. The exact public metadata-consumer function bodies and opening module verified all 500 full histories; that qualification did not invoke the complete neural arena or reencode the pool. Disjointness applies to the named recorded use indexes and protected starting identities. Later played trajectories, unretained search identities outside those indexes, external/pretraining exposure and broader family overlap remain unknown. There is no universal unseen-position claim.

## Bounds and failure handling

No training, HOLD rescoring, live adoption or external-engine match is included. A separate CPU engineering scope permits selector owner ≤300 seconds and synthetic guard controls ≤60 seconds, total ≤360, including all failures. The scientific scope permits one GPU owner ≤4500 seconds (inner ≤4470) and one CPU game-reader owner ≤3600 seconds (inner ≤3540). Source review and preparation are not compute admission.

Use fresh study UUID `38dfe848-415b-4a12-b8e8-d9507cab22f3` and a new exact-stage GPU scope, subject to the unchanged week-one deadline of November 19, 2026, 16:09:20 UTC and total GPU owner-wall cap 172800 seconds. Old study scopes cannot be inherited. Root remains the sole numerical launcher, with CPU affinity 20,22, two framework threads, all three existing leases, resource preflight and STOP/PAUSE/CANCEL/YIELD controls. User authorization permits bounded GPU work alongside COD; no idle-GPU gate is introduced.

No interim score peek, efficacy/harm stopping, automatic retry, reseed, panel extension, pooling or checkpoint replacement. A technical failure preserves all partial evidence as TECHNICAL_UNREAD; source diagnosis and any successor require a separate bounded disposition. The published record will append the actual closed result and costs. The broader goal of beating unhandicapped multicore Rofchade and approaching Stockfish/Lc0 remains active regardless of this diagnostic's class.

## Closed prerequisite qualifications and frozen identities

The single CPU guard owner passed all **31 cases** (16 adapter and 15 selector), with zero failures, errors or skips, in 1.2064751900034025 seconds. The selector owner closed in 6.345103987958282 seconds. Total engineering owner wall is **7.551579177961685 / 360 seconds**; the closing census contains 265 closed stages. The selector read 170,570,297 metadata bytes and recorded peak RSS 370,479,104 bytes. No neural prediction, training update or tablebase query was performed in these prerequisites.

Independent source review found that the unrun V1 arena wrapper named two selector-manifest fields incorrectly. V2 consumes the actual `helper` and `direction` keys and adds a regression case against that exact schema. V1 remains preserved as source-only CHANGES_REQUIRED. The execution observer, strict game/PGN/tablebase processing and paired-score implementation are unchanged; no failed runtime or scientific attempt is being rehabilitated.

The final scientific protocol changes only seven allowed identity/status leaves from its independently reviewed template. All model, panel, seed, classifier, search and resource choices are unchanged. The new model envelope retains the exact accepted producer A512 receipt chain. Later phase binding may fill only the declared publication/authority/science/model/role/book identity leaves and apply the separately admitted execution flag. Those late receipt identities cannot change the scientific choices. Default-CUDA repeatability limitations remain recorded in the [prior qualification](2026-11-14-sf-current-p0-cuda-qualification.md); no bit-repeatability acceptance rule is introduced.

External artifacts use `~/chess-artifacts/operations/rl-week-one-20261112-v1/`. Bulk weights, books, role rows and future raw games remain outside Git. Independent actual prerequisite reviews reconciled compact reports, source contracts and closed owners; they did not independently reread the full book, role rows or cached pool.

| Artifact | SHA-256 |
| --- | --- |
| `core-A512-A39-control-diagnostic-direction-v1/decision.json` | `f041dea202fe63ad381b9216e23b4428ee81a2d6eb9bfdea27bdd577371a0770` |
| `core-A512-A39-control-diagnostic-direction-independent-review-v1/review.json` | `b3c6d624ef5726daa833eae6d1e1ad77b7e5b6d02f948ae90d2c1bb78679cae8` |
| `core-A512-A39-control-diagnostic-scientific-freeze-v1/protocol.json` | `4d34d3621a17122ca09a39e2ac2a538041bffda61866e6f718feae9cc224e685` |
| `core-A512-A39-control-diagnostic-scientific-freeze-v1/root-freeze.json` | `f2e03b54e5a4b49358d32161beddbe724f7d605a31e1b46574bfb4e1b90cb451` |
| `core-A512-A39-control-diagnostic-playing-preparation-v2/manifest.json` | `5afc205ccc5b08313d0bfbfa7424bb0450af57c2b35ac18c4664da82723ea771` |
| `core-A512-A39-control-diagnostic-playing-preparation-v2/protocol.TEMPLATE.json` | `2213cd7e155e15a31ba3a4a1ee8e67de013888b5ccdade57d2efb38da4bd9fc5` |
| `core-A512-A39-control-diagnostic-playing-preparation-v2/commands.TEMPLATE.json` | `be0fa09bd621875cee5cb6a3e52a1a95afaf8596af06385948edac058aee7814` |
| `core-A512-A39-control-diagnostic-playing-independent-source-review-v1/review.json` | `a967e4da42f6a489efeb01e4ce24c57eb220301b7c54809a9be57f5fa20283bc` |
| `core-A512-A39-control-diagnostic-playing-independent-source-review-v2/review.json` | `593fb14f7b6670914288fbfb32c506dcb027f7f2f6f6e19d6bdc1518fd307e4e` |
| `core-A512-A39-control-diagnostic-role-selector-preparation-v1/manifest.json` | `f969912431c291c43abb74af14830246ce7c4b762a4f775e69d9e84bc85a1ea7` |
| `core-A512-A39-control-diagnostic-role-selector-independent-source-review-v1/review.json` | `b2b031e763b263e6c79f1d26f3d8e3ec6f062bd099c2fcc20bf0abccc91b9630` |
| `core-A512-A39-control-diagnostic-bindings-v1/controls/root-acceptance.json` | `0be510d476172e398c288ddce6c308eb3607f9c7f3adf3b2215b7a8950aa2454` |
| `core-A512-A39-control-diagnostic-guard-controls-independent-actual-review-v1/review.json` | `27843251057e643bfb15279f689c7423d27124295b8e7226760b7995cf581ffd` |
| `core-A512-A39-control-diagnostic-role-selector-results-v1/manifest.json` | `e1b44054ddcf55052734fedd9c591c1c116487bbafab62df33454ca6c4ffe600` |
| `core-A512-A39-control-diagnostic-bindings-v1/roles/root-acceptance.json` | `5d53a878cf77e9ee723e3a9f56b14f56cfef4213c37dd53e06855da33bed7959` |
| `core-A512-A39-control-diagnostic-role-independent-actual-review-v1/review.json` | `31a9a32d7c25fc4d4b0b6c778c92d7f3ba341ad164badeb073a0c98ab7fa4727` |
| `core-A512-A39-control-diagnostic-bindings-v1/models-binding.json` | `426d6e20fe18a7e7ad55f02528e6b61fb278ea1d89bea2f287a9388bec402731` |
| `core-A512-A39-control-diagnostic-bindings-v1/GPU-scope-plan.json` | `1666400fce0b25545fe35ac584a7925463735671d85118bf39fb32584af0276d` |
| `core-A512-A39-control-diagnostic-bindings-v1/resource-selection.json` | `ab5a3be72fce427260c8badd6b71f092e29d63886aba0d7c428df3844b0db40f` |
| `core-SF-current-P0-native020-pilot-closeout-v1/decision.json` | `8c43fc5a052a1b701705b56a59266de73f325b2833e76934b07e4cab1c8ce2fc` |
| `core-SF-current-P0-native020-playing-readout-main-publication-v1/publication.json` | `7b80b7e359f0e8842e3071c2509ab3523efedc5c4c4f303b48b5afec56ac7dbd` |


## Closed readout — November 14, 2026

The registered diagnostic completed all **1000 games / 500 color pairs**: A512 won 300, drew 413 and lost 287 against unchanged A39. Its score was **50.65%**, with the registered paired approximate 95% interval **[48.348010390918184%, 52.95198960908181%]**. Pair variance was 0.06897069138276554. The fixed classifier returned **UNRESOLVED_CONTROL_CONTRAST**: neither the upward nor downward rule passed.

This provides no confirmed improvement or regression at the tested fixed-bank 512-update horizon. It does not establish equivalence, no effect, teacher quality, an optimal recipe, global Elo or an external-engine milestone. The longer-horizon RL regressions remain unexplained. A39 remains the donor; A512 and B512 remain unadopted. This closes this diagnostic, while the first-week and longer strength goals remain active.

Both owners closed PASS. GPU owner wall was **2650.5146221950417 / 4500 seconds**; the separate CPU game reader used **40.23353753297124 / 3600 seconds**. The already reported metadata/guard engineering cost remains **7.551579177961685 / 360 seconds**. All 446 rule terminations and 554 strict Syzygy terminations passed the prescribed game/history/role/PGN audit. The serving process loaded both complete 494-entry, 175-plane, 61,444,448-unique-parameter models, used the fixed eager BF16/F32 transport profile, and performed zero updates. Served-state hashes matched the accepted CPU checkpoint state before and after serving: A512 `e67a51c23751272eb42369d27e7bd5c8055d707e8e228a1c3498e8a28118464f`, A39 `9796079cd73bef73ba2f5d5f2ad0779704498832e34a7c2a880f0d0b7f305229`.

The prospective record was merged through [PR #1323](https://github.com/jjoshua2/DeepFin/pull/1323), main commit `f24e1582b6a1629d310f0d2444bab2957b07ad3d`, before the GPU owner began at 2026-11-14 21:40:50.931 UTC. Independent actual review accepted the closed endpoint with no findings, using the compact reports, pinned source and terminal evidence; it did not independently reread the bulk raw payloads.

No further admissions, retries, reseeds, extensions, checkpoint selection or pooling are allowed under this closed study. All 500 consumed diagnostic roles remain protected. The prior native-.2 pilot remains closed NONPASS, with its unused 500 confirmation roles protected; this diagnostic cannot rescue it or authorize confirmation.

| Closed evidence under the same external artifact root | SHA-256 |
| --- | --- |
| `core-A512-A39-control-diagnostic-prereg-main-publication-v1/publication.json` | `5cfe080224544ab286b6f2caede9849ea0c8c2b637f764fce2a07a3e1be6cdb6` |
| `core-A512-A39-control-diagnostic-bindings-v1/A512_vs_A39_control_characterization/root-acceptance.json` | `4cb84e73fec6dbde474da12c96d0daa25f67e43af67185f6b3979cb5f396876d` |
| `core-A512-A39-control-diagnostic-readout-v1/readout.json` | `2e2ff30794fbcbe3bfa3041555fb5ea83e31484ab99e9b10ee6fcc40a409384a` |
| `core-A512-A39-control-diagnostic-independent-actual-review-v1/review.json` | `1724126399f214db376232e75124c4438e4a319fcbdaca6911535416b4175a9e` |
| `core-A512-A39-control-diagnostic-closeout-v1/decision.json` | `1d370c335c21124cb34b691e91cd6be5e81c49b2fbc8d300e8d38df96d48a89a` |
