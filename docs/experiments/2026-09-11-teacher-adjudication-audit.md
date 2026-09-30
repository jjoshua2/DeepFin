# SF / BT4 / Ceres teacher-adjudication audit

Registered September 11, 2026. This record is stacked on the Tactical300 retrospective calibration in PR #643. It broadens the question from one proposed >300cp override rule to the more general bootstrap-design problem:

> **what information should saved Stockfish search contribute when one-node neural teachers already provide strong policy/value targets?**

No new Stockfish search, BT4 inference, Ceres inference, target materialization, training, arena, live configuration change, or playing-strength claim is included in this PR.

## Motivation

The current development policy incumbent is B100: pure one-node BT4 policy sharpened at teacher temperature 0.5 with the historical SF-derived value target. The earlier broad Tactical100 attenuation experiment was unresolved at -13.6 Elo versus B100 [−52.4,+24.9] at 400 simulations. Tactical300 therefore tests a different sparse/strong correction, but its d9-only trigger is now blocked pending PR #643's deeper-search calibration.

Separately, the registered Ceres experiments are deliberately simple first anchors:

- CeresB50: 50/50 BT4/Ceres policy, each at teacher T=0.5, original SF value;
- B100CeresV25: fixed B100 policy with 50% SF / 25% native BT4 / 25% Ceres value, where Ceres is 60% primary@0.55 + 40% secondary@1.5.

Those mixtures are useful controlled baselines, but their weights are hypotheses rather than evidence that the teachers make complementary mistakes or are calibrated optimally.

## Questions

This audit addresses six related questions without spending a training epoch.

### 1. Does target surgery actually move probability toward better saved-search moves?

For every ordinary row where the final saved d10/d12 roster is available, report policy mass covered by that roster and, conditional on the covered mass:

- expected centipawn regret to the best rescored move;
- probability mass on the final best set.

Always report covered mass because G10's later rosters are narrowed. Missing moves receive no invented score.

Compare:

- B100 / BT4 T=0.5;
- the ordinary Tactical300 >300cp / 50% transfer preview;
- when a row-aligned Ceres bank is supplied, Ceres T=0.5;
- 50/50 arithmetic BT4/Ceres;
- 50/50 geometric/log-opinion BT4/Ceres on common positive support.

This is **target-geometry evidence**, not expected Elo. A lower conditional regret is useful for allocating training experiments, but cannot promote a recipe by itself.

### 2. Is SF more trustworthy as a ranking constraint than as a probability distribution?

For d9 best-versus-inferior move pairs separated by strictly more than 100/300/500/1000 effective cp, count:

- constraints whose two sides are both available in the saved final roster;
- constraints confirmed by deeper SF;
- constraints contradicted by deeper SF;
- ties;
- unscored constraints retained separately.

This directly informs a possible auxiliary policy-ranking loss such as

`L = L_neural_policy + lambda * max(0, margin - (logit_sf_best - logit_bad))`

without asserting that SF supplies the correct normalized policy probabilities.

A later ranking-loss training experiment must use a separately registered lambda/margin and historical matched control. This audit does not add that loss to the trainer.

### 3. Can teacher conflict route deeper SF efficiently?

Treat a saved deeper correction as observed only when the complete d9-best set is actually present in the later roster and another rescored move beats it. For that adjudicable population, retrospectively simulate these cheap routing rules:

- search everything;
- BT4 top set disagrees with d9 best;
- d9 best-vs-second margin <=100cp;
- B100 top probability <0.50;
- conflict OR low d9 margin;
- conflict OR low B100 confidence.

Report each rule's selected/search fraction and fraction of observed reversals captured. These are retrospective controller diagnostics, not a claim that the same rates transfer to 100M/1B future data.

### 4. Are BT4 and Ceres complementary or mostly redundant?

The audit can optionally consume an exact row-aligned completed Ceres manifest for the same G10 derived source. It verifies each Ceres bank through the existing qualified collector contract and actual source-array/legal/feed alignment before use.

Where Ceres exists, report:

- top-set agreement/disagreement;
- Jensen-Shannon divergence;
- BT4 and Ceres policy entropy;
- deeper-SF adjudication of disagreements when both top sets are actually scored;
- conditional regret for each teacher and arithmetic/geometric mixtures.

The existing original-18.91M Ceres bank is **not** silently treated as G10 coverage. If no exact G10 Ceres manifest is supplied, the BT4/SF audit still runs and all Ceres metrics remain empty.

### 5. Are the registered value weights/head temperatures well calibrated?

When the adapter manifest includes native BT4 WDL **and** an exact G10 Ceres bank includes both value heads, use the final saved SF score to construct a comparable deeper-SF WDL target with the same historical cp-to-WDL mapping. Report Brier and cross-entropy losses for:

- saved SF WDL;
- native BT4 WDL;
- Ceres primary@0.55;
- Ceres secondary@1.5;
- the registered 60/40 Ceres dual-head blend;
- the registered 50% SF / 25% BT4 / 25% Ceres mixture.

This can identify obvious redundancy/calibration failures before training. The deeper-SF-derived WDL remains a teacher target rather than game-theoretic ground truth, so these losses do not by themselves choose production weights.

### 6. Which positions deserve a bounded Ceres-on-G10 label spend?

Every audit emits a deterministic bounded selection of at most 4,096 rows: 1,024 smallest identity hashes from each preregistered stratum:

1. observed deeper reversal;
2. d9/BT4 disagreement with d9 gap >300cp;
3. other d9/BT4 disagreement;
4. agreement control.

The selection stores exact derived/raw physical identities and input keys. It is **selection evidence only**: it is not the existing qualified Soft-SF selected bank, does not satisfy a Ceres collector admission profile, and launches zero Ceres inference. A later collector PR must bind this selection and independently qualify its row materialization before GPU use.

## Position-type audit

The same pass reports descriptive reversal/regret strata for:

- in-check versus not-in-check;
- <=10, 11–30, and >30 legal moves;
- ply <40, 40–79, and >=80.

These are intended to reveal simple interpretable specialization before considering a learned router. No phase-specific rule is selected in advance.

## Implementation

`scripts/teacher_adjudication_audit.py`:

- consumes the same SHA-pinned `adapt_raw_bt4_sidecars` schema-1 manifest as PR #643;
- reuses its authenticated raw source / BT4 sidecar / row-provenance join;
- reconstructs B100 at teacher T=0.5 without new inference;
- verifies source game/ply/input/legal identity before analysis;
- retains narrowed-roster coverage instead of assigning values to omitted moves;
- computes policy geometry, ranking-constraint, routing and position-stratum aggregates;
- optionally verifies exact G10 Ceres banks and computes three-teacher policy/value diagnostics;
- emits `teacher_adjudication_ceres_selection.json` with the bounded deterministic sample;
- publishes atomically into a fresh output namespace.

Focused tests cover roster-conditional regret, JS/arithmetic/geometric mixtures, strict Tactical300 300/301cp behavior, ranking contradictions with unscored moves, routing capture accounting, deterministic bounded selection, value losses, and selection-stratum priority.

## Interpretation / next experiments

This audit should choose among **orthogonal hypotheses**, not start a broad tuning grid.

- If high-gap SF pairwise constraints are rarely contradicted on an adequately sized adjudicable set, a small auxiliary ranking-loss experiment is more attractive than another normalized SF-policy mixture.
- If Tactical300 preview systematically reduces conditional deeper-search regret while PR #643 shows few deeper reversals of the d9 winner, its training experiment becomes better motivated. If it raises regret, do not train merely because the mechanism exists.
- If a cheap conflict/uncertainty router captures most observed reversals at a much smaller search fraction, register a prospective adaptive-depth SF collection test before future 100M-scale labeling.
- If BT4/Ceres disagreement is common and deeper SF materially vindicates each teacher in different subsets, test an interpretable router or conditional mixture after the clean CeresB50 anchor. If they are nearly redundant, prefer the stronger/simple teacher instead of averaging by default.
- If the registered Ceres value mixture loses offline calibration to one of its components or the secondary head contributes no visible benefit, revise the next value hypothesis before spending another full training epoch. Do not retroactively change the already registered first Ceres value anchor.

No single retrospective threshold here is a promotion criterion. Any new policy target, ranking loss, router, value weighting, or adaptive-search controller requires its own prospective recipe and matched strength experiment.

## Boundaries

- Saved d10/d12 searches are narrowed, adaptively selected, and may share historical TT state. They are calibration evidence, not truth.
- Policy regret is conditional on the saved roster and must be read together with covered mass.
- The audit uses one-node neural teacher outputs; it does not change live Gumbel search.
- Original source/history, single-seed and development-panel caveats remain unchanged.
- Existing CeresB50 and B100CeresV25 remain the clean first Ceres strength anchors after their full original-corpus teacher bank qualifies; this audit informs second-generation recipes rather than silently replacing those registrations.
