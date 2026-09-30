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

### Estimator corrections before the first substantive pass

Schema 2 preserves the original Tactical300 preview and selection strata, but
makes their interpretation explicit. `d9_best_minus_next_lower_cp` is the best
score minus the next strictly lower score. The separate BT4-top best/worst gap
fields report both endpoints when neural top probability is tied; neither is a
renaming of Tactical300's gate.

Each policy cell counts all ordinary rows, unavailable policies, invalid final
rosters, coverage rows (including zero coverage), zero-coverage rows and regret
rows separately. Mean coverage uses coverage rows; mean conditional regret uses
only positive-coverage rows. It is an equal-row mean after each policy is
renormalized on the saved roster. Compare teachers on the common valid rows
using the banked paired regret differences; optional Ceres or empty geometric
support must not silently change the comparison population.

Unknown reversals remain null. Position reversal rates divide by adjudicable
rows. Routing reports both selection among all ordinary rows and its historical
`search_fraction` among adjudicable rows. Neither estimates prospective compute
savings or outcomes for moves never rescored.

Flat ranking counters retain their existential best-set meaning. The nested
`all_winner` counters conservatively require every tied d9 winner to beat an
inferior move; any losing winner contradicts that constraint. Missing winner or
inferior scores remain unscored. Each inferior move's BT4 probability is counted
once, with confirmed/contradicted/tied/unscored mass at the existing strict
100/300/500/1000-cp thresholds. Counts and probability mass on missing moves
remain visible; narrowed final rosters cannot establish their reliability.

`teacher_adjudication_rows.jsonl` banks every authenticated row, including
explicit mate-domain exclusions. The terminal summary records its count and
SHA256. Rows carry the qualified source namespace, raw shard/physical row,
worker/game/ply and input keys, derived location, both SF-gap concepts, absolute
best score, tied-best cardinality, final depth/reason/mate status, confidence,
coverage/regret, common-row paired differences, reversal/routing outcomes,
ranking counts/mass and actual Tactical300 probability mass moved. This compact
metric bank supports subsequent game-clustered analyses without another raw join.
It is diagnostic output, not a teacher-label or training admission. A failed
pass leaves its partial bank under `.writing`; only the successful terminal
summary authenticates a complete bank. The native-WDL/Ceres join remains separate
work, and no value usefulness or playing-strength claim follows from these fixes.

### Optional authenticated native BT4 value input

`--native-wdl-manifest PATH --expected-native-wdl-manifest-sha256 SHA256`
adds the existing historical native-WDL bank to a policy-only raw adapter. The
manifest must admit the complete original G10 cohort through its existing common
qualification, completed collection receipts and historical producer pins. Its
source, summary, BT4 model and `/output/wdl` head must match the audit's original
source and policy teacher. Only selected audit shards undergo payload checks:
source columns, game/ply identity, exact stored LC0 feed hashes, cached WDL content
and storage stability. Values are indexed by derived row, preserving the source
shuffle. A raw bank that also supplies WDL is rejected as ambiguous.

With both qualified Ceres heads present, the optional path banks all six existing
value losses, their saved final-SF WDL ruler and an explicit inclusion reason.
The registered temperatures and weights remain fixed: Ceres primary 0.55,
secondary 1.5, dual 60/40, and the combined value 50% saved SF, 25% native BT4,
25% Ceres dual. Mate-domain d9 rows remain excluded. Without the new flags,
existing policy and value behavior and metadata remain unchanged.

This joins already collected values; it performs no inference and grants no
training admission. Saved deeper-SF agreement measures this ruler's agreement,
not independent value accuracy or Elo. Native-WDL availability does not make the
convenience-prefix G10 panel representative. Actual value readouts require a
separate completed, pinned invocation; this implementation is not a result.

### First-four-shard native-value diagnostic preregistration

The next CPU pass supplies authenticated native BT4 WDL to the already qualified
first four original run06 G10 shards: 32,768 rows, selected as a convenience
prefix. The prior policy-only adapter supplied no value array, so its zero value
counts did not compare teachers. This pass fills that missing input using the
existing native collection and both saved Ceres heads; it collects no new neural
inference or Stockfish scores. The original source, shuffle and row namespace
remain unchanged.

The six fixed predictions are saved SF, native BT4, Ceres primary (temperature
0.55), Ceres secondary (1.5), their 60/40 dual, and the registered 50% SF / 25%
BT4 / 25% Ceres-dual blend. Compare Brier loss and cross-entropy against the
existing saved final-SF WDL ruler on common included rows. Report paired losses,
all-row inclusion/exclusion counts and ruler depth/reason. Retain each row's
source-qualified game identity, losses, ruler and inclusion reason for later
clustered readout; never cluster on bare game ID. Mate-domain d9 rows remain
excluded under the existing audit semantics. No temperature, mixing-weight or
sample-selection fitting is part of this pass.

This is a diagnostic of agreement with a censored, SF-derived ruler, not ground
truth, representative G10 performance or Elo. A lower diagnostic loss neither
promotes a training recipe nor cancels the registered Ceres anchors. Policy
outputs are incidental to the required same-row source join; no independent
policy rerun is being proposed.

The existing owned-process operator runs once with CPUs 2,3, GPU hidden, two
numeric threads, 8 GiB address-space limit, 150 GiB SSD reserve, 2 GiB sampled
output cap and a separate analysis lock. The inclusive limit is 15 minutes
(870-second outer TERM plus 30-second KILL); the child has at most 810 seconds.
The previous four-shard policy pass took 376.63 seconds with 2,000,152 KiB
peak RSS; the extra work is selected native-cache/content/feed verification. STOP, resource or
lineage failure retains failed output and does not trigger an automatic retry
or expanded budget. Only a terminal-success summary authenticates the new bank.
The concrete plan is `scratchpad/bt4_joint20/g10_ceres_first4_value_diagnostic_v1/plan.json`;
its final runtime commit and plan SHA must be frozen before validation or launch.

## Fixed >300 cp teacher complementarity diagnostic

The additive `policy_complementarity_gt300` row field compares BT4 T0.5,
Ceres T0.5 and their arithmetic 50/50 mixture on exactly the same moves whose
d9 score is more than 300 cp below the d9 maximum. It records unrenormalized
probability mass, partitioned by the existing conservative all-winner criterion:
confirmed, contradicted, tied or unavailable. Missing any d9 winner or the
flagged move from the final roster makes its outcome unavailable. Contradiction
means the move outranks at least one prior winner, not necessarily every winner.
Mate-encoded final scores retain the existing ranking semantics and explicit flags.

This field is null for excluded d9 mate-domain rows or missing Ceres; those rows
remain explicitly distinguishable through existing eligibility fields. Included
rows retain each legal move's policy index, UCI move, d9 and available final score,
mate-domain flags, both teacher probabilities and constraint outcome. Existing
source-qualified row/game identities apply; no feature or history copy is added.
The arithmetic mass is the mean of constituent masses up to floating-point
normalization. No conditional roster renormalization is performed.

The fixed question is whether Ceres removes probability from SF-flagged moves or
merely transfers it to other flagged moves, including within the existing balanced
`abs(d9_best_cp) <= 100` stratum. The per-move cutoff is distinct from Tactical300's
best-minus-next-lower d9 gap gate. Preserve unavailable mass, all-winner ties and
SF ruler dependence in the readout; this measurement cannot establish Elo.
A bounded first-four-shard saved-source run is prepared separately, retaining the
same original/native/Ceres bindings and all prior policy/value metrics. No new
inference, target rewriting, training promotion or threshold sweep is included.
