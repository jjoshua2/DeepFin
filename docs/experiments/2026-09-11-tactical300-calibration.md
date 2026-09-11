# Tactical300 retrospective BT4-vs-deeper-SF calibration

Registered September 11, 2026. This record is stacked on the Tactical300 mechanism
PR and must complete before any real Tactical300 corpus is materialized or trained.

## Question

The Tactical300 candidate assumes that a very large phase-zero d9 Stockfish gap is
strong enough evidence to move substantial B100 policy mass toward the d9 winner.
That is not automatically true. A one-node neural policy can encode strategic
information that a shallow search has not resolved, and the existing G10 evidence
already shows that deeper saved searches often change the preferred move set.

The calibration asks the specific retrospective question needed for this recipe:

> When d9 Stockfish strongly prefers one move/set, but B100's one-node BT4 policy
> prefers a different move/set, how often does the already-recorded d10/d12 search
> subsequently score the BT4 choice above the d9 choice?

This is a calibration of an intervention, not a claim that d10/d12 is ground truth.

## Inputs and provenance

Use `scripts/tactical300_calibration.py` with an existing schema-1
`adapt_raw_bt4_sidecars` manifest and its exact SHA256.

The diagnostic deliberately reuses the adapter's trust boundary instead of accepting
an ad-hoc policy dump. For every used raw shard it:

- pins the raw source manifest and immutable completed sidecar receipts;
- verifies the BT4 model/output/provider/remap identity;
- verifies the raw sidecar against the source shard;
- checks the derived row-provenance file against the derived summary and shard attrs;
- checks original and stored full-history input keys, worker/game/ply and source
  namespace/config identity;
- checks the raw board legal moves against the derived `legal_mask`;
- reuses the raw one-node BT4 probability vector and applies the exact B100 teacher
  temperature `T=0.5` without new inference.

The source row's saved G10 phase data supplies d9/d10/d12 observations. No new
Stockfish or neural-network inference is performed.

A partial shard slice is supported for implementation/cost checks, but it can never
produce the scientific decision. Only complete coverage of the manifest's derived
cohort may reach the decision rule.

## Ordinary-position analysis

Exclude any d9 row containing mate-domain scores from ordinary centipawn calibration;
those rows are reported separately.

For an ordinary row:

1. identify the exact d9 best-score set and its gap to the next distinct score;
2. reconstruct B100's sharpened `T=0.5` BT4 policy and exact top-probability set;
3. record whether the two sets are disjoint;
4. use the validated G10 adaptive selector to recover the final saved d10 or d12
   roster, or an explicit fallback;
5. never invent a score for a move absent from that narrowed roster.

Report d9-gap thresholds **strictly greater than** 100, 200, 300, 500 and 1000 cp.
The Tactical300 candidate boundary is `>300`, so exactly 300 cp remains outside it.

For the selected >300-cp rows also stratify by B100 top probability:

- `<0.25`
- `[0.25, 0.50)`
- `[0.50, 0.75)`
- `>=0.75`

and by final selected depth d10 versus d12 (with fallbacks separate).

### Roster adjudicability

The later G10 searches are narrowed. A pairwise comparison is counted only when the
**complete d9-best set and complete BT4-top set are both scored** in the final saved
roster.

For adjudicable rows report both:

- pairwise d9 win / BT4 win / tie using the best saved final score in each set; and
- whether the final roster's global best set is d9, BT4, tied between them, or a
  third move/set.

Separately report:

- BT4 complete while d9 is absent or partial;
- d9 complete while BT4 is absent or partial;
- both incomplete/partial;
- neither scored; and
- explicit adaptive-selector fallback.

These one-sided cases are scientifically useful but do **not** enter the pairwise
win-rate denominator.

## Mate analysis

Do not treat encoded mate-distance differences as centipawn gaps.

For rows with a d9 winning mate, report:

- whether B100's top set intersects the d9 winning-mate set;
- whether the final saved d10/d12 roster still contains a winning mate;
- whether a d9 winning-mate move remains winning at the final depth; and
- whether the B100 top set is itself a final winning-mate set.

Rows with losing-mate alternatives but no d9 winning mate are counted separately and
are not folded into the ordinary threshold grid.

## Primary decision

The predeclared Tactical300 question is the `>300 cp` **disagreement** slice.

Let:

- `N` = rows where both the complete d9-best and complete B100-top sets are present
  in the final saved d10/d12 roster;
- `B` = those rows on which the B100-top set has a strictly higher final saved score
  than the d9-best set.

The diagnostic reports `B/N`.

Decision rule:

1. A partial cohort always returns `PARTIAL_NO_DECISION`.
2. If full coverage has **fewer than 1,000 adjudicable >300-cp disagreements**, return
   `INSUFFICIENT_ADJUDICABILITY`; do not clear d9-only Tactical300.
3. If `N >= 1000` and `B/N >= 5%`, return
   `BLOCK_D9_ONLY_REQUIRE_DEPTH_STABILITY`.
4. If `N >= 1000` and `B/N < 5%`, return
   `NO_5PCT_BLOCK_CALIBRATION_STILL_REQUIRED_FOR_ADMISSION`.

The last state is deliberately **not** `PASS_TRAINING`. It removes only this one
mandatory reason to require a deeper-stable gate. A subsequent target-geometry pass,
producer/materialization qualification and explicit matched training admission remain
necessary.

The 1,000-row and 5% values are experiment-allocation thresholds, not statistical
confidence bounds or estimates of optimal-play error. Do not repeatedly tune them on
this same cohort and call the selected rule confirmed.

## Interpretation limits

- G10 d12 is selected adaptively from the d10 margin; depth strata are not randomized.
- Later searches are restricted to earlier top-k rosters, so absent moves cannot be
  treated as losing at the later depth.
- Historical searches may share transposition-table state.
- Stockfish at d10/d12 can still be wrong.
- The qualified cohort is a development cohort and does not establish the same rate
  on a future 100M-position distribution.
- This diagnostic says nothing by itself about playing strength.

## Implementation boundary

This stacked PR adds only the calibration analyzer, focused synthetic tests and this
registration. It does **not** run the full diagnostic, rewrite Tactical300 targets,
train a model, launch an arena, alter Gumbel search or change live configuration.

The completed output is a fresh directory containing `tactical300_calibration.json`.
The receipt binds the adapter manifest, derived summary, teacher identity, selected
shards, raw verification receipts, row-identity digest, threshold/confidence/depth
aggregates, decision and producer hashes.
