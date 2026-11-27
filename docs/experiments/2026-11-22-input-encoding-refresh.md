# Input and encoding refresh: evidence, first diagnostics, and training sequence

**Status: prospective.** The first deliverable is a bounded, read-only input-reliance
instrument and its CPU engineering checks. No A39 inference, feature comparison,
training, collection, arena, or production change has been run for this record.
Engineering checks on synthetic inputs do not fill any scientific result below.
The owner's request is to prepare the plan and first tests; later training stages
need their concrete data, implementation and measured cost recorded before launch.

**Question.** Which inexpensive chess information should DeepFin receive, and in
what representation, given the current network and data? Compare what the network
already uses, what it can learn equally well without, and information a different
representation makes easier to use. These are three different questions.

**Source snapshot:** main `3dd6210290aaf1e0dde140b5854990980d884fef`.
Follow [evaluation](../eval_protocol.md) and the repository's `experiment-readout`
workflow. This record owns the new measurements; old plans are evidence, not launch
instructions. Preserve stopped studies and their artifacts.

## Baseline and invariants

The primary frozen model is the retained SL base **A39**, checkpoint SHA-256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
Its relevant input contract is **175 planes: 112 history/metadata plus 63
`v2_threats` classical planes**, `lc0_root_legacy_meta`, compact `lc0_1858` policy,
and `history_rep_fix: true`. The
[epoch-two donor readback](https://github.com/jjoshua2/DeepFin/blob/3dd6210290aaf1e0dde140b5854990980d884fef/docs/experiments/2026-11-11-epoch2-rl-transition.md#baseline-and-evidence)
records 16 blocks, width 512, 16 heads, 61,444,448 unique parameters and 175 input
planes. Do not substitute the older 146-plane description or infer the donor's
architecture from the current production template. Read and stamp the actual
checkpoint architecture before evaluation; refuse a feature-layout mismatch.
The executable guard and observed architecture, not an old filename containing
`146`, decide the width.

For every accepted run retain checkpoint hash, code revision, dirty-patch hash if
applicable, native-build identity for any invoked native path, model architecture,
history/repetition regime,
policy mapping, device/dtype, batch size, and the ordered selected-row identities.
Use a clean reviewed checkout for the first scientific run. The CLI records the
commit, dirty flag and hashes of its two driver modules; any dirty patch and native
build receipt must be attached separately by the operator. The CLI does not claim
to hash every package file or native binary.
Data identity includes source, source-local game, shard/segment and row; a bare
`game_id` is insufficient across banks. Games split across shards retain one cluster
when the source supplies that relationship. If it does not, call the resulting
cluster a segment and do not claim game-level uncertainty.

Keep data, labels, checkpoints and outputs outside source checkouts, under a fresh
run directory. Existing source banks remain read-only. This work changes no live
configuration, ongoing RL collection, serving process, or checkpoint publication.

## Historical evidence: keep the measurements and their qualifications

Code and ledger links below pin immutable revisions; PR links retain the discussion
context. The large frozen ledger contains both early claims and later corrections;
its July 3 retrospective is the relevant qualification.

| Prior comparison | Recorded evidence | What remains established |
|---|---|---|
| v1 146 planes to v2 threats 175 | First frozen window: eval loss -0.039, policy loss -0.015; second window eval loss -0.028. Typed-attack mean-ablation increased loss by 0.067–0.078 across eight nets. | Replicated offline support for direct attack/threat information. |
| v2 playing comparison | 512 games, +4.8 Elo, 95% CI [-22.7, +32.2]. | Playing-strength effect unresolved; this is not an established Elo gain. |
| v3 checks, x-rays, SEE, passers | Loss changes +0.003 to +0.036; 3,000-puzzle/256-simulation accuracy v2 0.601 versus v3 0.570–0.583. | No convincing improvement in that campaign; loss changes alone did not establish harm. |
| v3 combined attack maps and king shelter | Archived PR #65 also contains the checks+x-ray+SEE union and five king-shelter summaries. | These were tried too; separate per-arm numerical results were not recovered from committed files. |
| v2 lean 143 planes | Kept six pin and 25 attack/threat/control planes, removed 32; one run +0.0217 eval loss, reported 14% speed improvement. | Removal's quality effect unresolved; the speed figure is historical, not a present throughput promise. |
| July 28 frozen-family probe / dropout removal | On 1,032 positions, zeroing old summary groups left 97.7–98.8% top-1 agreement and mean absolute Q change 0.006–0.009; threats gave 84.8% and 0.049. | Older-net reliance evidence. The subsequent dropout-off change was a simplification, not a proven strength gain; no completed post-change ablation readout was recovered. |
| Exact dynamic relation bias | Five attack/defence/pin/open-line/pawn-tension matrices already implemented; the historical summary says "offline ≡ threat planes; shelved." The underlying numerical run receipt was not recovered. | An unchanged generic relation-bias proposal would repeat existing work. The shorthand does not establish statistical equivalence. |
| Repaired corpus history, September 2 | Same 4,894,143 rows/labels; +342.0 Elo [302.1,390.4] at matched 9,680 steps; +358.8 [318.6,407.8] at matched 19,360 steps. | Correcting training/serving history mismatch mattered greatly in that corpus. It does not show eight frames beat consistently trained two-frame inputs. |

Sources:

- [v2 implementation, reuse and historical CPU costs, PR #42](https://github.com/jjoshua2/DeepFin/pull/42).
- [v3 implementation and experiment discussion, PR #64](https://github.com/jjoshua2/DeepFin/pull/64).
- [Archived six-variant and lean configurations, PR #65 head](https://github.com/jjoshua2/DeepFin/tree/173f449e30d74abf9e34d506bbe536e69f819400/configs).
- [July 3 seed-noise correction](https://github.com/jjoshua2/DeepFin/blob/3dd6210290aaf1e0dde140b5854990980d884fef/docs/experiment_ledger.md#L13432-L13480).
- [July 28 feature reliance and dropout preregistration](https://github.com/jjoshua2/DeepFin/blob/3dd6210290aaf1e0dde140b5854990980d884fef/docs/experiment_ledger.md#L10919-L10994).
- [Historical dynamic-relation summary](https://github.com/jjoshua2/DeepFin/blob/3dd6210290aaf1e0dde140b5854990980d884fef/docs/experiment_ledger.md#L3761-L3768).
- [September 2 history experiment and second pass](https://github.com/jjoshua2/DeepFin/blob/3dd6210290aaf1e0dde140b5854990980d884fef/docs/experiment_ledger.md#L72865-L72925).
- [June first-window driver](https://github.com/jjoshua2/DeepFin/blob/173f449e30d74abf9e34d506bbe536e69f819400/scripts/sidecar_sweep_2026_06_15.sh)
  and [second-window driver](https://github.com/jjoshua2/DeepFin/blob/173f449e30d74abf9e34d506bbe536e69f819400/scripts/sidecar_sweep_2026_06_15_wave4.sh):
  from-scratch, one offline replay epoch, batch 512, seed 0, 4,096 evaluation rows;
  1,463-shard and 962-shard frozen windows.

The retrospective measured seed variation of approximately 0.034 in eval loss and
0.019 in policy loss for a repeated May architecture. These are historical
variability estimates, not current rejection thresholds. Its correction explicitly
withdraws the inference that small v3/lean loss differences established harm.
Later shorthand such as "input features exhausted" does not supply stronger data.

### Two traps the new study must avoid

The old
[feature prescreen](https://github.com/jjoshua2/DeepFin/blob/6ede647e10b6b308134ccdc6b511b5c3409561c6/scripts/feature_prescreen.py)
predicts **spatially averaged family summaries**, not complete square maps. A high
R-squared for how much x-ray/check activity exists does not prove the network knows
which source attacks which target. The reported 0.82–0.91 redundancy therefore
does not close the case for spatial or pairwise information. Future probes must
retain square/source/target identity and compare with simple occupancy/count controls.

Likewise, removing or scrambling a family from a frozen net measures reliance
under an intervention, often outside its training distribution. It cannot establish
whether a net retrained without the family would improve. Historical near-inert
classical groups and the unresolved lean experiment illustrate that distinction.

## Stage 0 — seal the input bank and qualify the instrument

Use permitted, existing **prepared Zarr shards** with genuine recorded inputs, legal
moves and source-qualified game identities. Prefer an A39-generated bank from the
current RL evidence chain; do not reopen cancelled studies or silently borrow
their rows. The freezer accepts only recorded `x` with exactly 175 planes and
metadata `lc0_root_legacy_meta`, `history_rep_fix: true`, `lc0_1858`; stored
`legal_mask`/its has-flag and `game_id`/its has-flag are required. There is no FEN
regeneration, width upgrade or history fallback. The selected checkpoint must carry
the matching embedded architecture and load its raw state dictionary strictly;
only the `torch.compile` wrapper prefix may be removed. Shape/key migrations are
refused, including a 146-to-175 zero-column expansion that would fabricate zero
threat reliance. Relation-input models are refused by this plane-only instrument.
Legacy NPZ source shards are refused because their reader eagerly decodes before
the source-row budget can be checked. The resulting diagnostic bank is an NPZ;
it is self-contained and its saved input, row and cluster identities are rechecked
before inference.

No new games or Stockfish labels are needed for this first screen. Dataset path,
manifest hash, source inclusion list and complete selected-row list must be sealed
before inspecting feature-response results. A reused development bank remains
development data; it does not become a fresh strength confirmation set. No feature
report certifies training/holdout independence.

The first screen is bounded to **at most 4,096 selected rows**, drawn uniformly
without replacement from at most **50,000 explicitly listed source rows**, and one
frozen model. The source cap refuses an oversized manifest rather than silently
changing the population; use a documented smaller manifest when necessary.
Record the actual game/segment count, rows per cluster and source composition.
Sampling must be independent of output changes, loss, tactics found by the model,
or treatment response. Keep all transformations paired on exactly the same rows.
Use seed **20261122** for deterministic sampling, permutations and bootstrap;
record it with the run. If a source is too small, report its realized size rather
than drawing replacement positions and pretending they are new observations.

The first full execution is itself cost-bounded: **300 seconds for freezing and
900 seconds for evaluation**, CPU, two Torch threads, FP32, no compilation or shared
runtime changes. Deadlines are checked at shard/batch boundaries, so the final
in-flight operation can finish after the nominal deadline; these are cooperative
bounds, not a hard process-kill guarantee. Add actual wall time and peak RSS to the
execution receipt; the first CLI does not itself instrument peak RSS.
Read at most the stated source/selected-row caps; stop for insufficient RAM/disk
before starting. A synthetic cost smoke can check the CLI but cannot predict A39's
runtime. If the current-network pass cannot finish within budget, preserve it as
**INCOMPLETE**, not a null result; use the same sealed bank after a prospective
budget amendment rather than silently extending or choosing only favourable cases.

An explicitly scheduled GPU execution may instead use a **10 GPU-minute** ceiling,
after existing ownership/resource use is checked. It requires `--device cuda:0` and
an explicit `--gpu-mem-fraction` chosen from measured available capacity. No fraction
guarantees safety beside another job. This is the same sealed data and experiment,
not a new sample; label the backend change and establish its identity floor.
Use `--max-seconds 600` for that execution; the cooperative-deadline limitation
above still applies.

Required engineering witnesses:

1. **Identity:** the identity transformation leaves input bytes and legal support
   unchanged; repeated evaluation establishes the backend's numerical floor.
2. **Scope:** only the selected family changes. History, metadata, all other
   families, row order and legal policy mapping remain unchanged.
3. **Permutation:** the selected family's channel histograms are preserved under
   within-row spatial permutation, while a nonconstant fixture changes location.
   Permutations must not mix positions or games.
4. **Effectiveness:** a small deterministic synthetic model responds to a known
   changed feature; a no-op mutation must fail the test. A real trained model may
   legitimately ignore a family, so its lack of response is not an instrument failure.
5. **Policy:** use the checkpoint's policy mapping and the original position's
   legal support, renormalize on that support, and reject empty or invalid support.
   Padded/illegal logits cannot enter distribution metrics.
6. **Statistics:** a fixture with unequal game lengths and colliding game IDs from
   distinct sources demonstrates the advertised grouping and resampling behavior.

Invalid checkpoint/data contracts, nonfinite outputs, changed legal support, failed
identity controls or missing provenance stop the scientific readout. Correct the
instrument, version it, then reuse the same sealed rows; do not reroll a favourable
sample. No test below is a comparison with Stockfish playing strength.

## Stage 1 — frozen-input reliance and geometry diagnostics

### Hypotheses and treatments

**H1:** on A39, direct threats influence policy/value more than the older summary
families. **H2:** some of that reliance depends on where a feature occurs, beyond
its amount. These hypotheses describe A39, not the optimal trained encoding.

The first run fixes eight families: `king_safety`, `pins`, `pawn_structure`,
`mobility`, `outposts_space`, `attacks_typed`, `threats_all`, `extras_all`. Their
boundaries come from the encoder; the last two cover the whole 29-plane v2 addition
and all 63 extras. Composite and component rows overlap; do not add their effects.
Keep the 112 history/metadata planes intact. The kernel additionally supports
`attacks_all`, `attacker_counts`, `hanging`, `cheaper_attacker`, `safe_checks`,
`control`, `pawn_tension`, `pawn_storm`, and `v1_all`; expanding to these requires a
declared follow-up on the same bank, not choosing whichever family looks promising.

Each family has an identity reference and two interventions:

- **Zero replacement (`zero`):** set the selected channels to zero. Naturally zero
  feature rows have no exposure. This removes
  both the feature's location and its amount. It is descriptive reliance under an
  out-of-distribution intervention, not a valid alternate board and not the old
  mean-replacement prescreen.
- **Spatial permutation:** independently shuffle the 64 squares within each row,
  applying the same square permutation to all selected channels of that family.
  Preserve its per-row channel histograms and cross-channel local bundles, while
  breaking their alignment with occupancy and other families. Constant maps can
  remain unchanged.

The first CLI aggregates **all sealed rows**, including rows unchanged by a
treatment. Changed/no-op input-row counts are not emitted. If an exposure-specific
readout is needed, derive and bank those counts from these same inputs and seeded
transformations as an explicit reanalysis; never describe the initial aggregate as
an exposed-row estimate.

No feature is chosen after looking at its response. A later square-resolved linear
probe, if wanted, is a separate prospective diagnostic with a game-held-out split,
per-square target, occupancy/piece-count baseline and randomized-target control;
the old global-summary R-squared is not its result.

### Metrics and uncertainty

Bank one row per position/treatment containing identities and legal-policy/WDL
change metrics. Primary descriptive endpoints are:

- Legal-policy **total variation**, `0.5 * sum(abs(p_reference - p_arm))`.
- Absolute change in **Q = W - L**: Q is in [-1,1], and absolute Q change is in [0,2].

Report legal top-1 flip fraction, legal-policy KL (reference to arm, explicitly
stating any numerical flooring), WDL total variation and signed Q change as
secondary descriptions. The first CLI emits these metrics; it does not emit
draw-probability changes or source/phase-stratified summaries. Source-specific
summaries may be computed from banked rows in a labelled reanalysis; phase-specific
analysis also needs independently defined phase tags. Distributions use the
network's main policy and main `wdl`; auxiliary heads are not substituted.
Do not call any of these changes an improvement or regret reduction: their sign
does not say which answer is better.

Report both the position-weighted mean and the equal-game-weighted mean.
Uncertainty first averages each endpoint within its source-qualified game, then
resamples those game keys **without source stratification**, with replacement.
Thus the interval targets the observed mixture of games rather than fixed source
proportions; source-qualified keys prevent collisions but do not themselves make
the bootstrap stratified. Use 2,000 replicates and disclose the source/game mix.
All rows from the same game move together. Intervals are pointwise 95%
descriptions of predeclared families, not simultaneous claims that the largest
family is a winner. With fewer than 30 independent clusters, report descriptive
values and label interval-based readings **UNREAD: too few clusters**.

**Precommitted reading:** controls passing and a complete bank establish an
instrument-qualified description. For an endpoint, call dependence **material at
this screen's scale** only when its lower 95% bound exceeds both 0.01 and ten times
the largest per-row repeat-forward difference for that endpoint. Call it **small
at this scale** only when its upper bound is below 0.01; otherwise **UNREAD**.
These 0.01 cutoffs are one percentage point of probability mass or 0.01 Q, not Elo
equivalents. Report exact estimates even when a cutoff is unmet. If the numerical
floor exceeds 0.001, fix/qualify that backend before making a materiality claim.
Spatial sensitivity identifies a dependence on location; it does not prove that
a new pairwise encoding would add useful information or improve training.

### Runnable interface and readout

Create an explicit source manifest outside the checkout. `source_id` is the
conversion/generation namespace in which `game_id` is unique; preserve it across
shards that contain portions of the same game. Paths must name already prepared
shards, not raw FEN rows:

```json
{
  "schema_version": 1,
  "sources": [
    {
      "source_id": "a39-existing-bank-namespace",
      "shards": ["/absolute/shared/prepared-shard.zarr"]
    }
  ]
}
```

From the review checkout, with the appropriate already prepared development
environment, replace the three absolute paths below with the declared source
manifest, unique shared output directory and verified A39 checkpoint:

```bash
SCREEN_ROOT=/absolute/shared/input-encoding-refresh-run1
A39_CHECKPOINT=/absolute/shared/a39-checkpoint.pt
python -m scripts.input_encoding_screen freeze \
  --manifest /absolute/shared/input-encoding-sources.json \
  --out "$SCREEN_ROOT/bank.npz" \
  --max-rows 4096 --max-source-rows 50000 --max-seconds 300 --seed 20261122

python -m scripts.input_encoding_screen run \
  --bank "$SCREEN_ROOT/bank.npz" --checkpoint "$A39_CHECKPOINT" \
  --out-dir "$SCREEN_ROOT/report-cpu" \
  --families king_safety pins pawn_structure mobility outposts_space \
    attacks_typed threats_all extras_all \
  --modes zero spatial_shuffle --batch-size 64 --bootstrap 2000 \
  --max-seconds 900 --seed 20261122 --device cpu
```

The identity repeat control always runs first. The implemented numerical gate is
`allclose(rtol=1e-6, atol=1e-7)` plus zero legal top-1 flips; preserve the observed
differences rather than describing the entire backend as bit-exact. The report
directory is new: `run.json` records settings, `baseline.npz` banks raw logits,
each case has an NPZ of metrics/row IDs/cluster keys and a JSON summary, and
`report.json` records complete/failed status. A deadline/error retains completed
cases; there is no automatic retry or continuation. Re-execution uses a fresh
directory and the same bank. A failed report is not a complete family comparison.

The bounded engineering check is:

```bash
python -m pytest tests/test_input_feature_diagnostics.py tests/test_input_encoding_screen.py
```

These use synthetic inputs and small fixtures, not A39. **39 tests passed**
(22 kernel and 17 end-to-end/provenance checks), on Python 3.13.15 and PyTorch
2.14.0+cpu with two Torch threads. The integration test encodes actual move-history
fixtures, writes ordinary replay shards, freezes their inputs, loads a small real
transformer checkpoint and executes both interventions through the CLI entry point.
Independent review found that the generic checkpoint loader could expand 146 input
columns to 175. This screen now loads raw weights strictly; the mismatch regression
fails before inference. Tests also cover compile-prefix normalization, incomplete
weights, wrong history metadata, missing source/game presence flags, an unstable
identity control and changed bank input/legal/cluster identities.
Stage 1 is the implemented instrument in this PR; the training adapters,
alternative encoders and playing studies below are future work.

| Required readout | Current state |
|---|---|
| Instrument revision and synthetic CPU checks | **39 passed**; independent review's strict-state finding fixed and regression tested |
| A39 checkpoint/architecture readback | **NOT RUN** |
| Sealed source manifest and selected-row IDs | **NOT CREATED** |
| Actual A39 CPU/GPU cost and completed case count | **NOT RUN** |
| Family reliance and geometry estimates | **NOT RUN** |
| Current-network strength or encoding adoption | **NO CLAIM** |

## Stage 2 — matched training, before selecting a new encoding

This is the next experiment specification, not an executable training launch in
this PR. Stage 1 can prioritize engineering and expose dead input plumbing; it
cannot approve or reject an encoding. First restore a baseline trainer comparison
on one frozen, permitted train bank with a disjoint source-qualified game holdout.
Use the actual SL baseline's targets/losses, optimizer family, architecture and
schedule, not the concurrently changing RL recipe. Freeze those identities and the
independent seed pairs before any update. Keep mirroring, history, targets, row
order, presentations, checkpoint horizon and evaluation identical within each pair.

### First training comparison: compact control

The historical lean result remains unresolved, so test three arms:

1. **Full175:** current v2 input.
2. **Masked175:** retain tensor width, zero the 32 planes omitted by the historical
   lean subset during both training and serving.
3. **Compact143:** encode just the same 112 base + six pin + 25 attack/threat/control
   planes, with an actually smaller input projection and tensor transport.

Masked175 separates information removal from width/layout. Share the initial kept
input columns and trunk/head tensors across arms rather than allowing changed fan-in
initialization to decide the comparison. Use identical feature dropout (off), with
explicit take-effect checks on train and serve. Train each arm consistently from
the same fresh initialization scheme. A later A39 continuation is a separate
migration question; masking a co-adapted A39 is not this from-scratch comparison.

**Primary offline endpoint:** held-out legal policy cross-entropy against the same
frozen policy target. Main-WDL proper score is a separately reported guard; do not
rank on a total loss whose composition changed. Precommit the practical policy and
WDL margins after a control-only repeat estimates current seed variation, before
candidate updates. The allowed repeat estimates noise only, not candidate quality.
Use a paired-seed interval; a single favourable seed cannot select the arm.

**Decision:** lower policy loss with a CI excluding zero and no WDL harm beyond the
frozen margin nominates a quality candidate. For a speed-motivated compact arm,
the upper CI must be inside both preregistered noninferiority margins, and measured
end-to-end encoding plus inference must improve by at least **5%** at relevant
batch sizes. An unresolved interval does not establish equivalence. Upper bounds
outside the quality margins kill that dose/recipe for advancement; otherwise stop
unread at the budget. No automatic extension or best-checkpoint selection.

**Budget and launch completion:** one control-only cost/noise study, capped at
**30 GPU-minutes**; then at most **four paired seeds and two million stored-row
presentations per arm/seed**, with a combined **12 GPU-hour** ceiling. These are
upper bounds, not a claim of sufficient power or a required spend. Before candidate
updates, freeze the actual seed count, horizon, learning-rate schedule, margins,
sample-size justification, measured memory/cost and a final-checkpoint rule within
those caps. If the resulting design cannot resolve a useful effect, revise the
plan prospectively instead of launching a knowingly uninformative sweep. Freeze
training order or interleave arms to expose machine/loader effects.

### Second training comparison: typed source-to-target pair adapter

Test this separately after the compact comparison, initially on Full175. Existing
v2 maps identify attacked squares and attacker types; the existing five relation
matrices add untyped edges. The new hypothesis is that **attacker type, defender
type, and the exact source/target squares together** ease tactical learning beyond
those inputs. It is not the claim that Stockfish's NNUE network can be transplanted
unchanged into this transformer.

Compute sparse typed attack/defence edges using the same verified bitboard attack
kernel and perspective convention. Specify pseudo-legal attacks versus legal moves,
friendly blockers, pawns, kings and pins; do not mix these semantics silently.
Use a small adapter with zero initial contribution, and benchmark both encoding
and the actual model path. Prefer square-token features or an explicit bounded
pair contribution; do not lengthen the transformer sequence by one token per edge
without pricing that cost.

Required arms are baseline, **the existing five-relation bias** as the simpler
control, the typed adapter, and an equally sized adapter on existing piece/square
features. A shuffled-routing negative control permutes edge destinations while
preserving counts/types. It changes routing, not teacher targets, and must not
preserve the intended source-target relation.
Parameter count, initialization, optimizer groups and common data are stamped.
Identity at zero contribution, nonzero-gradient take-effect, mirror/perspective
consistency, and serialization/export parity precede training. Compare held-out
policy and WDL with the same control/noise procedure and maximum budget as above;
freeze the actual horizon/margins before running this separate stage. The adapter
must beat its simpler relation and equal-capacity controls to support the claim
that typed pair information adds something. A gain over bare baseline alone could
come from capacity or optimization; successful shuffled routing weakens the
specific source-target explanation.

## Stage 3 — confirm playing value and price the implementation

Only a nominated trained endpoint reaches this stage. Freeze one candidate and its
matched control, code/build, openings, tablebases, search settings and time controls
before play. Choose pair count from the minimum useful effect and measured draw/pair
variance; do not inherit 512 or 1,000 games as a universal gate. Register one final
read or a fully specified sequential design. Cap this stage at **8 GPU-hours** and
stop unread if the required resolution cannot fit.

Run matched-simulation games to isolate network quality, then a separately recorded
matched-time comparison for the intended deployment. Use actual deployed encoding,
batching and serving paths. A slower Python prototype establishes only its own
speed; do not extrapolate a native implementation's performance. A quality candidate
advances only on a positive paired strength interval; a compact efficiency candidate
requires its predeclared strength noninferiority margin and positive measured
throughput. Specify those thresholds and the arena command after the trained
endpoint and cost probe exist, before games. Promotion is a separate deliberate
deployment with checkpoint/optimizer/runtime recovery available.

Measure CPU feature construction, plane zeroing/materialization, host-to-device
transport, inference at representative batches, and end-to-end worker throughput
separately. A cheap bitboard expression can still add costly dense channels or
pair tensors. Keep feature-only microseconds and actual worker/game throughput as
distinct measurements. This PR does not implement a fresh encoding benchmark or
promise that the old lean/v2 timing transfers to today's native serving path.

## Subsequent questions, in order and with a reason to proceed

1. **Pawn pairs:** a compact sparse encoding of same-side and opposing pawn
   relationships, conditioned on relative file/rank and colour/perspective. Existing
   passed/doubled/isolated/tension summaries are the simpler control. Proceed only
   if a square-resolved probe or held-out pawn-structure errors identify a gap; price
   CPU construction and input transport. A full NNUE-sized feature table is not the
   default proposal.
2. **King-relative context:** piece/pawn positions relative to each king, with a
   small shared embedding or adapter and an equal-capacity absolute-square control.
   Preserve both king frames and castling semantics; compare beyond existing king
   zones, shield summaries and arc/Smolgen geometry. This tests conditioning, not
   another undifferentiated bundle of king-safety planes.
3. **History length and delta encodings:** consistently train and serve current
   state plus explicit rules state, two frames, eight frames, and optionally move/
   occupancy deltas. Keep repetition/rule50/castling/en-passant information fixed
   when testing piece-history length; separately test compression of that rules
   information. Never construct a shorter-history arm by erasing a frozen net's
   frames and call its result a trained comparison. Preserve real source histories
   and the train/serve identity guard. The old H8/H2/repeat ladder was proposed but
   no completed readout was found in this audit.

These are separate interventions, not a combined input-v4 bundle. Each successor
gets a prospective addendum with baseline, controls, deciding endpoint, margins,
cost and recovery before compute. A negative result is conditional on the tested
model, data and horizon; a positive offline result still needs the claim-appropriate
strength test.

## Recovery and next action

Stage 0/1 reads files and writes only its own output directory. On failure preserve
the receipt and partial observations, correct the code and rerun the same sealed
sample under a new output version. No live rollback is needed. Training stages use
isolated checkpoints/optimizers and never resume or prune live state; stopped or
failed arms remain recorded and cannot silently turn into a new random seed.

The bounded synthetic checks and independent review are complete. On the training
host, the first scientific
action is the declared checkpoint/data readback and bounded Stage 1 execution.
That report informs which trained comparison deserves work; it does not change
A39 or the live RL program.
