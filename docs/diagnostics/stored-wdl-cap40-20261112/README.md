# Cap40 stored-input prediction drift, 2026-11-12

This independently reviewed diagnostic compares donor and cap40 WDL predictions on the exact same 254 stored rows from one metadata-selected, corrected-history, pure-selfplay producer shard. It is **target-free descriptive drift**, not evidence of improved playing strength, correctness, calibration or loss.

- Source revision: `53f8908522e0659b0661f484586dc1e35c55e309`.
- Three supplied producer-game groups contain 70, 87 and 97 rows. The shard was training-ingested; exact sampled-row exposure is unknown. There is no heldout or independence claim.
- Five sequential CPU passes: donor batch 16, identical donor repeat, donor batch 1, cap40 batch 16 and cap40 batch 1. Repeat and batch controls passed at `atol=1e-6`, `rtol=1e-5`.
- Candidate-minus-donor expected-score prediction shift: +0.8194 percentage points row-weighted; +0.7235 points equal-group weighted. Mean **absolute** expected-score change was 6.0688 points; 133 rows moved up and 121 moved down. A small signed mean masks substantial bidirectional changes.
- The largest individual WDL probability change was 54.9444 points. This is a prediction change, not a measured accuracy change.
- Each child recorded 16 GiB address-space, 60 CPU-second and 90 wall-second limits, one CPU thread and CUDA disabled. These are execution settings, not measured peak-resource telemetry.
- W,D,L order and side-to-move perspective are source-declared. Real semantic POV validation remains unperformed.

`independent-review.json` is the unchanged reviewed compact receipt (SHA256 `9da1295da264a3a9488bde613f357d62420fa8a607db002398bca58cb13b8e9d`). It binds checkpoint, input, saved-output and plan identities and independently recomputed metrics. Raw input rows and probability datasets are not published here. Local paths in the receipt identify retained technical evidence, not publicly accessible artifacts. No training, GPU workload or model update occurred.

## Same-row native search-target agreement

A separately reviewed, bounded CPU extraction retained **all the same 254 rows** and their native `search_wdl` values. No model forward was repeated; scoring used the identity-bound saved donor/cap40 predictions above. All presence flags were true. Raw float16 targets were retained, then explicitly normalized in float64 under a predeclared absolute sum-error tolerance of 0.001. Maximum observed sum error was 0.000366211 and maximum component correction was 0.000243366. The log-probability floor was `1e-12`.

Lower is better agreement with this named target:

- Row-weighted Brier: donor **0.014763238**, cap40 **0.036587110**; candidate-minus-donor **+0.021823872**. Brier agreement worsened in all three supplied groups. Equal-group delta: **+0.020580016**.
- Row-weighted log loss: donor **0.693543415**, cap40 **0.707798328**; candidate-minus-donor **+0.014254913**. Two supplied groups worsened and one improved. Equal-group delta: **+0.010384260**.

This is agreement with recorded **producer search targets**, not Stockfish truth, real accuracy, calibration, playing strength or a causal diagnosis. As subsequently verified below, normalized search labels match the actual configured WDL-target semantics for present rows; these measurements are not the realized sampled/weighted overall training loss. The same exposure, grouping and semantic-POV limitations above apply. The producer can replace a nonfinite search target with `[0,1,0]`; presence flags do not identify those fallbacks, so fallback incidence remains unknown and all draw targets were retained.

`native-search-agreement-review.json` is the unchanged independent review receipt (SHA256 `bc6e7b25fd65075e87fcf488d285680a5edbe811645d8e3eba1fbb83c91bfc7d`). It binds raw/normalized target identities, original ordered rows, checkpoint/prediction receipts and independently recomputed metrics. Raw targets and datasets are not published. Extraction/scoring source revision: `155f2f887861520fb8976860fbbbbeaa5007c814`. Execution settings were 8 GiB address-space, 120 CPU seconds, 150 wall seconds, one CPU thread and CUDA disabled; no training or additional inference occurred.

## Configuration correction and draw/score decomposition

Independent source/config review confirms this specific run used `search_wdl_draw_mode=net_raw`: the stored search target retains the **producer root network's draw probability**, while the searched Q adjusts win/loss probabilities. On this step-zero producer shard, draw-target agreement is therefore partly self-referential, not an independent draw-quality signal. Metadata alone does not prove exact functional equivalence between the producer export and the donor checkpoint.

The actual WDL blend was search fraction 1, SF fraction 0, with adjusted outcomes, directional damping and terminal outcome transfer disabled. All eight retained iteration logs reported search effective fraction 1 and terminal outcome fraction 0. Thus normalized search labels match the configured WDL-target semantics on present rows. The earlier review receipt's blanket exclusion of “final mixed training loss” must not be read as evidence of a different WDL target here: exact sampled exposure, weighting and the overall multi-objective training loss remain distinct and unknown on these 254 rows. `w_future=0.01` trains a future **policy**, not future-outcome WDL transfer.

Using expected score `s=W+0.5D`, WDL Brier decomposes algebraically as `2*(s-s_target)^2 + 1.5*(D-D_target)^2`. The recorded candidate-minus-donor aggregate change consists of:

- Draw-error component: **+0.0222491304**.
- Expected-score squared-error component: **−0.0004252600**.
- Total Brier change: **+0.0218238715**, subject to floating-point rounding.

The Brier worsening is driven by disagreement with the partly self-referential draw target; the expected-score squared-error component improved slightly. This does not establish an independent quality improvement either. Expected-score mean absolute error increased (0.051828364 to 0.060572924), so even the direction depends on the named metric. No strength or causal gradient conclusion follows.

Unchanged independently reviewed evidence:
- `independent-config-review-20261112.json`, SHA256 `0939c7025b442927ea0bf5c92cf977b0beb40e8b30359909258629052ab92a50`.
- `independent-decomposition-20261112.json`, SHA256 `c78f44d4473415a17ed169a89f88d28856e1ac3b37e784f9d312d42535961e5b`.

This correction used saved outputs, metadata and pinned source only. No additional model inference, corpus decoding or training was performed.

## Donor shared-value-body gradient probe

One independently reviewed, finite CPU diagnostic used the same 254 rows with exact native categorical labels and all network/categorical masks present. It compared the WDL and categorical gradients on only `value_wdl.token_proj.{weight,bias}` and `value_wdl.net.0.{weight,bias}` at the **donor** checkpoint. The trunk was detached, private output projections frozen, objective weights fixed at 1 and 1, and reduction was a uniform fixed-panel masked mean, not the unknown realized training sample weighting.

Independently recomputed from saved gradient vectors:
- Dot product: **+0.0699337494**.
- Cosine: **+0.0897392491**, weak positive aggregate alignment rather than aggregate opposition on this subset.
- WDL norm: **0.4775996354**; categorical norm: **1.6317003057**, approximately 3.42 times larger. Magnitude alone does not establish harmful dominance.

There was no optimizer step or trunk backward. The execution receipt reports identical whole-model state hashes before/after and untouched parameter `.grad`; independent review verified the receipt binding, not a separate model reload. Actual repeat and batch-invariance controls were **not run**; corresponding tests used synthetic models. This local shared-body result does not establish full-trunk compatibility, optimizer/clipping effects, the cause of the observed training trajectory, or playing strength.

`shared-body-gradient-review.json` is the unchanged compact independent review (SHA256 `f77d9c166f302276362b6fe3a2c46f32a6574745fc390395b684789e6893a0ea`). Source revision: `e9e029edb6501703652cf6141c595c5569cb6359`. The one child used recorded limits of 16 GiB address space, 60 CPU seconds, 90 wall seconds, one CPU thread and CUDA disabled. Raw gradient vectors, inputs and targets are not published here.


## Stored Stockfish coverage (same fixed 254 rows)

Independent saved-artifact review: [sf-coverage-review.json](sf-coverage-review.json),
SHA256 `5d6cd9f445e11701038cb9cdfc1dde0e521f6a5a7ec9954984d846385f77260c`.
All 254 rows have SF labels and metadata; all labels are finite, nonnegative,
component-bounded by one, and within 0.001 of unit sum (maximum error
0.0003662109375). Recorded search nodes range from 3,065 to 5,047 and recorded
depth from 2 to 15. These are observed metadata values, not a strength estimate
or evidence of the historical 100k-node configuration. No normalization was
performed.

These SF labels describe action-conditioned child positions, not independent
root-position truth. Exact played-action and full successor/history binding
are absent; the recommended SF move is not evidence of the played action.
Consequently no donor/candidate root prediction was scored against these
labels, and no child comparison is qualified. The finite coverage read did
not launch Stockfish, run model inference, or change training. Raw labels and
row data remain outside this repository.


## Exact successor-key coverage

Independent saved-artifact review: [ply-coverage-review.json](ply-coverage-review.json),
SHA256 `910371e3c0c34705a9c6f8ed3b7fae6852e0c4bc4aeeac7c4afe27eb30ca74f2`.
All 254 original rows have valid, unique supplied game/ply keys. Exactly 227
have a unique same-game `ply + 1` key in this fixed panel; 27 do not. Group
coverage is 86/97, 61/70 and 80/87. There are no duplicate keys or ambiguous
successor keys. This does not use storage adjacency.

Full-panel successor coverage therefore fails. These 227 candidates have not
been qualified as legal transitions or exact child positions with complete
history. This audit performs no model/checkpoint validation, inference or
scoring. No subset was substituted for the original 254-row panel. Raw ply
values and masks remain outside this repository.


## Fixed partial-cohort transition validation

[Independent transition review](transition-review.json), SHA256
`f3abf94acfbecee2615715de1a6cc52415d94529f5908d7339fbab3013ee5661`,
separately recomputed all 227 predetermined pairs. Each has one legal move
matching successor position metadata and exact seven-slot stored-history
overlap. The 27 missing successors remain exclusions, not replacements.
This is narrow child-position/input consistency, not older-history or full
game provenance, repetition truth, extra-feature correctness or adjudication
qualification. No model or engine ran for this check.


## Partial-cohort child Stockfish agreement

[Independent saved-output review](child-sf-agreement-review.json), SHA256
`f8582c8be0cd63abcc58aa4e1c78f468c42659ef79d01db4835acdd3ec4d08b2`,
recomputed the fixed 227-pair comparison. Existing successor-row predictions
were flipped W/L exactly once into parent POV and compared with the parent
row's action-conditioned child SF target. Float16 target rounding was handled
by the declared bounded normalization; no rows were selected by prediction
or label outcome. The 27 missing successors remain excluded.

- Brier: donor 0.154634, cap40 0.129602; 172 pairs improve, 55 worsen.
- Log loss: donor 1.329923, cap40 1.209763; 193 improve, 34 worsen.
- Both losses decrease in all three supplied game groups.
- This partial cohort records 3,065–5,047 SF nodes and depth 2–11. The earlier
  coverage report's depth 2–15 range describes the original 254-row panel,
  not this subset.

This direction differs from the producer-search-target agreement result,
whose draw component partly preserves the producer network's own prediction.
The two results address different targets and positions; neither establishes
root-position truth, held-out generalization, Elo, or a causal explanation
for training changes. The partial cohort is successor-available, not random.
No model inference, engine execution or new corpus acquisition was needed
for this saved-output comparison. Raw per-pair evidence stays outside the
repository.


### Published receipt path redaction

The three receipt copies listed below replace the local home-directory prefix
with `~/`. Their metrics, artifact identity hashes and all other fields are
unchanged. Earlier SHA256 references for these receipts identify the original
retained review bytes, not the path-redacted repository copies. This is a
presentation-only derivative, not a reissued independent review.

| Repository receipt | Original review SHA256 | Published copy SHA256 |
| --- | --- | --- |
| `child-sf-agreement-review.json` | `f8582c8be0cd63abcc58aa4e1c78f468c42659ef79d01db4835acdd3ec4d08b2` | `9692d01ff0ffb49bb73fb86647e0ffdbfc8b8dc0bee42f56218a5f42e14458ee` |
| `independent-review.json` | `9da1295da264a3a9488bde613f357d62420fa8a607db002398bca58cb13b8e9d` | `ff488948d6c90db5131b1f91086be3dd3e319d646138686a1acfa636da5453ba` |
| `transition-review.json` | `f3abf94acfbecee2615715de1a6cc52415d94529f5908d7339fbab3013ee5661` | `db9afe7d141545c81c14fe8e2d9cb1c29ccf615651a6fc7eed5006db5359bdda` |


## Raw legal policy-prior drift

[Independent saved-output review](policy-drift-review.json) found 30/254 top-1
changes and mean total variation 0.16415 between donor and cap40. Mean entropy
decreased from 1.00374 to 0.69079 nats; all three supplied game groups had lower
entropy. This is a sharper, changed raw legal-move prior at temperature 1, not
evidence of better moves, deployed search-temperature behavior, or Elo.

All 254 original inputs were retained. Saved repeat/batch probability controls
passed the predeclared combined tolerances. Dense-versus-sparse agreement and
whole-model state preservation are supported by execution receipts, not an
independent model rerun. No raw logits, probabilities or input rows are
published here.

This review contains no absolute home paths, so publication required no
redaction. The published copy and original retained receipt are byte-identical.
Original SHA256: `307190cfae5cfe3b8a202fe8c7114b37b2a34a9135703afa193e11c02cf7a7db`.
Published copy SHA256: `307190cfae5cfe3b8a202fe8c7114b37b2a34a9135703afa193e11c02cf7a7db`.
