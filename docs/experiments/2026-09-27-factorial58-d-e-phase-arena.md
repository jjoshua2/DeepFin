# SF-free D/E phase-start arena: completed readout — September 27, 2026

The preregistered phase interaction is **inconclusive**. E121's score relative
to D121 was 48.242% from early starts and 50.000% from late starts. The primary
late-minus-early contrast is **+1.758 percentage points**, with a paired-root
95% bootstrap interval of **[−2.734, +6.445] points**. Its point estimate is
opposite the predicted negative direction, but the interval supports neither
direction. The [independent CPU readback](evidence/factorial58-phase-de-20260927/independent-readout-review.md)
passed the complete game, terminal-rule and statistical checks. This is a
conditional result for one pair of trained checkpoints and one fixed source panel.

## Question and frozen comparison

[The completed factorial](2026-09-22-factorial-readout-next24h.md) trained D121
with an equal BT4/Ceres policy target and an equal-third Stockfish/BT4/Ceres main
value target. E121 used the same policy target and replaced the value target with
50% native BT4 and 50% calibrated Ceres, without Stockfish. Both policy teachers
were separately sharpened to T=0.5 before equal mixing; Ceres value retained its
registered calibration. Each checkpoint came from seed 121 and one complete
58,090,688-row, 113,459-update epoch. E versus D is the **whole value-recipe**
contrast: removing Stockfish also raises both BT4 and Ceres value shares from
one-third to one-half. It cannot isolate a Stockfish coefficient or the benefit
of either replacement teacher.

The registered phase hypothesis predicted that removing Stockfish value would
help E more at early starts and help less, or hurt, at late starts. The deciding
quantity was E's two-color paired-root score at late starts minus its score at
early starts. A 95% interval entirely below zero would support that direction;
an interval entirely above zero would support the opposite direction. Otherwise
the directional verdict was inconclusive. A stronger early-positive/late-negative
reversal required the early score interval entirely above 50%, the late interval
entirely below 50%, and the interaction interval below zero. These rules were
frozen in the phase protocol before play; its source SHA256 is recorded in the
[compact readout](evidence/factorial58-phase-de-20260927/compact-readout.json).

The frozen history-bearing panel had 384 distinct source-game roots, with 128 in
each of the early, middle and late bins and 64 roots per starting side in each
bin. Each root was played twice with candidate colors swapped: **384 pairs / 768
games**. The panel bins were absolute ply 16–39 with at least 26 pieces, ply
40–79 with 15–25 pieces, and ply at least 80 with 7–14 pieces. The matched-
simulation arena used 400 simulations per side, training search shape, policy
prior temperature 1.0, game temperature 0.1, seed 2026092703, compiled CUDA,
and strict rule50-aware six-man Syzygy root, leaf and terminal adjudication. Its
10,800-second and 1,000-ply limits were recovery bounds, not shorter decision
samples. Both checkpoints, evaluation config, panel, history seeds, arena code,
and output files are hash-bound in the [compact evidence](evidence/factorial58-phase-de-20260927/compact-readout.json).

## Completed readout

The primary analysis averages E's two color-swapped scores within each root,
then weights the two starting sides equally in each phase. Its 20,000-draw
bootstrap resamples source-game-distinct roots within starting side, using
NumPy PCG64 seed 2026092704 and linear-quantile 95% percentile intervals.

| Start phase | Roots | E score | 95% paired-root interval | E−D score margin |
| --- | ---: | ---: | ---: | ---: |
| Early | 128 | 48.242% | [43.750%, 52.539%] | −1.758 points |
| Middle | 128 | 46.289% | [43.164%, 49.414%] | −3.711 points |
| Late | 128 | 50.000% | [48.633%, 51.367%] | 0.000 points |

The late-minus-early interaction is **+1.758 points [−2.734, +6.445]**.
Therefore the primary rule returns **inconclusive**; the middle-bin estimate is
descriptive and does not replace that interaction. The strong reversal also
fails its prespecified rule.

Thirty-nine early roots had fewer than eight available moves after their source
PGN's initial FEN. A witnessed irreversible move made unavailable earlier
repetition history irrelevant to the selected roots, but cannot recreate older
neural input frames. The prespecified exclusion leaves 89 longer-history early
roots: E scored 48.609% on them, versus 50.000% on late roots. The corresponding
late-minus-early contrast is **+1.391 points [−4.206, +6.996]**. It does not
recover the predicted negative direction. The 39 excluded early roots scored
45.250% after side standardization; this sensitivity is conditional on the
recorded source histories, not a new selected arena.

The secondary overall GSPRT used the single terminal look at all 384 pairs,
with `elo0 = −15`, `elo1 = +15`, `alpha = 0.05`, and `beta = 0.10`. E's combined
score was 185/384 = 48.177%; the ascending pentanomial counts were
`[15, 67, 242, 51, 9]`. The likelihood ratio was −8.455, below the H0
boundary −2.251, so the declared overall screen returned **H0** without an
early stop. This favors the registered −15-Elo hypothesis against +15 Elo for
this panel. It does not establish equality, E inferiority to zero, or a
30-Elo-optimal value mixture. It cannot decide the primary phase interaction.
The earlier 128-pair strict ordinary-opening E–D result was +13.58 Elo
[−14.35, +41.68]; it used a different starting-position panel and is not
pooled with this phase-start result.

## Integrity, limits and next decision

The independent review matched all 384 ordered roots to the frozen 23 source
PGNs, checked both color halves and replayed all 768 PGNs from their saved
histories. All 412 natural-rule endings and 356 strict six-man Syzygy endings
agreed with direct rule checks; there were no max-ply endings. It recomputed
the phase intervals, sensitivity and overall GSPRT from the raw bank. The
review was a CPU readback of frozen artifacts, not an independent model run.
Tablebase files were checked by filename/size metadata and terminal probes,
not full content hashes.

Only one training seed per model was tested. The chosen source PGNs, phase-bin
selection, and unavailable older neural history on 39 early roots limit
generalization. The result neither supports a phase-specific Stockfish value
benefit nor tests whether rare deeper Stockfish corrections help on selected
tactics or endgames. It does not settle a 30-Elo practical margin or the best
mixture at larger scale. Keep the phase claim unresolved; no target, training,
or deployment change follows from this readout. The held second-seed plan
remains a separate allocation decision.

The [compact readout](evidence/factorial58-phase-de-20260927/compact-readout.json)
retains both game scores, phase, starting side and short-prefix flag for each
ordered root, plus aggregate statistics, settings and SHA256 identities. The
verbatim [independent PASS note](evidence/factorial58-phase-de-20260927/independent-readout-review.md)
has SHA256 `81b0701c35bb4c484b32cdbd64d679c91bcb4689595c48c9a38cdc8a1f86a53b`.
The raw JSONL game bank, PGN and arena result remain in the external artifact
store at the relative directory recorded in the compact evidence; their hashes
allow exact retrieval and re-analysis without replaying games. No public bulk
download URL is registered for this snapshot.
