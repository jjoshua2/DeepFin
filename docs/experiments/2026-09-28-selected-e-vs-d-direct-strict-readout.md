# Selected-E versus D direct strict six-man arena

This record concerns the seed-121, one-epoch Selected-E checkpoint against the
matched-horizon D checkpoint. Selected-E routes each row to one BT4 or Ceres
teacher for both policy and main WDL, with half of rows assigned to each in
the frozen routing recipe. D blends BT4 and Ceres policy on each row and
blends equal thirds of Stockfish, BT4 and Ceres in its main value target.
The comparison is a recipe-strength check on the historical 58,090,688-row
corpus and one training seed. Training-source and schedule-plan hashes differ;
matching the initial tensor, epoch row count and realized update count does
not isolate the effect of SF, one-teacher routing, or either head.

The preregistered question was whether the cheaper Selected-E recipe is within
30 Elo of D on a fixed strict evaluation panel. The sealed packet SHA-256 is
`f4ce6a5069bfd9717694239892b71df6f47728edcb52b12fc2ca16d9515763b1`;
its pre-result decision text is SHA-256
`6a34f815d88ba56d156aa30467ef291b476337222747681c5cf30fe1ffc94ad2`.
It fixes 576 two-color opening pairs from the prior Selected-E versus E arena,
400 matched simulations per side, training search shape, prior temperature
1.0, play temperature 0.1, a 1,000-ply cap and strict rule50-aware six-man
Syzygy WDL+DTZ in search and adjudication. The arena limit is 7,200 seconds,
with an enclosing 7,800-second watchdog. All 1,152 games, exact opening and
color coverage, PGN and tablebase telemetry must pass before a strength
decision. No early stop, resumed game or post-result bank enlargement is
permitted by this packet.

The completed run met the fixed 1,152-game and 576-pair roster. The arena
reported 609,292 successful search tablebase hits from 820,861 probes, with
both WDL and DTZ tables open. The postrun verifier checked the game log, PGN,
opening/color pairs and the strict rule50-aware six-man telemetry. There were
no resumed or truncated pairs. The deciding run took 4,606.4 seconds in the
arena, or 4,684.5 seconds including the enclosing verification. The terminal
receipt and game log have SHA-256 values
`78d4d6f68194355d61c02dda19a137a0601e2363f7232f869cc7aa4f4013bce4` and
`3f8905892362c3bd57a4a467f5ea25243b6d02baa1ae1d78d0fb54319e99370e`.
An independent audit reconstructed all 576 scores, replayed 1,152 PGNs and
re-probed all 593 Syzygy terminal boards with strict rule50 WDL+DTZ. Its
sealed manifest SHA-256 is
`e961a2364441697e07091e3d312ff4d85894a2c5b98f84954a5cd9dc72e82f90`.

For the primary decision, score each two-color opening pair from Selected-E's
point of view. Take the mean and sample standard error of the 576 pair scores;
transform mean ±1.96 standard errors from score to Elo. The preregistered
conditional noninferiority gate passes only if the 95% lower Elo endpoint is
strictly above −30. An upper endpoint below −30 is a negative result;
otherwise the fixed-budget result is inconclusive. This interval describes
opening-pair sampling for these checkpoints, not variability from training
seeds, source mixtures or the eventual 500M-row run.

The observed pair score was 0.47265625, equivalent to **Selected-E minus D
−19.02 Elo**, with paired normal 95% interval **[−33.43, −4.68] Elo**. The
pentanomial counts, in candidate view, were 34/116/226/153/47 for
WW/WD-DW/DD-WL/LD-DL/LL. The lower interval endpoint crosses the −30-Elo
margin and the upper endpoint exceeds it. The exact preregistered decision is
**inconclusive; no advance**. The central estimate favors D, while this fixed
bank cannot establish that Selected-E is more than 30 Elo worse or within 30
Elo. We will not enlarge this observed bank to force a verdict.

This one-seed checkpoint comparison cannot assign the difference to removing
Stockfish value, changing the BT4/Ceres policy from an on-row blend to one
teacher per row, or changed training/source schedule pins. It does not overturn
the separate Selected-E versus E cost screen, nor validate a 500M target
distribution. The next data decision should measure actual one-teacher label
wall time and qualify cross-source row identity/admission before committing a
large corpus; a new training contrast needs an independently selected seed or
new recipe, with its decision rule fixed before a new arena.

The [compact evidence](evidence/2026-09-28-selected-e-vs-d-direct-strict.json)
records the terminal identities, paired statistics and audit receipt without
publishing checkpoints, full PGN, game log or host paths.
