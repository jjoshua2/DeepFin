# Conditional fresh confirmation design — draft for review

September 6, 2026. Written before observing any E0T05-versus-C playing outcomes.
This is preparation, not authorization to launch confirmation or a new active
preregistration. The final manifest and chosen pair need a dated registration
and independent launch review after the current complete screen is interpreted.

C is the current supported leader. Its completed direct 100 screen against G20T05
is +29.95 Elo [11.89,48.18]. The current E0T05 screen asks whether stored SF top
ties with BT4 T0.5 can beat C's wider SF-close set at the same BT4 temperature.

Proposed selection:

1. If the complete current screen favors E0T05 under its registered interval rule,
   confirm E0T05 versus C with both recipes freshly trained at the same nonzero seed.
2. Otherwise retain C as the provisional leader. Choose the stronger plausible
   alternative from E0T05 and G20T05 by their completed development scores against
   this exact C checkpoint at 100 simulations. E0T05's score is reported directly;
   G20T05's score is 1 minus C's score in the completed C-versus-G bank. Prefer E0T05
   if these scores are exactly tied. Both banks have the same registered opening
   pairs and protocol; validate that alignment before using the comparison.
3. The alternative choice is a heuristic to select a useful confirmation opponent,
   not a direct E0T05-versus-G result or a significance claim. Do not subtract Elo
   estimates or use this selection as independent confirmation evidence. A current
   interval crossing zero remains inconclusive; it does not prove equal recipes.

Proposed confirmation uses training seed 1 for both fresh runs and one 100-simulation
match of 1,000 games / 500 pairs on the reserved history-bearing FEN bank:
`scratchpad/bt4_joint20/confirmation_openings_v1/openings.fen`, SHA256
`e0d13b2ea70c0ac278570a0e463c3c1c3030a18256522bcba864db23cdc07c98`.
Use a separately registered arena random seed (proposed 20260907), the same full
qualified training search, both prior temperatures 1.0, maximum 300 plies, move temperature 0.1,
compilation on, 128 concurrent games, batch 4096. Keep the final full epoch for each role.

100 simulations matches the deciding development screens and configured base
`mcts_simulations: 100`; the live config also has fast/start budgets, so this is
not a claim that every production search uses 100. A deeper 400-simulation question
remains distinct; do not automatically run all three budgets for confirmation.

Expected compute is about 5.9–6.1 GPU hours for two epochs and one arena. Proposed
hard cap 10.5 GPU hours: 4.5 reference training + 4.5 candidate training + 1.5 arena, including
termination allowances. Independent CPU schedule/readout limits are separate.
Recheck actual game/batch order under seed 1, completed summaries/checkpoints,
current metadata provenance and frozen runtime before the arena. Both models
must be trained fresh; no adoption of seed-zero or existing seed-one checkpoints.

Use one fixed horizon with the same paired estimator. Interval wholly above zero
favors the named candidate, wholly below favors the reference, crossing zero is
inconclusive. This fresh result should be compared with the selected development
contrast. A reversed or unresolved result does not establish robust replication
and should trigger an explicit scientific decision; no automatic retries, more
games or further seeds are authorized by this draft. Preserve all results,
including contradictory evidence. Do not claim a training-seed variance estimate
from one fresh paired seed.

The reserved bank has not been used for development match outcomes, and its
terminal positions do not overlap the prior development opening bank. This does
not certify exclusion from training data. Preserve the existing purity and
historical-control limitations and keep 100M observation-selection/RL transfer
claims separate.
