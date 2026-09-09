# B100 search-prior calibration — September 9

Completed development comparison: sharpening B100’s search prior from 1.0 to 0.7 scored **52.34%**, or **+16.30 Elo** with a nominal paired 95% interval of **−20.83 to +53.80**. This misses the preregistered follow-up threshold. Keep prior 1.0 for recipe comparisons and prioritize value, tactical-policy and teacher-family experiments. This does not establish that 0.7 is ineffective or that 1.0 is optimal.

## Registered comparison

Use the same completed seed-zero B100 checkpoint on both sides, changing only search-prior temperature: candidate 0.7, reference 1.0. No retraining or target-temperature change. Run 256 games as 128 color-swapped opening pairs at 100 simulations, with 16-ply opening history and opening seed 20260909. The panel has 128 unique endpoints and no endpoint overlap with the existing 500-opening recipe panel or final-confirmation panel.

Common settings: training search shape, flat root noise on, move temperature 0.1, maximum 300 plies, no tablebases, rolling pool 128, evaluator batch 4096 and compilation requested. The arena’s flat root noise differs from production’s per-ply schedule; this is calibration of that common arena profile. The whole invocation was capped at 1800 seconds, with a 1560-second internal arena limit. All 256 games were required; no SPRT or optional bank extension.

The precommitted resource-allocation rule was to consider a separately registered follow-up only when estimated Elo was at least +15 and its nominal lower 95% bound exceeded zero. It is a development rule, not a promotion test.

## Completed result and review

| Measure | Result |
| --- | --- |
| Games / opening pairs | 256 / 128 |
| Candidate score | 0.5234375 |
| Paired score 95% interval | [0.470064, 0.576811] |
| Elo / nominal 95% interval | +16.30 [−20.83, +53.80] |
| Pentanomial WW / WD / DD-or-WL / LD / LL | 22 / 20 / 51 / 18 / 17 |
| Actual invocation | 783.07 seconds, exit 0 |
| Follow-up condition |Point estimate passes; lower bound fails |

Independent review verified all unique game keys, color swaps, panel endpoints, realized settings, bank/contract/readout identities and actual stage completion, and independently reproduced the paired arithmetic. The producing reader verified checkpoint content; the completed review reused that hash proof and checked unchanged input identities. This is one checkpoint, one development panel and one simulation budget, with no independent training-seed confirmation or search-scaling conclusion.

The first operational attempt failed before any games because its GPU-check executable path did not match WSL. A separate corrected attempt retained the same scientific settings and panel; only that completed attempt contributes results. The original failure remains recorded.

[Compact result and artifact identities](artifacts/b100-prior-calibration-20260909/result.json) preserve the result and audit hashes. Full game-bank and operational records remain retained separately; this compact record does not publish them.
