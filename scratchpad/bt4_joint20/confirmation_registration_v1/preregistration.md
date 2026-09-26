# Fresh-seed confirmation: SF-close versus sharpened stored ties

Registered September 7, 2026 UTC, after independent review of the completed
development screen and before either fresh training run or confirmation outcome.

## Selection and hypothesis

C20T05 remains the provisional leading recipe. In the completed direct screen,
E0T05 scored 347 wins, 221 draws and 432 losses against C: score 0.4575,
−29.60346 Elo, paired nominal 95% interval [−48.72166, −10.66261]. Independent
review passed; receipt SHA256
`245a9baf86c36ebeebefa0ccfeb54dc0b72c8f3b08e474eda78cdf264eba0973`.

The prewritten conditional selection rule chooses E0T05 as the fresh challenger:
its development score against C was 0.4575 versus G20T05's 0.457. That 0.0005
difference is only an opponent-selection heuristic, not evidence that E beats G.
C is the reference throughout; E is the candidate. The hypothesis is that C's
advantage survives a fresh paired training seed and fresh match openings.

C redistributes existing target mass over all stored SF maxima plus d9 top-three
moves within 20 effective centipawns, using BT4 temperature 0.5. E uses the same
BT4 temperature but redistributes only inside the stored maximal-target ties.
Stored ties can reflect quantization or saturation rather than underlying SF
score equality. This comparison tests those two complete target recipes.

## Fixed experiment

Use the exact published corpora and frozen runtime identified by the accompanying
`manifest.json`; change no targets, architecture, optimizer or search parameters.
Train both models from scratch with seed 1, in reference-then-candidate order.
Use a full game epoch, batch 512, 16 planning and 16 loading workers, and 88-step
training windows. Derive expected counts from the fresh seed's actual plan;
do not substitute seed-zero counts. Require complete planned/realized rows and
batches, matched canonical schedules, finite window loss and gradient norms,
zero nonfinite skips and CUDA retries, and the exact final checkpoints.
Intermediate checkpoints are evidence, not candidates.

Run one fixed 1,000-game / 500 color-swapped-pair match at 100 simulations each,
arena seed 20260907, prior temperature 1.0 on both sides, maximum 300 plies,
move temperature 0.1, 128 concurrent games, evaluator batch 4096 and compilation
on. Use both complete search dictionaries from the qualified C100 readout.
100 simulations is the deciding base budget, not every production search budget.

Use the reserved 500 history-bearing openings, each with 16 legal opening moves:
SHA256 `e0d13b2ea70c0ac278570a0e463c3c1c3030a18256522bcba864db23cdc07c98`.
The final overlap receipt
`confirmation_openings_v1/completed_bank_overlap_final_E0T05.json`, SHA256
`85e88d4803e04a57fa8887c66c4171b91b2bf59753421aaf7a85add31076b0c1`,
establishes development alignment across all 14 completed banks and zero terminal
opening overlap with the reserved bank. It does not establish training-data purity.

## Decision, budget and recovery

Use the registered paired sample-variance normal 95% score interval transformed
to Elo, with opening pairs as the sampling unit. Require exactly 1,000 canonical
games, 500 complete pairs, no orphans, and matching realized identities/settings.
An E-minus-C interval wholly below zero confirms C's expected direction; wholly
above zero contradicts the decisive development finding and favors E in this
fresh match; crossing zero is unresolved. Get independent completed-result and
provenance review before the scientific conclusion. Preserve contradictory or
inconclusive evidence; neither triggers automatic extra games, seeds or retries.
No rolling outcome inspection, early stopping, checkpoint selection or grid.

Hard GPU cap: 37,800 seconds (10.5 hours), comprising 16,200 seconds per training
stage and 5,400 for the arena, each including 30 seconds for termination. Expected
cost is about six GPU-hours. CPU schedule verification is capped at 1,800 seconds
and readout at 120. Queue waits and CPU stages do not consume GPU charges.
Use the qualified launcher at commit `12f26da49f83225086e237ad0257494346abdb05`,
SHA256 `07fea41523fbb8068cfd910d0ff5e629622ccb81a180e7160188106d4040c09c`.

Preserve live generators and archival work. Share the existing GPU lease; release
it for CPU-only stages. Keep the independent timeout supervisors, stop handling,
two numerical threads, low scheduling priority and 150 GiB disk reserve. Output
paths must be new. Preserve partial state on failure; diagnose any interrupted
stage before separate recovery. Do not adopt seed-zero checkpoints or relaunch
over existing outputs. Merging code does not update this frozen runtime.

A passing result supports the best-tested 20M bootstrap recipe under this search
budget. It does not establish universal optimality, training-seed variance from
one fresh paired seed, a search-scaling slope, 100M transfer or RL benefit.
Retain the historical control/purity limitations and keep the larger-corpus
observation-selection contract and eventual RL evaluation as separate questions.
