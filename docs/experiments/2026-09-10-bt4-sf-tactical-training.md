# SF tactical guidance for BT4 policy

Status: training and the registered 256-game match completed. Tactical attenuation scored **−13.6 Elo versus B100, paired 95% interval [−52.4, +24.9]**, at 400 simulations. The result is unresolved; retain B100 and move to a different teacher-mixture question without extending this match. This updates the [earlier readiness record](2026-09-08-b100-tactical-policy-readiness.md).

## Question and treatment

Can SF's large move-quality gaps and mate findings improve pure BT4 policy supervision? The baseline is B100: stored BT4 policy sharpened at teacher temperature 0.5, with original SF value targets.

The candidate multiplies that policy by `max(0.1, exp(-max(0, gap_cp - 100) / 100))`, then normalizes. Categorical mate rules handle mate scores separately. This retains BT4's relative ranking among moves inside the 100 cp allowance while penalizing substantially worse moves. It cannot restore moves whose stored BT4 probability is zero. All 16 nonpolicy arrays, including SF values, remain byte-verified copies; value supervision is unchanged.

## Completed preparation and recovery

The producer completed 18,910,484 rows in 2,309 shards from the original 20 M raw prefix, exiting successfully after 21,242.78 seconds. Its final supervisor admission failed because the producer serialized cp-domain endpoints as floats and the checker expected integer JSON spelling. [PR620](https://github.com/jjoshua2/DeepFin/pull/620) fixes that mismatch. The original failure receipt is preserved; no data regeneration or rewriting was needed.

The corrected admission passed 31 focused tests and independent Grok review. A separately reviewed recovery inspector checked the genuine recipe, raw/source identities and every shard's metadata/layout, inheriting the pinned producer's legal joins and 16 nonpolicy byte-copy proofs. Recovery qualification passed in 59.29 seconds. The subsequent original-runtime schedule check passed: 18,910,484 rows, 36,935 batches and the same canonical schedule as the completed control.

## Registered training and deciding match

Training uses the unchanged historical trainer/runtime, seed 0, batch 512, 16 planning/16 loader workers, 36,935 updates and 420 windows. The training cap is 16,200 seconds; the whole operation has a 21,630-second bound including waiting and realized schedule verification. The child inherits the shared GPU lease, so existing labeling yields at its normal boundary. The coordinator executes training only; arenas remain separate.

The registered comparison is against the original one-epoch B100 checkpoint: 400 simulations per side, 128 swapped opening pairs, priors 1.0, opening seed 20260909, maximum 300 plies and no tablebases. Its paired 95% score interval determines a bounded win/loss/unresolved readout; an overlapping interval does not trigger automatic extension. No automatic extra depths, temperature sweep or promotion is registered. Fresh-seed and fresh-opening confirmation remain necessary before a strong general claim.

This remains one training seed on a development corpus/panel. Original source/history/score-bound and trainer purity caveats remain; it does not establish a result for 100M rows or resumed RL.

## Completed training

The candidate completed all 18,910,484 rows and 36,935 updates. Actual staging and realized sampling share physical schedule hash `fb08f981…`; separate reconstruction matched canonical schedule `dc687fc3…`, the original control protocol. Training took 9,063.98 seconds and the whole operation took 9,355.24 seconds (2.60 hours), both exiting successfully. The final checkpoint hash begins `e2d4ed70…`; the compact evidence contains full identities. The GPU lease was released before CPU schedule verification, and existing BT4 labeling resumed.

The producer reported 1,841,617 rows with a winning-mate move; every such row already had positive B100 probability on at least one reported winning-mate move. This rules out an all-mate-moves-missing support intervention on this corpus, not insufficient probability or tactical search mistakes. The 6,711 storage support losses count individual move entries, not positions; at most 0.0355% of rows are affected. Source score-bound limitations remain.

## Match preparation and launch

Both final checkpoints loaded on CPU with the same 61,444,448-parameter architecture and the frozen runtime. The actual opening histories, 400-simulation search settings, priors 1.0 and no-tablebase configuration matched the registration. CPU preparation completed in 113.80 seconds with CUDA uninitialized. Independent Grok review found no integration defects in the preparation/launch path; the parent checked the completed CPU package and final contract.

The fixed match acquired the shared GPU lease at a labeling boundary after 208.18 seconds of waiting and started its arena process. Its 5,400-second owned stage and 10,230-second whole-operation bound remain unchanged. It completed all 128 pairs and exited successfully. The whole match operation, including lease waiting and readout, took 1,669.77 seconds (27.83 minutes).

## Completed match and decision

The candidate scored 48.0469% across 256 games / 128 swapped opening pairs. The candidate-oriented pentanomial counts were WW=19, WD/DW=21, DD/WL=40, LD/DL=27, LL=21. The paired standard error was 0.028226; the nominal 95% interval transforms to −52.4 through +24.9 Elo around an estimate of −13.6 Elo.

The interval crosses zero, so the registered rule gives an unresolved result. This is no demonstrated benefit from this particular 100 cp allowance / 100 cp decay / 0.1 floor recipe; it does not reject all Stockfish tactical guidance. Retain pure BT4 T0.5 policy with SF values as the incumbent. Do not extend the match or promote the candidate.

The strict package reader checked the finished bank and checkpoint content. A separate parent calculation reconstructed all candidate scores from result/color, checked 128 unique swapped pairs and matching opening endpoints, and reproduced the paired estimate and interval. Launch runtime/history qualification comes from the previously reviewed preparation receipts, not from the bank reader alone. The final interpretation and this prose are parent-reviewed; no new independent playing-result review is claimed.

## Evidence and next decisions

[Compact launch and completion evidence](artifacts/b100-tactical-training-20260910.json) records qualification, completed training, realized schedule and checkpoint identities. Bulk corpus, logs and checkpoints remain outside Git. The completed bank, package contract and independently reproduced numerical readout are identified in that artifact.

Weighted teacher mixtures are the preferred next family for both heads. Ceres policy mixtures should initially keep value fixed; adding Ceres to SF/BT4 values should be tested with policy fixed. Next prepare a 50/50 BT4–Ceres policy mixture at teacher temperature 0.5 for each component, keeping SF values fixed. A later separate value candidate is 50% SF / 25% BT4 / 25% Ceres with policy fixed. Collection and producer qualification are still prerequisites; neither mixture has a trained strength result. These directions are not a mandatory queue.
