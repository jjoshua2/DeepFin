# SF tactical guidance for BT4 policy

Status: registered training and realized-schedule verification completed successfully. CPU preparation passed and the fixed-match operator has started; no match result or strength gain is claimed. This updates the [earlier readiness record](2026-09-08-b100-tactical-policy-readiness.md).

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

The fixed-match operator has started and acquires the shared GPU lease at a labeling boundary. Its 5,400-second owned stage and 10,230-second whole-operation bound remain unchanged. It produces one 128-pair bank and a strict completed-bank readout; no result is available at this snapshot.

## Evidence and next decisions

[Compact launch and completion evidence](artifacts/b100-tactical-training-20260910.json) records qualification, completed training, realized schedule and checkpoint identities. Bulk corpus, logs and checkpoints remain outside Git. The next deciding readout is the registered match, not teacher agreement alone.

Weighted teacher mixtures are the preferred next family for both heads. Ceres policy mixtures should initially keep value fixed; adding Ceres to SF/BT4 values should be tested with policy fixed. The tactical result and actual collection readiness will choose the next comparison; these directions are not a mandatory queue.
