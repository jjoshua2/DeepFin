# Fixed-input BT4 and Ceres batch profiles on 58,820 rows

Larger batches reduced measured label-call work on the same 58,820-row roster. BT4 batch 128 saved **28.558%** against batch 32; Ceres batch 512 saved **74.070%** against batch 32. These are separate four-arm, fixed-input numerical-profile screens. They measure conversion, inference and raw-head writing inside the call core; charged shape-matched warmup, session startup, target materialization, forensic readback, fresh source generation and integrated production writing are outside that clock.

| Teacher and completed arms, in execution order | Call-core seconds by arm | Mean baseline → candidate | Saving |
| --- | --- | --- | ---: |
| BT4: B32a, B128a, B128b, B32b | 37.315552, 26.625312, 26.724909, 37.360918 | 37.338235 → 26.675110 s | 28.558% |
| Ceres: C32a, C512a, C512b, C32b | 156.517484, 38.798012, 38.899173, 143.121593 | 149.819539 → 38.848593 s | 74.070% |

Both paired savings were positive: BT4 28.648% and 28.468%; Ceres 75.212% and 72.821%. Each arm processed 58,820 fixed rows with a shape-matched warmup. BT4 used a short final call; Ceres repeated the last real feed to fill its physical tail. The original independent audits reopened all raw-head files, checked their full-file hashes, geometry and finite values, and reproduced the fixed 4,096-row quality sample with the screen's target math. Repeated runs at the **same** batch profile produced identical raw bytes. A later [direct compact-order readback](2026-09-30-full58820-corrected-selected-targets.md) corrected the Ceres top-1 count below without changing either raw-head bank or the measured call times.

| Numerical check, batch candidate versus batch 32 | BT4 batch 128 | Ceres batch 512 |
| --- | ---: | ---: |
| Legal-policy top-1 agreement | 100% | 99.609375% |
| Legal-policy total variation, p99 / max | 0 / 0.000000119 | 0.013068 / 0.026917 |
| Training WDL total variation, p99 / max | 0 / 0 | 0.004378 / 0.016953 |
| Frozen numerical-profile budget | Pass | Pass |

The sample is 4,096 rows and 8,192 paired comparisons per teacher. The training WDL is native for BT4 and combines two value heads for Ceres. Ceres has nonzero policy and WDL differences; its batch-512 output is an allowed numerical variant, not a byte-preserving replacement for batch 32. Reported policy-TV maxima for the endgame, high rule-50 and mate-in-one slices remain below the overall policy-TV maximum budget; subgroup top-1 and WDL quantiles were not separately assessed. Mate-in-one coverage does not establish behavior for longer mate sequences.

**Compact-order correction, 2026-09-30.** The first Ceres quality conversion paired board-order legal logits with compact-sorted legal slots. On the same frozen sample, a direct raw-head-to-compact reconstruction changes Ceres top-1 disagreement from 36/8,192 (99.560546875% agreement) to 32/8,192 (99.609375%). Policy-TV p99/max remains 0.013068130314350137 / 0.026916921138763428; calibrated WDL-TV p99/max remains 0.004377536773681641 / 0.0169525146484375. The frozen numerical-profile budget still passes. These are paired numerical-profile facts, not a strength readout. The original raw quality audit remains historical evidence of the mistaken mapping; the corrected direct audit and all 8,192 pair records are pinned in the [correction record](2026-09-30-full58820-corrected-selected-targets.md).

**Execution and credit boundary.** The BT4 quartet came from the interrupted v5 run03. Its four BT4 jobs finished, but the attempted C32a worker then failed at import, leaving the run's `FAILED_ZERO_CREDIT` receipt. The BT4 result is therefore an independently audited completed-quartet diagnostic, not a completed eight-arm run. A later v6 run04 repaired the Ceres worker import environment and completed **only** the Ceres quartet with a `COMPLETE_BATCH_PROFILE_ZERO_CREDIT` terminal. The two quartets have separate source packets, output roots and audits; there is no combined eight-arm terminal.

Historical factorial collection already used Ceres batch 512/group 4 for 22,776,111 missing labels ([factorial record](2026-09-19-58m-policy-value-factorial.md)). The 74.070% comparison is against this screen's audited fixed-32 diagnostic baseline, not a new improvement over every earlier labeler. Likewise, the [500M capacity sketch](2026-09-29-500m-source-target-candidate.md) used unverified, mismatched 800.19/329.64 rows/s label assumptions. These savings cannot be subtracted from its total-day figure. The next adoption measurement is an integrated producer-to-target-writing rate on qualified, retained positions, with numerical acceptance and source/readback costs carried through. Neither screen awards source, corpus, target, training, Elo or 500M-capacity credit.

## Compact evidence identifiers

The identifiers below are SHA-256 digests of the named immutable files or audit content streams. They identify the private evidence without embedding machine-specific artifact paths in this public record.

| Artifact | SHA-256 |
| --- | --- |
| BT4 v5 source packet manifest | `eb2f0a830afb7af74b6b0bd82f79d09af713f48f072b866dcbf9118977d6a26e` |
| BT4 run03 failure receipt | `e430ed9995bf483888bd6283964e1310130f6e55d54b559b185d0ad599ea58e6` |
| BT4 completed-quartet independent audit | `2d4a9d3471a0198dbff0ea071bc63836fdd1229f213811ab67b377ca6ef9cff6` |
| BT4 raw heads, batch 32 repeated / batch 128 repeated | `02ac624b79337f0902bd9f9b6efcd0af27b31dc3b9487d5f0584653f4ffdaf59` / `4087320aab0deffefd1eba63efcd2f3cf236f0415b89f6d93355dc4e2dfd8303` |
| Ceres v6 source packet manifest | `c56998188d06de207e3a26de5a2d6a89f2c4fac5e5f2360fcac5976d270dc058` |
| Ceres run04 complete terminal | `144d1bec3ec06f1e3a6d4bb2aecd87acbd01fb026da98ea7c4d7de5639261200` |
| Ceres completed-quartet independent audit | `cefb946fcb9b5ad0cf5dd507c8c4efb15e07eef51e6b3d9ea5da9a818ae5f856` |
| Ceres raw heads, batch 32 repeated / batch 512 repeated | `a1f9e40a967796d00ce26a4b0ed7658b9cc3e4ace64995b9934e0cdfba04c907` / `4795dc6e5c37fed558ce9c43b8f65955dd554b5a17c9fe174146a7144098a702` |
| Shared quality-sample indices | `bdaa911debf32e7f2d9da473ad24324e56941568c919602fac422c12a122ac18` |
| BT4 audit's adapted quality selector | `bbd8a57ca2cfbaee0589a971d32baea2b25c1b85b2508614cb0512aa33f4814f` |
