# Selected-E versus E: seed-121 strict arena readout (2026-09-27)

**Decision: PASS the preregistered noninferiority gate.** The selected one-teacher-per-row policy/value target candidate finished the same 58,090,688-row, 113,459-update seed-121 training horizon as the saved original-E reference. Their fixed 576-pair, 1,152-game, color-swapped arena completed without truncation or resumed pairs. Candidate-minus-E paired Elo was **−3.016**, with a normal 95% paired interval **[−17.737, +11.694]**. Its lower bound exceeds the preregistered **−20 Elo** margin, so the exact precommitted rule advances the recipe to a *future real one-teacher annotation-cost screen*. No annotation-rate or cost saving was measured here.

The hypothesis was that selecting BT4 or Ceres once per row for both main targets would retain strength close enough to original E's blended two-teacher targets to justify measuring a cheaper collection route. The control was the saved original-E seed-121 checkpoint, with the same initial tensor identity. The arena used the fixed 576 opening pairs in both colors, 16 opening plies, 400 simulations per side, and strict rule-50-aware six-man Syzygy search/adjudication. The realized candidate score was 0.495660; the paired pentanomial counts were 40 / 146 / 192 / 160 / 38 (WW / WD-DW / DD-WL / LD-DL / LL). The decision used the full fixed bank and its paired uncertainty, rather than the point estimate or an early stop.

| External artifact identity | SHA-256 |
| --- | --- |
| Arena plan | `4bfdc1ecc539873d693f5e1db3f581ae13603ede483ff840eb9d50eeaf048a86` |
| Complete arena terminal (`teacher-e-selected-vs-e-rule50-20260926/complete.json`) | `9a74429bb2c39ae848b487ef25a213cd1df4a4d30f63c1d898cb7ba4c4e17c03` |
| Arena game log / PGN / results JSONL | `e142b455c184420b6ef22f32a90ca8d958ebd6b87281add8f092506a498fd053` / `884096fcdf36c33bd789051f7265d020026659a4efe7abb22e3fd6e1d1296840` / `fb87c046688282b70a1a1eb4f48bd712d98f92ef10ea8f068137f58a5e6c747c` |
| Selected-E training completion (`teacher-e-selected-training-registration-v4-20260926/complete.json`) | `fc9ab8c34007d02321eb10646095387a3ad479734cdcbd23362c584ed8d6c8b2` |
| Selected-E checkpoint | `154e768ce1a76dda98b0077244227df1591ef3d4a4e2e438e16c49b1a58ac203` |
| Original-E checkpoint | `83e10fb3bcf5d09541b3e8cddfd7c6bfae55865913cbfff095970d7d6a870168` |

The saved game log and PGN begin from each opening FEN without independently recording the actual pre-opening 16-ply move stack. The terminal reports zero natural results left unreproven without book history, but these records cannot retrospectively authenticate every executed pre-opening history or a repetition depending on it. Keep that frozen evidence limit when interpreting the otherwise completed strict arena. The interval is conditional on these two seed-121 checkpoints and this fixed opening sample; it neither proves superiority/equivalence at every margin nor estimates training-seed or 500M-scale effects.

**Next step:** preregister and independently qualify a same-row, one-teacher collection cost/rate screen after the sole-GPU source work, with exact target/readback and physical-call accounting. Existing labels let this experiment compare targets and strength, but do not measure new teacher inference, owner reuse, storage, qualification, or trainer-admitted unique throughput. This result grants **zero new corpus or owner-label credit** and does not settle the 500M SF/BT4/Ceres source mix.
