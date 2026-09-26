# Direct SF-close versus global sharpened BT4

Registered September 6, 2026 after the SF-close three-budget screen completed.
The independent full-screen readout and launcher review must pass before execution.
Use the pinned runtime preregistration and launch manifest for execution identities.

## Question

Which of the existing C20T05 and G20T05 checkpoints is stronger in a direct
matched-search screen? C20T05 redistributes existing SF mass among all stored
SF maxima plus d9 top-three moves within 20 cp, using conditional BT4 at T0.5.
G20T05 mixes 80% normalized SF with 20% global BT4 at T0.5. Value targets and
architecture are unchanged. Both use one full epoch of the same 18,910,484-row
source, training seed zero, and the frozen wise-cloud runtime.

Completed global comparisons against S0 favored sharpened over raw global BT4,
but did not separate G20T05 from E0 reliably. Completed C20T05 comparisons against
E0 favored C at all three budgets: +37.32 Elo at 25 simulations [18.93, 55.92],
+50.74 at 100 [31.74, 70.04], and +39.08 at 400 [21.98, 56.36]. The paired
400-minus-25 score difference was +0.25 percentage points [-3.35, +3.75],
so increasing relative search advantage was not established. Because those recipes faced different
opponents, subtracting their Elo estimates cannot rank C against G. Reusing their
checkpoints in a direct match answers that question without another training epoch.

## Prospective decision and protocol

Run one **100-simulation** match. Expected cost is approximately 0.7 GPU hours;
the hard arena/supervisor cap is **5,400 seconds (1.5 GPU hours)**, including
its 30-second TERM-to-KILL allowance. The internal arena deadline is 5,370 seconds.
Choose 100 simulations because this screen selects the next trained challenger;
the completed C400 result establishes that C remains competitive against E0
at the deeper budget. Reserve a direct deployment-budget comparison for finalists
rather than automatically running all three budgets for every contender.

The primary measure
is C20T05's paired 95% Elo interval against G20T05: wholly above zero favors C,
wholly below zero favors G, and an interval crossing zero is inconclusive. An
inconclusive screen does not establish equivalence or require more games.

Use 1,000 games / 500 color-swapped opening pairs, matched_sims with training
search shape, both search-prior temperatures explicitly 1.0, original book seed
42, 16 opening plies, maximum 300 plies, move temperature 0.1, compilation on,
128 concurrent games, and evaluator batch 4096. Preserve the original book for
this development screen; reserve the fresh history-bearing confirmation opening
bank. No rolling decision, SPRT, automatic retries or extending the horizon after
seeing results. A timeout or incomplete bank is unfinished evidence, not a loss.

The frozen reader's read_arm function checks the complete paired bank and realized
search settings. The launcher additionally binds exact candidate/reference paths
and hashes, book identity, execution settings and runtime. Use the analytic paired
interval supplied by the existing pentanomial estimator, not independent-game
error bars. This one-budget match cannot estimate a search-scaling slope.

## Identities and resource handling

- C20T05: runs/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05_epoch_v1/checkpoint.pt,
  SHA256 8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3.
- G20T05: runs/armB/qtemp_0.0005_hist_20m_bt4_global_G20T05_epoch_v2/checkpoint.pt,
  SHA256 bd8c208a95247373f423be0649329e9c100db64ab1b5a5e68fd7a6aec3769a74.
- Frozen runtime checkout: .dev/worktree/wise-cloud,
  revision 7ec261509fb7345cf1ca0ad73809193fc2749bb1; Python 3.10.12,
  Torch 2.11.0+cu128, CUDA 12.8, NumPy 1.26.2. Native builds remain pinned.
- Frozen reader: /tmp/deepfin-bt4-prior-one/scripts/bt4_joint_readout.py,
  SHA256 bdb6b2e2a2dc04c087b7ae36628211015aaba50ac96c4110a5f372c56cefd79f.

Use a new output directory and the shared scratchpad/gpu0_experiment.lock,
low CPU priority, and at least 150 GiB free disk. Wait for the existing SF-close
match to finish and for the GPU lease. Preserve the generators, raw BT4 labeler,
archive queue and all input checkpoints. A surviving timeout supervisor must
bound the arena even if its coordinator exits; cleanup affects only the owned
arena session. Preserve partial artifacts for explicit recovery. Record actual
runtime and bank identities, process/command, completion checks and GPU charge.
Launch only after the existing C400 completion receipt and independent review pass.
The raw BT4 labeler may take its normal GPU lease between experiments.

## Interpretation and follow-up

This is a nominal, exploratory comparison of existing seed-zero checkpoints on
a reused development opening bank. It is not independent recipe confirmation,
100M transfer evidence, an RL result or a claim of optimality. Reuse completed
training provenance, including the previously disclosed holdout and historical
control limitations; do not relabel this as a fresh matched-training experiment.

If the close family remains competitive, the already materialized sharpened
exact-tie recipe is a substantive possible challenger: it may change the winner,
not just attribute C's gains. Its bounded 8,192-row independent data check passed;
prospective and realized training schedules still need qualification if selected.
If the global family clearly leads, a larger sharpened global dose is another
substantive candidate. Do not train a grid simply because its corpora exist.

After choosing a recipe and strongest relevant alternative, retain fresh matched
training-seed and fresh-opening confirmation, including deployment-relevant search.
Larger-corpus observation selection and subsequent RL benefit remain separate
questions in the broader bootstrap research.

## Execution record

State parent: scratchpad/bt4_joint20/direct_close_global_run01. The pinned
preregistration.md is this prospective snapshot; launch_manifest.json supplies the
new arena output directory, exact executable/input hashes and the 100-simulation,
5,400-second choice. From the reviewed launcher checkout:

```bash
/usr/bin/python3 scripts/bt4_direct_screen.py --manifest /home/josh/projects/chess/scratchpad/bt4_joint20/direct_close_global_run01/launch_manifest.json
/usr/bin/python3 scripts/bt4_direct_screen.py --manifest /home/josh/projects/chess/scratchpad/bt4_joint20/direct_close_global_run01/launch_manifest.json --execute
```

This record has no playing result yet. The launch manifest is finalized after the
launcher review; the immutable snapshot is not edited while a job is running.
The completed SF-close reader artifact is
scratchpad/bt4_joint20/sf_close_run02/completed_full_screen_readout.json,
SHA256 2dde2ce5b46bc69b662be129b3752ffe37f93aa383f2d3bd06dbeb5d581a7257.
