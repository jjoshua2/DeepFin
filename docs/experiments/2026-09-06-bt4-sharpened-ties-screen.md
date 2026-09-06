# Sharpened stored-tie BT4 versus SF-close BT4

Registered September 6, 2026 after the complete C20T05-versus-G20T05 direct screen.
This registration precedes training of E0T05 and all outcome observation for it.

## Question and selection

Does restricting sharpened BT4 to the stored SF target top ties beat the wider
SF-close recipe when both use BT4 temperature 0.5? C20T05 is the current leading
tested checkpoint: it beat E0 at 25, 100 and 400 simulations, and directly beat
G20T05 at 100 simulations by +29.95 Elo [11.89, 48.18]. The latter match was
414 wins, 258 draws and 328 losses in 1,000 games / 500 opening pairs.

E0T05 is a substantive remaining challenger. This comparison isolates restricting
redistribution to stored top ties versus C's broader candidate set at the same
BT4 temperature. Stored target ties can arise from float16 quantization or target
saturation; they are not necessarily exact ties in underlying SF centipawn scores.
This is not an E0-versus-E0T05 sharpening-only comparison. Do not automatically
train G50, a larger grid, or more games based on partial or inconclusive results.

## Fixed targets and training

Use the already published corpus
`data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_toptie_t050`:
BT4 temperature 0.5, alpha 1, `stored-top-set-only-v1`. Redistribute the existing
SF mass of the tied maximal stored targets using normalized sharpened BT4 inside
that set; preserve targets outside the set. Unique-max rows do not change.
C instead uses all stored SF maxima plus d9 top-three moves within 20 effective
centipawns, redistributing that selected set's existing mass with BT4 T0.5.

E0T05 mix summary SHA256:
`a85ba1403c2477018bdc59b622311094027ca17cf045dc78090f6c0ee9f5463d`.
Derived summary SHA256:
`4354bafe8e1435bb030faaa161810bbfff667a2b83c647940ad371eb2ab1caf1`.
The existing independent 8,192-row formula/non-policy check is
`sf_close_run02/optional_sharpened_ties_data_qualification.json`, SHA256
`48897206e4af6d37bb4fa351cebe1e1d4b493e07b63126fd8a65332bd85af8c1`.
It is a bounded payload check, not a full-corpus payload identity proof. No new
teacher inference, rematerialization or target change is selected.

Train one seed-zero full game epoch from scratch using the same frozen trainer,
architecture, optimizer and non-policy targets as C. Runtime is wise-cloud
revision `7ec261509fb7345cf1ca0ad73809193fc2749bb1`, Python 3.10.12,
Torch 2.11.0+cu128, CUDA 12.8 and NumPy 1.26.2. Use
`scripts/lc0_control_train.py`, config `configs/lc0_positive_control.yaml`,
steps 0, batch size 512, sampling mode `game_epoch`, 16 planning workers,
16 loading workers, seed 0, CUDA, train-window steps 88, and the same
`--allow-invalid-control` setting. Use a fresh output directory; no adoption,
resume, retries, step extension or checkpoint selection after looking at games.

The full epoch is 18,910,484 rows, 2,309 shards, 97,968 games and 36,935 batches.
Require the final checkpoint, all planned/realized batches and rows, zero
same-game repeats, zero nonfinite skipped gradients, zero CUDA retries and finite
window loss/gradient summaries. Preserve intermediate artifacts as evidence, but
they are not candidates for this screen.

The completed prospective metadata/schedule proof is
`sharpened_ties_run01/prospective_schedule.json`, SHA256
`1aca8a7dfd7e2917927560c9eea285fe26e3db591ff8ebdd2cfa00dfba08cfd8`.
It matches the source and completed C ordered game metadata and batch/load order.
Canonical schedule SHA256:
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Prospective E0T05 physical schedule SHA256:
`c81bd16b2589c31c2090501312c67ac313315866a12892e092d2460494776103`.
After training, use the pinned verifier on actual staged shards and the completed
summary. Require source/C canonical equivalence and equality of saved, recomputed
physical and realized schedule hashes before evaluating. This is game/row-order
evidence under the pinned planner, not feature/target byte identity. Do not rerun
the successful prospective scan without changed inputs or a specific concern.

## Direct screen and decision

Evaluate only the completed E0T05 checkpoint directly against C20T05:
`runs/armB/qtemp_0.0005_hist_20m_bt4_sfclose_C20T05_epoch_v1/checkpoint.pt`,
SHA256 `8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3`.
Bank and pin the new candidate checkpoint hash from its completed training receipt.

Run one fixed match: 1,000 games / 500 color-swapped opening pairs, 100 matched
simulations on each side, training search shape and the full qualified C100
settings. Explicit search-prior temperature 1.0 for both sides. Use the original
book seed 42, 16 opening plies, maximum 300 plies, move temperature 0.1,
128 concurrent games, evaluator batch 4096 and compilation on. Preserve the fresh
history-bearing confirmation FEN bank for later. No rolling decisions, SPRT,
optional stopping, automatic retries or horizon extension.

The primary measure is E0T05's nominal paired 95% Elo interval against C, using
the existing registered pentanomial estimator with the opening pair as the unit.
An interval wholly above zero favors E0T05; wholly below zero favors C; an
interval crossing zero is inconclusive and does not establish equivalence.
Require 1,000 valid canonical rows, 500 complete pairs, no orphan rows and exact
roles, checkpoint/book identities, execution and realized search settings.
Independent result review must precede interpretation and the next selection.

## Budget and recovery

Expected cost is approximately 2.6–2.7 GPU hours for training plus 0.65–0.7 for
the arena. Hard GPU-stage budget is 6 hours: 16,200 seconds (4.5 hours) for
training and 5,400 seconds (1.5 hours) for the arena, each including a 30-second
TERM-to-KILL allowance. Charge actual stage duration. Queueing and CPU-only
schedule verification do not consume this GPU budget. Bound the latter to
30 minutes; a timeout or incomplete epoch/bank is unfinished evidence, not a loss.

Use the shared GPU lease with an independently surviving timeout supervisor and
owned process-group cleanup. Release the lease for post-training CPU schedule
verification and reacquire it for the arena so the raw BT4 labeler can progress.
Preserve live SF generators, raw BT4 driver and archive queue. Use nice 19,
low I/O priority and two numerical threads. Restrict schedule checks to CPU0,1;
do not impose that affinity on the 16-worker training run. Maintain the existing
150 GiB disk reserve. Respect stop markers and preserve partial artifacts and
failure receipts. Inspect any interrupted attempt explicitly before recovery.

## Limits and follow-up

This is a nominal development screen using an existing seed-zero C checkpoint
and reused development openings. It is not fresh recipe confirmation, a training
seed-variance estimate, a search-scaling slope, an RL result or 100M transfer
validation. Preserve historical control limitations: no held-out purity receipt,
architecture/trainer checked against a committed pin instead of the live config,
and game-epoch sampling differs from the old replacement-sampled control.
Do not relabel `valid_control: false` as a valid historical control.

After this challenger screen, select the best-supported recipe and strongest
relevant alternative for fresh matched-seed training and the reserved fresh
opening bank, including deployment-relevant search. If the screen is inconclusive,
choose the subsequent confirmation comparison from that uncertainty rather than
claiming equality or silently adding games. The larger-corpus phase/observation
selection contract and eventual RL benefit remain separate work in the broader
100M bootstrap program.

The reviewed launch manifest supplies exact new output paths, launcher hashes and
runtime/provenance pins. Preserve this prospective snapshot once execution starts.

## Prepared execution

The prospective snapshot is `scratchpad/bt4_joint20/sharpened_ties_run01/preregistration.md`.
The matching `launch_manifest.json` pins the corpus, completed C reference,
prospective schedule, frozen runtime and both launchers. Inspect the plan with:

```bash
python scripts/bt4_one_epoch_screen.py --manifest /home/josh/projects/chess/scratchpad/bt4_joint20/sharpened_ties_run01/launch_manifest.json
```

After separate launch review, `--execute` runs the single epoch and qualified
fixed match. Launch the coordinator at low priority without a CPU affinity clamp;
the CPU-only schedule subprocess applies its own two-core cap. New outputs are
`sharpened_ties_run01/experiment` and
`runs/armB/qtemp_0.0005_hist_20m_bt4_toptie_t050_epoch_v1`. An interrupted attempt
preserves these outputs and requires explicit recovery; the tool has no resume mode.
