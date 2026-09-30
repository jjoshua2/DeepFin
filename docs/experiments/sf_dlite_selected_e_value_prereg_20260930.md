# D-lite Selected-E main-value screen: preregistration

Status: frozen before the 2,048-row transport pilot; pilot labels are unadmitted and there is no full target, training, or Elo credit. Freeze
the exact census roster, d8 labels, runtime, pack adapter, and opening panel
before either GPU arm. This screen asks whether a fixed shallow Stockfish scalar
helps an already trained network when the chosen Selected-E policy stays fixed;
it does not recreate historical D or predict the best 500M-row data mix.


The evidence paths in this document are relative to the local chess artifact
root. The prelaunch proof bundle is
`operations/sf-dlite-legacy-screen-prereg-20260930/MANIFEST.json` (SHA-256
`210534bcc3de673b9566b3d9169589bd2a4b92a44290258eba678292ee2860fd`).
It seals the packet/operator/launcher sources and selection proofs before any
full label campaign. This preregistration is an experiment declaration, not an
admission of the historical overlay into current production training. After
publication, changes to the hypothesis, sample, or decision rule require a
dated appended amendment.

| Frozen evidence | SHA-256 |
|---|---|
| Census terminal | `f7880a99dc09a43519aeea884beccc22cc2c0c373cf2574df6e5dd21e611d85b` |
| 2,500,000-row selected roster | `8766e5745a6f36e4ecbed90cc861d012e29f2d21356958fa1477bead4dcc7941` |
| Independent selected-roster audit | `159648aa60fa3cdc4f11e79c0ff629a47d2a538316cc8011bee5aa1030865d6a` |
| Selected-E qualification | `2f985f12410e82d6ecc5f4d18012d0f3d030fad475d1d6be8d1c54c13f40222d` |
| Depth-8 label packet / operator / launcher | `186abebb31fa9f3800cb3cb9ecc6a58dcc13c73b1d207a330ea78064adc8f282` / `52d97829d2a40219dcf50a5ecd6735494be74ad4bd0ca34a75f0b3f03e1bea1d` / `bd5e943f3a46be0a13eec2402aa75780e1412e02e43a4fbc34adaf34bf781ca8` |
| Pilot selection / selector manifest / independent selection audit | `c441fea802a04e6c340fceb7ace6e7bcda1f81d9dcacfb67c2d59786218b181d` / `aa70a90dd315631f5a22d5401cd884499aba5939a874d59664e68a39a599337b` / `eecc85b0b490ff5c3bbbb5f3a3d0c5c68711642584dd510a2bc7f00b13ab4283` |
| Deep-audit selection / source joins / independent audit | `7108922e3ea3d3901f753d48496f9023db49a1c4d677e4af4edd14761708a51d` / `1869d022d3fd38d3d483c4e788c1af325e223d92b5477ed56091a0f2cd4d1c22` / `c2591cd26dd95e5cb6a3640eeba75b01be97ffabea61ea4f64a3132807ffcf9a` |
| Game-ID injectivity proof / target scheme | `2b10f425d98d5f788ec8b06f359094c8cc655749e24d4ef641b9f2b318422b88` / `716ef4da2c928a437ca001f7ea769c8e555091221cb22ca7ee884e5deeaefdfa` |

The deep selection uses within-cohort/material-bin medians. The original
cohort-wide-median availability check had only 9 and 7 distinct games in the
high-material late cells; both availability receipts are retained in the proof
bundle. All 16 final cells have 64 distinct games and 64 distinct physical
source files, with 1,024 games globally distinct. This change was based only
on roster availability, before viewing any Stockfish score. The independent
selection audit reproduced the exact ordered 1,024 indices and joins.

## Inputs and direct target gate

Use exactly the 2,500,000 unique stored inputs selected by the reviewed G10
cohort25/27 census, after an independent all-selected-row source/overlay audit.
The audit should stream the selected roster by source file, verify each selected
physical raw row and full history, legal mask, stored input, chosen route,
policy bytes and control WDL, and match the roster hash. It need not revalidate
every unselected saved row; the original census already has separate receipts
for those. Preserve source-qualified game keys
`(source_namespace, worker_id, game_id)` and the exact game-epoch schedule.

Use the reviewed, root-authorized d8 packet
`operations/sf-dlite-legacy-screen-prereg-20260930/label_packet_v6.json` (SHA-256
`186abebb31fa9f3800cb3cb9ecc6a58dcc13c73b1d207a330ea78064adc8f282`)
only after its
transport pilot and operator have been reviewed. On every selected row the
control uses the saved Selected-E neural FP16 `search_wdl`; the candidate uses
one FP16 rounding of `(calibrated fixed-d8 SF + 2 * widened control WDL) / 3`.
All 1,858 policy entries, legal masks, input planes, and every auxiliary and
nonmain target remain byte-identical. Bank the unrounded SF CP/mate/native WDL,
PV, nodes, history and exact source/roster join. D-lite is a main-value-only
intervention. Existing auxiliary arrays and masks remain unchanged in both
arms; the outcome and SF auxiliary loss fractions are zero.

Before GPU training, freeze an outcome-blind, source/game-stratified 1,024-row
subset of the selected roster for a direct deep-SF value-target audit: 64 rows
per cohort × four piece-count bins (7–10, 11–16, 17–24, 25–32) × two
game-ply bins (below versus at/above the median computed separately inside
that cohort and piece bin). This availability-only change was frozen before
any D-lite labels or deep-SF scores: a cohort-wide median leaves the 25–32
late cell with only 9 source-qualified games in cohort 25 and 7 in cohort 27;
the within-bin split has at least 4,716 distinct games in every cell. Preserve
both availability receipts and the exact median/counts.

For each of the 16 cells, first keep one representative per source-qualified
`(source_namespace, worker_id, game_id)`: the row with the smallest binary
stored-input SHA-256, ties by source ID, physical source row, saved shard ID,
and stored row. Rank these games by SHA-256 of the fixed domain
`sf-dlite/deep-audit-game/v1\0`, the 32-byte source namespace, and little-endian
signed 32-bit worker ID and signed 64-bit game ID; ties use the original triple.
Traverse cells in cohort, piece-bin, then below/above order, taking the first
64 games absent from earlier cells and one physical source file per cell.
This yields 1,024 rows with 1,024 distinct source-qualified games and 64
distinct source files per cell, or HOLD if a cell cannot fill. Store the full
source/roster joins and selection hash before any deep search, D-lite result,
or training. Do not use label values or outcomes to choose rows.

The project's existing `data/audit_set_v1.jsonl` is pinned at SHA-256
`d8e26efa0b010450abf9374693afc45027db6d146571785ab897af5061144df2`;
it has FEN-level rows without this selected-roster source/history identity and
therefore cannot simply be substituted. The prior G10 first-four value bank
SHA-256 `09d8e43676ec9e0c91218081a41da07d86583ef71411b3b112a329c820adef54`
also concerns a different saved slice; reuse any exact matching row only after
proving source-qualified identity and history-window equality. Otherwise bank
new fixed depth-12 scalar scores on the frozen sample using the same full UCI
history, strict six-man profile, CP/mate/native WDL/raw lines and source pins.
If the depth-12 exact score is absent, record the predeclared inclusion reason
and do not impute. HOLD if source replay fails or more than 5% of audit rows
lack the exact ruler; do not silently resample.

Compare control and candidate Brier and ECE to the same deep-SF WDL ruler,
with mate/inclusion reasons, per-stratum results, and source-qualified game
cluster uncertainty; compare to completed source-game outcomes separately when
available. `docs/eval_protocol.md` requires this direct target gate before
training. Because the ruler is SF-derived and shares a CP-to-WDL map, SF
agreement is a diagnostic and cannot establish outcome calibration or strength.
No Brier/ECE sign alone kills or promotes D-lite here; source validity and
complete audit reporting are the gate. Do not fit temperature or change the
candidate formula after seeing this bank.

## Paired adaptation

Primary initialization: the completed historical Selected-E seed-121
`checkpoint.pt` at
`runs/factorial58_selected_e_seed121_run01/checkpoint.pt` under the artifact root,
expected SHA-256
`154e768ce1a76dda98b0077244227df1591ef3d4a4e2e438e16c49b1a58ac203`.
Recheck architecture, tensor identity, optimizer and scheduler compatibility
before accepting it. Both new arms independently load this same checkpoint,
including the same optimizer, scheduler, step and ZClip states, then apply
exactly the same declared LR/schedule treatment. Hash and compare those states
and the initial model before the first optimizer step. The old checkpoint by
itself is not the control result; both control and D-lite are newly adapted.
`Trainer.load()` may perform name-based optimizer remapping or fall back to fresh
moments: compare the loaded optimizer slots to the donor checkpoint and HOLD on
any partial or silent reset. For this first short screen, inherit the donor
optimizer moments, scheduler state and step identically in both arms; no
unregistered LR override is allowed. A change to a fresh schedule would be a
new preregistered pair, not a continuation of these arms.

Train each arm on one identical frozen 2.5M-row `game_epoch` at batch 512 and
seed 121, with identical model, runtime, optimizer, LR profile, mirror setting,
row order, per-step batches, masks and loss fractions. Outcome and SF auxiliary
fractions are zero; the active main WDL uses the stored control or D-lite vector.
Read back every consumed row/target schedule and verify that only the declared
main WDL bytes differ. Training is serial on the one GPU. Freeze the exact
trainer command and full source/config hashes after the paired native-Zarr or qualified
short-roster adapter passes; the current pre-roster overlay helper only admits
the older frozen-E chain and must not be weakened to admit Selected-E.
For this local 2.5M-row diagnostic, paired archives are native compressed
`shard_XXXXXX.zarr` directories, which the historical reader and
`game_epoch` path already enumerate. This does not qualify a new 500M-row or
external packed loader.

The historical `lc0_control_train` game-epoch path has the right one-row-per-game
semantics and 88-step windows but no plain `--init-checkpoint`; the fixed-epoch
runner can load the donor checkpoint but its sampling order is a different
route. A minimal reviewed adapter must join the donor checkpoint to the
historical game-epoch route and certify the exact consumed row/target schedule
for both arms. Neither existing CLI alone is qualified for this paired screen.
The existing drivers do not persist a verified exact cursor/RNG state for this
new short route. Use an external 55-minute wall cap per arm, measure the real
wall on a bounded warmup, and restart a crashed short arm from the common
initial checkpoint in a fresh output directory. No partial arm is accepted.
If either complete arm cannot fit under one hour, HOLD this short screen and
add independently verified exact model/optimizer/scheduler/RNG/data-cursor
resume before extending it. Preserve raw logs and each checkpoint hash.

## Strength decision

Use the fixed 576 opening pairs (1,152 color-swapped games) at 400 MCTS
simulations per side with strict six-man WDL+DTZ rule50 tablebases. Verify
opening/source purity, model identity and exact equal search settings. Complete
all pairs, save paired raw outcomes and per-game logs, and do not early-stop.
Treat one color-swapped opening pair as the smallest committed arena unit: write
the two raw game logs, fsync them, then atomically publish a source/model/search-
bound pair receipt. Resume by verifying and reusing sealed pairs and replaying
only an incomplete pair. Bound any unsealed group to 30 minutes and retain
the fixed opening order and deterministic seeds across restarts. A full arena
terminal requires all 576 verified pair receipts.
The primary D-lite score is the mean of 576 opening-pair means, each averaging
its two color-swapped D-lite scores. Compute a normal 95% score-space interval
`mean ± 1.96 * sampleSD(pair means) / sqrt(576)` using n−1 variance; transform
interior endpoints to Elo with `400 log10(s/(1−s))`.

With clean source, target, training and arena gates, a lower score endpoint
strictly above 0.5 is positive one-seed/panel evidence; an upper endpoint
strictly below 0.5 is negative one-seed/panel evidence; otherwise unresolved.
Incomplete games, zero-width/undefined interval, changed arena settings or
failed identity/tablebase checks are HOLD. The direct audit's Brier/ECE does not
replace this playing-strength result. A close one-panel result needs another
seed or disjoint panel before any claim about a much larger corpus.
