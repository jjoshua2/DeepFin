# Fresh confirmation: SF-close versus sharpened stored ties

Registered September 7, 2026 UTC, after independent review of the
[completed development screen](2026-09-06-bt4-sharpened-ties-screen.md) and before
fresh training or outcomes. **Completed September 7: the fresh match confirms
C20T05 over E0T05 at 100 simulations.** E-minus-C is −37.67 Elo
[−56.85, −18.72], from 1,000 games / 500 opening pairs. Both models were
trained from scratch with seed one; no production recipe was changed.

## Selection and hypothesis

C20T05 is the provisional leader and reference; E0T05 is the candidate. The
hypothesis is that C's advantage persists with both models trained from scratch
at a fresh paired seed and evaluated on fresh openings. The expected direction
is therefore a negative E-minus-C interval.

The prewritten selection heuristic chooses E because its completed development
score against C was 0.4575 versus G20T05's 0.457. The 0.0005 difference selects an
opponent; it does not establish E-versus-G superiority. E lost its direct screen
against C by −29.60 Elo [−48.72, −10.66].

Both target recipes use sharpened BT4 at temperature 0.5. C redistributes existing
mass over all stored SF maxima plus d9 top-three moves within 20 effective
centipawns; E redistributes only within stored maximal-target ties. Those ties can
reflect quantization or saturation, not underlying SF-score equality. Compare the
two complete recipes without changing their published corpora.

## Fixed comparison and deciding rule

Train the reference, then candidate, from scratch with seed **1**, using the
frozen runtime, architecture, optimizer and full game-epoch schedule: batch 512,
16 planning and 16 loading workers, 88-step windows. Require finite losses and
gradient norms, zero nonfinite skips/CUDA retries, complete planned and realized
rows/batches, matched canonical schedules and the exact final checkpoints.
Derive counts from this seed's plan; do not reuse seed-zero batch counts or select
intermediate checkpoints.

Run **1,000 games / 500 color-swapped pairs at 100 simulations per side**, arena
seed **20260907**, prior temperature **1.0** on both sides, using the full
qualified C100 search dictionaries. Maximum 300 plies, move temperature 0.1,
128 concurrent games, evaluator batch 4096 and compilation remain fixed. Use the
reserved 500 history-bearing openings, each containing 16 legal opening moves.
Final overlap qualification covers all 14 completed development banks and finds
zero terminal-opening overlap with the reserved bank; this is not training-data
purity evidence.

The deciding statistic is the opening-pair sample-variance normal 95% score
interval transformed to Elo, reported as **E minus C**. Require exactly 1,000
canonical games, 500 complete pairs, zero orphans and matched identities/settings:

- Entirely below zero: confirms C's expected direction in this fresh match.
- Entirely above zero: contradicts the decisive development finding and favors E.
- Crossing zero: unresolved; it does not establish equivalence.

Independently review the completed bank and provenance before concluding. No
rolling outcome inspection, early stopping, extra games, seeds, retries, grid or
checkpoint selection follows automatically from any result.

## Budget, recovery and evidence

The hard GPU cap is **37,800 seconds / 10.5 hours**: 16,200 seconds for each
training stage and 5,400 for the arena, including 30 seconds of termination
allowance per stage. Expected cost is about six GPU hours. CPU schedule checking
is capped at 1,800 seconds and readout at 120; neither CPU stages nor lease waits
consume GPU charges.

Use the qualified [confirmation launcher](../bt4_confirmation.md), frozen at
revision `12f26da49f83225086e237ad0257494346abdb05`, with the shared GPU lease,
independent timeout supervisors, two numerical threads, low scheduling/I/O
priority and 150 GiB disk reserve. Release the lease for CPU-only stages. Preserve
live generators and archival work. All outputs are new; preserve partial state
and diagnose interruptions before separate recovery. Do not adopt seed-zero
checkpoints or overwrite existing attempts. Merges do not update the frozen runtime.

The original prospective registration and manifest were frozen at
`scratchpad/bt4_joint20/confirmation_registration_v1/`. Byte-identical snapshots
are now published in the evidence catalog. This record summarizes their unchanged
scientific semantics. Artifact paths below are relative to
`scratchpad/bt4_joint20/`:

| Artifact | SHA256 |
| --- | --- |
| `confirmation_registration_v1/preregistration.md` | `f89a5a719aeb6c189f1ecf1d85b73815bd0bfbc1be145c5ebe4e4d25113cf528` |
| `confirmation_registration_v1/independent_launch_review.json` | `c07453d71ca01daf7dd45b93fc5d9fd0161a5f1ac5f50a25918559bd5f2b6d88` |
| `confirmation_registration_v1/launch.json` | `345ac427b8559107bbb24a091974fdb621f340841a9fb2e2d6eeded5e91d7a68` |
| `confirmation_registration_v1/manifest.json` | `33d4845ddcc131522e572c42b2ff6338850c4f5b4500295bf9f413fe073011a9` |
| `confirmation_openings_v1/openings.fen` | `e0d13b2ea70c0ac278570a0e463c3c1c3030a18256522bcba864db23cdc07c98` |
| `confirmation_openings_v1/completed_bank_overlap_final_E0T05.json` | `85e88d4803e04a57fa8887c66c4171b91b2bf59753421aaf7a85add31076b0c1` |

A result supports a comparison of these tested 20M bootstrap recipes at this
search budget. One fresh paired training seed cannot estimate seed variance or
establish universal optimality, a search-scaling slope, 100M transfer or RL benefit.
Historical control/purity limitations remain explicit; larger-corpus observation
selection and eventual RL evaluation are separate questions.

## Completed fresh training and schedule verification

Both seed-one epochs completed from scratch with their registered corpora. Each
realized **18,910,484 rows, 36,935 batches and 420 training windows**, including
the final 63-step window. Every window had finite loss and mean gradient norm;
both runs reported zero nonfinite gradient skips and CUDA retry batches. Final
checkpoints were selected by the registered rule, without selecting an
intermediate checkpoint.

| Stage | Charged GPU seconds | Final checkpoint SHA256 |
| --- | ---: | --- |
| Reference C20T05 | 9,789.061950 | `ef42d11276529178480a9d255315ee4bf07b2eb39a8516ecb0a7799d03392745` |
| Candidate E0T05 | 9,284.944868 | `bc9f029ef1874ae0aa613def4295331e1a894d1a23e12fb466ee3e9ef6bae740` |

The two epochs consumed **5.2983 GPU hours**. The CPU schedule verifier completed
in 460.10 seconds, below its 1,800-second cap. Both actual staged runs matched
the source metadata and canonical schedule
`4d75a181dea1e94e592f2bc78036e3b233452a3f85b8a1d6024e69fbb731510e`.
Each physical-path schedule also matched its own completed training summary.
The proof uses identical ordered game columns, planner inputs, pinned scheduler
and NumPy 1.26.2; row-offset equality is a code-backed inference. This check does
not independently establish feature/target payload equality or held-out purity.

Independent review passed both completed epochs and the actual matched-schedule
proof. Both summaries retain `valid_control: false`: they lack a held-out purity
receipt, judge architecture/trainer premises against committed pins, and use a
game-epoch sampler different from the historical replacement-sampled control.
These are the registered limitations of this paired recipe comparison.

The fresh arena launched after the launcher’s training and schedule checks. Its actual
command, working directory, live-config path and first metadata header matched
the registered settings, including both full qualified search dictionaries.
The launch observation read no game outcomes. This section establishes training
and launch provenance; it is not a playing-strength result.

Paths in the following table are relative to `scratchpad/bt4_joint20/`:

| Artifact | SHA256 |
| --- | --- |
| Reference run `summary.json` under `runs/armB/confirmation_seed1_E0T05_vs_C20T05_reference_C20T05_v1/` (repository-relative) | `034f463d59f2b0a7263a29daf5a8c957e674d217a4a124c041753586f0fe56ae` |
| Candidate run `summary.json` under `runs/armB/confirmation_seed1_E0T05_vs_C20T05_candidate_E0T05_v1/` (repository-relative) | `c28d2067501b263368cf2699b5b6ec4ff5df01ed6afbedd37bc4422942c9a662` |
| `confirmation_seed1_E0T05_vs_C20T05_v1/matched_schedule.json` | `6b4281eb68cf07404205166402011930497e251b6d69c3caa8a6d2651b0da334` |
| `confirmation_registration_v1/independent_fresh_training_schedule_review.json` | `c4190f1bbfe34aa321664656ab10d3c7708232b84058b4a6580628ffa405e380` |
| `confirmation_registration_v1/parent_arena_start_observation.json` | `7f8d16cbccd1b3d5659329a55c7c17a56b3885926b401602743f6617065af5ac` |

## Completed match and decision

The fixed match completed without a restart, truncation or orphan pair. E0T05
scored **350 wins, 192 draws and 458 losses**, or **44.60%**. Its opening-pair
sample variance is 0.0955250501002004; the registered normal 95% score interval is
[41.8909%, 47.3091%]. Transforming its endpoints gives **E minus C = −37.67 Elo
[−56.85, −18.72]**. The pentanomial counts, in LL / LD / DD-or-WL / WD / WW
order from E's perspective, are **102 / 86 / 185 / 72 / 55**.

The whole interval is below zero, so the prospective rule **confirms C's expected
direction in this fresh match**. Equivalently, C scores 55.40% and leads by
+37.67 Elo [18.72, 56.85]. This corroborates the seed-zero direct screen;
it does not pool the two seeds into an estimate of training-seed variance.
C20T05 is the selected baseline for the next bootstrap research stage.

The arena charged 2,163.371250 GPU seconds. Total fresh-confirmation compute was
**21,237.378068 seconds / 5.8993 GPU hours**, within the 10.5-hour cap and each
stage cap. The runtime remained frozen throughout. The parent independently
recomputed W/D/L, all color-swapped pairs, the interval and completed-file hashes.
The independent completed-bank review also passed: it reconstructed every result,
replayed all 500 opening histories and verified settings, identities, receipts and
closed budgets. Its receipt is
`confirmation_registration_v1/independent_completed_confirmation_review.json`
under `scratchpad/bt4_joint20/`, SHA256
`cab8e9ccbf5d379d6dc03c9fa5c0502d2b865b50a649d5bf6c3f8093661505e0`.
The readout's `training_provenance_verified: false` describes the standalone
reader's scope: training, schedule and launch evidence are separate receipts,
reviewed alongside the bank; it is not a new failed training check.

The next useful questions concern the larger corpus's SF observation selection
and the separate graded-objective experiment in
[PR #517](https://github.com/jjoshua2/DeepFin/pull/517), including interactions
with SF-only and BT4-mixed targets. Do not launch a larger teacher-dose grid simply
because prepared corpora exist. A fresh 400-simulation comparison could test the recipe at a higher search
budget if that decision requires it; this 100-simulation
confirmation does not answer that question. No additional arena or training job
was automatically added after seeing this result.

The [published evidence catalog](evidence/bt4-bootstrap/README.md) includes the
completed game bank, reserved opening histories, readout and completion/provenance
receipts, so the paired result can be independently recomputed from GitHub.
Checkpoints and training corpora remain external, identified by hashes and paths.

| Artifact under `scratchpad/bt4_joint20/confirmation_seed1_E0T05_vs_C20T05_v1/` | SHA256 |
| --- | --- |
| `arena.games.jsonl` | `88d7c7a915ca6c9889f55237462862bacf0c6baae6373eca67d668276a3a69f6` |
| `readout.json` | `3719bd291ca7a7a2625eca2075eac1eb2b5cd1906e978bf6fc7a59fe2be355f7` |
| `complete.json` | `82121c284aac074b466f1d5a58a3e440fd672311d8558e81db9c0df77f07cdff` |
| `arena.gpu-charge.json` | `99dae086f46afe8ea6d5b5f9591d090c4e183b117c053624b4d1071a2ce85eb8` |
