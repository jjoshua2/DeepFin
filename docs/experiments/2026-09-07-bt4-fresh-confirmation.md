# Fresh confirmation: SF-close versus sharpened stored ties

Registered September 7, 2026 UTC, after independent review of the
[completed development screen](2026-09-06-bt4-sharpened-ties-screen.md) and before
fresh training or outcomes. **Coordinator launched after independent review; outcomes unread.** At the
launch observation it was queued for the shared GPU lease held by the raw BT4
labeler; no fresh trainer child had started. This is a launch snapshot, not a
claim that training has completed or a current process-status monitor.

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

The authoritative prospective registration and manifest remain outside git at
`scratchpad/bt4_joint20/confirmation_registration_v1/`. This record summarizes
their unchanged scientific semantics. Artifact paths below are relative to
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
