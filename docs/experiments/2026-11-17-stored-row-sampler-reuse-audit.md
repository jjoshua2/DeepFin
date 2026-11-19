# D1/D2 stored-row sampler reuse audit — completed read-only census

The accepted historical journals show replacement sampling, not complete passes
or unique chess positions. This audit measures each stored identity
`(generation, source_sha256, row_index)` before augmentation. Original and mirrored
views share that row's quota; the journals do not record which orientation was
chosen. Accepted learner observers previously joined draw counts to applied B32
views. Naturally duplicate board/opening contents remain distinct stored rows;
semantic uniqueness is unknown and semantic deduplication is not required.

One bounded CPU owner streamed two representative immutable journals once each:
S0 for the short cohort and K1 for the long cohort. Accepted boundary/observer
proofs establish byte-equal draw journals within S0/S1 and within D1
K0/K001/K01/K1, respectively. Thus the table describes each individual run in its
cohort, not pooled exposure summed over six models.

| Per-run cohort | Batches / draws | Distinct stored rows drawn | Zero-draw rows | Maximum draws of one row | Rows drawn >2 times | Excess draws beyond two per row |
|---|---:|---:|---:|---:|---:|---:|
| Short S0/S1 | 512 / 16,384 | 10,678 | 6,890 | 8 | 1,160 | 1,461 |
| Long K0/K001/K01/K1 | 4,096 / 131,072 | 17,562 | 6 | 20 | 17,210 | 96,016 |

Each population contains 17,568 stored rows from 192 source memberships. “Excess”
is the sum of `max(draw_count - 2, 0)`, not the number of rows above two.
The census read 17,082,977 raw journal bytes plus 1,562,219 bounded metadata bytes.
Only the admitted read-only CPU owner accessed journals; it imported no Torch or
NumPy, made no CUDA query, and loaded no models or games and performed no training.
Its actual owner-wall charge was 1.2553332509705797 CPU seconds. All427 prior
owners were closed with three leases free at the accepted cutoff.

[Compact accepted audit evidence](evidence/2026-11-17-stored-row-sampler-reuse-audit.json)
pins the actual readout, Root and independent acceptances, closed custody and
histograms. Counts identify historical reuse and coverage gaps. They do not prove
that reuse caused overfitting or the [fresh confirmation's non-confirmation](2026-11-17-s0-a39-fresh-confirmation.md#november-17-completed-fresh-confirmation-nonconfirm--inconclusive),
or identify a value-head cause, optimum λ, recipe or external strength gain.

Keep all completed D1/D2 results, models, failed attempts and actual costs.
Historical runs are not relabeled cap-compliant and the old population is not
called unseen. The user's current cap is at most **two total deliberate
presentations per stored row**, counting original and mirror together; it is not
two of each orientation. Proposed exact-once sampling uses one total presentation.
For the illustrative 17,568-row / B32 population, that is 549 full updates; no bank
or training recipe is selected by this arithmetic.

The exact-once source package and review (`160d49e7…` / `3e7d15b2…`, PR1378) are
software preparation only. Corrected-head hosted checks and adoption were pending
when this record was prepared. The source enforces a fixed lineage's one-pass,
no-rewind ledger; controlled prior-use accounting must also prevent a new UUID or
new ledger from resetting the claimed quota. Future training remains held until
the selected runtime, sampler manifest and actual prior-use/cap enforcement are
validated. This audit authorizes no training, inference, HOLD access, retry or
additional match.
