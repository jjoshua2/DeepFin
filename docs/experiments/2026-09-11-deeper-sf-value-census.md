# Deeper Stockfish value observations in G10

Status: the registered CPU diagnostic is running. No deeper-value corpus, training comparison or playing-strength result has been produced.

## Question and fixed comparison

G10 already pays for narrowed d10 searches and sometimes d12 searches, while its current value derivation uses the latest available d9 observation for each root move. The earlier readiness check examined only 128 positions from three games. This diagnostic asks whether the deeper observations are broadly usable and change the value targets enough to justify a controlled training comparison.

Reuse the historical 64 closed shards: 531,412 physical rows, including 528,545 rows with results and valid phase-zero policy structure. This is a fixed development bank, not a representative sample of the whole G10 corpus.

Compare the current composite latest-phase d9 maximum with the recorded adaptive-final maximum: d12 when the G10 gate extended, otherwise d10. Use the same side-to-move cp-to-WDL mapping, slope 0.006 and draw width 120, without fitting it. Report optimum move sets rather than treating an arbitrary tied winner as disagreement. Keep mate-band observations separate from ordinary centipawn aggregates.

The diagnostic checks selected depth blocks, completeness, unique rank/move rosters, legal membership, narrowing widths, ancestor-set membership and the recorded gate. It follows production's rank-one-minus-rank-two gate calculation. It does not independently establish that each narrowed roster equals the preceding block's top-k choice; that is a remaining production-admission check. Ambiguous current observations and forced-stop cases are reported explicitly rather than silently repaired.

The scale constraint is to make better use of already collected SF searches and neural-teacher labels. This allocation adds no search depth and no engine calls. A later comparison can test these stored SF values with BT4/Ceres mixtures on the same cohort; it must not assume that raising corpus-wide search depth is affordable at 100M–1B positions.

## Registered allocation and decision

One CPU process, two cores and two numerical threads; no GPU or new Stockfish inference. Limits are 2 GiB address space, 30 minutes including termination grace, a 512 MiB compressed-output cap checked every 2,048 rows and a 150 GiB SSD reserve. Preserve the existing generation and Ceres collection jobs. A partial or failed execution is not a completed result.

If at least 99.9% of result-and-phase-zero-valid rows pass the stated selector checks, consider broader qualification and a value-only producer. Otherwise inspect the failure reasons and define a prospective common-cohort or fallback rule first.

A mean WDL total-variation change of at least 0.005, or a centered-value change of at least 0.05 in at least 1% of eligible rows, nominates the recipe as a substantive later G10 training candidate. These are allocation thresholds, not accuracy or strength thresholds. Preserve source/config/shard/game identities and descriptive group sums; do not attach a population confidence interval to this purposive bank.

The original 18.91M B100 corpus is a different single-phase source. These observations cannot retrofit its rows. Any eventual training test must compare both value recipes on the same qualified G10 cohort, holding policy, rows, schedule and non-value targets fixed. Deeper restricted search is neither uniform full-root d12 nor a ground-truth value label.

## Evidence

Host artifacts: `scratchpad/bt4_joint20/g10_deeper_sf_value_census_v1/`.

- [Preregistration](evidence/deeper-sf-value/preregistration.json) SHA256: `0b5cbda7987a4df36abc5158e67a47a3761a7f2060cf9096de42dae78132c3bf`.
- Analysis SHA256: `035b97e8ee5d4c8e73687020a8902acfd0fd4a093b6abbd4402ef9856c4eb7e9`.
- Eleven fixture cases and plan validation passed before launch. Parent source/plan review and independent static review passed. Completed readout remains pending. [Independent review](evidence/deeper-sf-value/independent_review.json) SHA256: `ea2bad85f459414298c0ff297031f0864923b1fd01abe067b9d37f38cf9d655d`.

The compressed row bank and raw sources remain host-local. Publish the compact completion, counts and review with the readout before choosing the next experiment.
