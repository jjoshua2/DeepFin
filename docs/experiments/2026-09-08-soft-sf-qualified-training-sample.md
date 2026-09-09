# Soft-SF control from a qualified training sample

The target-blind sample completed: **4,096 rows across 64 derived shards, 127 raw
shards and 2,082 source-qualified games**. Among the preregistered temperatures
10/20/40/80 cp, **10 cp** is closest to actual C20T05's weighted mean entropy:
**0.477670 versus 0.493582 nats**, an absolute difference of **0.015912**. This
selects a descriptive control temperature on the sample; it does not select a
training run or establish playing strength.

[Compact readout](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/completed_readout.json)
· [independent review](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/root_completed_review.json)
· [original completion](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/bank/complete.json)
· [evidence and source hashes](evidence/bt4-bootstrap/soft-sf-qualified-sample-manifest.json).

## September 8 update: training and both arena cells complete

SoftSF10 completed the same original one-epoch protocol used for B100/G50:
**18,910,484 rows, 36,935 updates and 420 finite windows**, seed 0 and batch 512,
with no skipped updates or retries. The GPU stage charged **9,519.0066 seconds
(2.64 hours)**; the subsequent CPU realized-schedule stage took **319.2711 seconds**.
The canonical schedule identity remains
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
The final checkpoint is
`3cf979d0a2f2b4d0f8fd5a9398c36d6d7c4f39158144b475e08f68a2ac44b1bd`.

This follows [completed full-corpus materialization](2026-09-08-value-collection-and-horizon-readiness.md#softsf10-materialization-complete):
raw-effective-cp softmax at 10 cp changes policy only, preserving all 16 non-policy
columns, including SF search value and the inherited history inputs. This did not
adopt the newer two-epoch runtime. The original `historical_valid_control=false`
remains: no held-out purity receipt, architecture/trainer judged against committed
pins rather than a live config, and game-epoch sampling differs from the historical
replacement-sampled control. Schedule agreement is a code-backed ordering proof,
not an independently emitted row stream or a retroactive purity qualification.

CPU preparation for **SoftSF10 versus B100** completed in **139.5785 seconds** and
passed independent launch review. The prepared low-depth cell uses 100 simulations,
ordered SPRT 0/+15 Elo (alpha .05, beta .10), first 128 pairs then 64-pair looks,
a 500-pair cap and low-only 64-pair lookahead. The protected high-depth cell uses
400 simulations and a fixed 128-pair opening prefix. Both retain prior temperature
1 and the original qualified arena runtime. Any valid low verdict, including a
negative or inconclusive one, proceeds to the high cell; an invalid low cell stops.
The earlier frozen launch snapshot records coordinator startup at **22:38:27 UTC
on September 8**. Both arena cells have now completed successfully; their final
results follow below. That earlier snapshot remains unchanged.
The lease wait is separate from GPU charges; each arena stage has a 5,400-second
cap, within the 27,000-second package GPU cap including training.

[Training completion](../../scratchpad/bt4_joint20/publication_20260908_softsf_ceres_v1/softsf10/training.complete.json)
· [realized schedule, lossless](../../scratchpad/bt4_joint20/publication_20260908_softsf_ceres_v1/softsf10/realized_schedule.json.gz)
· [independent training and arena-readiness review](../../scratchpad/bt4_joint20/publication_20260908_softsf_ceres_v1/softsf10/independent_arena_launch_review.json)
· [coordinator start snapshot](../../scratchpad/bt4_joint20/publication_20260908_softsf_ceres_v1/softsf10/arena_operator.actual_start.json)
· [exact snapshots and hashes](evidence/bt4-bootstrap/softsf-training-ceres-gpu-manifest.json).

## Completed SoftSF10 versus B100 result

SoftSF10 loses this matched, original-corpus one-epoch comparison at both search
budgets. All scores below are **SoftSF10 minus B100**, using the same seed-zero
training schedule and the registered opening pairs.

| Search | Deciding sample | SoftSF10 W / D / L | Score | Result |
| --- | ---: | ---: | ---: | --- |
| 100 simulations | First 128 pairs / 256 games | 56 / 39 / 161 | 0.294922 | Ordered SPRT reaches H0 at the first look; LLR −4.43405 |
| 400 simulations | Fixed 128 pairs / 256 games | 50 / 44 / 162 | 0.281250 | −162.99 Elo, paired 95% interval [−205.52, −124.59] |

The low rule was H0=0 versus H1=+15, alpha=.05 and beta=.10, with looks at 128
then every 64 pairs up to 500. Its stopped Elo is −151.41, with an ordinary
interval [−197.17, −110.13]; these are **descriptive after stopping**, not a
sequentially calibrated confidence interval. H0 is not equivalence. The protected
400-simulation probe ran despite the unfavorable low result.

On the same first 128 opening pairs, the score interaction
`SoftSF10 advantage at 400 − advantage at 100` is **−0.013672**, with paired
bootstrap interval **[−0.082031, +0.058594]** (10,000 PCG64 draws, seed 20260903).
There is no clear relative recovery with more search. This aligned score contrast
retains covariance across budgets; subtracting two Elo estimates would not do so.
Two budgets do not establish an entire scaling curve.

The 64-pair lookahead admitted 384 games: 371 finished, 13 remained in flight and
616 never started. Only the 256 games belonging to the first 128 canonical opening pairs enter
the decision; 115 finished games lie outside it. Independent arrival-order replay
finds that the first declared prefix becomes complete at finished record 371.
The limit reduced speculative admission as designed, but comparison with older
matches does not isolate a speedup: checkpoints, trajectories, stopping samples
and compilation state differ.

Charged arena time was **726.26 seconds low + 1,340.47 seconds high = 34m26.73s**.
Including training, the package charged **3h13m05.74s**, below its 7.5-hour cap;
CPU preparation and lease waits are separate. Both stages exited zero and their
banks, commands, settings, checkpoint identities and runtime commit agree.

[Independent result review](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/review/receipt.json)
· [low bank, lossless](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/arena/low/arena.games.jsonl.gz)
· [high bank, lossless](../../scratchpad/bt4_joint20/publication_20260908_softsf_results_v1/arena/high/arena.games.jsonl.gz)
· [exact terminal evidence](evidence/bt4-bootstrap/softsf-results-value-priorities-manifest.json).

B100 remains the control for the next value contrast. This rejects the tested
entropy-selected cp10 policy recipe at this checkpoint and horizon, not every SF
softening temperature, SF tactical signal or teacher-specific head. No promotion
or fresh-seed confirmation follows. The [next research priorities](2026-09-08-bootstrap-next-value-and-policy-contrasts.md)
keep the fixed-policy value contrast separate from policy filtering and joint
policy/value correction.

## Sampling and observed identity

The [preregistration](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/preregistration.md)
and [exact selection plan](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/plan.json.gz)
were fixed before reading targets. The original **18,910,484-row** training corpus
has 2,309 shards. Its ordered shards were divided into 64 contiguous strata; a
seed-labelled hash chose one shard per stratum and 64 rows within each chosen
shard. No target-dependent filtering, replacement, teacher inference or new labels
were used. The [public prelaunch record](https://github.com/jjoshua2/DeepFin/pull/559#issuecomment-5584529887)
preceded execution.

This is a stratified cluster design, **not an iid or held-out sample**. The tables
use self-normalized inverse inclusion weights; rows within games and shards are
correlated. The bank retains those weights and source/game identities for later
cluster-aware summaries. It spans the corpus more broadly than the earlier
[128-row, one-shard geometry diagnostic](2026-09-08-bt4-target-geometry.md), whose
mean entropies are not substituted here. The earlier FEN-only 4K audit lacked this
training-row join.

All selected rows joined unambiguously to retained original raw observations.
Re-encoding each history validated its original `input_key` and matched stored
float16 inputs and SF policy exactly. All **16 non-policy columns** matched
between SF and C. BT4 source keys matched stored inputs, and reconstructing C from
stored SF, original BT4 and the d9 rank3/20cp rule matched actual stored C exactly.
All legal d9 scores were present, with original effective-cp mate encoding retained.
These are checks on the selected bank; they do not qualify a complete Soft-SF
corpus or a new derivation pipeline.

## Target geometry

Entropy is in nats. “Stored” means the original float32→float16 storage path,
followed by legal renormalization for the descriptive metrics. Support counts
strictly positive entries; it is not a count of equally plausible moves.

| Target | Ideal entropy | Stored entropy | Stored top-move mass | Stored support |
| --- | ---: | ---: | ---: | ---: |
| Original SF | — | 0.549470 | 0.795418 | 6.519 |
| Actual C20T05 | — | 0.493582 | 0.822702 | 6.600 |
| Raw-cp 10 | 0.477670 | 0.477670 | 0.808892 | 8.481 |
| Raw-cp 20 | 0.769158 | 0.769159 | 0.716768 | 14.395 |
| Raw-cp 40 | 1.250886 | 1.250886 | 0.580627 | 19.544 |
| Raw-cp 80 | 1.831957 | 1.831957 | 0.422043 | 22.021 |

Float16 storage barely changes these mean entropies but removes tiny tails: the
10cp candidate's mean support falls from **22.318** before storage to **8.481**.
The comparison therefore retains both forms. “Ideal” here is the numerical
float64 softmax, which can itself underflow for extreme score differences.

The independent bank check reproduced every stored row metric and all output hashes,
including the 64 array parts; the entropy-matching choice also agreed.

The raw-cp construction has different effects across score regimes:

| Rows | Count | SF entropy | C entropy | Stored 10cp entropy |
| --- | ---: | ---: | ---: | ---: |
| No mate score present | 2,967 | 0.313897 | 0.363446 | 0.546435 |
| At least one mate-encoded score | 1,129 | 1.168621 | 0.835614 | 0.296937 |

Thus 10cp softens the nonmate subset on average while sharpening the subset with
mate-encoded scores. It is not simply a uniform entropy increase. Raw mate-distance
scores remain intact rather than being replaced with an invented score map. The
383 nonmate rows whose best score lies within ±200cp are retained as a separate
stratum in the compact readout; no balanced-position population claim follows.

## Cost, evidence and next boundary

Collection took **285.73 seconds wall time**, **247.56 CPU seconds**, and **648.18 MiB
maximum RSS**, inside the 600-second cap on CPUs 6–7 with two numerical threads,
low I/O priority and GPU hidden. No model was loaded or evaluated.

The [lossless observations](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/bank/observations.json.gz),
[raw rows and histories](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/bank/raw_rows.jsonl.gz)
and [consumed-input identities](../../scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/bank/consumed_inputs.json.gz)
are published for reuse. The 64 array-bank parts and full corpora remain external;
the original completion lists their exact sizes and hashes. Historical paths in
archived originals describe provenance, not portable executable defaults.

The sample establishes a distinct, source-aligned raw-cp control and its declared
entropy match. The later full-corpus and training completion is recorded above;
the completed gameplay comparison is reported above. Temperature matching alone
was not evidence of better labels or a stronger, more search-scalable model.
