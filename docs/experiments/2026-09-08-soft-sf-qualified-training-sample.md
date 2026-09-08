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
entropy match. Full-corpus preparation, matched training and gameplay evaluation
remain separate decisions. Temperature matching is neither evidence of better
labels nor evidence of a stronger or more search-scalable model.
