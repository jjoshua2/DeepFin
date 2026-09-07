# G10 transfer readiness and target alignment

September 7, 2026. This preparation work supports the larger-data stage of the
[adaptive bootstrap research](2026-09-07-bt4-hybrid-endpoints.md). It does not choose
another recipe before H20's completed comparison or establish playing strength.

## Available data

A receipt/metadata inventory observed from 22:07 to 22:10 UTC found the following
closed physical rows. These are source-qualified row occurrences, not a count of
globally deduplicated positions or an atomic snapshot of the still-running generators.

| Source | Closed raw rows | Receipt-backed BT4 rows | BT4 backlog |
| --- | ---: | ---: | ---: |
| run06 G10 | 24,476,040 | 20,186,464 | 4,289,576 |
| run07 G10 companion4 | 6,155,799 | 4,738,228 | 1,417,571 |
| Total | 30,631,839 | 24,924,692 | 5,707,147 |

The 3,003 completed BT4 shard receipts matched source metadata, sidecar attributes
and array layouts. Active labeling advances these intended G10 sources. Complete
all-row SF observation coverage has not been qualified. **The qualified new G10
matched-recipe training corpus count is still zero**: derivation, ranking and
raw-to-derived row joins remain to be completed. Merely reaching 35M or 100M labeled
rows does not resolve those semantic requirements.

## Bounded alignment diagnostic

The preregistered sample used the first 64 physical rows of one closed shard from
each source: 128 rows across three source-qualified games. The same saved raw bank
was retained when correcting the estimator. This is a construction diagnostic,
not a random sample, prevalence estimate, held-out test or strength comparison.

All 128 sampled rows had complete legal-move d9 observations in phase zero. The
legacy deriver can instead select later-phase d9 observations. Switching between
those choices changed centipawn scores on 126 rows, stored SF maximum sets on
40 rows, and consistently constructed near-close candidate sets on 53 rows.
For 36 rows, the phase-zero rank-one move was outside the legacy stored maximum.
Thus mixing a phase-zero rank sidecar with the legacy target can violate the
intended shared SF observation semantics.

Value selection is a separate intervention. Counterfactual phase-zero searched WDL
differed on 114 rows, with maximum component difference 0.139746. Both actual
writer diagnostics retained the current latest-phase searched value unchanged;
these observations do not show which value choice plays better.

The input join also needs more than a board or game/ply key. All 128 full-history
input keys changed when the current writer quantized float32 inputs to float16,
even though every quantization-aware writer comparison passed. There were 35 bare
game/ply key collisions across the two sources. Recomputing an original BT4 input
key from stored float16 inputs therefore cannot establish the intended join.

## Preparation decision

Expose policy/rank observation selection independently from value observation
selection, retaining historical defaults. For the G10 transfer pilot, align policy
and near-close ranks on explicitly selected complete phase-zero d9 observations.
Keep the existing searched-value convention identical across recipe arms unless a
separate value intervention is deliberately registered.

Carry source-qualified raw row references and original full-history keys through
the exact filtering, grouping and shuffle used by derivation. Verify the stored
quantized input separately, then publish a source-bound derived BT4 sidecar.
Use compact indexed provenance at 100M scale rather than repeating full paths and
configuration metadata for every row. The implementation and small real pipeline
pilot must qualify this contract before any full G10 derivation or recipe training.
H20 continues to use its already frozen, single-phase development corpus.

## Registered bounded pipeline pilot

[Machine-readable registration](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_pipeline_pilot_v1/preregistration.json)
and the two original closed receipt snapshots are included in the evidence manifest.
Registered before operational derivation or adaptation. This tests whether the
new tools can produce aligned inputs for a later recipe comparison. It does not
train a network or choose the next target family.

Use the complete first closed `w00-00000.jsonl.zst` shard from each source, selected
by filename before inspecting further rows. The existing closed receipts identify
8,236 run06 rows and 8,243 run07 rows (16,479 physical rows total). Keep the sources
in separate derived directories. Freeze a copy of each selected completed BT4
receipt and record the source manifest, raw-shard, teacher, implementation and
command identities before execution. Source hashes are:

- run06: `3c93d2d127bc6d11db9eb31425e0c795c96b417106d737632ef4b689569ae17d`
- run07: `a499d2921043065b388778435efa2e2e63d3e206ab1310264b75dafa39d9dea0`

Derive once per source with `uniform-d9`, policy observation `phase0`, value
observation `latest-phase`, value scheme `search`, temperature `0.0005`, floor `0`,
seed `0`, 8,192 rows per output shard, one worker and `--row-provenance`.
`--limit` is the selected source's complete physical row count, including any
rows dropped by derivation. Before launch, verify that this is exactly the first
shard in the tool's resolved input inventory. Do not substitute a different shard
if a qualification fails.

On those exact surviving rows, adapt the existing raw BT4 labels and construct
phase-zero d9 ranks with rank cap three. Use the ordinary mixer admission path to
materialize C20T05 and its H20T05 hybrid, with identical non-policy arrays. These
small recipe artifacts are pipeline fixtures, not additional training arms.
Reuse compatible frozen descriptive audit receipts; if admission requires a new
receipt for the registered context, reanalyze the same saved observations and
identify it as reanalysis, without rerunning a teacher or interpreting its score
as new evidence.

Success requires complete source-qualified row joins, original and quantized
history-key checks, legal rank/target coverage, ordinary sidecar admission,
unchanged non-policy arrays, and identical source-qualified game/ply schedules
between the two recipes. Report all actual exclusions and counts. Every emitted
row must meet these invariants; a partial output or metadata-only match fails.
This qualifies only the selected inputs and paths, not all G10 rows or 100M-scale
throughput. The existing loader may combine the separate source parents; do not
flatten colliding game IDs into one physical source directory.

Use a frozen reviewed checkout with the qualified CPU development interpreter,
GPU hidden, at most two numerical threads and two CPU cores, nice 19 and idle I/O
priority. Stages run sequentially with a **30-minute total wall-clock cap**,
including termination grace, **8 GiB of new output/cache allowance**, and the
existing **150 GiB free-space reserve**. The adapter's identity-cache allowance
is 16 MiB for these two raw shards. Preserve failures and logs, stop dependent
stages on failure, and do not automatically retry or expand the input. Active H20
preparation and GPU labeling retain priority. After this pilot, decide whether a
larger preparation batch is justified from its actual failures, throughput and
coverage; a full transfer training is a separate result-dependent selection.

## Evidence

The [publication manifest](evidence/bt4-bootstrap/g10-readiness-manifest.json) binds
the original inventory, 3,003-shard membership record, 128 raw rows, exact diagnostic
arrays and row-level joins/readouts, estimator corrections and producing script.
[Readiness receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_readiness_v1/readiness.json)
and [diagnostic/recommendation](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/G10_readiness_v1/sample_diagnostic_v1/diagnostic_and_recommendation.json)
retain complete hashes, observed settings and limitations. Host commands and PIDs
are archival provenance. Bulk source corpora and temporary writer directories remain
host-local; this evidence publication is not a portable launch bundle.

[Tool validation and independent reviews](evidence/bt4-bootstrap/g10-tools-validation-manifest.json)
bind the final derivation, adapter and rank changes. The affected derivation suite
passed 317 cases, adapter tests passed eight, rank tests passed 13, and the combined
repository lint passed with no findings. Independent checks covered actual mixer
admission, shuffled source joins and rejection of changes during publication.
These checks qualify the tools on their exercised paths; the registered real-data
pilot remains unlaunched.
