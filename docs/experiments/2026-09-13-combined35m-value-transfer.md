# Matched value transfer on original plus G10 data

Status: preregistered; multi-corpus admission implementation is in progress. Neither
training arm nor evaluation has launched. Existing target products are complete;
their combined training compatibility and schedule are still to be qualified.

## Question and allocation

Does replacing half the SF value distribution with native BT4 WDL improve a
bootstrap trained on the larger corpus we can already assemble? Keep B100 policy
(BT4 teacher temperature 0.5) identical in both arms. This tests the value mix at
an increased, shared data/training scale, rather than refining another tactical
policy threshold after the unresolved Downside comparison.

The existing inventory provides **35,314,577 accepted rows**: **18,910,484** original
rows plus **16,404,093** G10 rows in 20 qualified cohorts. Both the B100/original-SF
and corresponding SF50/native-BT450 products already exist. Use those products;
no fresh labels, duplicate epochs disguised as new rows, or copied mega-corpus.
The inventory's distinctness refers to source observations, not unique chess
positions. Target completion alone does not establish combined training admission.

| Arm | Main value supervision | Policy |
| --- | --- | --- |
| SF100 | Existing original SF distribution | Existing B100 T0.5 |
| V50 | Existing normalized 50% SF / 50% native BT4 WDL | Same B100 T0.5 |

Other input and target arrays, including game-result targets, remain matched.
The existing value writer's arithmetic and float16 storage semantics stay fixed;
this is not a new calibration or head architecture. Retain the exact existing
teacher/model and source qualifications in the combined manifest.

## Training and identity

Train both fresh models with initialization and schedule seed **101**, the same
frozen historical trainer/optimizer/window behavior and one complete game epoch
at batch size **512**. Use two planner and two loader workers. Run the arms
sequentially, SF100 then V50. Do not continue old checkpoints, reset a partially
finished arm silently, change the learning-rate schedule, or adopt main's newer
sampler during this comparison.

An ordered manifest maps each original/G10 logical cohort and shard to both
physical arm roots. Basenames alone are ambiguous across cohorts. Corresponding
inputs, game identities and unchanged targets must match before deriving one
new canonical combined schedule; the old 18.91M canonical hash is not applicable.
The prospective planner determines actual updates/windows. The arithmetic
estimate is 68,974 updates; it is not a fabricated completion receipt.

Existing G10 metadata gives 1,990 disjoint raw shards. The qualified generator
rotates only at game boundaries and assigns unique game IDs within each raw run;
under that contract, current cohort-parent grouping does not split games across
these selected cohorts. Preserve separate run namespaces. The original corpus
is a third source namespace. Historical manifests lack per-shard generator source
SHAs; record that limitation rather than inventing an attestation. No sampler
change or repeat corpus census is warranted by the current evidence. A failed
actual history/value/encoding compatibility gate stops admission; it is not
permission to weaken the gate.

## Resources and recovery

The recent 18.91M epoch took 7,832 seconds. Linear extrapolation gives about
**4.06 GPU-hours per arm**, or **8.13 hours** for both, excluding admission and
evaluation. This is a planning estimate, not a measured 35.3M runtime.

CPU-only admission/prospective work has a **3,000-second inclusive cap**, two
numeric threads and a separately reviewed bounded-memory command. Reuse saved
proofs where sufficient; read required identity metadata once, not feature/value
payloads repeatedly. Each training stage has **21,600 seconds**, with **27,000
seconds inclusive** for its coordinator and cleanup. Require **48 GiB available
host RAM at startup**, **32 GiB while running**, and **150 GiB SSD free**. Keep
Stockfish at its reduced 2+1 execution concurrency. Use the shared GPU lease;
Ceres collection must finish or stop cleanly before training takes it.

The larger game/row plan can use more memory even with two loaders. Sampled
headroom is a stop guard, not a guarantee against every transient allocation.
Retain owned-process cleanup and all failure receipts. Preserve partial artifacts
and resume only through a qualified recovery path; no automatic whole-arm retry,
budget extension or changes to host WSL configuration.

## Deciding evaluation

After both complete and their actual schedules/receipts qualify, compare V50
(candidate) directly with SF100 at **400 simulations**, **512 games / 256
color-swapped opening pairs**, both search-prior temperatures **1.0**. Use the
existing qualified opening-generation procedure with new panel seed **20260913**;
freeze its actual complete panel/hash before either training launch. Register
its source/book and endpoint/history contract with the package. Panel identity
is a prelaunch artifact still to be produced, not a hash asserted here.

Keep the qualified training search shape, no tablebases, 300-ply limit, rolling
128 concurrent games and inference batch cap 4,096. Evaluation gets **7,200
seconds per owned stage**, **12,030 seconds inclusive** of lease wait/cleanup.
There is one depth and no temperature sweep. Evaluate after all pairs finish,
without interim result-driven stopping or extension.

Use mean candidate score over opening pairs and its nominal paired 95% interval,
with descriptive Elo. An interval wholly above 0.5 supports V50 at this corpus,
seed and search setting; wholly below favors SF100; crossing remains unresolved.
A positive estimate alone does not promote the recipe. No automatic additional
bank, seed or dose follows. A supported improvement motivates confirmation nearer
the intended final bootstrap scale; a clear loss favors SF100 and a distinct
teacher/value alternative; an unresolved outcome keeps both viable while later
allocation uses effect size, cost and remaining hypotheses.

## Scope of the conclusion

This is a matched value-treatment comparison within a shared combined corpus.
Against historical 18.91M results, data amount, G10 distribution, update count and
seed all change. It does not isolate the effect of more rows, establish search
scaling from one depth, cover training-seed variance, or establish a 100M/RL
result. A fresh panel reduces direct reuse but does not make a single seed a
replicated finding. Existing historical-control limitations remain explicit.

Ceres coverage proceeds separately using otherwise available GPU time. It is not
silently mixed into V50 or used to choose a checkpoint from this comparison.
