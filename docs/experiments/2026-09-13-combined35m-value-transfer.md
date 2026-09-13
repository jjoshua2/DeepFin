# Matched value transfer on original plus G10 data

Status: prospective admission, the fixed opening panel and the frozen selected-subset
preflight are complete. Neither training arm nor evaluation has launched. Actual
staging and realized training remain to be verified.

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
its source/book and endpoint/history contract with the package. The panel is now frozen before either training launch; its actual identity and
construction evidence appear below.

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

## Prepared admission implementation

`scripts/combined_corpus_schedule.py` adds an explicitly pinned, ordered corpus-set
manifest and optional prospective schedule pass. Existing single-source tools and
the historical sampler are unchanged. The manifest maps `(cohort, shard_index)`
to the original source, B100 and V50 paths; unqualified basenames cannot collide.
It checks the two fixed recipes and original SF lineage, qualified raw-roster
disjointness, root aliases, actual shard inventory and source/qualification pins.
The original corpus uses its existing qualification; G10 uses the existing
metadata identity assessment with its historical source-code caveat, without
requiring a new per-shard generation attestation.

The prospective pass imports the hash-pinned historical sampler, witnesses its
small game-ID/presence-column reads once, and compares full ordered columns and
canonical game grouping. Physical schedules use the manifest seed. The source
is a proof reference: emitted training directories include **B100 and V50 only**.
The report binds the manifest, logical map, ordered game-column hashes, actual
runtime versions, seed/batch and per-arm physical/canonical schedule identities.
It is explicitly prospective, not a completed training receipt.

At implementation registration, the prepared local manifest (then not admitted or executed) was:
`scratchpad/bt4_joint20/combined35m_value_scale_feasibility_v1/admission_preparation_v1/manifest.draft.json`,
SHA256 `18830f7e403035cb8b0a2303d5c3877a792e7f3d28e894a2d1bd0680a49fcec6`.
It names 21 source/target groups, 4,321 shards per arm and 35,314,577 rows.

After immutable code review, the proposed one-pass invocation is:

```text
python scripts/combined_corpus_schedule.py \
  --manifest /absolute/path/manifest.draft.json \
  --expected-manifest-sha256 18830f7e403035cb8b0a2303d5c3877a792e7f3d28e894a2d1bd0680a49fcec6 \
  --output /fresh/path/prospective.json --execute --deadline-unix <launch+2970>
```

Use the qualified historical interpreter, GPU hidden, CPUs 4 and 5, two numeric threads,
nice 19 and 2,970 seconds to TERM plus 30 seconds for KILL
cleanup within the registered 3,000 seconds. Require 32 GiB available memory and
150 GiB disk reserve. Resource checks occur during admission/scan callbacks; they
are not a separate watchdog inside a long planner call. No address-space cap is proposed: the qualified Torch import uses large virtual
mappings. This is not an RSS-cap claim; the outer timeout remains mandatory. Do not run a redundant default pass first.

Focused fixture tests cover CLI admission at seed 101, duplicate-basename mapping,
swapped/missing cohort order, game-key mismatch, overlapping raw selection,
modified values, wrong dose/head/parent, extra shards, input pin drift, fresh
output and single-read metadata witnessing with loader restoration on failure.
No real corpus admission, array scan, training or evaluation ran during that implementation validation; the subsequent bounded admission is recorded below.

The implemented training-only adapter consumes the actual admission report,
retains frozen trainer history/value compatibility gates, and requires actual staging
and realized completion against this new logical map. It uses the new canonical
identity and the B100 policy roots for SF100; training execution remains pending.

Final validation: 13 focused fixture tests passed. Scoped host type checking passed
with zero findings; final-file Ruff and Vulture passed. The whole type gate reported
236 missing-native-module source findings in this unbuilt worktree, with none in
the new files. Native modules were not built or changed to clear that environment
limitation. A separate 0.10-second summary-only check confirmed that all 21 actual
source/B100/V50 recipes fit the implemented contract; this did not walk shards,
run admission or decode arrays.

## Frozen opening panel — completed before training

The registered seed **20260913** produced **256 unique opening endpoints / 512
color-swapped games**, stored as root FEN, sixteen legal moves and replayed final
FEN. The complete saved panel SHA-256 is
`14470ee9bcf5fdfc822bb19988941ecf4bbe2a1ea739d46487d3663b1735340c`.
One small-panel review replayed every history, verified valid nonterminal endpoints
and exact final FENs, and found **zero normalized endpoint EPD overlaps and zero
exact root/history overlaps** with the previous 128-opening panel `3c955d68…`.
This establishes disjointness only from that panel, not all training/development
positions or an independent opening distribution. No overlap filtering or reseeding
was used.

The first preparation attempt failed after **1.583 seconds** because importing
the whole arena module imported unused Torch, whose library mapping exceeded the
4 GiB address-space limit. No panel or model was built; the compressed-book digest
had already been checked. The preserved correction executed the exact untouched
`load_paired_openings` AST function from the pinned arena source, with the exact
opening module loaded directly to avoid package initialization. It retained the
same book, frequency-weighted PGN prefixes, NumPy 1.26.2 RNG, seed and history rules.
It did not change the sampler or raise the memory cap.

The corrected run completed with exit 0 in **71.872 seconds**, maximum RSS
**498,344 KiB**, and no Torch import. Its 298-second bound plus the failed attempt
fit the original 300-second allocation; CPUs 4–5, two numeric threads, hidden GPU,
4 GiB address space, 48 GiB startup headroom and 150 GiB disk reserve were retained.
[Compact panel evidence](evidence/combined35m-fixed-panel-20260913.json) pins the
failed attempt, exact invocation/source/book, independent preparation review and
completed panel. Review did not reread the book, rerun generation or load models.
The later actual match package must reproduce this full panel and the registered
256-pair/seed20260913 settings; the old 128-pair helper constants are not sufficient.


## Completed prospective admission and selected-subset amendment

The one CPU admission pass completed with exit 0 in **382.722 seconds**. It
admitted the ordered **21-cohort, 4,321-shard, 35,314,577-row** source/B100/V50
mapping and computed each prospective schedule at seed 101 and batch 512. Each
contains **182,188 namespaced games and 68,974 updates**: 68,863 full batches and
111 batches of 511 rows. Full ordered game-ID/presence witnesses and canonical
game grouping matched across the three arms. The canonical source identity is
`ca3922c459b321dd5e890f8b3e21f6fee6a24aa6c7f0d76653165616aeca67bd`;
physical hashes differ because the target directories differ. The report SHA-256
is `00c28c92a530eb56d5051bb3606ecff6516eb580f94325c8da7fd16ccd692bb9`.

This is a prospective identity proof, not realized training. It inherits the
qualified generator/whole-game contract and disjoint raw selections without
inventing historical per-shard source-code attestation. No feature/target arrays
or models were loaded, and the review did not repeat the game-column pass.
The source arm is only a proof reference; SF100 training uses B100 policy roots.
The invocation retained the registered 3,000-second inclusive CPU bound, two
threads, hidden GPU, 32 GiB available-memory and 150 GiB disk floors. Memory
checks are callbacks, not an independent watchdog inside a planner or an RSS cap.

The frozen trainer's partial-corpus gate identified a separate, explicit launch
condition: the selected G10 derivations are finished products of global raw runs
that remain unfinished. Before training, register the existing
`--allow-partial-corpus` opt-in **only for the exact complete selected union**.
Require every selected G10 shard's `derive_run_finalized` to be true and require
the original corpus to be nonpartial. No attributes are rewritten; unfinished
selected derivations remain ineligible. Leak, history, architecture, replay and
value gates remain unchanged, with no `--allow-leak` or mixed-history override.

A separate **1,200-second CPU qualification allowance** covers both arms
sequentially through the unchanged frozen preflight, including its two label
presence columns and exact selected-shard metadata. Its command uses CPUs 6–7,
two threads, hidden GPU, 32 GiB host headroom and 150 GiB disk reserve; no
address-space cap is claimed. The later training receipt must reproduce the
qualified partial-corpus record exactly. This allowance is additional to the
completed 3,000-second prospective allocation; training/evaluation budgets and
targets are unchanged. The actual qualification result is recorded below.

For the eventual arena, **20260913 is both the opening-generation seed and arena
seed**. The existing book path, sixteen-ply sampler and fresh seeded RNG regenerate
the immutable panel before model loading; this is not a switch to a panel-supplied
arena method. Completed training, actual staged identity and the fixed match
package remain required. No combined training or strength result is claimed.


The corrected selected-subset preflight completed with exit 0 in **88.668 seconds**
(87.675 seconds inside the helper), with maximum RSS **1,045,604 KiB** and zero
swaps. Both arms covered all **4,321 selected shards**. Each has **2,012 G10
partial-source stamps**, all with finalized derivations; the original corpus is
not partial. The active `search_wdl` column covers **35,314,577/35,314,577 rows**;
the unused optional `sf_wdl` label field covers zero rows. The config uses
`sf_wdl_frac=0` and `search_wdl_frac=1`: `search_wdl` is SF-derived in SF100
and mixed in V50, so absent `sf_wdl` labels do not mean absent SF value. The unchanged frozen
architecture, trainer, replay, history and value preflights passed.

The preserved first attempt failed after **65.820 seconds**: the frozen B100
preflight had passed, but the new wrapper incorrectly required both label-presence fields to
have full coverage. Its correction requires exactly the registered optional-zero
and active-full counts; it does not weaken the trainer's gates or change data.
Three small metadata fixtures checked that distinction. The retry kept a fresh
output namespace and a **1,134-second inclusive remaining cap**, so its maximum
plus the first attempt's rounded-up 65.821-second charge remained below the
original 1,200-second allocation. Actual combined elapsed time was **154.488
seconds**. No failed-attempt partial record was reconstructed or adopted.

The actual qualification SHA-256 is
`fb72304dccd0cb277d3fcc80b57287964723760fcca5ddcd0bbc6b917039d8b7`.
The [compact evidence](evidence/combined35m-fixed-panel-20260913.json) binds both
attempts, the correction, independent reviews and completed admission/panel.
These are completed preparation checks; neither combined model has been trained
in this record, and no strength result is claimed.
