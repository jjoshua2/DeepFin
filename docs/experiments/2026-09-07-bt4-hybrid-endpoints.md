# Adaptive bootstrap exploration: H20 first

Registered September 7, 2026, after the [fresh C/E confirmation](2026-09-07-bt4-fresh-confirmation.md)
completed and before training or playing outcomes for the new arms. **Status:
H20/C100 completed at +45.07 Elo [27.00, 63.39] and H20/G100 at +85.78 Elo [67.25, 104.80]. The registered C400 probe has launched; its outcomes remain unread. The full package is incomplete.**
The previous screen is complete. This registration selects H20 as the next bounded
comparison. Subsequent tests are chosen from completed results, not a mandatory
three-arm or 168-hour queue. Routine choices within the research goal do not need
individual user approval. Production adoption remains separate.

## Completed H20/G100 comparison — September 8

H20 scored **62.10%** against G20T05 at matched 100 simulations: **493 wins,
256 draws and 251 losses**, giving **+85.78 Elo [67.25, 104.80]**. All 1,000 games
and 500 color-reversed opening pairs completed, with zero orphan rows. The nominal
95% interval uses opening-pair mean scores and the same registered Elo transform
as the primary C100 screen. This conditional comparison favors H20 over G20T05
for the registered selected-move-set construction at a shared 20% global BT4
dose, conditional on the previously verified outside-set target equality within
storage tolerance.

The [independent review](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/g100_increment_publication_v1/H20_G100/independent_review.json)
recomputed the completed bank and checked both checkpoint identities, complete
realized search dictionaries, book, paired openings, execution settings and
training/schedule provenance. Prior temperature was 1.0 for both networks.
The [completion receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/g100_increment_publication_v1/H20_G100/complete.json)
records **2,595.36 seconds / 0.7209 charged GPU hours**, ending at **07:13:10 UTC**,
within the 90-minute cap. Three games reached the registered 300-ply limit; none
was cut short by the wall-time cap. The earlier process receipt retains `complete:false`
from before scientific qualification; `complete.json` separately records the
subsequent successful qualification.

The [readout](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/g100_increment_publication_v1/H20_G100/readout.json)
and [losslessly compressed game bank](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/g100_increment_publication_v1/H20_G100/arena.games.jsonl.gz)
are published with original and compressed hashes in the
[evidence manifest](evidence/bt4-bootstrap/h20-g100-increment-launch-manifest.json).
These results concern one seed-zero epoch on reused development openings. They
do not measure seed variation, remove adaptive-selection uncertainty, establish
search scaling or authorize production promotion. The historical-control
limitations below remain unchanged.

The [C400 launch snapshot](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/g100_increment_publication_v1/C20T05.s400.process.snapshot.json)
records the next registered probe starting at **07:18:51 UTC**: H20 versus C20T05,
400 simulations, 500 games / 250 paired openings, prior temperature 1.0, and the
same 90-minute cap. Arena PID 316336 is owned by timeout 316335 and coordinator
288450. This snapshot establishes launch only; no C400 games, results or rolling
logs were read for this publication. The completed C100 and G100 cells favor H20
at the development budget; the registered C400 cell remains necessary before a
verdict on the full package.

## Completed H20/C100 primary screen — September 8

H20 scored **56.45%** against C20T05 at matched 100 simulations: **448 wins,
233 draws and 319 losses**, giving **+45.07 Elo [27.00, 63.39]**. All 1,000 games
and 500 color-reversed opening pairs completed, with zero orphan rows. The
nominal 95% interval uses the standard error of opening-pair mean scores followed
by the Elo transform. The estimate exceeds the registered +15 Elo threshold and
its lower bound exceeds zero: **promising for this primary cell**.

The [independent review](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/c100_increment_publication_v1/H20_C100/independent_review.json)
recomputed the raw bank and checked the complete search dictionaries, checkpoint
identities, paired openings and qualified training/schedule provenance. Both sides
used prior temperature 1.0. The arena completed at 06:24:22 UTC in **2,410.62 seconds
/ 0.6696 charged GPU hours**, within its 90-minute cap. Two games reached the
registered 300-ply limit; neither was truncated by the wall-time cap.
The [readout](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/c100_increment_publication_v1/H20_C100/readout.json)
and [losslessly compressed game bank](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/c100_increment_publication_v1/H20_C100/arena.games.jsonl.gz)
are published with the original and compressed hashes in the
[evidence manifest](evidence/bt4-bootstrap/h20-c100-g10-increment-manifest.json).

This is one seed-zero epoch on reused development openings. The interval measures
match sampling, not training-seed variation or adaptive-selection uncertainty.
The recorded historical-control purity/config/sampler limitations below remain.
It does not establish search scaling, a universal recipe winner or production
promotion, and it selected no new training arm. At this C100 snapshot, H20/G100 and H20/C400
were still needed to complete the package; the later G100 result above supersedes
that status. The
[next-arena launch snapshot](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/c100_increment_publication_v1/G20T05.s100.process.snapshot.json)
records H20/G20T05 at 100 simulations starting at **06:29:55 UTC**; its outcomes
were not read for this record. The optimized labeler's
[first completed group](2026-09-07-bt4-label-legal-reuse.md#first-optimized-group-completed--september-8)
is a separate preparation milestone.

## Qualified training completed — September 8

The [completed process receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/training.process.json)
records **9,851.44 seconds / 2.7365 charged GPU hours**, within the registered
4.5-hour training cap. The trainer started at 02:53:54 UTC and exited zero at
05:38:06 UTC. Its [complete trainer summary](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/training.summary.json.gz)
records all **18,910,484 rows**, **36,935 steps/loss calls** and **420 windows**
with finite losses and mean gradient norms, zero nonfinite skips and zero transient
CUDA retries. Sampling seed zero, batch size 512 and window size 88 are unchanged.
Planned and realized row/batch counts and the internal sampling hash agree.

The final checkpoint is `checkpoint.pt`, SHA256
`0a711fcf10ff87fc8360d3fd4b3035b170a7c15172616317c4a9687ae99d7017`.
This identity is recorded in the trainer summary; this documentation pass did not
rehash the checkpoint or rescan the corpus. The mid-run checkpoint was not selected.

The separate CPU [schedule stage](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/schedule.process.complete.json)
then exited zero in **303.22 seconds**, within its 30-minute cap. The
[realized report](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/realized_schedule.json.gz)
verifies actual staging and matching source metadata. Its canonical hash is
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`, matching
the registered source/C schedule; physical order hash
`7349d7287be32b5ec0119c45ace5ef4e3e9762c52675401a942fdf6347983192` matches
the completed trainer summary. The resulting
[qualified receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/training.complete.json)
now records `complete:true` and binds the final checkpoint, summary and schedule.

The verifier does not decode feature/target payloads; their preservation relies
on the separately qualified mixer provenance. Its plan hashes cover game choices
and shard paths, not within-game row offsets; row-order equivalence remains a
code-backed inference. The earlier
[05:41:29 pending snapshot](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/schedule.process.snapshot.json)
is preserved unchanged, as is the training process receipt's pre-qualification
`complete:false` flag. The later qualified receipt supplies the completed status.

The summary retains `valid_control:false`: no held-out purity receipt, the
architecture/trainer premise checked against a committed configuration rather
than the live file, and the game-epoch sampler's intentional departure from the
historical replacement-sampled control. This is the registered development arm,
not a newly validated historical control or evidence of strength. The existing
H20/C100, H20/G100 and H20/C400 comparisons remain the deciding package; no new
training recipe is selected here.

The [first-arena process snapshot](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/C20T05.s100.process.snapshot.json)
records the registered H20/C20T05 comparison starting at **05:44:12 UTC**:
1,000 games, 100 simulations per side and prior temperature 1.0 on both sides,
with arena PID 304943 under timeout 304942 and coordinator 288450. The
[B100 handoff receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/B100.handoff.json)
records the matching live first-arena identities at **05:44:24 UTC**, when waiter
PID 301740 transitioned to the unchanged reviewed CPU materialization supervisor.
The [initial materialization status](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/B100.materialization.status.snapshot.json)
records `RUNNING`. The [05:46:08 host observation](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/training_completion_publication_v1/transition_observation.json)
confirms mixer PID 304973 on CPUs 0–1 at nice 19 beneath the reviewed materializer.
This establishes actual startup, not a completed B100 corpus. The separate four-hour
CPU allowance and existing resource guards apply. The same observation confirms
the qualified optimized labeler adopted its runtime and was waiting for the GPU;
it establishes no new optimized inference or throughput result. No game scores
were read in that launch snapshot; B100 training remains unselected.

The [completion evidence manifest](evidence/bt4-bootstrap/h20-training-completion-manifest.json)
binds exact process, summary, qualified completion, schedule and handoff bytes, retaining
the earlier pending-stage snapshot. The summary and schedule are losslessly
compressed with both original and compressed hashes retained. Earlier recovery,
failed receipts, qualification and training-start snapshots below remain unchanged.

## Recovery completed and training started — September 8

The separate recovery completed in **200.46 seconds**, processing the remaining
17 shards / 134,420 rows and publishing all **18,910,484 rows / 2,309 shards**.
The original timeout receipt remains failed; it has not been rewritten as success.
The validated 2,292-shard prefix was reused through the reviewed process proof,
without another prefix payload scan. Full empirical aggregates remain unavailable;
the analytic mass-error bound is 0.000545, below `2^-10`. The suffix's 133,623 changed
rows and measured mass drift are a non-inertness witness, not population estimates.
See the [completed recovery readout](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20_recovery_v1/completed_readout.json).

The first post-recovery qualifier rejected four intentional native-library
symlinks before producing a qualification receipt. A narrowly reviewed v2 accepts
only those exact runtime aliases, checking canonical targets, content hashes and
link/target stability; corpus inputs still require regular files. No binary or
corpus payload changed. The [correction record](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20_recovery_v1/qualification_alias_correction.json)
retains the failed execution and original qualifier bytes. The corrected qualifier
passed in **8.15 seconds**, binding the final summaries, transitive C/SF/BT4/rank
lineage, complete validation stamps and stable metadata.

The [dataset qualification](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20.dataset_qualification.json)
and [prospective schedule](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20.prospective_schedule.json.gz)
passed before training. H20 and the completed C anchor share canonical schedule
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`:
18,910,484 rows, 97,968 games, 36,935 batches, seed zero and batch 512. H20's schedule
was prospective at launch; the later training-completion snapshot above records
the completed realized checks.

The coordinator launched at 02:48:37 UTC; its surviving timeout supervisor started
the actual trainer at **02:53:54 UTC**. The [process snapshot](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/completed_readiness_publication_v1/H20.training.process.snapshot.json),
captured at **02:56:44 UTC**, records coordinator 288450, timeout 288561 and trainer
288562, with training incomplete. It binds the original Python 3.10 / Torch 2.11
CUDA runtime, seed-zero epoch and 4.5-hour training cap. That historical receipt establishes the actual start; the later completion
snapshot above supersedes its training status, without adding an arena result. The registered H20/C100, H20/G100 and
H20/C400 package remains unchanged; no follow-up candidate was added.

The [completed-readiness manifest](evidence/bt4-bootstrap/completed-readiness-manifest.json)
publishes completion, correction, qualification, schedule, independent reviews and
launch evidence alongside the separately completed G10 common-input preparation.
Large text records use lossless gzip with original and compressed hashes. Earlier
failure and preregistration files retain their original bytes and dated statuses.

## B100 readiness while H20 trains — September 8

The distinct pure-BT4 policy endpoint now has its own completed cached audit and
reviewed bulk-preparation plan. At **2026-09-08 04:44:37 UTC**, the first-arena
handoff was armed and waiting (PID 301740). **Bulk materialization had not launched,
and B100 had not been selected for training.** H20's complete registered comparison
package remains the next deciding evidence.

The [B100 audit](../../scratchpad/bt4_joint20/B100_preparation_v1/audit_B100T05.json.gz)
completed in **8.37 seconds** on the existing **4,000-position FEN-only BT4 cache**,
with global alpha 1 and teacher temperature 0.5. Treatment invariants passed.
Candidate-minus-original-SF expected deep-SF regret was **−0.32 cp [−2.79, +2.13]**;
top-choice regret was **−8.08 cp [−11.30, −4.93]**. These are position-bootstrap
95% intervals (10,000 replicates, seed 20260903). Expected-regret improvement is
unresolved at this precision. Neither result measures playing strength, search
scaling or a trained recipe; the cached teacher is not the history-aware teacher
used for eventual corpus materialization. No teacher inference or GPU work ran.
The legacy audit's `training_permitted` field records descriptive tool admission,
not a new training decision. The [completed receipt](../../scratchpad/bt4_joint20/B100_preparation_v1/completed_readout.json)
retains exact input pins, timing and limitations.

The [bulk plan and independent review](../../scratchpad/bt4_joint20/B100_preparation_v1/materialization/independent_review.json)
prepare the same 18,910,484 original SF/history rows and 2,309 shards. The policy
becomes legal-normalized sharpened BT4; all SF value and non-policy fields remain
from the original source. The normal mixer retains full source/history/teacher
admission and policy write/read/legal-mass checks. Final corpus qualification and
prospective schedule checks still follow any successful materialization.

The measured G50 global materialization cost, **7,382.36 seconds / 2.05 hours**,
is a planning reference rather than a B100 runtime prediction. B100's child cap
is four hours including termination grace, with two CPU cores and numerical
threads, low CPU/I/O priority, a shared preparation lock, a 150 GiB free-space
reserve and a sampled 32 GiB output allocation limit. These are sampled guards,
not disk quotas; timeout survives coordinator loss but disk/STOP polling does not.
Failures and partial outputs are retained without automatic retry. Bulk copying
and compression can contend with H20's prefetch, so launch timing is reserved for
a suitable arena-phase I/O window rather than inferred safe from hidden CUDA.

The [armed receipt](../../scratchpad/bt4_joint20/B100_preparation_v1/arena_handoff_v1/armed.json)
records a maximum four-hour wait for H20's **first `C20T05.s100` arena**. The gate
requires both original training handles absent, qualified completed training and
schedule, and a live arena whose PID identity, command, working directory and
coordinator → timeout → arena ancestry match the fixed owner and manifests.
Missing/reused owners, STOP, failure or a missed/completed first arena end the
attempt; it does not fall through to a later match. It reads no game scores and
signals no other jobs. On acceptance it replaces the waiter with the unchanged
reviewed B100 materializer, whose separate four-hour allowance and the resource
guards above then apply. This is a start-time gate, not a guarantee that the arena
will overlap the entire CPU build.

The [independent pre-arm review](../../scratchpad/bt4_joint20/B100_preparation_v1/arena_handoff_v1/independent_review.json)
passed, including a real-host refusal while the training handles remained present.
Six disposable fixtures cover acceptance, live training, wrong ancestry/command,
reused owner and STOP. The [new evidence manifest](evidence/bt4-bootstrap/b100-arena-handoff-manifest.json)
preserves all eight compact originals, including prepared plans and the later
arming receipt. It establishes an armed waiter, not a completed handoff or corpus.
After execution replaces the waiter, the materializer's own/parent STOP paths
apply; the handoff-only STOP no longer controls it.

The [readiness and handoff manifest](evidence/bt4-bootstrap/b100-readiness-labeler-handoff-manifest.json)
contains the full losslessly compressed audit, exact command/runtime bindings,
supervisor fixtures and unlaunched plan. It also links the separate labeler-runtime
qualification below; neither record selects another training experiment.

## Original timeout and recovery registration — September 8, 2026

The original mixer reached its registered GNU timeout after 14,370.09 seconds
(exit 124). Its coordinator and two child PIDs were confirmed absent; the failed
receipt was preserved and the partial retained for the separate continuation. The [timeout readout](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20_timeout_v1/readout.json)
and [all-shard metadata inventory](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20_timeout_v1/shard_metadata.jsonl)
record 2,292 contiguous fully stamped shards / 18,776,064 rows. At that timeout
snapshot, seventeen shards / 134,420 rows remained and the final corpus was not
published. This was an operational
preparation failure, with no training or playing-strength result.

The registered continuation would reuse the completed prefix and process the
remaining shards from their first chunk with the original checks. It would retain the original
failure and attach separate recovery provenance. The original validated prefix is
accepted as executed-process evidence, the same kind of evidence used by the
previously prepared normal qualifier. This is not a new independent payload
checksum or a claim of protection against later external file mutation.

The original process lost its aggregate statistics at timeout. They would remain
unavailable, including full-corpus entropy, changed-row counts and measured mass
mean/max. The remaining shards provide a measured non-inertness witness, without
being presented as a representative sample. The original final empirical
normalization gate did not run. For this fixed normalized H20 arithmetic, the
[independent recovery design review](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20_recovery_minimal_design_review.json)
derives a conservative uniform float64 → float32 → float16 mass-error bound of
0.000545 across at most 1,858 coordinates, including subnormal rounding. That is
below the existing row and mean limit of `2^-10`. It is an analytic guarantee,
not an observed statistic; the normalization limit is unchanged. Prefix source,
parent and teacher checks were not to be repeated merely to reconstruct descriptive
aggregates.

The registered continuation was limited to 30 minutes including termination grace,
with CUDA hidden, two numeric threads, low CPU/I/O priority and the existing
150 GiB free-disk reserve. It requires the failed writer to be absent and the
original preparation lock. The recipe, source rows, stored C parent, teacher,
training seed/runtime, exact epoch, canonical schedule and three playing
comparisons below remain unchanged. Qualification of the recovered publication
and matched schedule precede training. A failed continuation is preserved and
inspected; it is not automatically retried.

The recovery implementation passed eight focused recovery tests and 24 existing
normal-mixer/H20 regressions, scoped Ruff/type/dead-code checks, and an independent
correctness review. A regression forbids every prefix payload read and checks the
completed prefix remains byte-identical while the remaining shard is processed.
Malformed metadata, an unknown producing version, a live original PID, changed
recipe and damaged suffix non-policy data refuse publication. These implementation
tests preceded the separate host launch review and completed recovery recorded above; passing tests alone did not establish completion.

The [recovery evidence manifest](evidence/bt4-bootstrap/h20-recovery-manifest.json)
links exact original failure, validation logs, implementation/design reviews, CPU
runtime import, preparation registration and launch bindings. The complete prefix
metadata is losslessly gzip-compressed, with both compressed and original hashes
in that manifest. Archived operational scripts identify this host-bound attempt;
they are not portable instructions to relaunch it.

Only H20 is selected for training now. Subsequent substantive alternatives remain
adaptive choices from completed evidence. Routine trial selection within the
research goal does not require individual user approval.

## Preparation readout — September 7

The implementation merged in [PR #527](https://github.com/jjoshua2/DeepFin/pull/527)
after independent review, 69 mixer tests, 85 launcher/readout tests and whole-repository
lint. Execution is frozen at `054042f990c0e1b86787ad1366ccb8abcacc6aab`; the live
training runtime remains unchanged. The [publication manifest](evidence/bt4-bootstrap/h20-preparation-manifest.json)
links the complete audit, source identities and independent implementation/preparation
reviews. Archived host commands and PIDs are historical evidence, not instructions
to restart jobs.

The actual [descriptive audit](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/audit_H20.json)
completed in 10.40 seconds with target invariants passing. Across the existing
4,000-position bank, H20 minus original SF expected deep-SF regret was −5.10 cp
(position-bootstrap 95% interval [−6.64, −3.67]); top-choice regret changed by
−8.13 cp [−10.17, −6.28]. These diagnostics use the older FEN-only BT4 cache,
whereas materialization uses history-aware training sidecars. They do not measure
playing strength or compare H20 against C20T05. The registered arena package remains
the deciding evidence.

Corpus materialization started at 21:51:14 UTC under the reviewed CPU supervisor,
with a four-hour cap and a 150 GiB free-space reserve. It must complete its full
source/parent/target checks before corpus and prospective schedule qualification.
No H20 training or playing result is available at this snapshot.

A [readiness note](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/next_experiment_options.md)
identifies pure BT4 as the most distinct available endpoint and G50 as the easiest
prepared alternative. Longer training and raw-cp SF targets address separate major
uncertainties, but require qualified implementation changes. These remain conditional
options: no next candidate has been selected before H20's completed readout.

## Question and candidate options

C20T05 remains the incumbent. Its fresh seed-one advantage over E0T05 is
+37.67 Elo [18.72, 56.85]. C's earlier win over G20T05 compared complete recipes:
G20 both added global BT4 and weakened BT4's control within C's selected set.
It did not establish that outside coverage itself is harmful.

Let S be the legal-normalized stored SF target, B be the legal-normalized raw
BT4 policy sharpened at teacher temperature 0.5, and C be the legal-normalized
stored C20T05 target. Preserve all other training targets and fields.

| Arm | Target | Question and status |
| --- | --- | --- |
| H20 | 0.8 C + 0.2 B | Does broader BT4 help while retaining C's conditional distribution among selected moves? **Selected next.** |
| B100 | B | Does the missing BT4-only policy endpoint outperform the current mixture? Candidate follow-up. |
| G50 | 0.5 S + 0.5 B | Was the tested global dose too small? Prepared candidate follow-up. |

C's set A is the union of all stored SF maxima and d9 top-three moves within
20 effective cp. It can exceed three moves because stored maxima may reflect
quantization or saturation. C redistributes the set's entire existing mass using
BT4 T0.5, while preserving the original target outside A.

Before storage rounding, H20 preserves C's conditional distribution within A.
At the same global dose, H20 and G20 have identical outside targets, so their
comparison addresses selected-set redistribution. H20 versus C addresses the
broader change in allocation while preserving internal conditional probabilities.
These are complementary contrasts; do not subtract their Elo estimates to infer
another matchup. Outside mass changes by 0.2 times (B(outside) minus C(outside)),
which can be negative. A 20% mixture is not necessarily 20% outside probability.
Whole-row normalization of stored float16 C and subsequent storage cause small
additional deviations; qualify and report those rather than claiming byte identity.

## Preparation and training identities

Reuse the original 18,910,484-row, 2,309-shard, 97,968-game corpus and existing
raw history-aware BT4 and d9 rank sidecars. H20 must retain explicit original-SF,
C-parent, raw-BT4 and rank lineage; a generic bypass of source identity checks is
not acceptable. Its SF-agreement audit is descriptive: arithmetic, alignment,
legal mass and non-policy preservation decide preparation fidelity; SF agreement
does not replace the registered playing comparison.

The [initial H20 feasibility receipt](../../scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation/H20.feasibility.json)
(SHA256 `554fe4c09f73bce3bdb23269f96285a1c00d8b4cc4cf4eca1f5fcde431d9b6c3`)
reused 128 banked training rows from 43 games. It reconstructed stored C and G20
exactly, checked all 16 non-policy fields, and found H20/G20 outside differences
at most 0.0000610352, within the combined normalization/storage tolerance.
This is a bounded construction check, not a full-corpus or held-out strength check.
At initial registration the existing mixer rejected C as a nested source and its
ordinary global audit reconstructed S. The explicit composition/provenance support
in PR #527 and subsequently completed recovery resolve that preparation gap.

Train each selected new arm once from scratch at **seed zero**, using the qualified frozen
wise-cloud runtime `7ec261509fb7345cf1ca0ad73809193fc2749bb1`: Python 3.10.12,
Torch 2.11.0+cu128, CUDA 12.8 and NumPy 1.26.2. Retain the pinned trainer/config,
optimizer and initialization procedure, batch 512, one complete game epoch,
16 planning/loading workers and 88-step windows. Expected rows and batches are
18,910,484 and 36,935. Require finite window diagnostics, zero nonfinite skips or
CUDA retries, complete realized rows/batches, and actual source-normalized schedule
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Select the registered final checkpoint, not an intermediate winner.

Reuse qualified seed-zero anchors rather than retraining a seed-one G20 bridge:

| Anchor | Final checkpoint SHA256 |
| --- | --- |
| C20T05 | `8a355d29f7d3eee5deec4b3a16a6625d23baebe1302e3a8f2dea9136939e1db3` |
| G20T05 | `bd8c208a95247373f423be0649329e9c100db64ab1b5a5e68fd7a6aec3769a74` |

Their completed training/schedule/runtime evidence is in the
[published catalog](evidence/bt4-bootstrap/README.md). Bind new corpus summaries,
prospective schedules, actual executable/configuration hashes and reference
identities in the launch manifests before execution. Check actual staging after
training. Preserve the historical `valid_control:false` limitations: no held-out
purity receipt, committed configuration premise, and a game-epoch sampler different
from the old replacement-sampled control. These remain development comparisons.

## First comparison and reusable follow-up protocols

Train H20 and complete its three registered comparisons first. Then independently
review the result and choose the next experiment that can most change our decision.
B100 and G50 are substantive candidates, but neither is automatically queued.
Preparing reusable support for them does not commit their training compute. Existing
raw BT4 labeling retains the GPU during preparation and between experiment stages.

The first three rows below are H20's selected package. The remaining rows define
available follow-up protocols; their status stays unselected until a dated decision
records the rationale. If another comparison is more informative, register that
comparison before launch without asking for routine user approval.

| Candidate / reference | Simulations per side | Games / opening pairs | Role |
| --- | --- | --- | --- |
| H20 / C20T05 | 100 | 1,000 / 500 | Primary hybrid strength screen |
| H20 / G20T05 | 100 | 1,000 / 500 | Selected-set construction at equal global dose |
| H20 / C20T05 | 400 | 500 / 250 | Predeclared higher-search probe |
| B100 / C20T05 | 100 | 1,000 / 500 | Policy endpoint strength screen |
| B100 / C20T05 | 400 | 500 / 250 | Predeclared higher-search probe |
| G50 / C20T05 | 100 | 1,000 / 500 | Larger global-dose screen |

Run H20's higher-search probe after valid completed training even if it loses at
100 simulations. If the B100 protocol is selected later, its higher-search probe
also survives a shallow loss. G50's listed protocol is limited to 100 simulations;
it cannot dismiss that family at higher budgets. Once a comparison is launched,
finish its declared horizon unless validity, resource or operational limits require
stopping. Adapt the next experiment using completed evidence rather than changing
the current match after seeing partial results.

Use the same qualified training search dictionaries with explicit prior temperature
1.0 on both sides. Keep move temperature 0.1, maximum 300 plies, compilation,
128 concurrent games and evaluator batch 4096. Use original development book seed
42 with 16 opening plies. The 400-simulation bank uses exactly the first 250
opening pairs of its candidate's 100-simulation comparison; verify endpoint and
color identities. Neither the seed-one reserved confirmation bank nor future
confirmation seeds are development inputs here. Bank original game rows, full
settings, identities and charges. Read each fixed bank only after it completes;
no rolling outcome decisions or outcome-dependent extension.

## Readout and allocation rules

Use candidate-minus-reference Elo from the opening-pair sample-variance normal
95% score interval, transforming score endpoints to Elo. Require exactly the
registered complete pairs, with no duplicates or orphans, before reading strength.
For the three primary candidate-versus-C screens:

- Estimate at least +15 Elo and lower bound above zero: promising.
- Otherwise upper bound at least +15: still competitive at this precision.
- Upper bound below +15: deprioritize this tested recipe/budget for ordinary
  refinement. This is not equivalence or a rejection of its entire family, and
  never cancels a predeclared higher-search probe.

For H20 versus G20, a wholly positive interval favors H20, a wholly negative one
favors G20, and crossing zero is unresolved. Interpret the result as a comparison
of the specified inside-set target constructions, conditional on shared outside
targets up to verified storage tolerance.

The 400 probes have lower precision and describe performance at that budget.
Any secondary 400-minus-100 score contrast must use the same 250 opening pairs
at both budgets, with an aligned-pair bootstrap (10,000 PCG64 replicates, seed
20260903); label it exploratory. A win at 400 alone is not proof of a growing
search advantage. Keep per-arm results visible, including losses and invalid runs.
Independent completed-bank/provenance review precedes the next research choice.

## Resource envelope and continuation

H20 is expected to use approximately **4.5–5.5 GPU hours**. Its hard stage caps,
including termination allowance, are 4.5 hours for training, 1.5 hours for each
100-simulation match, and 1.5 hours for the 500-game higher-search probe: **9 hours**
total. The available B100 and G50 protocols cap at 7.5 and 6 hours respectively.
If all three were eventually selected, their expected total would be 12–14 hours
and stage caps would sum to 22.5 hours; this is a cost scenario, not a queue.

The initial planning envelope is **30 GPU hours**. Log actual expenditure and
remaining allocation; unused allowance does not justify extra arms or repetitions.
Necessary GPU qualification, if any, is logged within that envelope and limited to
0.5 hour. Before exhausting the envelope, publish what was learned, compare the
remaining opportunities and record the next bounded allocation under the standing
research authorization. This is not a per-test approval gate. Do not start a stage
without its declared remaining allowance.
CPU-only schedule verification is capped at 30 minutes per stage and releases the
GPU lease; lease waits do not count as GPU compute.

Keep the shared GPU lease, surviving timeout supervisor, owned-child cleanup,
STOP handling and durable completion/failure receipts. Preserve partial outputs
for explicit diagnosis rather than automatic retries. Use the established low
CPU/I/O priority and two numerical threads, preserve generators, and keep at least
150 GiB free. Provisionally allow 32 GiB for each new H20/B100 corpus, then check
actual growth; this is a planning allowance, not measured compression. G50 needs
no rematerialization. Account for training outputs and concurrent generation too.

Publish preparation findings, implementation PRs, launches and completed readouts
in this record and its linked compact evidence. A plan or successful target check
must not be reported as launched training. Explore substantive alternatives before fine-tuning one family. After each completed
package, record which uncertainty matters next and why the chosen test is worth its
cost. Potential directions include the global endpoint/dose, wider C support,
raw-score SF targets, training horizon, search calibration, graded objectives and
consistent G10 transfer. Evidence determines their order; prepared corpora or tools
are not assignments. Preserve unused seeds/openings for finalist confirmation.

Continue while there are plausible improvements with a favorable information or
strength gain relative to their cost. Finishing H20, publishing a PR, or exhausting
one planning block does not complete the broader goal. Report diminishing returns
only with an explicit assessment of the remaining alternatives and relevant-scale
evidence, not merely a narrow local winner. No production adoption is implied.
