# Global BT4 policy targets and search-budget scaling

## Decision before training

On September 5 the user explicitly reopened the six-arm near-tie design. The A-F
coordinator was stopped through its STOP marker at 15:56Z, before any A-F training
or arena. Its recorded operational `failed` status means intentional supersession,
not a negative scientific result. Preserve A's external materialization and all
teacher banks; do not train A merely because part of its preparation is paid for.
B-F and the conditional rank/window ladder are removed from the queue.
This record supersedes the execution plan in
[the near-tie record](2026-09-05-bt4-joint-targets.md).

The recovered wise-cloud ledger at lines 73243-55 excluded the literal global
80/20 mixture because expected deep-SF regret increased by 4.43 cp, despite top-1
regret improving by 5.08 cp. This was teacher-agreement evidence, not a playing loss.
The old exact-tie treatment gained about 74 Elo at 100 simulations and 77 at 25;
that establishes persistence over those budgets, not improved search scaling.
The current SF targets are very sharp distributions with ties, not literally
one-hot on every position.

Hypothesis: broader BT4 policy supervision can improve the trained network's
playing strength and its relative benefit from a larger search budget, even when
it assigns probability to moves the deep-SF ruler dislikes. BT4 is a one-network-
evaluation teacher with zero search nodes here; this does not reproduce training
on LC0's search-visit distributions. Teacher ranking quality and smoothing are
plausible mechanisms, not established explanations.

## Frozen data and treatments

Use the same 18,910,484 positions / 2,309 shards and full legal BT4 sidecar
specified in the near-tie record. Let S be the stored SF target normalized over
legal moves, and B_T the full legal BT4 distribution normalized after raising
its probabilities to power 1/T. Do not truncate B_T to SF ranks or gaps.

| Treatment | Target | Role |
| --- | --- | --- |
| S0 | Unchanged stored SF target | Fresh matched control |
| G20T1 | 0.8 S + 0.2 B_1 | Literal broad mixture |
| G20T05 | 0.8 S + 0.2 B_0.5 | Teacher sharpening at fixed dose |
| E0 | Existing completed exact-tie treatment | Practical incumbent benchmark |

S0 and both global arms train from scratch for one exact epoch with seed0,
batch512, 36,935 realized batches, game_epoch sampling, planner/loader workers
16/16, window88, the pinned wise-cloud trainer/config/native extensions and
Python3.10.12/Torch2.11.0+cu128. Values, features, legal masks, priorities and
game/ply identities remain unchanged. Require complete planned/realized schedules
with no same-game repetition. Source path namespaces differ; compare the underlying
source-qualified game/row order after mapping the copied directory back to its
original, rather than requiring equal path-containing plan hashes across arms.
The same config and seed use the same initialization path. Preserve realized
initialization evidence where the existing trainer records it; do not claim a
measured initial-weight hash if it was not captured.

S0 removes the replacement-sampler confound of the older SF checkpoint. E0's
historical Torch version is not independently stamped, so its comparison retains
that provenance limit. Newly trained S0 and global treatments share a currently
verified runtime. Do not migrate that runtime during this comparison.

Audit and target materialization use the main-based toolchain's locked Python3.13
CPU environment, including its matching native extensions. Both global treatments
use that same toolchain. Training and playing remain on the frozen Python3.10 CUDA
environment; the development environment is not substituted into those jobs.

## Integrity and descriptive audit

Use existing frozen teacher observations without repeating inference. Bank SF
regret, entropy, top-1 changes, effective support and tail mass as descriptive
quantities. SF-regret degradation cannot veto these playing-strength treatments.
Use the mixer's explicit descriptive audit mode with a hashed immutable copy of
this preregistration. Keep treatment identity, finite/legal normalized targets,
source/sidecar row fingerprints, unchanged non-policy arrays and float16 mass
accuracy mandatory. For global mixtures the conserved mass is all legal moves;
preserving the old top-tie subset's mass would defeat the intervention.

## Fixed playing screen and inference

Play each of G20T1, G20T05 and E0 against S0 at 25, 100 and 400 simulations
per side. Each cell requests 1,000 games / 500 color-swapped opening pairs,
seed42, the same pinned production opening book, opening plies16, max plies300,
temperature0.1, matched_sims, training search shape, compile on, no rolling.
Preserve the realized search settings, checkpoint/book hashes and raw games.
Run every budget even after a shallow loss: a crossover is part of the question.
Simulation budget is the manipulated quantity, not a promise of identical depth
or realized node count. Report measured search work and wall cost when available.

The primary strength result is each global arm versus S0 at100 simulations.
Report the producing arena's pair-unit 95% Elo interval: positive lower bound is
promising; negative upper bound is a loss at that budget; otherwise inconclusive.
The400-simulation result is a separately registered deployment-relevant result.
A candidate promising at either100 or400 merits a bounded confirmation decision;
a100 loss does not erase a400 gain. Report all cells, including losses.

For search interaction, use aligned opening-pair scores against S0. Bootstrap
10,000 paired resamples, seed20260903, of score400 minus score25. A positive95%
lower bound establishes increasing relative advantage over S0 on this opening
distribution; it does not establish absolute improvement with search. Report the
400 strength result separately. Compare the two G arms using the same aligned
openings; common-opponent score differences are not direct head-to-head Elo.

This is an exploratory one-training-seed screen with nominal intervals and no
family-wise correction. Do not declare an optimum or promote a selected winner
from these results alone. Before promotion, prospectively record an independent
training seed and fresh-opening direct comparison against E0 at the relevant
budget. If candidates remain indistinguishable, prefer the simpler raw-temperature
treatment for confirmation. Global-versus-E0 changes scope and effective dose;
without an entropy-matched control a gain cannot be attributed specifically to
BT4 ranking knowledge. Larger doses, pure BT4 and an entropy control are possible
next experiments, chosen from the uncertainty exposed here, not an automatic grid.

## Execution, budget and recovery

Use a new `scratchpad/bt4_joint20/global_run01` state directory with an immutable
`preregistration.md` snapshot. A baseline phase trains S0 first; a treatment phase
prepares global corpora, trains them, and runs the nine playing cells. Separate
phase manifests permit S0 to start while the global mixer is being validated.
The old `af_run01` directory and its STOP marker remain intact.

Budget30 GPU hours for this screen: roughly8.5 hours for three epochs plus the
nine arenas, with measured stage times charged to the cap. This is an estimate,
not guaranteed completion. Do not launch a stage that exceeds the remaining
declared allowance; preserve an incomplete cell as unread, never as a loss.
Record a prospective budget amendment if measured costs justify one.

Use the shared GPU lease, one GPU stage at a time, and prepare at most one next
corpus concurrently. Preserve150GiB free disk plus one source-copy footprint
before materialization. The existing G10 labeler fills preparation gaps and yields
at16-shard boundaries; its100M-bank work has a separate existing budget.
Leave both CPU G10 generators running. No live training configuration, generator
recipe or production model is changed by this screen.

STOP in a phase directory cancels its owned children and preserves partial output.
Training partials require explicit recovery; arena resume validates identity and
preserves raw orphan-half history. Do not edit code loaded by active processes.
Useful toolchain fixes go to main through a separate PR and are adopted only at
an identified boundary.

## Readout

UNREAD. No S0/global training or playing result existed when this plan was recorded.
Independent design review found the three new trainings adequate for the initial
screen, subject to the interpretation and confirmation limits above.

### September 5 execution evidence

The active immutable preregistration snapshot has SHA256
`0efd4e6e5217237b90ebeffa33edf400019b02b360258b32233a2e3366bc1565`.
A prior snapshot is retained as `preregistration.before_runtime_correction.md`;
the correction identified the CPU materialization interpreter before successful
audits or training. Two failed Python3.10 audit logs are preserved; they failed
at import because the new worktree's native extensions target Python3.13.

The corrected mixer and raw/rank tools reached main in PR519, merge
`7aa2ec3317932451ec9ebc94b1a5d3433c038fc8`, after independent review,
69 focused CPU tests, whole-repository lint and green CI. Materialization remains
pinned to its clean `4394bd2cddc05ea780ce303a1f731730a1f4dfff` checkout.

Both descriptive audits passed integrity on the4,000-position frozen ruler:

| Treatment | Expected SF regret delta cp (95% CI) | Top-1 regret delta cp (95% CI) |
| --- | --- | --- |
| G20T1 | +4.4328 [+3.9645,+4.9014] | -5.0760 [-6.9180,-3.4019] |
| G20T05 | -0.0634 [-0.5575,+0.4263] | -5.1050 [-6.9483,-3.4272] |

These are descriptive results, not strength admissions or verdicts. Receipts
`audit_G20T1.json` and `audit_G20T05.json` retain the SF comparison verdict
separately from fidelity and the hashed descriptive-admission rationale.

S0's coordinator started at16:25Z and the trainer acquired the GPU at16:28Z;
the existing G10 labeler yielded and now waits on the shared lease. The baseline
driver SHA256 is `da9afa748f7f947986fd096c38d7dc2536932728d33322c0a3288ed2055ce205`.
Source rows, trainer target temperature1.0 and unchanged61,444,448-parameter
architecture were observed at launch. Subsequent windows reached approximately
23 seconds per88 steps after startup. This is progress, not a completed result.

The obsolete A materializer was interrupted at16:26Z; its partial `.writing`
corpus remains unpublished and preserved. G20T1 preparation started alongside
S0 using the CPU environment. Independent readback of its first8,192-row shard
found all16 non-policy arrays identical,8,138 changed policy rows,161,350 newly
positive policy entries, zero illegal positive entries, and maximum total-mass
error0.000350. This confirms the broad-tail intervention reaches real output.
Evidence is `G20T1.first-shard-integrity.json` in the new state directory.

### September 5 WSL restart and recovery

The kernel restarted at16:52:06Z (boot ID changed and uptime reset), terminating
all prior processes. S0's last logged progress was50/420 windows /4,400 steps;
its output contains staged shards but no checkpoint or optimizer/sampler state.
This attempt cannot resume an exact epoch. Preserve it under its original v1
path and restart from scratch into `_epoch_v2`, with the same seed and settings.
The new coordinator state is `global_run02`; the scientific preregistration and
audit bytes remain unchanged. No unfinished run becomes a negative result.

The original incomplete GPU charge remains in run01. Run02 carries a reconciled
copy charging1,441.670314 seconds through the next kernel's boot time, a
conservative elapsed-time upper bound. That cost counts against the existing
30-hour allowance; the reboot does not reset the budget.

Both G10 generators resumed through their existing `--resume` path with their
exact manifest settings and unchanged verified generator/encoding/Stockfish code.
Before resume, preserve manifests, progress files and all16 unlisted interrupted
shards under `scratchpad/reboot_recovery_20260905`. Published inventory before
resume was14,377,828 primary rows plus2,796,072 companion rows, totaling17,173,900.
Closed shards remain authoritative; in-flight games are replayed by the existing
deterministic recovery protocol. The unchanged G10 BT4 banker resumed after its
output was checked for partial sidecars and CUDA availability was verified.


## September 6 corrected-prior readout

The [completed bootstrap results](2026-09-06-bt4-bootstrap-results.md) record the
prospective correction to search-prior temperature 1.0, completed nine-cell screen,
independent reviews, current SF-close follow-up and remaining confirmation. Its
results supersede an interpretation based on the inherited prior 1.5. Training
recipes and the original evidence above remain preserved.
