# RL research program: first week, November 12–19

Status: **ACTIVE. No demonstrated RL gain or external-engine milestone.**

The research objective is to convincingly beat unhandicapped multicore Rofchade,
then approach strong Stockfish and Lc0 deployments and investigate exceeding
them. This is a months-long program. A qualified pipeline, closed experiment,
merged PR, lower loss or finished first week does not achieve that objective.
The previous overall completion label was too broad: only its bounded execution
and publication were complete.

This week runs from November 12 at 11:09 a.m. Eastern through November 19 at the
same time. The operating deadline is `2026-11-19T16:09:20Z`. The user explicitly
extended the work to a first week; the earlier experiment's 48-hour limit remains
part of its historical record, not this week's deadline.

## Starting evidence and ownership

- Preserve epoch-two donor SHA256
  `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`.
  No candidate has been adopted.
- The [initial RL experiment](2026-11-11-epoch2-rl-transition.md) ended with a
  valid but inconclusive 1,000-game confirmation (315/371/314; score 0.5005).
  Its longer endpoint failed its donor screen.
- The [cap-40 adjustment](2026-11-12-epoch2-rl-cap40.md) completed 320 updates
  but failed its fixed screen (56/80/120; score 0.375). Those games and its
  unused confirmation panel retain their original identities. No pooling,
  favorable extensions or reinterpretation as learning success.
- Existing [PR 1229](https://github.com/jjoshua2/DeepFin/pull/1229) owns the
  stored-input diagnostic implementation. Snapshot
  `4957d667df3ee5e541fc62c5fc7ede5c43c0fdea` already contains actual WDL,
  child-SF, policy-drift, limited shared-body gradient and tiny-history work,
  plus a one-search crossover resource pilot. Reuse and refresh that owner's
  evidence; do not create a second crossover implementation or controller.
- The first-week development base is main
  `192af638e6787fc90bc3ee2eb1b37417648a911b`. Compared with measured source M,
  its runtime changes are operational SF-budget/topology guards; model,
  optimizer, targets, encoding and search sources are unchanged. Measured
  runtime checkouts remain pinned. This development checkout owns the
  independent external-benchmark instrument repairs.

External evidence bundle:
`~/chess-artifacts/operations/rl-week-one-20261112-v1`.
Readiness, diagnostic-design and research-audit packets there record exact
source/path identities and limitations. They are planning evidence, not engine
results or launch authority for another owner's jobs.

## Week-one work and decisions

| Window | Work | Evidence needed to advance |
| --- | --- | --- |
| Nov 12–13 | Repair and qualify the external match instrument; capture actual engine capabilities and off-clock readiness; calibrate one paired match | Complete histories reach the engine and PGN/log, required options fail closed, correct clocks/colors/results and bounded process closure |
| Nov 13–15 | Establish a donor-versus-multicore-Rof baseline; consume the existing owner's crossover controls | Frozen binary/net/checkpoint identities, declared hardware/clock/book, complete paired games and uncertainty; actual identity/repeat/batch crossover controls |
| Nov 15–17 | Select one falsifiable mechanism and run the smallest justified controlled update comparison | Frozen starting state, legitimate untouched measurement cohort, explicit targets/sampling/update budget and a prediction that can fail |
| Nov 17–19 | Test a promising endpoint on fresh playing evidence and, where justified, an independent training realization; report the week | Predeclared horizon and margin, independent confirmation before a gain/adoption claim, all failed/inconclusive outcomes and generation-inclusive costs retained |

These are working windows, not promises that a candidate will improve on schedule.
Correctness failures stop the affected lane; useful independent work continues.
An inconclusive or negative run gets a scientific next decision. It does not end
the broader research program or authorize an unrecorded hyperparameter sweep.

## First external benchmark

Use the real Rofchade binary and NNUE on this machine, full strength, with an
explicit multicore setting and equal wall-clock controls. The conservative
shared-machine starting point is **eight Rof threads**, leaving room for the
two-thread neural engine and neighboring work. Record affinity/topology; eight
configured threads is not proof of eight physical cores continuously executing.
Higher-core-count comparisons remain part of the program and require their own
recorded resource window. No Elo/skill/node handicap is allowed on Rofchade.

The readiness packet proposed a small engineering pair at 30 seconds plus
100 milliseconds per move, zero clock grace, eager donor inference, off-clock
warmup and one match at a time. Before execution, the admitted protocol was
revised to use the same indexed three-through-six-piece tablebases in both
engines, with theoretical harness adjudication disabled. The exact source, commands,
engine options, net identities, resource checks, fresh outputs and stopping rule
must be bound before launch. The first pair establishes operational readiness,
not strength. A following small full-history panel checks the complete instrument.
Choose the substantive baseline's fixed panel and horizon after measuring cost,
before looking at its results. Count openings, not individual correlated games,
as the sampling unit. Missing/degenerate uncertainty remains unavailable.

The initial audit found Rofchade 3.1 installed. The subsequent capability probe
verified its UCI handshake, NNUE load acknowledgement, eight-thread setting and
affinity to eight distinct physical cores. The tablebase qualification found all
510 expected indexed material families for both WDL and DTZ, reproduced five
fixed rule/DTZ fixtures and obtained actual six-piece hits from both engines.
This does not attest every table file's contents or every position.
Installed Lc0 0.32.1 and 0.33dev builds are CPU-only;
neither qualifies as a strong GPU-Lc0 reference. GPU backend/network readiness
and the chosen Stockfish binary/net need separate verification. An installed
version alone is not a current-best-engine claim.

An internal fixed-simulation donor comparison measures learner quality at that
search budget. A wall-clock external match measures the deployed engine including
its latency. Report both under their own settings. Do not translate a handicapped
SF curriculum result or one external reference into a public/absolute Elo rating.
Before claiming the Rofchade milestone, freeze the meaningful winning margin,
time controls, hardware and independent confirmation protocol prospectively.

## Mechanism before another recipe

The PR 1229 bank is 254 **training-ingested** rows from three supplied games;
it is not an untouched holdout. Its 227 child comparisons and limited gradient
probe answer narrower questions than the rejected playing match. Three qualified
distinct prefix positions do not provide complete histories for all 254 roots.

At snapshot `a039b723e86379b9eb57d966d6119fa7cf51d853`, the existing owner had
completed 24 all-node policy/value crossover searches with 12 identity, repeat
and batch controls on three correlated opening positions. All controls passed
(maximum policy difference about 1.79e-7). Donor/donor and candidate/candidate
selected the same move at all three roots; one mixed-output condition changed
one move. The fourth search return is a legal-action mask, not search WDL.
Use the corrected saved readout; do not repeat those controls or infer a harmful
mechanism from them. The Python/CPU reference path is not automatically
equivalent to the native GPU screen. These data show search sensitivity on those
positions; they cannot label moves as worse, prove generalization or explain the
match by themselves.

Use its outcome to choose between target/perspective/calibration measurements,
actual optimizer/trunk-update measurements and exposure/sampling measurements.
No current evidence selects a guaranteed SF blend, clip limit, LR or sampler
change. In particular, producer-draw agreement is partly self-referential and
the auxiliary hard-clip scope excludes Aurora matrix updates. Historical SF-anchor
anecdotes are not current causal evidence.

A new learning comparison needs a genuinely untouched, complete-history cohort
with declared game groups, target provenance/perspective and exposure exclusion.
Do not relabel an ingested shard, old inspected arena panel or the three tiny
prefixes as that holdout. Preregister the next intervention, simpler control,
deciding metric, effect threshold, uncertainty, training realizations, compute
budget and kill rule before training. Additional inputs and teacher supervision
are hypotheses to measure, not an established advantage over Leela.

## Budget, controls and continuing status

The first-week planning ceiling is **48 hours of GPU-intended stage wall time**,
including generation, readiness, failed runs and cleanup. This is a ceiling, not
a target or an allocation of exclusive hardware. Each job is finite (at most
two hours) and records its own smaller hypothesis-specific budget before launch.
The initial engineering pair has at most 600 seconds; a small subsequent
history-bearing panel has at most 1,200 seconds. Heavy CPU diagnostics also need
finite limits; routine tests retain two Torch threads with CUDA hidden.

Use existing local hardware, no paid compute, one GPU experiment at a time, and
the canonical leases. Honor intentional-stop/STOP/PAUSE/CANCEL/YIELD controls,
including the previous operation's controls, and explicit gaming/work requests.
Never clear markers or disturb user-stopped THREE, the epoch-three hold, a closed
publisher or unrelated jobs. The week deadline stops new compute unless extended.
Preserve the donor; exact interrupted RL resume remains unqualified unless
separately established. No autonomous deployment follows a favorable screen.

Week-one success is a trustworthy external baseline, a control-qualified mechanism
readout and useful controlled learning evidence. A headline learning success
requires actual repeatable playing improvement with independent confirmation.
Negative results are valuable but do not satisfy that strength milestone.
Publish the week readout and costs with achieved/missing milestones. Keep the
months-long strength objective explicitly open.

## November 12: clock readiness failed; deadline repair

The first engineering pair used source
`5c4cff29a3832d7c6de94f8a6a542780c1c9ffbf`, donor A39, eager CUDA inference,
two PUCT walkers with gather one and default chunk 2,048, Rofchade 3.1 with eight
threads/1,024 MiB hash, equal 30 s + 100 ms clocks and zero grace. Both donor
first responses were legal but took **73.872421 s and 73.160766 s**. The driver
correctly recorded two time forfeits and excluded the late moves from the PGNs.
This is a failed deployment profile, not an estimate of chess strength or RL gain.

The owner closed all three tracked process identities after 303.956620 s;
its PASS status means clean execution and cleanup. Independent readout:
`engineering-pair-readout-v1/readout.json`, SHA256
`25e90b639f961fe1b166598c8e76109a6f26896c770903bf49bc84a6d62b5a11`
under the external evidence bundle. No guarded search error, incomplete-search,
fallback or NNUE-load diagnostic appeared. GPU-intended readiness plus pair
owner wall time totals **469.566925 s**, including startup and cleanup.

The first-chunk bootstrap assumed 6,000 nodes/s and allowed a full 2,048-node
chunk at the computed clock budget. Workers polled their stop event but were
not signalled when the deadline expired; the caller waited for the whole chunk.
Reducing the clock allocation alone cannot fix this. The proposed repair gives
each finite UCI phase a deadline timer bound to that phase's stop event and
cancels/joins it before returning. Cooperative pools can finish their in-flight
batch and return truthful completed nodes. This does not preempt a GIL-held
native operation or an opaque classic-Gumbel chunk; it is not universal hard
real-time safety. No model, target, optimizer or allocation formula changes.

A real slow-evaluator walker test fails against the original phase and passes
with the repair. Nine deadline tests, 176 lifecycle/ponder/search regressions
and one self-play smoke passed with CUDA hidden and two Torch threads. The
source change still needs independent review and actual donor qualification.

Before another pair, freeze the repaired source and qualify it with **no games**:
same donor/device/search/chunk/tablebases, at most 300 GPU-intended stage seconds,
off-clock startup, two full-history roots (start and `g1f3 d7d5`) at the same
30 s + 100 ms clock allocation, one explicit 200 ms search and one exact
64-node control. Require legal moves, positive genuine non-tablebase nodes,
no guarded faults/fallbacks, each timed response within its computed hard
budget plus 500 ms, exact 64 nodes for the node-only control and owned process
closure. Record startup separately. Failure preserves the output and blocks
the next match; no favorable retry or training/adoption follows automatically.
