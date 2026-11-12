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
and one self-play smoke passed with CUDA hidden and two Torch threads. At this
preregistration, independent review and actual donor qualification were pending.

Before another pair, freeze the repaired source and qualify it with **no games**:
same donor/device/search/chunk/tablebases, at most 300 GPU-intended stage seconds,
off-clock startup, two full-history roots (start and `g1f3 d7d5`) at the same
30 s + 100 ms clock allocation, one explicit 200 ms search and one exact
64-node control. Require legal moves, positive genuine non-tablebase nodes,
no guarded faults/fallbacks, each timed response within its computed hard
budget plus 500 ms, exact 64 nodes for the node-only control and owned process
closure. Record startup separately. Failure preserves the output and blocks
the next match; no favorable retry or training/adoption follows automatically.

## November 12: qualification and repaired pair

Readout cutoff: **2026-11-12 18:36 UTC**. Subsequent stages retain separate
results; the pending work below describes this cutoff.

The match-history/option fix merged in [PR 1232](https://github.com/jjoshua2/DeepFin/pull/1232)
at `0db6e954861ebae91526c6cd4c3a08dd37ce1655`. The deadline repair merged in
[PR 1235](https://github.com/jjoshua2/DeepFin/pull/1235) at
`9a0a825b2b147d72522b406c52d051b7a2a97f80`. Both composed revisions received
independent review and passed all four remote CI jobs. Measured repaired runtime
remains pinned to `f914f51eb349681300fe50147029e98b880de63e`.

All four preregistered donor probes passed with no fallback/fault diagnostic.
The two 940 ms clock allocations returned in 1.028 s and 0.995 s; the 200 ms
request (170 ms after move overhead) returned in 0.251 s, within the declared
500 ms tolerance. The node-only control returned exactly 64 nodes. Startup took
82.137 s off clock; the owner closed after 90.408 s. This qualifies the sampled
cooperative route, not every clock or an opaque native chunk.

The fixed successor pair kept the original eager PUCT2/gather-one/chunk-2,048
profile, 30 s + 100 ms clocks, eight-thread Rofchade and common tablebases.
It completed **0 wins, 0 draws, 2 losses** for the donor, with actual checkmates
at 132 and 127 plies. Independent replay reconciled all 259 legal played moves,
PGNs, clock charges and protocol responses. No clock forfeit, illegal move,
ply-cap draw or guarded engine fault occurred; all three owned process births
were absent after closure. Wall time was 288.221 s including startup/cleanup.
There is no between-opening interval from one pair and no strength milestone.

Of the donor's 129 replies, 26 were root tablebase shortcuts reporting one node
and one hit. The remaining 103 replies reported 1,844 nodes in total, median 19
and range 1–33. Do not count the shortcuts as neural search work or interpret
this small deployed-search budget as an RL-quality measurement.

## November 12: serving cost and compilation

The fixed raw study loaded one strict A39 model, retained actual 175-plane/history
metadata and observed CUDA BF16 autocast. It completed 48 forward calls/516 rows,
including 40 timed calls/430 rows. Same-batch route/repeat outputs matched exactly;
batch-size changes did produce small numerical differences. Direct serving
median latencies for batches 1, 2, 8 and 32 were 28.846, 30.022, 29.321 and
29.880 ms. Larger batches amortize a nearly fixed call cost on this small panel;
this is not search NPS or an attribution to one specific CPU/GPU operation.
CUDA-event stream intervals include eager host-launch gaps and are not accumulated
kernel-active time. The owner closed after 7.148 s.

The first compiled-study attempt failed during observer setup: it tried a
nonexistent dispatcher attribute after two eager factory calls/three rows.
It never constructed the compiled comparison route. Its 23.222 s and failure
artifacts are retained. The corrected observer uses the actual delegate API;
real CPU constructor controls reproduce the old failure. A separately reviewed,
fixed successor preserved the original comparison schedule and 900 s ceiling.

The successor completed with actual owner-thread graph capture/replay, finite
outputs and bit-identical same-route repeats. Two Dynamo graphs were recorded;
separate hooks observed three CUDA capture objects and 22 balanced replays.
The 36 measured requests/48 rows gave these completed request latencies:

| Batch | Eager median | Compiled median | Median latency ratio |
| --- | --- | --- | --- |
| 1 | 29.726 ms | 3.052 ms | 9.74 |
| 2 | 29.127 ms | 3.663 ms | 7.95 |

These small-sample serving measurements include dispatcher completion, not search
or playing strength. Inverse-median throughput differs from aggregate rows divided
by summed durations; both are retained in the compact evidence. Cold compilation,
imports and cleanup are included in the 107.500 s owner wall time.

**Strict numerical equivalence remains unqualified:** none of the 24 eager versus
compiled comparisons met the original `atol=1e-5, rtol=1e-4` criterion. Maximum
WDL probability difference was 0.0006151; maximum legal-policy total variation
was 0.0039340. No sampled legal top action changed on the four correlated opening
prefixes. These observations do not prove numerical equivalence on other boards.
Treat compilation as a distinct experimental serving profile. An actual UCI
clock/node/history qualification is the next step before binding its command to
the frozen eight-game/four-opening engineering panel; neither is an adoption or
strength gate.

## Continuing work and cost checkpoint

Seven completed GPU-intended stages total **986.065154 s (16 min 26.065 s)** at
this readout cutoff, including the failed observer, load, warmup and cleanup.
This is owner wall time, not GPU kernel time. Manual CPU tests and remote CI are
outside that total; the earlier closed RL experiments retain their separate cost
records. Compact results, per-stage identities and bulk-artifact hashes are in
[day1-readout.json](evidence/rl-week-one-20261112/day1-readout.json).

The learning lane found that default distributed holdout selection splits rows
within games and that capped games can receive SF/default-draw labels. The optional
finite-game provenance API in [PR 1237](https://github.com/jjoshua2/DeepFin/pull/1237)
exposes full histories, pre-outcome start ordinals, raw record roles, final native
FEN and separate terminal/rescore paths. It preserves current labels; it does not
create or qualify a holdout. A separate two-game plumbing pilot is being prepared,
with its physical games excluded from the later research cohort. No new research
cohort, full learning comparison or substantive external baseline is qualified yet.

There have been **zero new week-one training updates and no candidate adoption**
at this checkpoint. Keep the donor and prior negative results. The first week and
the months-long Rofchade/Stockfish/Lc0 objectives remain active and unachieved.

## Audited eight-game panel — November 12, 19:29 UTC

This checkpoint follows the earlier 18:36 UTC readout; it does not rewrite its
pending statuses. The compiled public UCI profile completed four separately
qualified probes: the two 940 ms hard-budget requests returned in 945.2 and
944.5 ms with 410 and 430 reported nodes, the 170 ms hard-budget movetime probe
returned in 174.5 ms with 81 nodes, and the node-only control returned exactly
64 nodes in 140.4 ms. Startup took 98.49 s off-clock. The owner closed all eleven
observed process identities. This qualifies those measured operations; the
distinct compiled profile still fails all 24 earlier raw eager/compiled strict
tolerance comparisons. It is not numerical equivalence or checkpoint adoption.

The fixed full-history engineering panel then completed all **eight games**
against **Rofchade 3.1 on eight physical cores**, with its NNUE acknowledged,
1 GiB hash, common indexed three-through-six-piece tablebases, equal
30 s + 0.1 s clocks, zero grace and a 450-additional-ply cap. Four convenience
openings were played in both colors. The unchanged A39 donor scored:

| Donor wins | Draws | Losses | Score | Opening pairs |
| --- | --- | --- | --- | --- |
| 0 | 4 | 4 | 25% | 4 |

Independent replay reconciled all 774 clocked responses, full opening histories,
PGNs, UCI traces and clock charges. The four losses were checkmates; the four
draws were claimable next-move threefold repetitions under the frozen driver's
`claim_draw=True` rules. There were no time forfeits, illegal moves, ply-cap
endings or guarded search faults. All three recorded owner/engine process
identities were absent after closure. Whole-stage owner wall time was 484.080 s.

The donor reported a median 345 nodes over 386 responses. Thirty eligible-root
tablebase shortcuts each reported one node and one hit; the other 356 responses
had median 357 nodes. These counters include search conventions and reused or
terminal trees, not necessarily neural forwards. They are not comparable with
Rofchade's node totals as neural throughput. No fresh graph/precision observer
ran during these games.

This establishes a functioning external instrument on four convenience pairs.
No general-strength interval or Elo is reported. Do not pool these games with
the earlier eager pair or attribute a playing improvement to compilation:
openings and numerical profiles differ, and no weights were learned. The
multicore Rofchade strength milestone remains unmet.

The generation lane has **no qualified data** at this cutoff. Its first bounded
attempt failed closed after 3.253 s because its runtime `process.json` collided
with the owner's reserved stage namespace. Its unchanged files were preserved
under a nested failure-artifact directory; the owner guard was retained. A
separate preparation corrected the receipt name and exercised the actual owner
accounting function against both the old collision and the new filename. The
second attempt failed closed after 5.145 s at a raw architecture-dictionary
equality guard, after strict donor loading but before evaluator construction or
generation. It produced no games or shards. The architecture comparison needs
diagnosis against the existing donor contract before another admission.

The proposed four-arm loss-presence study is superseded for the SF-anchor
question: every arm set SF supervision to zero, and removing entire objectives
also changes gradient and optimizer dose. The next proposed contrast keeps all
losses, prepared batches, the full trunk, optimizer and schedule fixed, changing
only the main WDL target's SF/search shares. Stored categorical targets would
remain unchanged. No share, teacher effort, deciding threshold or training run
is selected. Actual whole-game provenance, target perspective/availability and
the real Trainer consumer still need qualification. The old SF-removal anecdote
is not mechanism evidence: the ledger records the operator's August 23 rejection
of that uncontrolled result. Shared guidance now reflects that qualification.

At this cutoff, eleven completed GPU-intended stages total **1585.066561 s**
(26 min 25.067 s), including both failed collectors. This is complete owner wall
time for those stages, not GPU kernel time or all CPU work. Their identities,
exact costs, source/readout hashes and compact results are in
[engineering-panel-readout.json](evidence/rl-week-one-20261112/engineering-panel-readout.json).
There are still **zero new week-one training updates, no qualified research
cohort and no candidate adoption**. The first week and longer strength program
remain active.
