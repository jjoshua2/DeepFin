# SF generation depth and concurrency for the 500M bootstrap

## Question and previous evidence

Can fixed full-width d8 supply result-bearing, derivable positions much faster
than the current G10 ladder (`all:9,8:10,4:12`)? The old bounded fleet had three
active workers, added547,022 closed rows in roughly8h (~68.4k/h), and included
startup and shutdown. That historical rate is not a steady-state benchmark.

This CPU-only screen leaves the frozen58M training/labeling queue and all existing
corpora untouched. Each cell writes a new namespace. Same book, Stockfish binary,
selfplay temperatures, hash64MiB, dedup2M/worker, and maximum400 plies are retained.
D8 explicitly uses the `fixed` gate; it never masquerades as validated G10.

## Prespecified comparison

Six short cells: fixed d8 and G10, each with execution concurrency1,2,4. Logical
worker partition remains4, seed20260920 and requested game quota64 throughout.
Run paired depth cells at each concurrency; no8-worker arm in this first screen.
Each cell gets at most100 wall seconds including cold startup. Shard threshold256
banks complete games frequently; it changes serialization amortization from the
old8192-row setting and must be reported as a confound.

Primary metric: closed-game rows per wall second that retain a result and pass
production stored-input, phase0 depth/support, and latest-phase value-support
checks. Also report banked rows, no-result/invalid fractions, CPU time, observed
memory/disk bounds and startup/closed-shard censoring. Unlisted tails are excluded.
Depth changes actual play; this is end-to-end generation throughput, not search
speed on identical positions. No claim about trained strength follows.

Select the fastest valid d8 concurrency if it materially improves the measured
frontier (2x is the practical next-stage target). Tiny or zero closed-shard samples
are inconclusive and trigger a longer bounded measurement only if budget allows.
One sample per cell cannot establish stable confidence intervals. Compare the
observed rate with the ~161 accepted rows/s needed for another418M in30 days;
do not extrapolate worker scaling beyond measured resources.

## Bounds and recovery

At most8 affinity CPUs, nice19, GPU hidden, numerical threads2. Launch and every
new cell require100GiB free disk; running reserve96GiB and available RAM40GiB.
Output hard target<=2GiB with early stop at1.5GiB. Observed descendant CPU cap1500s
reserves300s for missed short children and accounting uncertainty within30 CPU
core-minutes. Total wall cap30min. Shutdown authenticates owned PID start times,
including Stockfish's separate sessions, and checks no live owned process survives.
No resume, in-place source rewrite, or GPU launch is permitted.

## Completed short screen

Run v3 completed all six cells in 611.1 seconds: 1,345.1 observed descendant
CPU-seconds plus 7.47 controller CPU-seconds. All 4,513 banked rows passed the
production eligibility checks; none lacked results or failed validation.

| Policy | Active workers | Closed rows | Closed shards | Eligible rows/s |
|---|---:|---:|---:|---:|
| G10 | 1 | 261 | 1 | 2.59 |
| d8 | 1 | 383 | 1 | 3.80 |
| G10 | 2 | 261 | 1 | 2.58 |
| d8 | 2 | 1,285 | 3 | 12.74 |
| G10 | 4 | 261 | 1 | 2.58 |
| d8 | 4 | 2,062 | 5 | 20.40 |

D8 with four workers supplied about 73.5k eligible rows/hour during this short
window. That is a useful measured direction, but the apparent 7.9x advantage
over G10 at four workers is **not an established steady-state speedup**. Every
G10 cell closed exactly the same one-game, 261-row shard before the 100-second
cutoff. Additional workers spent their budget on games that had not closed.
These results cannot establish G10 concurrency scaling. D8 concurrency results
also mix execution scaling with game-completion boundaries.

A longer equal-concurrency comparison that closes several games per worker is
the next informative measurement; do not extrapolate the G10 denominator or
claim this screen establishes a monthly 500M-generation rate. Even d8's measured
20.4 rows/s is below the approximately 161 rows/s planning target. This is
whole generation plus full-width SF banking, not just labeling previously
available positions, and does not measure eventual network strength.

## Failure recovery and validation

The first attempt failed in readout because direct-file Python execution found
a user-site `scripts` package. Setting `PYTHONPATH` inside the already running
interpreter did not repair its import path. The second failed because readout
expected numeric staircase widths, while the production manifest writes string
width labels. Both failed runs remain preserved. Their first G10 cells each
contain 261 closed rows, all recoverable and valid under the corrected reader;
neither failed attempt is counted as a completed comparison.

The fixes bind imports to the pinned runtime and test manifests produced by the
actual generator. Seven focused tests passed and were independently rerun. They
include a real registered-interpreter readout with blank PYTHONPATH and foreign
working directory, plus escaped-process ownership/cleanup. The trigger separately
passed stubborn-root, setsid-grandchild, unrelated-child preservation, and natural
root-exit cleanup checks. No GPU queue or existing corpus was changed.

Frozen benchmark source: `401a93bf8`. Compact full cell evidence, plan hash and
receipt hash: [JSON artifact](artifacts/2026-09-20-sf-throughput-v3.json).
Raw results are under
`/home/josh/chess-artifacts/corpora/generation_throughput_20260920/run_v3`;
plans, logs and recovered-failure readouts are under
`/home/josh/chess-artifacts/operations/sf_generation_throughput_v3_20260920`.
The corpus and operations siblings for v1/v2 preserve the failed attempts.

The independently reviewed longer follow-up has now completed; its prespecified
protocol and results follow.

### Longer confirmation protocol

The explicit `confirmation` profile runs G10 first, then d8, each with four
workers and a 600-second maximum. The ordering is fixed and unreplicated;
thermal drift and changes in the concurrent Ceres workload remain possible
confounds. Both request 256 games with the original seed, book and teacher
settings so the game quota should not end the faster cell prematurely.

The observed CPU cap increases to 5,000 seconds, reserving 400 seconds within
a 90 CPU-minute allocation for accounting uncertainty and cleanup. The overall
wall cap remains 1,800 seconds, affinity remains CPUs 8–15, and RAM/disk/output
limits are unchanged. This is a new explicit allocation, not a silent expansion
of the pilot's 30 CPU-minute limit.

Every 30 seconds, record progress-listed closed rows/games/shards separately
from unlisted file bytes. These live counts are explicitly unvalidated; final
eligible rows still require full production identity/support validation. Rows
still in worker memory are unknown and never counted. Compare early and later
closure increments to diagnose completion censoring; even the longer test does
not supply independent order/seed replication or a trained-strength result.

### Completed longer confirmation

| Policy | Wall seconds | Eligible rows | Closed games | Closed shards | Eligible rows/s |
|---|---:|---:|---:|---:|---:|
| G10, four workers | 600.54 | 10,431 | 53 | 28 | 17.37 |
| d8, four workers | 601.18 | 38,615 | 200 | 106 | 64.23 |

All 49,046 banked rows passed eligibility; no invalid or missing-result rows.
The complete run used 1,240.0 wall seconds, 4,486.7 observed child CPU-seconds
and 42.9 controller CPU-seconds, within its explicit bounds. Frozen source was
`93537e6b0`. [Full compact evidence](artifacts/2026-09-20-sf-throughput-confirmation.json)
includes 30-second closure counters and exact plan/receipt hashes.

D8 achieved **3.70x** G10's eligible generation throughput in this comparison.
During approximately the final five minutes, closed-row increments were 20.82/s
for G10 and 74.16/s for d8. This confirms that the original 100-second pilot was
strongly censored and supplies substantially more closed-game evidence. It does
not remove the fixed-order, single-seed or concurrent-workload limitations.
The observed average d8 rate is about 231k rows/hour, or 166M rows in 30 days
if maintained continuously. That is a planning extrapolation from ten minutes,
not proof of month-long throughput or linear scaling to additional CPUs.

Crucially, both arms generate new positions **and** score every legal move.
Neither measures the cheaper task of adding one root SF value to an already
banked position. The next planned cost screen compares root d8, root d10 and
full-width d8 on identical history-preserving rows, keeping that question
separate from generation distribution and target quality. Existing bootstrap
contracts and the GPU queue remain unchanged.

## Planned matched-row scalar-value cost screen

The separate `benchmark_sf_values.py` screen fixes 64 midpoint-spaced games
from the 200 closed d8 confirmation games, selecting each game's midpoint row.
Selection is deterministic in `(worker-progress-file, game_id, source-shard)`
order. The external sample preserves the full raw row and history, source-row
index, canonical row hash, source-shard hashes and qualification receipt hash.
This measures a d8-selfplay-derived position sample, not a population estimate.

On those identical rows, compare root MultiPV1 at depth8, root MultiPV1 at
depth10, and full-legal-width depth8. Three independent SF engines use Hash64MiB,
one thread and the same Syzygy files. Every row/arm resets its own TT; rotate
arm order by row. The stored history is sent as `position fen ... moves ...`.
Record effective UCI commands, raw info including mate/cp/bounds, requested and
realized depth/width, nodes, per-label search-plus-reset time and complete
supervised wall time. Final-depth support must match the requested width.

No policy-complete training corpus is created and no target recipe changes.
Scalar-root scores can differ from a full-width search's best score; this cost
screen does not measure which value is more accurate or which trains better.
The existing all-move generation measurement and this existing-row post-label
measurement answer different questions.

Bounded allocation: CPUs8–9 only, nice19, no GPU, observed CPU<=1500s plus300s
reserve, wall<=1800s, available RAM>=40GiB, launch disk>=100GiB/running>=96GiB,
and output<=128MiB with an early75% stop margin. The reviewed subreaper tracks
owned engine sessions and cleans them on normal completion, STOP or failure.
Plans and outputs remain external; launch follows independent review.
