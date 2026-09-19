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

Local artifacts: `scratchpad/bt4_joint20/generation_throughput_20260920/`.
Current status: preparation only; launch held until disk reserve restored.
