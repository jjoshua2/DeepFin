# Faster recipe matches with ordered sequential decisions

**Status:** ordered sequential tools are merged, and the six-cell batch-capacity
component probe is complete. The H20 package retained its original fixed banks;
its completed strength results are reported in the
[H20 record](2026-09-07-bt4-hybrid-endpoints.md). No match in this record used the
new sequential path, and no whole-match rolling speedup has been measured.

## What is worth changing

Use **100 and 400 simulations**, with a specific reason required for any third
budget. Broad exploration should identify useful differences rather than spend
1,000 games estimating every large win precisely. Repeated mechanistic matchups
are useful only while their answer can change the next research choice.

The proposed future allocation is:

| Purpose | Allocation and deciding evidence |
| --- | --- |
| Broad recipe versus incumbent | 100 sims; paired GSPRT with logistic H0=0/H1=+15, alpha=.05/beta=.10; first look at 128 pairs, then every 64, hard cap 500 pairs/1,000 games |
| Protected new-family search probe | Fixed 128 pairs/256 games at 400 on the same first 128 low-budget opening pairs, regardless of shallow sign; use 250 pairs when finer resolution is needed |
| Confirmation | Freeze recipe/calibration, use two fresh paired training seeds (three preferred), initially 250 fresh opening pairs per seed at 400; add 100 only when the interaction remains a deciding question |

These are future allocation choices, not a mandatory queue or production
promotion. A substantially different family can retain its higher-search probe
without passing the shallow screen. A change to the selected hypotheses or
sample horizon belongs in the next registration, not a running match.

The completed H20/C100 bank scored 0.5645 and H20/G100 scored 0.621. Their pair
variances were 0.086137 and 0.084027. At that variance, a fixed 128-pair screen has
roughly ±35 Elo precision near the null, versus about ±18 at 500 pairs. Small
screens distinguish large differences; they do not reliably resolve +15 Elo.

A **post-hoc illustration** applied the proposed 0/+15 stopping rule to canonical
prefixes of those completed banks: C100 first crossed at 320 pairs/640 games,
G100 at 128 pairs/256 games. This is not a prospective test, a calibration of its
error rates, or an expected-savings estimate. Original fixed-bank results remain
the reported strength measurements.

## Keep throughput and statistical efficiency separate

A banked duration diagnostic found mean active-game counts of **76.77 and 78.03
out of 128**, with about **28% of chunk time below 32 active games**. This estimates
slot occupancy from completed-game durations, not GPU utilization. Rounded
timestamps, compilation/stalls inside chunks and the smaller final chunk affect
it; startup and between-chunk costs are excluded. Unused slots are not removable
wall time. The separate component probe below measures search calls on a fixed
opening panel, not the occupancy or elapsed time of a rolling match.

Rolling execution can refill slots while long games finish, but its old SPRT
path used whichever pairs completed first. Both colors being complete does not
remove selection toward fast-finishing outcomes. The prepared change buffers
results by canonical opening ID and evaluates every newly available declared
prefix look in order. A late pair may release many looks; the earliest crossing
owns the result. Extra completed and in-flight games remain separately recorded.

The optional syntax is:

```text
--sprt 'elo0=0,elo1=15,alpha=.05,beta=.1,first_pairs=128,step_pairs=64'
```

Both look controls default to 1. The final pair cap is a look even when it falls
between steps. Resume requires the same hypotheses, look schedule and new prefix
protocol. Legacy or different sequential specifications and fixed-N/sequential
conversion are rejected; completed observations cannot acquire prospective
meaning by changing a resume flag. Fixed-N execution and its strict BT4 readers
remain unchanged. A future sequential experiment launcher/readout still needs to
bind these outputs to its registration before scientific adoption.

## Completed batch-capacity component — September 8

All six registered cells completed once, using the same 1,024 measured opening
positions per model and two checkpoints (H20 and C). Each cell therefore contains
2,048 root decisions. A separate 128-position prefix was used for warmup. All
1,152 retained opening histories contain 16 legal plies and replay to their
unique banked endpoints. Search retained the qualified full settings, prior
policy temperature 1.0, Gumbel noise on, move temperature .1 and compilation on.

| Simulations | Width per model | Measured roots/second | Timed seconds |
| ---: | ---: | ---: | ---: |
| 100 | 32 | 60.98 | 33.59 |
| 100 | 64 | 53.10 | 38.57 |
| 100 | 128 | 63.53 | 32.24 |
| 400 | 32 | 19.49 | 105.10 |
| 400 | 64 | 25.15 | 81.43 |
| 400 | 128 | 22.76 | 89.98 |

At 100 simulations, width128 was 19.6% faster than width64 in this observation,
but only 4.2% faster than width32. At 400, width128 was 9.5% slower than width64;
width32 was also slower. The actual fixed order was 100:32,128,64 then
400:128,32,64. There was one observation per cell, without randomized order or
independent compilation/cache state; these percentages are descriptive, with no
causal speedup interval or throughput-optimality claim.

Use **rolling pool256 at 100 simulations and pool128 at 400**, both with
**evaluator capacity4096**, as provisional choices for the next registered
matches. The actual models had both dynamic-relation flags false, and the leaf
requirement covered all calls through pool256 without lowering a bound. A new
model must pass that same actual-model capacity check. Balanced width128 means
256 total roots across two models; a rolling pool256 can temporarily place all
256 on one side. Capacity4096 covers that case, but its throughput and memory
shape were not measured here. Do not equate these component rates with complete
match speedup: gameplay, refills, long-game tails and concurrent GPU load remain
outside the measurement.

The owned stage took **820.74 seconds**, within its 1,200-second cap including
termination grace. Timed cells totaled 380.91 seconds; warmups totaled 297.12,
including a 269.89-second first warmup; other stage work took 142.72 seconds.
Warmup is not a complete measure of compilation cost, and timed cells may still
include compilation. Peak allocated memory was 1.56 GiB and peak reserved memory
5.43 GiB; retained allocator cache prevents an independent per-width comparison.
The frozen producer checked every returned action for legality, but retained
only action hashes, so the independent review could not replay action vectors.
These are throughput observations, not playing-strength observations.

The [original cells](../../scratchpad/bt4_joint20/match_efficiency_v1/batch_capacity_v1/execution/cells.jsonl),
[full-history panel (gzip)](../../scratchpad/bt4_joint20/match_efficiency_v1/batch_capacity_v1/execution/panel.json.gz),
and [independent completed review](../../scratchpad/bt4_joint20/match_efficiency_v1/batch_capacity_v1/independent_completed_review.json)
are bound in the [capacity and launch evidence manifest](evidence/bt4-bootstrap/capacity-g10-launch-manifest.json).
It also preserves the original CPU-only preflight failure: raw dataclass settings
were compared with effective arena settings. The corrected serializer and passing
CPU preflight preceded the GPU probe; no unsuccessful GPU measurement was
repeated. Original source, runtime, plan, process and operator receipts are
included; bulk weights, opening book and compile caches remain external.

## What a result would establish

The GSPRT verdict favors one separated hypothesis over the other. Passing H1 is
not a confidence lower bound of +15; H0 is not equivalence. Reaching a cap without
crossing is inconclusive. Stopped Elo and ordinary fixed-N intervals are
selection-biased/descriptive. Generalized-MLE SPRT guarantees are asymptotic,
not exact finite-sample Wald bounds, and local logistic Elo differs from
Fishtest's usual normalized bounds. See [Fishtest's statistical methods](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
and the [underlying GSPRT paper](https://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf).

For search interaction, compare the aligned fixed-core pair-score differences
`score400 - score100`, retaining both colors when resampling. Winning at 400 does
not show an increasing advantage. Two budgets measure one contrast, not a full
scaling curve. Fresh training seeds address recipe variability; extra games on
one checkpoint do not replace them. Fishtest also documents the
[selection bias of passing tests](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html).

## Evidence and qualification

The [compact evidence manifest](evidence/bt4-bootstrap/faster-arena-manifest.json)
binds original independent reviews, planning calculations, occupancy and the raw
design note. Its correction preserves a historical provenance mistake: the note's
blanket 1,000-game critique described the old deployed documentation; current main
already starts with staged decisions and no universal game-count requirement.
The current [evaluation protocol](../eval_protocol.md) now documents the ordered
sequential contract and conditional precision.

Validation and independent implementation review are recorded with this PR.
Tests exercise delayed prefix release, earliest crossing before a reversing
suffix, explicit look cadence, persisted suffix accounting, changed-spec refusal,
noncontiguous resume and fixed-N behavior. Synthetic CPU checks qualify the sequential control flow. The separately
registered GPU component above qualifies provisional capacity choices; it does
not run a match or alter production.
