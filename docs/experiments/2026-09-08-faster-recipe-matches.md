# Faster recipe matches with ordered sequential decisions

**Status:** future protocol and tool preparation. The existing H20 package keeps
its original fixed banks. Its C100 and G100 comparisons are complete; the
registered C400 probe is outside this readout. No match used the new sequential
path, and no rolling throughput gain has been measured.

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
wall time. A separate bounded component probe is prospective at this snapshot.

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
noncontiguous resume and fixed-N behavior. Synthetic CPU checks qualify control
flow; no new match, inference or production change is part of this record.
