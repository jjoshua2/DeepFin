# Pre-registered predictions — can the bad tail be SCREENED at label time?

Written 2026-08-16 BEFORE running `tailscreen.py`. Author: main session.

## Why this exists

The 2026-08-15 phase-2 entry closes: *"A full-width, non-fabricated teacher is still the
only lever identified that reaches the tail. No arm launched; this needs an explicit go."*
The blocker is COST — MultiPV 40 costs 7x (`sf_multipv_width_costs_7x`), and a blanket
re-label is unaffordable.

**A blanket re-label is unaffordable. A TARGETED one may not be.** If the bad tail is
predictable from features that are already stored on every row — no extra SF compute —
then full-width re-labeling can be spent only where it changes the target, and the cost
multiplier collapses from 7x to `1 + f*6` for a selected fraction `f`.

Two possible good outcomes, and the second is cheaper than the arm it was meant to enable:
- **(A) SCREENABLE** ⇒ targeted full-width re-label becomes affordable.
- **(B) ALREADY VISIBLE** ⇒ if the *stored shallow* `sf_cp_regret_tgt` already separates
  the bad tail, then no re-labeling is needed at all: the rows can be DOWN-WEIGHTED using
  a number we already have. That is a loss-weight change, not a teacher change.

A third outcome is the informative negative:
- **(C) NOT SCREENABLE** ⇒ the bad tail is invisible to everything available at label
  time, the repair arm genuinely requires blanket full width, and it stays cost-blocked.
  This kills the targeted-repair idea cheaply, which is the point of running it first.

## Population

The phase-2 split: `dQ = Q[:,0] - Q[:,1]` (target minus SF), BAD = `dQ <= -0.10`.
Ledger records n=153 BAD / 66 GOOD / 788 non-tail (total 1007 of the 2000 sampled rows).
⚑ First check is that I reproduce those three counts exactly. If I do not, I am on a
different population and every number below is void ([[same_name_different_population]]).

## Predictions (pre-committed)

| # | quantity | prediction |
|---|---|---|
| P1 | reproduce n=153 / 66 / 788 | EXACT match, or I stop and re-derive the filter |
| P2 | AUC of `sf_cp_regret_tgt` for BAD membership | **0.75-0.88** — high, but NOT ~1.0 |
| P3 | AUC of `tgt_listed == False` | 0.62-0.75 |
| P4 | AUC of `n_legal` | 0.55-0.63 (weak) |
| P5 | AUC of `top1_mass` | 0.50-0.60 (weakest; sharpness is not wrongness — this is the "sharp and wrong" falsification carried forward) |
| P6 | best combined screen, top **15%** selected | captures **>=50%** of the BAD rows |
| P7 | cost multiplier at the P6 operating point | ~`1 + 0.15*6` = **1.9x**, vs 7x blanket |

## Pre-committed reading rule

- **SCREENABLE** if P6 holds: >=50% of BAD rows captured in the top 15%. Then outcome (A).
- **ALREADY VISIBLE** if `sf_cp_regret_tgt` alone reaches AUC >= 0.85 **and** the BAD rows
  it selects have a materially worse mean dQ than the rows it does not. Then outcome (B),
  and the cheap down-weight arm is preferred over the expensive re-label arm.
- **NOT SCREENABLE** if the best combined AUC < 0.65. Then outcome (C) and the arm stays
  blocked; I record that and do not propose it again without a new lever.

## ⚑ Known trap, priced in advance

`sf_cp_regret_tgt` is derived from the SAME MultiPV-6 data whose inadequacy is the subject.
It is **not circular with dQ** — dQ is measured against a deep full-width Q, the regret is
the shallow narrow one — but a high AUC here means "the shallow teacher already knows its
own target is bad", which is a claim worth stating explicitly rather than sliding past.
The interesting failure is the opposite one: **rows the shallow teacher is CONFIDENT about
and is wrong about** are the ones no screen built from it can ever see. I will report that
subgroup separately (`sf_cp_regret_tgt` ~ 0 AND `dQ <= -0.10`), because it is the
irreducible residual and it bounds what any label-time screen can achieve.
