# Recipe selection after the completed global screen

This is a September 6 decision note, written while C20T05 is training and before
any of its arena outcomes exist. It does not alter the frozen SF-close protocol
or queue another GPU job. The intended deliverable remains a strong, confirmed
bootstrap and an exercised RL restart in roughly one to two months.

## What the completed screen changes

The registered raw-versus-sharpened global comparison favors sharpened BT4 at
all three search budgets against common S0. Prioritize T0.5 for the next global
mixture-weight comparison. This is evidence at 20% BT4, the current CE objective,
one training seed and this search configuration; it does not eliminate raw BT4
at another mixture weight or under another objective.

The additional `global_vs_exact_ties_exploratory.json` uses the same six completed
G20T05/E0 banks. The G20T05-minus-E0 common-S0 score differences are:

| Simulations | Difference | Nominal paired 95% interval |
| --- | --- | --- |
| 25 | +0.75 percentage points | [-2.40, +3.95] |
| 100 | +2.30 percentage points | [-1.00, +5.55] |
| 400 | +2.75 percentage points | [-0.30, +5.70] |

The difference between their 400-minus-25 score changes is +2.00 percentage
points [-2.50, +6.30]. These are post-hoc, opening-pair bootstrap estimates,
not direct-match Elo or retraining uncertainty. Neither superiority to E0 nor
better relative scaling than E0 is established. Keep both recipes in contention.

## What the next comparisons need to resolve

Finish the already registered C20T05-versus-E0 25/100/400 matches. C20T05 combines
SF-close support expansion and teacher sharpening; its result cannot attribute
an effect to widening alone. The existing sharpened-exact-tie corpus is a possible
attribution control if that distinction would change the recipe choice, not a
mandatory extra epoch before every decision.

A direct leading-global versus competitive-close/exact-tie match can resolve the
current ranking more directly than another common-S0 match. Select its opponent,
budgets, fixed horizon and compute cap after the C results. Do not add Elo from
different reference opponents to infer that result. Preserve the reserved fresh
confirmation opening bank until a confirmation design is fixed.

For substantive new target research, a coarse larger sharpened-global dose is a
useful candidate: it addresses whether the present 20% BT4 choice leaves strength
unused. Pure BT4 remains a broader teacher alternative, especially for interaction
with a new objective. Neither is ruled out by agreement with SF or the small
gradient bank. Select a small comparison that can change the bootstrap decision,
rather than treating every target banked for diagnostics as a queued training arm.

For objective research, PR #517 still needs real offline label plumbing and
step-cost qualification before a training comparison. Compare its graded order
against the same-reward expected-value control and ordinary CE, and retain a
teacher/mixing contrast where it tests an actual interaction. No current gradient
diagnostic establishes a stronger checkpoint or a search-scaling improvement.

Final recipe recommendation still requires independent training-seed and fresh-
opening evidence. Larger-scale training and the bounded RL restart remain part
of the deliverable; continued recipe exploration must justify delaying them.
