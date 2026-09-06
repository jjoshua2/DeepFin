# Graded TailRL objective on the bootstrap corpus

Follow-up to [PR #517's diagnostic](2026-09-05-tailrl-policy-coverage.md).
The user requested testing its idea on the existing 20M bootstrap data while
the global BT4 and SF-close experiments continue. This record covers preparation
and a bounded CPU mechanism screen, not a training or playing-strength result.

## Question and distinction

Can a graded upper-tail objective provide a useful update direction beyond
ordinary sharp teacher cross-entropy and expected mover reward?

The original union-event loss does not protect each action inside its event.
For a single best-move event it is ordinary one-hot cross-entropy. A finite-order
binary variant only rescales that per-position gradient. Use original graded
per-move qualities to test a materially different objective; neither BT4 prior
probabilities nor float16 sharpened SF targets reconstruct those qualities.

`finite_tail_loss` implements the negative finite-order objective in
[TailRL, equations 23–25](https://arxiv.org/html/2609.02987v1): sum the negative
Best-of-k reward objectives with weights 1/k, for k=1 through a fixed order T.
For known finite actions, sort reward levels and exactly integrate their actual
interval widths. Retain the policy-independent cost above the largest reachable
reward; do not normalize each position's reward range or drop empty thresholds.
T=1 equals one minus expected reward. T=32 is the single proposed graded-tail
screen. This enumerates fixed teacher-labelled actions; it does not run the
paper's sampled rollout estimator or equate Best-of-k samples with MCTS visits.

## Bounded CPU screen

Freeze 128 rows sampled without replacement at seed 20260906 from the first
published bootstrap shard, source-qualified game/ply identities, actual encoded
inputs, legal masks, original all-legal-move d9 effective-centipawn scores, and
S0/G20T05 checkpoint logits. Use the corpus's existing .006/120 WDL mapping and
reward `(W - L + 1) / 2`. Require exact move coverage and row alignment; missing
scores are not zero rewards. Keep original SF targets alongside the quality bank.

These are seen training positions and final cross-recipe checkpoints. The screen
does not measure temporal collapse, held-out generalization or representative
corpus-wide effects. Bank selection, raw source identity and runtime before the
readout under `scratchpad/tailrl_bootstrap_v1`. CPU only, two threads at low
priority, at most 15 wall-clock minutes for collection. Preserve all GPU jobs,
generation, original checkpoints and source data. A collection failure leaves
its partial evidence and does not trigger an automatic larger sample.

On each model's same logits, compare legal-masked SF target CE, a further
sharpened CE control, expected reward (T=1), and graded tail (T=32). Inspect
per-position gradient norms, cosine and residual after norm matching. Report
directional changes in probability on rare high-quality and near-best moves,
with cohort definitions fixed before reading gradients. Zero or nearly identical
directions would make a new full epoch difficult to justify on this evidence.
Distinct directions justify considering a training ablation, not a strength claim.
No IID position confidence intervals or automatic recipe promotion.

## Subsequent decision

If the mechanism screen warrants training, preregister a bounded matched-start
ablation that separates graded-tail targeting from ordinary reward optimization
and gradient magnitude. Preserve the chosen base policy loss and all other heads;
choose the auxiliary coefficient and budget before training. The existing BT4
and SF-close comparisons remain queued and retain their frozen runtimes.

Evaluate any trained candidate at declared shallow and deeper matched-search
budgets against its matched control. A one-node quality score, entropy change,
or IID sampling curve cannot decide search scaling. Fresh-data/seed confirmation
is a later requirement for the selected training recipe.

## Qualification so far

The original installed-repository integration test failed because `fixture.zarr`
was outside the shared `shard_*.zarr` discovery contract. Using a real shard name
made all 42 original tests pass, including actual checkpoint loading and lazy
Zarr collection. The new finite-objective helper passed 24 focused numerical
tests, including independently enumerated rollout expectations and gradients.
Independent review and whole-repository lint are recorded after completion.

## Completed first mechanism readout

The bank completed in 76.7 seconds of elapsed time on CPU: 128 positions from 43 games, with
complete legal-move d9 scores, exact raw-row joins and unchanged model/source
hashes. The largest reconstructed SF-policy discrepancy was 0.000241593,
within the preregistered float16 tolerance. No network training occurred.

| Observation | S0 logits | G20T05 logits |
| --- | --- | --- |
| Mean T32 gradient cosine with SF CE | 0.628 | 0.585 |
| Mean T32 gradient cosine with expected reward/T1 | 0.943 | 0.949 |
| T32 unit descent increases near-best union mass | 117/118 | 116/118 |
| T32 unit descent decreases rare-near-best mass | 12/15 | 11/15 |
| SF CE unit descent decreases that rare mass | 2/15 | 3/15 |

The gradients differ from CE and are substantially closer to expected reward.
Improving a quality-set's total mass can still suppress particular rare moves.
The rare cohort contains only 19 moves in 15 positions from 12 games, seven of
them exact SF-best moves. This is not evidence for rejecting TailRL or predicting
a playing loss. These are derivatives under unit changes to logits, not actual
parameter updates or trained-model improvements.

Ten positions have exactly constant mapped rewards (nine all zero, one all one),
so both finite-tail objectives have zero gradients there. Six still have differing
raw centipawn scores: the existing WDL map saturated. Treat the reward mapping
as an experimental choice, not a faithful representation of every score gap.

Independent recomputation from the bank using a separate analytical gradient
formula agreed within 3.59e-16. Evidence: `scratchpad/tailrl_bootstrap_v1/bank.npz`
(SHA256 `541a353090e375b528e27ea6c58df770651b214f04e49ebea50945c3d6cf444e`)
and `gradient_readout_v1/report.json`
(SHA256 `ca21e70ee045f1b9dfa3b1244d42cd59bf822229beb82b2b77669d82627e344b`).
All 66 focused tests and whole-repository lint passed on the isolated checkout.
The initial sandbox checker could not discover its interpreter dependencies;
host discovery with the existing CPU environment resolved that environmental
failure. Native sources matched the borrowed build artifacts; no rebuild or
dependency upgrade was performed.

## Teacher source, mixture and objective must be considered together

The user emphasized that an SF-only reward screen does not settle how the loss
interacts with BT4 supervision or with the still-unresolved mixing recipe. The
two checkpoints above do not constitute that training ablation: their targets
differ, but the gradient probe used the same SF quality ruler on both.

Extend the same fixed bank with actual SF, exact-tie, global raw/sharpened and
SF-close supervision targets, plus prospective BT4-only and a larger sharpened
BT4 mixture. Reuse the banked logits. Compare each CE direction with the fixed
SF quality objectives to expose agreement or conflict without new inference.
Keep materialized/trained recipes distinct from prospective arithmetic targets.

This examines the data axis cheaply; it still cannot choose a winning training
combination. Subsequent training comparisons should cross selected teacher/mixing
recipes with the loss intervention at matched budgets, followed by matched
search-budget comparisons. Do not first declare a mixture optimal under CE and
assume that answer transfers to another loss. BT4 policy probabilities provide
supervision, not independently measured move rewards; any BT4-derived reward
proxy needs its own explicit definition and interpretation.

The [bootstrap roadmap](2026-09-06-bootstrap-research-roadmap.md) connects this
method study to the user's intended RL restart in roughly one to two months.

## Completed teacher interaction screen

Eight supervision targets were banked on the same rows: SF, exact ties, raw
and sharpened 80/20 global mixtures, SF-close, pure raw/sharpened BT4, and a
prospective 50/50 sharpened global mixture. The last three have no trained
checkpoint in this comparison. The SF-close target is materialized, but its
checkpoint is also absent from this bank. Exact input/row/legal alignment passed.

Comparing their CE gradients with the same SF-quality T32 gradient gave:

| CE target | Opposing direction on S0 logits | Opposing direction on G20T05 logits |
| --- | --- | --- |
| SF | 12/118 | 19/118 |
| SF-close | 6/118 | 7/118 |
| Sharpened 80/20 mixture | 13/118 | 16/118 |
| Prospective sharpened 50/50 mixture | 15/118 | 14/118 |
| Pure raw BT4 | 66/118 | 74/118 |
| Pure sharpened BT4 | 45/118 | 46/118 |

Here "opposing" means a negative inner product between two local logit gradients.
It is not a training outcome or a teacher-strength ranking. Pure raw BT4 CE also
increases the frozen rare-near-best mass in all 15 eligible positions on both
models. The SF-based reward objective and BT4 supervision can therefore pull in
different directions; greater agreement with SF is not automatically desirable.

No BT4 probabilities were used as rewards. This screen varies CE supervision
while holding the SF reward ruler fixed, and uses only two existing checkpoints.
It does not test a trained teacher-by-loss factorial experiment or search scaling.
Independent recomputation matched gradients, cosines and derivatives within
1.2e-14. The analysis reused banked logits and took 2.62 seconds of elapsed time
on CPU, with no new inference.

Evidence under `scratchpad/tailrl_bootstrap_v1/teacher_targets_v1`:
`targets.npz` (SHA256 `325ec0ac868ea50863d6bbe1e55b54ec6a286e955153675e08874ec49c204175`),
`interaction_readout_v1/report.json`
(SHA256 `f52671260539ebfd970fe891c3d28d595d4c64baf279c37a880497626d50a0f1`),
and `interaction_independent_review.json`
(SHA256 `540760a5a3265a01e2c7352a569094e035c7e1d87fdbdd467d891bcad7edb8ad`).
