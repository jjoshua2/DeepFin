# October 12, 2026 — Frozen-root I-opt mechanism screen

**Status: bounded frozen-root diagnostic completed by the science owner; corrected CPU readout verified October 13. No operator adoption, training, arena or playing-strength claim.**

Paper: [arXiv:2610.01566](https://arxiv.org/html/2610.01566).
Parent source: main at 265adece34a6dbe8d4a32122fff4e5c65a5d84c0.
Reuses [search_gain_probe.py](../../scripts/search_gain_probe.py), the actual native
Gumbel root returned policy and the all-legal [one-ply backup machinery](../../chess_anti_engine/mcts/one_ply.py)
where useful for a later controlled extension.

## Scientific question and constraints

Discriminate (A) exaggerated search confidence when very small Q spreads are
rescaled to [0,1], (B) a good move absent from root search visits, and (C)
visited children whose Q rankings disagree with an independent stronger
reference. These can overlap. The first scope is a frozen tree and frozen
checkpoint, **not** a comparison of RL policies adapted to different target
operators. There is no new inference head or production setting.

No experiment has established that an I-opt-like operator improves DeepFin.
The variance input in this PR is explicitly an *uncalibrated independent-Q-error
proxy*, not the paper's trained Q head, epistemic variance, or proof that MCTS
errors are independent. For an unvisited move, DeepFin's mixed-value completion
shares the same estimate with other unvisited moves, so the independence
assumption is particularly poor there.

## Intervention matrix (same recorded native search for every arm)

1. **current:** native C returned completed-Q policy, reconstructed with
   its own root value, visited child Qs, all legal priors, mixed-value FPU,
   and root q scale. A max-abs comparison against the actual C distribution
   must pass before a root is included. This is NOT raw visit counts.
2. **unscaled:** completed-Q with the same original Q and node-scale factor,
   but without dividing by max-min Q range. No tree/search/ranking changes
   beyond the target distribution.
3. **iopt_proxy:** pi(a) proportional to prior(a) times
   exp(reliability(a)*(Qcompleted(a)-E_prior[Qcompleted])/beta).
   reliability = tau2/(tau2+variance_advantage), with independent-action
   variance proxy variance_scale / max(1, visits), and an explicit larger
   unvisited factor. Advantage variance includes the contribution of the
   prior-weighted baseline (sum(p² var) + (1-2p) var).
4. **iopt_matched_kl:** same action-specific I-opt advantages, but scale
   its score per position to match the current target's KL to the prior
   whenever reachable. This checks whether action-specific weights help
   beyond merely softening the target. Unreachable roots are excluded from
   this arm's aggregate and paired comparisons; `current_matched_kl_subset`
   reports the baseline on exactly the same eligible rows. The scorer records
   exclusion counts. Centered, normalized score bracketing preserves this
   control under tiny score scales and large offsets; tied maxima retain
   their prior probability ratios at the limiting KL.
   There is deliberately **no** per-position KL-matched unscaled arm:
   unscaled and normalized completed-Q differ by only a positive scalar,
   so per-position KL-matching would return the same distribution.

Starter parameters beta=0.1, tau2=0.08, variance_scale=0.04,
unvisited_multiplier=4 are heuristic, *not fitted, calibrated or promoted*.
Treat any favorable screen as conditional on these choices; preregister a
second frozen held-out bank before adjusting them.

## Source and correctness gates

Invoke the existing probe with --operator-trace and --out. Each JSONL row then
retains legal UCI/actions, exact root prior, native visited Q or null,
native child visits, raw root W-L prediction, root scale, actual returned
policy and **only** directly reported SF MultiPV cp for each action; an
unreported move is null, NOT the existing move_regrets() worst-line floor
or the stored P0 midpoint. The original metadata header holds the
checkpoint, config-derived search shape, seed, and tablebase state.

The current probe regenerates boards from FEN and therefore does not retain
all training history; it refuses dynamic-relation checkpoints. The screen
must NOT claim production equivalence until the identical history/encoding
route is qualified. Operator capture refuses Syzygy root probing because the
separate TB overwrite is not included in the logged raw root WDL.
A >5% native policy parity failure rate disqualifies the result; every
failure is counted and must be investigated before improving the instrument.

No "best move missed" assertion is stronger than **best *listed* reference
move unvisited**. Direct cp is partial, often only top MultiPV moves.
Expected cp regret is reported only on roots with *all legal moves directly
scored*. Other roots retain reference-unknown target probability mass,
plus paired top-one regret only where BOTH selected actions are scored
by the independent reference. Each directly labelled action's regret is
clipped to [0, 1000] cp before either metric, matching the existing audit cap;
raw cp labels are preserved. This bounds mate-valued cp conventions without
turning mate distance into ordinary centipawns. Do not rank policies using a synthetic cp
tail. This also precludes claiming full expected-regret strength for
the entire population unless labels become complete.

Use independent SF-depth stability checks on selected disagreements;
SF is a reference, not an oracle, especially for horizon/fortress cases.

## Bounded first collection

These are opt-in commands on a frozen checkpoint copy (no live YAML edit).
Validate actual search shape and source lineage before running:

    PYTHONPATH=. python scripts/search_gain_probe.py \
      --checkpoint /absolute/frozen/checkpoint/trainer.pt \
      --shape main_selfplay \
      --shape-yaml /absolute/actual/live-config.yaml \
      --sims 32,64,128,256 \
      --positions 1000 --batch-size 32 --seed 42 \
      --gpu-mem-fraction 0.25 --operator-trace \
      --out /absolute/experiment-root/roots.jsonl

    PYTHONPATH=. python scripts/iopt_operator_screen.py \
      --input /absolute/experiment-root/roots.jsonl \
      --out /absolute/experiment-root/screen.json

Do not run these on the active GPU until the current workload, memory and
canonical lease are checked. Stop at 1,000 positions and 256 sims initially;
do not silently expand to a large bank or consume Stockfish label CPU. The
second command is CPU-only, reads exactly the banked search observations and
may be rerun to test a different explicit parameter without a new search.

## Readout, uncertainty and interpretation

Use paired differences per simulation rung, root/FEN-cluster bootstrap,
positive centipawns meaning a candidate reduced regret. Report counts of
all actual reference-labelled moves, mass on unknown moves and total parity
exclusions. The 95% intervals are descriptive (not independent chess games);
the same positions appear across rungs. For root-candidate failure cases,
bank examples with source/position identity before adjudicating.

Primary mechanism flags: (1) current KL >=0.3, completed-Q range <=0.03,
current covered top-one regret >=50 cp; (2) strongest direct SF-listed move
has zero visits; (3) visited Q chooses a move at least 50 cp worse than
the best *directly labelled visited* move. These are diagnosis thresholds,
not a mutually exclusive classification or proof of an Elo effect.

**Promotion gate:** require >=200 roots with paired, directly labelled top-one
choices at the 256-sim rung; a candidate should improve top-one cp regret by
at least 5 cp with a bootstrap lower bound >0, **and** not increase full-label
expected regret where fully labelled. If full-labelled rows are insufficient,
classify the quality comparison as *unread* and consider targeted independent
all-move labeling on a frozen preselected subset; never fill unknown tails.
Even a promoted offline arm must still pass bounded offline training and then
paired search-simulation and wall-time arenas before any Elo claim.

**Kill/no-go:** a parity or lineage failure, mismatched histories, label
coverage that invalidates the primary comparison, or no stable positive
signal at the declared horizon. A negative result is specific to the tested
checkpoint/budget and heuristic parameters, not a general rejection of I-opt.

## Next research stage, NOT implemented here

If search's best-listed moves are frequently unvisited, reuse all-legal
one-ply evaluations as a counterfactual Q estimate for unvisited actions,
then compare with a hypothetical strong-reference replacement to distinguish
missing information from value-network blindness. Keep successor-history
correct and replay-identical; there is not yet a one-pass neural Q head.
If visited Q is biased, calibrate error vs visits and search budget before
using a visits-only variance proxy for production. Root sequential-halving
elimination history is not banked here, so **premature elimination is not
separately measured**.

## Artifacts and status

- Source: scripts/search_gain_probe.py, optional --operator-trace.
- Offline evaluator: chess_anti_engine/eval/iopt_operator.py and
  scripts/iopt_operator_screen.py.
- CPU guards: tests/test_iopt_operator_screen.py.
- Bulk traces remain outside git with SHA256 provenance. The compact corrected
  [CPU readout](artifacts/2026-10-13-iopt-corrected-readout.json) records the raw
  bank identity, capture plan, checkpoint/config/audit identities and parameters.
- Independent code review is performed separately from the author. Synthetic CPU
  tests do not establish native C parity; the prior bank retains its actual parity
  checks and excludes mismatches.


## October 13 corrected readout of the existing bank

The science owner previously captured 1,000 fixed roots at 32/64/128/256 simulations
using `runtime-fixed-v1`. Its final diagnostic receipt retained the completed
collection and reported no adoption. This PR completion collected no new roots,
queried no teacher, and ran no GPU work. It reused exactly the existing 4,000-row
bank (SHA256 `b9a23bbcd6423fc34dfd7be3f4221d33ea9455af290c17756f7e4523ef7d0926`).
The corrected scorer accepts 998 roots per rung and retains eight total native
parity exclusions, unchanged from the original CPU report. All accepted roots in
this bank have reachable KL matches; the unreachable-subset controls are exercised
by separate CPU regression fixtures. Full legal labels cover only 117 roots.

At the preregistered 256-simulation horizon, positive gain means lower capped regret:

| Arm | Paired top-one gain cp (95% interval) | Full-label expected gain cp (95% interval) |
| --- | --- | --- |
| Unscaled | 1.916 (-0.923, 5.361) | -16.571 (-31.204, -4.460) |
| I-opt proxy | 1.046 (-2.742, 4.717) | -17.998 (-33.754, -4.124) |
| I-opt matched KL | 0.035 (0.000, 0.105) | -0.059 (-0.191, 0.009) |

**No-go under the registered rule:** no arm reaches a 5 cp paired top-one gain with
a positive lower bound and nonworsening full-label expected regret. This preserves
the science owner's previous verdict. It is conditional on the frozen checkpoint,
heuristic variance proxy, FEN-only history, partial MultiPV reference and capped
mate-valued cp conventions. It does not establish an Elo effect or reject I-opt
generally. Production target/search behavior remains unchanged.
