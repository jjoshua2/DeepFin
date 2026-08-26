# Experiment ledger

## 2026-08-26 — S1 global-margin lazy-eval calibration on qsearch-DAG

**Question.** Should the preregistered S1 design in `docs/nnue_speed_plan.md`
be implemented for the qsearch-DAG provider? The registered design uses one
global margin `m = p99.9(|full - psqt|) + 10% slack` and kills the implementation
if fewer than 20% of logged stand-pat probes can be served by a PSQT bound.

**Population.** 201,671 stand-pat probes from the 467-position qsearch-DAG parity
corpus under the production qsearch configuration used by the calibration run.
This is a qsearch-DAG population. FastQ was not separately instrumented or
calibrated.

**Observed calibration.** The run reported:

- `p99.9(|full - psqt|) = 2551` internal units;
- registered margin `m = 2806.1` after the fixed 10% slack;
- held-out margin miss rate **0.0099%** (registered rejection boundary 0.2%);
- predicted bound-served probe rate **0.488%** (registered implementation floor
  **20%**);
- probe-window diagnostics: **41.31%** fully open and **44.12%** half-open;
- `m = 0` probe-level bound opportunity **34.07%**;
- residual diagnostics: median `|full - psqt|` **521**, mean **633**,
  `sd(full) ≈ 1950`;
- `full ~ psqt` diagnostic regression slope **0.9175**, Pearson **r = 0.9410**.

**Registered verdict.** **KILL the preregistered global-margin S1 implementation
for qsearch-DAG.** Its own fixed decision statistic is 0.488%, far below the 20%
implementation floor. No search implementation is needed to reach that verdict.

**Important limits discovered in review.** The calibration rows are probe-level
records and do not contain canonical DAG node identity or created/hit/upgrade
state. Therefore the 34.07% `m = 0` probe opportunity is **not** a saved-FC-
propagation fraction and must not be multiplied by the 27.2% fresh-propagation
wall share. Repeated probes may hit a node whose FULL value was already paid for.
A future wall-saving estimator must record node identity and replay the actual
first-probe/upgrade sequence.

The run also does **not** kill a FastQ-specific lazy evaluator or every possible
smaller/conditional margin. FastQ has a different node/window distribution, and
those alternatives were not the preregistered experiment. Any such revisit
needs a new population and decision rule rather than inheriting this verdict.

**Reproducibility status.** This entry was added during review because the first
version of PR #477 claimed the measurement was ledgered when this canonical file
was empty. The summarized statistics above are the figures used to author #477,
but the referenced calibration branch/raw dump and exact analysis command are
**not present in the current canonical repository state available during this
review**. Do not invent an artifact path or command. Recover the original raw
artifact, or rerun the registered calibration, before claiming raw-data
reproducibility beyond these recorded summary statistics.
