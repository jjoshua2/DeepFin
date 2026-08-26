# NNUE speed plan — copy Stockfish's engineering, not its architecture

Successor to the measured profile of the qsearch labeling path (py-spy `--native`,
2,254 samples, banked; numbers below are from that run). Status: PLAN — each
section becomes its own PR, in the order given. The qsearch-on-DAG retrofit and
FastQ (`docs/fastq_design.md`) cut *nodes*; this plan cuts *cost per node*.

## The measured baseline

- `cae_nnue_propagate` (FC forward): **27.2%** of qsearch wall. Untouched by
  #469 — incremental state feeds it, it still runs per fresh eval.
- Accumulator arithmetic (add/sub rows + FullThreats delta): **~30%**.
- Move generation: **~18%** — sliders are ray-walkers; no magic/PEXT tables.
- Threat-feature maintenance: **~5.6%**.
- Flat ~5.6µs/node; #469's real-net win was only **1.047×** because our
  FullThreats deltas are wide where SF's HalfKA deltas touch 2–4 rows.

Eval ≈ 65–70% of qsearch wall. That is the target list, in profile order.

## PR-S1 — lazy eval (two-tier node value on the DAG)

SF skips the expensive positional network when the cheap material/PSQT part is
already far outside the window. Our complication: the DAG stores the static
value as a **window-independent node fact**, and lazy truncation is
window-dependent — storing a lazily-cut value would poison every transposition
reuse. A window-dependent fact must never enter the DAG
(`docs/fastq_design.md` §4.2), so laziness has to change *when* the value is
computed, never *what* is stored:

- Node payload carries a value **tier**: `PSQT_ONLY` (cheap: PSQT accumulators
  only, no FC propagate) or `FULL`. Both tiers are window-independent facts;
  the tier is part of the stored fact, upgrades are monotone
  (`PSQT_ONLY → FULL`), and a `FULL` value is never recomputed.
- At a stand-pat probe with window `(alpha, beta)`: if the node is `PSQT_ONLY`
  and `psqt_value ± lazy_margin` is entirely outside the window, use the bound
  and do NOT upgrade. Otherwise compute `FULL` (one propagate), store it,
  proceed. A transposition under a different window sees `PSQT_ONLY` and makes
  its own decision — correctness cannot leak between windows because only
  tiered exact facts are stored, never window outcomes.
- `lazy_margin` calibrated from data before the default is chosen: dump
  `(psqt_value, full_value)` pairs on a labeling corpus, set the margin at the
  observed p999 of `|full − psqt|` plus slack, and ledger the calibration.
  Predict the skip rate from the same dump BEFORE the PR is built
  (predict-the-exact-count rule) — if the predicted propagate skip is <20% of
  evals, the PR is not worth its complexity and dies at design time.
- **Oracle discipline**: lazy is a knob, default-off. The retrofit's
  bit-identity corpus runs with lazy OFF, always. A separate corpus asserts the
  lazy arm's *search values* differ only where a stand-pat bound was provably
  sufficient (the mutant: margin=0 must change fixture values; margin=∞ must be
  bit-identical to lazy-off).

## PR-S2 — magic/PEXT sliders

Replace ray-walking slider attack generation with PEXT bitboards (BMI2 is
present on the production CPU; keep the classic magic-multiply fallback for
distributed wheels, chosen at build time the way the existing SIMD paths are).
Feeds movegen (~18%) AND the SEE swap loop FastQ needs (x-ray reveal becomes a
table lookup). Acceptance: perft parity vs the existing generator on a fixture
set (exact node counts, the strongest oracle available), plus a measured
µs/node drop on the bench corpus consistent with the profile's movegen share —
state the predicted range in the PR before the measurement.

## PR-S3 — FullThreats delta cost (bench-gated, may die)

The single biggest structural difference from SF: their HalfKA delta touches
2–4 accumulator rows per move; our FullThreats delta recomputes wide threat
sets. Two candidate cuts — incremental threat-plane maintenance (only squares
whose attack sets a move can change), or narrowing the verifier input. **Do
not build this before the verifier-net bench reads out**: if the bench picks
the SF big net as the labeling verifier (the operator's registered prior), the
verifier path stops paying the FullThreats tax entirely and PR-S3 is dead at
design time. Sequence: bench first, S3 only on a bench outcome that keeps our
net in the verifier seat.

## The regression gate (rides with PR-S1, applies to all)

A `bench`-style fixed-corpus harness: deterministic position set, reports
µs/node, evals/s, node counts, and per-provider counters; run before/after
every speed PR and quoted in the PR body. Node counts are the correctness
tripwire (a speed PR must not change them; value-changing knobs like lazy are
asserted separately as above). No silent caps: any corpus truncation is
printed, not defaulted.

## Sequencing

qsearch-on-DAG retrofit (in flight) → FastQ-4+ → **S1 → S2** → verifier-net
bench → S3 (or its funeral). S1 before S2 because the profile says propagate
skips are worth ~2× more than movegen, and S1's calibration dump reuses the
retrofit's corpus machinery.
