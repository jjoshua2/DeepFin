# NNUE speed plan — copy Stockfish's engineering, not its architecture

Successor to the measured profile of the qsearch labeling path (py-spy `--native`,
2,254 samples; banked and ledgered 2026-08-25, including the correction entry that
replaced the earlier wrong Amdahl inference). Status: PARTLY SHIPPED — S2 measured
at −14.3% qsearch wall (#475), the **preregistered qsearch-DAG S1 global-margin
design was killed at design time** (§2), and S3 is still bench-gated (§4). The
qsearch-on-DAG retrofit and FastQ (`docs/fastq_design.md`) cut *nodes*; this plan
cuts *cost per node*.

## 1. The measured baseline

All from the banked profile (ledger 2026-08-25) unless marked:

- `cae_nnue_propagate` (FC forward): **27.2%** of qsearch wall. Untouched by
  #469 — incremental state feeds it, it still runs per fresh eval.
- Accumulator arithmetic (add/sub rows + FullThreats delta): **~30%**.
- Move generation: **~18%** — sliders are ray-walkers; no magic/PEXT tables.
- Threat-feature maintenance: **~5.6%**.
- Flat ~5.6µs/node across qply depths (same profile run); #469's real-net win
  was **1.047×** (ledgered bench, order-alternated).

Eval-side total ≈ **63%** of qsearch wall (27.2 + ~30 + 5.6 — the earlier
"65–70%" draft claim was arithmetic drift; use the sum of the measured parts).

## 2. PR-S1 — lazy eval (two-tier node value on the DAG) — ⚑ REGISTERED QSEARCH DESIGN KILLED 2026-08-26

**STATUS: DO NOT BUILD THE PREREGISTERED GLOBAL-MARGIN S1 FOR QSEARCH-DAG.** The
§2.3 gate fired on the population it named: predicted bound-served stand-pat
probes were **0.488%**, against the pre-committed **20%** floor, over 201,671
qsearch-DAG stand-pat probes. The registered margin was `m = 2806.1` (p99.9 of
`|full − psqt|` plus the preregistered 10% slack), with held-out miss rate
**0.0099%**.

That result is enough to kill **this registered design** because its kill rule
was explicitly probe-weighted and fixed before measurement. It is *not* evidence
for the stronger statements an earlier revision made:

- FastQ was **not** separately calibrated. Its SEE/delta/certificate/bounded
  recursion changes both the node population and the `(alpha, beta)`
  distribution, so qsearch's window frequencies are not FastQ's frequencies.
- The dump did not carry canonical DAG node identity / created-vs-hit state.
  Therefore probe skip rate cannot be multiplied by the 27.2% fresh-propagation
  wall share: repeated probes of one already-upgraded node do not pay another FC
  propagation. The earlier `34.07% × 27.2% = 9.27%` "oracle wall ceiling" is
  withdrawn.
- A smaller or conditional margin was not preregistered or swept. The current
  evidence does not prove that every possible lazy-eval policy is useless until
  the network architecture changes.

Useful diagnostics from the same qsearch-DAG dump remain informative, but are
**probe-population diagnostics, not wall-time ceilings**:

- **41.31%** of logged stand-pat probes carried a fully-open window
  (`alpha,beta = -200000,+200000`), and another **44.12%** were half-open. A
  fully-open visit cannot be served by a stand-pat bound at that visit.
- At `m = 0`, the probe-level skip opportunity was **34.07%**. This is useful as
  a window-distribution diagnostic only; without node identity and upgrade
  ordering it is not a saved-propagation fraction.
- `|full − psqt|` was p50 **521**, mean **633**, p99.9 **2551**, against
  `sd(full) ≈ 1950`. Regressing `full` on `psqt` gave slope **0.9175** and
  Pearson **r = 0.9410**, which rules out a simple scale mismatch as the reason
  the registered margin became wide.

The practical planning verdict is still simple: **do not spend an implementation
PR on the registered qsearch S1.** S2 already measured as a strong engineering
win. Revisit lazy evaluation only with a new preregistration and population —
for example a FastQ-specific dump or a genuinely different conditional-bound
scheme — and measure unique-node upgrade savings rather than converting probe
frequency directly to wall time.

SF skips the expensive positional network when the cheap material/PSQT part is
already far outside the window. Our complication: the DAG stores the static
value as a **window-independent node fact**, and lazy truncation is
window-dependent — storing a lazily-cut value would poison every transposition
reuse. A window-dependent fact must never enter the DAG
(`docs/fastq_design.md` §4.2), so laziness changes *when* the full value is
computed, never *what* is stored.

### 2.1 Stored facts and the accessor

This is the specification that would apply if lazy evaluation is revisited; it
is retained because it captures the correctness boundary independent of the
failed qsearch economics.

- Node payload carries a value **tier**: `PSQT_ONLY` (cheap: PSQT accumulators
  only, no FC propagate) or `FULL`. Both are exact, position-intrinsic facts —
  the tier records how much of the deterministic evaluation has been computed,
  not anything about a visit, window, or search path. Upgrades are monotone
  (`PSQT_ONLY → FULL`); a `FULL` value is never recomputed. Nothing
  window-dependent is stored.
- Consumers never read the raw value field. The single accessor is
  `dag_value_for_window(node, alpha, beta) -> (kind, score)` where `kind` is
  `EXACT` (FULL value, computed now if needed) or `LOWER`/`UPPER` (a certified
  bound sufficient for this visit, node left at `PSQT_ONLY`). Any future
  qsearch/FastQ consumer must route through this accessor rather than bypassing
  tier semantics.
- Tier upgrade is a second write to a published payload. It inherits the DAG's
  threading rule (GIL held; single-threaded enforced). A future concurrent
  consumer needs probe/upgrade publication inside the same critical section.

### 2.2 Fail-soft semantics (exact, so two implementers match)

Let `m` be the calibrated margin and `p` the PSQT_ONLY value. The calibration
property is `|full − p| ≤ m` (empirical, not proven).

- If `p − m ≥ beta`: stand-pat cutoff. Fail-soft return is **`p − m`**, not `p`.
- If `p + m ≤ alpha`: fail-low; the score contribution is **`p + m`** and search
  continues per the caller's recursion.
- Otherwise, and at every point where the position's own value is *returned*
  rather than merely cut on — terminal quiet/max-qply returns, and in-check
  nodes — compute `FULL`, store it, use it. **Only a cutoff may use a bound;
  every returned static value is FULL.**

### 2.3 Calibration, and the pre-committed qsearch kill

- The registered dump instruments the **qsearch-DAG retrofit provider** and logs
  `(psqt_value, full_value, alpha, beta)` at every stand-pat probe. The windows
  are required because a windowless `(psqt, full)` dump cannot predict whether
  a visit is bound-serviceable.
- `m` = observed p99.9 of `|full − psqt|` plus slack fixed at **10% of p99.9**
  before the dump. Held-out miss rate above **0.2%** kills the margin and forces
  recalibration wider.
- Registered predicted skip rate = fraction of logged **probes** where
  `psqt ± m` clears the logged window. **Kill: predicted probe skip < 20% ⇒ the
  registered qsearch S1 dies at design time.** It measured **0.488%**, so that
  decision is closed without implementing the search accessor.
- ⚑ This registered probe metric is sufficient for its own go/no-go rule, but
  it is **not a wall-time estimator** for an evaluate-once DAG. To estimate
  saved FC propagations, a future dump must also identify canonical nodes and
  whether each visit created, hit, or upgraded the node, then replay the actual
  first-probe/upgrade sequence.
- Mutant discipline for the margin itself, if revisited: `margin=0` must run on
  a fixture where `|full − psqt| > 0` is first asserted, and `margin=∞` must be
  bit-identical to lazy-off.

### 2.4 Counters (the evaluate-once identity, redefined on purpose)

A future implementation would define:

- `full_evals` — FC propagates. Identity: **`full_evals == nodes_at_FULL`**.
- `psqt_probes` — PSQT_ONLY bound uses.
- The DAG's `state_inits + state_makes == node_count` remains untouched.

Both identities must be asserted at every stats read.

### 2.5 Oracle discipline

Lazy would remain a default-off knob. The retrofit's bit-identity corpus runs
lazy-off. A separate corpus must assert the lazy arm's *search values* are equal
to lazy-off wherever no bound is used and bound-consistent wherever one is.

## 3. PR-S2 — magic/PEXT sliders

Replace ray-walking slider attack generation with PEXT bitboards (BMI2
verified present on the production CPU via cpuinfo, 2026-08-26; classic
magic-multiply fallback for distributed wheels, chosen at build time like the
existing SIMD paths, with PEXT-vs-fallback parity asserted in CI on the same
fixtures). Feeds movegen (~18%) AND the SEE swap loop FastQ needs (x-ray
reveal becomes a table lookup on the same attack tables).

Acceptance, pinned now: perft node-count parity vs the existing generator on
the standard perft suite plus crafted castling/EP/promotion fixtures (exact
counts — the strongest oracle available); predicted wall win stated in the PR
before measurement, with the prediction bounded by the profile's movegen
share: **expected 8–14% qsearch-wall reduction** (18% share, minus residual
table-lookup and non-slider cost). Below 5% measured ⇒ report as a miss
against the prediction, investigate before merging.

## 4. PR-S3 — FullThreats delta cost (bench-gated, may die)

The single biggest structural difference from SF: their HalfKA delta touches
2–4 accumulator rows per move; our FullThreats delta recomputes wide threat
sets. Candidate cuts — incremental threat-plane maintenance, or narrowing the
verifier input. **Do not build before the verifier-net bench reads out**: if
the bench seats the SF big net as the labeling verifier (the operator's
registered prior), the verifier path stops paying the FullThreats tax and S3
is dead at design time. The bench's own preregistration carries the seating
decision rule. S2 is net-independent; S3 is our-net-specific. The qsearch S1
verdict above is already closed for its registered population and does not need
the verifier bench.

## 5. The regression gate (specified with S1, applies to all)

A `bench`-style fixed-corpus harness that **fails, not reports**: nonzero exit
on any node-count change (the correctness tripwire; value-changing knobs are
asserted by their own oracle) or on a µs/node regression **> 3%** vs the banked
baseline. Reports µs/node, evals/s, node counts, per-provider counters. No silent
caps: any corpus truncation is printed, not defaulted.

## 6. Sequencing

qsearch-on-DAG retrofit ✅ (#472) → FastQ-4+ ✅ (#473) → registered qsearch-DAG
S1 calibration ✅ → **registered qsearch S1 killed by its own 20% probe gate** →
S2 (#475), which measured **−14.3% qsearch wall** and remains the next speed win
to land.

The calibration does **not** establish “S1 never” for FastQ or every possible
conditional margin, and it no longer tries to compare a probe-weighted `m=0`
rate to S2's wall-time result. Those are separate experiments if they ever
become worth running. Given FastQ's already tiny tactical work, there is no
current reason to schedule them ahead of higher-value work.

Remaining after S2/readout integration: the verifier-net bench → S3 or its
funeral (§4), plus whatever the production-Gumbel FastQ readout says about
quality, persistence and CPU scheduling.
