# NNUE speed plan — copy Stockfish's engineering, not its architecture

Successor to the measured profile of the qsearch labeling path (py-spy `--native`,
2,254 samples; banked and ledgered 2026-08-25, including the correction entry that
replaced the earlier wrong Amdahl inference). Status: PLAN — each section becomes
its own PR, ordering per §6. The qsearch-on-DAG retrofit and FastQ
(`docs/fastq_design.md`) cut *nodes*; this plan cuts *cost per node*.

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

## 2. PR-S1 — lazy eval (two-tier node value on the DAG)

SF skips the expensive positional network when the cheap material/PSQT part is
already far outside the window. Our complication: the DAG stores the static
value as a **window-independent node fact**, and lazy truncation is
window-dependent — storing a lazily-cut value would poison every transposition
reuse. A window-dependent fact must never enter the DAG
(`docs/fastq_design.md` §4.2), so laziness changes *when* the full value is
computed, never *what* is stored.

### 2.1 Stored facts and the accessor

- Node payload carries a value **tier**: `PSQT_ONLY` (cheap: PSQT accumulators
  only, no FC propagate) or `FULL`. Both are exact, position-intrinsic facts —
  the tier records how much of the (deterministic) evaluation has been computed
  for this position, not anything about any visit, window, or search path, so
  it is history-free in §4.2's sense. Upgrades are monotone
  (`PSQT_ONLY → FULL`); a `FULL` value is never recomputed. Nothing
  window-dependent is ever stored, which is what separates this from the §9
  non-goal (a TT with bound flags stores *search outcomes*; this stores
  degrees of completion of one pure function).
- Consumers never read the raw value field. The single accessor is
  `dag_value_for_window(node, alpha, beta) -> (kind, score)` where `kind` is
  `EXACT` (FULL value, computed now if needed) or `LOWER`/`UPPER` (a proven
  bound sufficient for this window, node left at `PSQT_ONLY`). Routing every
  consumer through one tier-aware accessor is what makes the no-leak claim
  hold *for the actual readers*, not just in principle: the retrofit provider
  and FastQ's §3.3 `stand_pat = dag static value` line both switch to this
  call, and **the S1 PR amends `docs/fastq_design.md` §3.3/§4.1 in the same
  commit** — a consumer reading the value field directly is the named mutant
  (test: a direct-read arm must diverge on a crafted PSQT_ONLY node).
- Tier upgrade is a second write to a published payload. It inherits the DAG's
  threading rule (GIL held; single-threaded enforced — see the history section
  of `docs/nnue_position_dag.md` for why that rule is written in blood): the
  upgrade happens under the same discipline, and the comment at the write site
  must say a future concurrent consumer needs the upgrade inside the same
  critical section as the probe.

### 2.2 Fail-soft semantics (exact, so two implementers match)

Let `m` be the calibrated margin (§2.3) and `p` the PSQT_ONLY value. The
calibration property is `|full − p| ≤ m` (empirical, not proven — §2.3).

- If `p − m ≥ beta`: stand-pat cutoff. Fail-soft return is **`p − m`** (the
  proven lower bound), NOT `p` — returning the unproven point estimate would
  smuggle an uncertified value into parent bookkeeping.
- If `p + m ≤ alpha`: fail-low; the score contribution is **`p + m`** (the
  proven upper bound) and search continues per the caller's recursion.
- Otherwise, and at every point where the position's own value is *returned*
  rather than merely cut on — terminal quiet/max-qply returns, and in-check
  nodes (which never stand-pat, per §3.3, so never take the lazy path at
  all) — compute `FULL`, store it, use it. **Only a cutoff may use a bound;
  every returned static value is FULL.**

### 2.3 Calibration, and the pre-committed kill

- The dump instruments the **retrofit provider** on its corpus runs and logs
  `(psqt_value, full_value, alpha, beta)` at every real stand-pat probe — the
  windows are recorded precisely so the skip prediction is computable; a
  windowless `(psqt, full)` dump cannot predict a window-dependent skip.
- `m` = observed p999 of `|full − psqt|` on that dump **plus slack fixed at
  10% of p999, chosen here, before the dump exists**. This is an empirical
  tail bound, not a proof: misses beyond `m` are possible off-distribution.
  Quantify the miss rate on a held-out half of the dump and report it in the
  PR; a held-out miss rate above 0.2% at the chosen `m` kills the margin and
  forces recalibration wider.
- Predicted skip rate = fraction of logged probes where `psqt ± m` clears the
  logged window (computable BEFORE any implementation, from the dump alone —
  the predict-the-exact-count rule). **Kill: predicted propagate-skip < 20% of
  probes ⇒ S1 dies at design time**, with the dump banked as the evidence.
- Mutant discipline for the margin itself: the `margin=0` arm must run on a
  fixture where `|full − psqt| > 0` is first *asserted* (else the mutant test
  is vacuously green), and `margin=∞` must be bit-identical to lazy-off.

### 2.4 Counters (the evaluate-once identity, redefined on purpose)

Two tiers split "evaluation" and the identity must not silently loosen (the
DAG's own history: a strict `==` once drifted to `≤` and went blind — see
`docs/nnue_position_dag.md`). S1 defines:

- `full_evals` — FC propagates. Identity: **`full_evals == nodes_at_FULL`**
  (monotone upgrade ⇒ at most one propagate per node, ever).
- `psqt_probes` — PSQT_ONLY bound uses (cutoffs served without propagate).
- The DAG's `state_inits + state_makes == node_count` is untouched.

Both identities are asserted at every stats read in the S1 tests.

### 2.5 Oracle discipline

Lazy is a knob, default-off. The retrofit's bit-identity corpus always runs
lazy-off. A separate corpus asserts the lazy arm's *search values* satisfy:
equal to lazy-off wherever no bound was used, and bound-consistent (the §2.2
inequalities, re-checked against a lazy-off replay) wherever one was.

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
is dead at design time. The bench's own prereg (its ledger entry, not this
file) carries the seating decision rule. Note the asymmetry: S1's tier scheme
is verifier-agnostic (every candidate net has a cheap PSQT part) and S2 is
net-independent, so neither is stranded by any bench outcome — only S3 is
our-net-specific.

## 5. The regression gate (rides with PR-S1, applies to all)

A `bench`-style fixed-corpus harness that **fails, not reports**: nonzero exit
on any node-count change (the correctness tripwire; value-changing knobs like
lazy are asserted by their own §2.5 oracle instead, and run through the gate
with the knob off) or on a µs/node regression **> 3%** vs the banked baseline.
Reports µs/node, evals/s, node counts, per-provider counters. No silent caps:
any corpus truncation is printed, not defaulted.

## 6. Sequencing

qsearch-on-DAG retrofit (in flight) → FastQ-4+ → **the S1 calibration dump**
(cheap instrumentation, no search change — it needs only the retrofit's
provider) → **S1-vs-S2 order decided by that dump**: build S1 first only if
its predicted propagate-skip saving (skip rate × 27.2%) exceeds S2's expected
8–14%; otherwise S2 first. (An earlier draft hard-ordered S1 first on a "~2×"
share ratio — the honest ratio is 27.2/18 ≈ 1.5×, and at S1's own 20% kill
floor its saving is ~5.4 points, *below* S2's expectation, so the order is a
measurement, not an assumption.) Then the verifier-net bench → S3 or its
funeral.
