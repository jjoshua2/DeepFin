# NNUE speed plan — copy Stockfish's engineering, not its architecture

Successor to the measured profile of the qsearch labeling path (py-spy `--native`,
2,254 samples; banked and ledgered 2026-08-25, including the correction entry that
replaced the earlier wrong Amdahl inference). Status: PARTLY SHIPPED — S2 measured at −14.3% qsearch wall (#475), **S1 killed
at design time** (§2), S3 still bench-gated (§4); ordering per §6. The qsearch-on-DAG retrofit and FastQ
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

## 2. PR-S1 — lazy eval (two-tier node value on the DAG) — ⚑ KILLED 2026-08-26

**STATUS: DEAD. The §2.3 gate fired; the whole section below is kept as the
spec the gate was written against, not as work to do.** Predicted
propagate-skip **0.488%** against the pre-committed 20% floor, on a banked
201,671-probe dump (ledger 2026-08-26; `m` = 2806.1, held-out miss 0.0099%, so
the margin is honest — it is not a calibration that can be retried wider).

⚑ **Do not re-open this on the 0.488%.** That number reads as "the margin came
out too wide, tune it", and the kill is not that. Two facts from the dump close
the lane structurally:

- **41.31% of stand-pat probes carry a fully-open window** (α, β = ∓200000),
  and 44.12% are half-open. A fully-open window is uncuttable by *any* margin,
  so ~41% of the population is unreachable before precision enters the
  argument at all.
- **The oracle ceiling — skip at `m = 0`, a hypothetically perfect
  zero-uncertainty margin — is 34.07%**, worth `34.07% × 27.2% =` **9.27% of
  qsearch wall. S2 measured 14.3%.** Even a *perfect* lazy eval loses to a PR
  that is already built and measured.

MECHANISM: our PSQT term is not a cheap approximation of our own evaluation.
`|full − psqt|` is p50 **521**, mean 633, p99.9 2551, against `sd(full)` ≈
**1950** — the residual is a third of the signal's own scale. Regressing `full`
on `psqt` gives **slope 0.9175, Pearson r 0.9410**: the two are on the same
scale, checked precisely because a scale mismatch is the one defect that would
have faked a wide margin and produced a false kill. SF's lazy trick works
because HalfKA's PSQT part *is* most of its eval. Ours is not; the positional
network carries the signal.

Re-open only if the net architecture changes such that PSQT carries most of the
eval — and then re-measure the **window** distribution first, because a
skip-rate metric over a population that is 41% unbounded windows is capped
before any margin is chosen.


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

qsearch-on-DAG retrofit ✅ (#472) → FastQ-4+ ✅ (#473) → the S1 calibration
dump ✅ (banked, ledger 2026-08-26) → **S1-vs-S2 order RESOLVED BY THAT DUMP:
S2 first, and S1 never** (§2). S2 (#475) measured **−14.3% qsearch wall**,
inside the §3 band; S1's *oracle ceiling* was 9.27%, so the ordering question
answered itself more decisively than the share ratio could have. The dump was
worth running for exactly this reason: an earlier draft hard-ordered S1 first
on a "~2×" share ratio, the honest ratio was 27.2/18 ≈ 1.5×, and the truth was
that S1 had no viable version at all.

Remaining: the verifier-net bench → S3 or its funeral (§4). Note §4's
asymmetry still holds in the direction that matters — S3 is the only
our-net-specific item, so it is the only one a bench outcome can strand.
