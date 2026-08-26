# FastQ-4+ — bounded tactical verifier on the position DAG

Design spec for the first search-policy consumer of `CaePositionDag` (PR #470,
stacked on #469's `CaeNnueState`). Status: SPEC — build starts after #470 lands.

## 1. Objective

Answer one narrow question per MCTS leaf: **"is the static NNUE value tactically
unstable here, and if so, what is the corrected value?"** — in ~5–20 NNUE
evaluations, against the current qsearch's ~72 average with a tail in the
thousands. This is a verifier, not a miniature general-purpose engine.

Gate math: the Q-labeled bootstrap arm failed its throughput gate ~6×. #469
(incremental state) and #470 (evaluate-once) supply part of that; FastQ's move
policy and pruning supply the rest. Deterministic budgets are what make the row
rate *stable*, not just fast.

Evidence constraint (bend-rule inversion, ledger 2026-08-25 08:40): added search
depth AMPLIFIES the static evaluator's bias away from deep SF (+7.5cp per sim
doubling, CI excludes 0). Shallow, well-pruned, bounded resolution is the
evidence-preferred shape — depth is capped by design, not by budget pressure.

## 2. Position in the stack

- New provider `nnue-fastq` registered alongside the existing providers. The
  existing `nnue-qsearch` is NOT modified or removed: it is the quality
  reference arm for the comparison harness (§8).
- Branch stacks on `feat/nnue-position-dag`; retarget to `main` after #470
  merges.
- Single evaluator ownership rule is inherited: FastQ evaluates through the
  `dag_open` handle (same mmap'd weights, no second evaluator path — the
  fifth-policy-path trap is the named hazard here).

## 3. Search semantics

### 3.1 The quiet certificate (the spec's centerpiece)

```
quiet(node) :=
       !in_check(node)
    && no promotion available
    && no capture with SEE >= 0
```

`quiet` is **window-independent and history-free** — computable from the
structural position alone — and is therefore a cacheable node fact (§4).
Delta-style reasoning ("no capture can affect THIS caller's alpha") is
window-dependent by construction and is a per-visit pruning decision (§3.4),
never part of the stored certificate. Tests pin this split (§8).

### 3.2 Move policy

- Not in check: generate **captures and promotions only**. Never generate quiet
  checks — "gives check" is not a reason to search a move.
- In check: generate **all legal evasions**; no stand-pat while in check.
  Evasion recursion is OWNED by FastQ (do not bounce through the generic check
  resolver — the node budget must mean what it says). Terminal helpers and
  scoring constants are reused from the resolver.
- **Checks are resolved, never generated**: when a searched tactical move gives
  check, the reply ply is exact evasion resolution (counted against depth and
  budget). This yields capture+ / forced-move / capture tactics without ever
  enumerating candidate quiet checks (the measured 17× explosion).

### 3.3 Recursion shape

```
fastq(node_id, ply, alpha, beta):
    if on current DFS path(node_id): return draw_score     # cycle guard, §4.3
    if in_check: search all evasions (no stand-pat), return best
    stand_pat = dag static value (evaluate-once)
    if stand_pat >= beta: return stand_pat
    alpha = max(alpha, stand_pat)
    if ply == max_qply or quiet(node): return stand_pat/best
    for move in ordered tactical moves:                     # SEE order, §5
        if SEE < 0 and not recapture-square exemption: skip
        if delta-prune(stand_pat, victim, alpha): skip
        child = dag_intern_child(node, move)                # #470 API
        score = -fastq(child, ply+1, -beta, -alpha)
        ... alpha/beta bookkeeping ...
```

- `max_qply = 4` default (variant knob, §6). Depth is the primary cost control.
- Fail-soft alpha-beta. Search VALUES live in the recursion only — never in the
  DAG (§4.2).

### 3.4 Pruning

- **SEE gate**: skip SEE-negative captures, except on the recapture square
  (the square just captured on), which stays exempt so forced recaptures are
  never blinded. Checking captures use the same gate initially; permissiveness
  for forcing tempo is a measured follow-up, not a day-1 assumption.
- **Delta pruning**: skip a capture when
  `stand_pat + value(victim) + margin <= alpha` (margin a constant knob,
  initial ~200 internal units; realized value logged).
- **Node budget = 32 as a TRIPWIRE, not a tuned knob.** With depth-4 + SEE +
  delta the cap should bind only in pathology. Every trip increments a counter
  that is part of the run report; a nonzero trip rate outside crafted fixtures
  is a finding to investigate, not a knob to raise. On trip: return current
  bound (deterministic stand-pat fallback).

## 4. DAG integration rules

### 4.1 What FastQ reads/writes per node

- get-or-create via `dag_intern_root` / `dag_intern_child` (which reuse #470's
  canonical hash + verified structural equality);
- stored ONCE per node: `CaeNnueState` (from #469 make-on-copy), the static
  NNUE value, and the `quiet` certificate bits of §3.1;
- the DAG's own counters (probes/hits/inserts/collision_steps/edge_reuses) are
  the proof that reuse actually occurs — surface them in FastQ's stats.

### 4.2 What is FORBIDDEN in the DAG

**No backed-up search values, ever.** Alpha-beta results are window-dependent
(fail-low/high bounds) and, with cross-call persistence and structural-only
identity, repetition-unsafe. The DAG stores window-independent, history-free
facts only. A mutant that caches a fail-high value as a node fact must be
killed by a test (§8). A conventional TT with bound flags is an explicit
non-goal (§9) — if measurements later justify one, it is a separate, search-
local structure, not the canonical DAG.

### 4.3 Cycles

Structural identity admits back-edges (reversible chess). FastQ keeps its
current DFS path as a small id set; re-entering a node on the path returns
draw-score (repetition adjudication in the search overlay, exactly as #470's
identity-boundary section prescribes). A crafted repetition fixture must
terminate and return draw-score (§8).

### 4.4 Persistence and reroot

The graph survives across leaves/batches and across root advances via
`set_root` (nodes stay alive; `reset` is explicit). **Persistence policy is
chosen from measurement, not assumed**: FastQ reports cross-call hit rate
(hits attributable to nodes created by earlier calls) and
`cae_position_dag_memory_bytes`; the eviction/reset cadence decision waits for
those numbers. Until then: reset per game, persist within a game.

## 5. SEE

A real static-exchange evaluator is part of this PR (none exists today).

- Swap algorithm on the target square with x-ray reveal through sliders;
  handles en-passant and promotion values; side-to-move alternation with early
  stand-pat cutoffs.
- **Pins are deliberately ignored** (static SEE, Stockfish-style). Documented
  as a known approximation; the parity oracle below quantifies it.
- Test oracle: brute-force capture-sequence minimax on the target square over
  crafted fixtures (x-ray stacks, EP, under-promotion, pinned attacker rows
  where SEE is knowingly wrong — asserted as EXPECTED divergences so the
  approximation is pinned, not hidden).
- Used for BOTH ordering (best SEE first) and gating — one computation.
  MVV-LVA exists only as the pre-SEE tiebreak.

## 6. Knobs (all threaded to the C provider; realized values logged)

| knob | default | variants |
|---|---|---|
| `max_qply` | 4 | 2 / 4 / 6 / 8 |
| `node_cap` | 32 | 16 / 32 / 48 / 64 |
| `delta_margin` | 200 | fixed this PR |
| `see_recapture_exempt` | on | on/off |

The signature defect rule applies: every knob must provably reach the C
search (mutation: a knob set to an extreme that must change a fixture's node
count — run it, watch it change). A Python-only knob silently dropped by the C
path is the named historical failure.

## 7. Instrumentation (per run, and per call in debug)

FastQ calls · nodes created vs canonical hits (within-call / cross-call split)
· DAG probes/hits/collision_steps · NNUE evaluations (must equal nodes
created — the evaluate-once invariant as an ASSERTABLE counter identity) ·
budget-tripwire count · SEE prunes / delta prunes / beta cutoffs · wall per
call p50/p99 · `dag_memory_bytes`.

## 8. Verification plan

Unit tests with NAMED MUTANTS (a test that has not killed its mutant does not
count as done):

1. Certificate split: `quiet` computed with a window argument → must fail a
   test asserting window-independence (mutant: fold delta into the stored
   certificate).
2. Evaluate-once: same structural position reached via two move orders → NNUE
   eval count 1 (mutant: skip the canonical lookup → count 2 → FAIL).
3. No-search-values-in-DAG: mutant caches a fail-high value and reuses it
   under a different window → a two-window fixture must FAIL.
4. Cycle guard: repetition fixture terminates with draw-score (mutant: drop
   the path set → hang/overflow caught by the budget tripwire test).
5. Budget integrity: evasions count against the budget (mutant: exempt
   evasions → crafted check-storm fixture exceeds cap → FAIL).
6. Check policy: a quiet check is never generated (fixture where the only
   tactic is a quiet check: FastQ must return stand-pat); a searched capture
   that GIVES check gets exact evasion resolution (fixture: capture+ / forced
   king move / recapture resolves to the material result).
7. SEE oracle fixtures per §5.
8. τ=knob threading per §6.

Reference-arm harness (same-population, same rows, both arms through their own
providers): mean |Δ|, p95 |Δ|, identical fraction, buckets >50/100/250
internal units, and **sign agreement** as primary — mean |Δ| alone is gameable
by never correcting anything. Reference is the existing qsearch-4 arm.
The DECIDING readout for any production claim is the downstream standardized
primary vs deep SF (the az_purity prereg framework), not similarity to
qsearch — qsearch-4 is a cheap first-pass reference, not the target.

## 9. Non-goals (deferred until measurements justify)

Quiet-check generation · mate heuristics (the trigger-gated mate verifier is
its own later PR) · history heuristics · LMR · null move · conventional TT
with bound flags · NNUE-instability certificates (needs training) · boundary-
targeted verification (later PR; but the `(position, alpha, beta, budget)`
call shape is chosen NOW so a Gumbel caller can hand cut-boundary candidates
tighter windows without an API break).

## 10. Sequencing

#469 blockers close → #469 merges → #470 retargets to main + review →
**qsearch-on-DAG retrofit** → this PR.

The retrofit interns the EXISTING qsearch (unchanged move policy) on the DAG:
evaluate-once + cross-call persistence via `set_root`, node-budget flag
DEFAULT-OFF with a trip counter, full counters surfaced. Its acceptance
criterion is the oracle FastQ cannot have — **bit-identical values to the
pre-DAG qsearch on a corpus** (one variable: the substrate) — plus measured
evals/call reduction and cross-call hit rate. That hit rate sizes the DAG's
real benefit BEFORE FastQ relies on it, the retrofit doubles as the DAG's
correctness proof, and it makes the §8 reference arm cheap to run at matrix
scale. The budget flag stays off wherever the oracle property is being
asserted; enabling it for production labeling is a measured decision made
against FastQ, not a default.

Layers: FastQ = layers 1–2 of the four-layer verifier picture; the mate
verifier and boundary-targeted verification are later PRs on the same DAG.
