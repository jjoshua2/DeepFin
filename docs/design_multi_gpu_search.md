# Design: root-parallel Gumbel over evaluator groups (multi-GPU search v2)

Status: DESIGN (2026-07-09). Target: TCEC 8×RTX 5090 (PCIe, no NVLink).
Companion: the existing `MultiGpuPucvPool` (PUCT+virtual-loss, shared tree)
stays as the throughput baseline; this design is the quality-preserving
alternative. Both get compared on rented multi-GPU hardware before the event
(protocol in §7). Pondering is NOT allowed at TCEC — no opponent-time
compute anywhere in this design.

## 0. Why (one paragraph)

All of the engine's validated search strength lives on the Gumbel path
(sequential halving, c_scale 0.025 ≈ +301 Elo @8k sims, sim-invariant
root-log transform). The existing multi-GPU mode abandons that path for
untuned PUCT+virtual-loss and pays quality for throughput at unknown
exchange rates (~8k leaves in flight under vloss on 8 GPUs). This design
parallelizes the *validated* search instead: within a sequential-halving
phase, each surviving root candidate's simulation budget is independent
work — embarrassingly parallel across devices — so root decisions remain
mathematically identical to serial Gumbel while the bulk of the compute
scales with GPUs.

## 1. Layer 1 — evaluator groups

An **EvaluatorGroup** is `s` GPUs presenting the single-evaluator interface
search already consumes (`evaluate_inplace_async`, pinned buffers, 2-slot
pipelining). `s = 1` is today's `DirectGPUEvaluator` unchanged. `s = 2`
splits each batch in half, submits both halves concurrently on per-device
streams, and joins — cutting per-submit latency, which is the measured
single-game bottleneck (forward-bound, batch ~408 saturates one 5090).

Topology is a single config: `groups g × stripe s`, `g·s ≤ #GPUs`.
PCIe reality (no NVLink on 5090s): striping pays host↔device scatter/gather
per submit (~5 MB planes at batch 512 ≈ 2-3 ms ≈ the forward itself), so
the expectation is `s ∈ {1, 2}` useful, `s ≥ 4` not. Striping is an
OPTIONAL optimization built last; every layer above depends only on the
group interface, not on `s`.

Per-group state mirrors the PUCV pool's proven pieces: evaluator factory
invoked on the group's own thread (cudagraph TLS), `torch.cuda.set_device`
first (the TLS trap is in project memory and the 2026-07-09 review flagged
its absence in the pool), shared compile cache, per-device warmup at
install (ALL devices, not the first 2-3 — review bug #3).

## 2. Layer 2 — candidate-owned root-parallel Gumbel

### Semantics contract
The root runs exactly today's Gumbel: same candidate sampling (Gumbel
noise + prior + topk), same sequential-halving schedule, same completed-Q
formula, same c_scale/root-log value transform, same final move selection.
The ONLY change: within a phase, per-candidate simulation budgets execute
concurrently on different evaluator groups. Phase decisions (halving)
happen at a barrier, single-threaded, from per-candidate results — bitwise
the same inputs a serial run would have produced given the same per-
candidate sim outcomes.

### Ownership rule (the heart)
Each root candidate's subtree is owned by EXACTLY ONE group at a time.
Consequences:
- No cross-group contention on tree nodes → the shared-tree mutex and the
  non-atomic `W` accumulation race (review bug #4) are structurally
  irrelevant across groups.
- Virtual loss is needed only INSIDE a candidate's subtree (a group batches
  its own descents, `gather ≤ ~256`, virtual-mean pending) — the mild,
  single-GPU-validated regime (`UseVL` path: +112% vs sync at Threads=1),
  never the 8k-pending regime.
- Trees can be per-candidate arenas: allocation locality, trivial memory
  accounting against Hash, and free reclamation of pruned candidates'
  arenas at halving (the current tree never reclaims pruned siblings —
  review bug #5 — this design gets reclamation for free).

### Scheduling
- Phase p has `m_p` surviving candidates, each with per-candidate budget
  `b_p` (unchanged Gumbel arithmetic). Candidates go into a work queue;
  each of the `g` groups pulls the next candidate and runs its FULL `b_p`
  for that phase (one candidate = one group = no handoffs mid-phase).
- `m_p ≥ g` (early phases, 32→16→8 candidates): all groups busy,
  near-linear scaling, zero semantic deviation.
- `m_p < g` (late phases): idle groups SPLIT a candidate's remaining budget
  with the owner — intra-candidate parallelism, both groups descending the
  same candidate arena under virtual-mean. This is the only place two
  groups share nodes; pending count stays ≤ 2×gather. Sequential halving
  concentrates budget exactly here, so the tail matters and the mitigation
  is bounded and local. (v1 may ship without splitting: accept ≤2× idle
  tail in the final phases, measure, then add.)
- Chunking/clock: the existing chunk contract survives — the phase runner
  checks stop/deadline between per-candidate gather-batches; a phase
  barrier is also a natural quiescent point for `info` emits and the
  batch-margin clock guard.

### Tree reuse across moves
Per-candidate arenas make `advance_root` a re-labeling: the played move's
candidate arena becomes the next root's base tree; other arenas are freed.
When the next search starts, root expansion + Gumbel sampling happen as
today; child subtrees inherited from the kept arena seed the new
candidates' arenas (or v1: drop reuse entirely for simplicity — measure
the cost; reuse at TCEC classical TC is worth real nodes, so v2 restores
it).

### What is explicitly NOT changed
- No Gumbel math changes, no new search hyperparameters beyond `g`/`s` and
  the existing gather/vloss knobs scoped intra-candidate.
- Single-group degenerate case (`g=1, s=1`) must be BIT-IDENTICAL to the
  current engine (same RNG stream consumption, same node ordering) — CI
  regression gate, not a hope.

## 3. Config surface

```
--devices cuda:0,...,cuda:7      # unchanged
--search-parallel gumbel|pucv    # NEW: which multi-GPU mode (default gumbel once shipped)
--eval-groups G                  # NEW: evaluator groups (default = #devices)
--eval-stripe S                  # NEW: GPUs per group (default 1)
--vl-gather / --pucv-pending-mode / --vloss-weight   # reused, intra-candidate scope
```
UCI mirrors: `SearchParallel`, `EvalGroups`, `EvalStripe` (combo/spin).

## 4. Test plan (the part that works WITHOUT 8 GPUs)

1. **Bit-identity gate** (CI, CPU evaluator): `g=1,s=1` vs current Gumbel —
   identical chosen moves, visit counts, and root Q across a position suite
   and seeds.
2. **Semantics-under-parallelism** (CI, CPU evaluators): `g ∈ {2,4}` with
   deterministic per-candidate eval stubs — halving decisions and final
   move identical to serial for the same per-candidate results; ownership
   invariant asserted (no node touched by two groups outside split mode).
3. **Same-GPU smoke** (dev box, GPU-free window): `--devices cuda:0,cuda:0`
   → 2 groups sharing one physical GPU. Exercises every threading/stream/
   cudagraph path except actual cross-device transfer. (The review found
   this was never done even for the PUCV pool — it goes into the standard
   pre-flight for BOTH modes.)
4. **Rented-box protocol**: §7.

## 5. Existing-PUCV hardening (parallel track, shipped regardless)

From the 2026-07-09 review, in priority order:
- (#1) `--devices` >1 without the pool serializes through the coalescer —
  one GPU busy at a time. Fix: auto-enable the pool for >1 devices (or
  hard-warn + refuse); nobody should ever benchmark the serialized path.
- (#2) `torch.cuda.set_device(dev)` first thing in each pool worker's
  factory (cudagraph TLS trap; pattern exists in inference_threaded.py).
- (#3) warm up EVERY device at install, not the 2-3 the round-robin hits.
- (#4) make `W` accumulation atomic (CAS loop on the double, or per-worker
  accumulators folded at chunk barriers) — quality noise at 8 workers.
- (#6) `run()` returns actual sims (not target) on early stop; skip GPU
  submit for terminal-only batches.
- Emit the "c_scale inert on this path" warning when the pool is active
  (exists for walkers>1, missing for the pool).
- Docs/defaults: Hash ≥ 65536 guidance for ≥4 GPUs (tree at 8× node rate).

## 6. Expected outcomes (pre-committed hypotheses)

- Root-parallel Gumbel at g=8: ~5-6× effective sims at equal per-sim
  quality (Amdahl tail from late phases), i.e. strictly better play than
  1-GPU Gumbel at fixed time. Risk: per-candidate budgets too small at low
  chunk counts to fill 512-wide batches early in a move (mitigation:
  intra-candidate gather already batches; early phases have many
  candidates × moderate budgets = good width).
- PUCV at 8 GPUs: higher raw nps than root-parallel Gumbel, lower Elo/node;
  net Elo-at-time unknown — that is exactly what §7 measures.
- If PUCV-at-8 beats Gumbel-at-8 at fixed time, we ship PUCV for the event
  and eat the tuning debt; if Gumbel wins, we ship the validated path and
  keep all our tuning. Either way the loser stays as the bench reference.

## 7. Rented-machine comparison protocol (~half a day of box time)

All runs same checkpoint, warmed compile cache, exclusive GPUs.
1. **nps ladder** (autotune, ~30 min): 1/2/4/8 GPUs × {pucv virtual-mean
   g512, pucv legacy g768, root-parallel gumbel} → sims/s scaling curves.
2. **Strength ladder** (`scripts/match_vs_uci.py`, engine A vs engine B,
   fixed clock, paired openings): the decisive reads, ~100 games each:
   a) 8-GPU PUCV vs 1-GPU Gumbel  ← is the existing mode even net-positive?
   b) 8-GPU root-Gumbel vs 1-GPU Gumbel  ← does the new design pay?
   c) winner(a) vs winner(b)      ← what TCEC actually runs.
3. Decision rule: highest Elo-at-time wins the event slot; ties break
   toward the Gumbel path (carries all validated tuning + simpler failure
   modes under clock pressure).
