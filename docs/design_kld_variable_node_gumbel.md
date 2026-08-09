# Design: KLD-gain-driven variable-node search under Gumbel

**Status: DESIGN ONLY. Nothing here is implemented, launched, or promoted.**
No ledger entry exists and none should be opened until the offline screen in §11
is run. Measurements below are read-only observations of the live run
(trial `379f6_00000`) on 2026-08-09 and offline reads of banked replay shards.

**Recommendation up front: NO-GO on the mechanism as proposed. DEFER the
underlying idea behind one cheap offline screen (§11).** The throughput
argument is dead on arithmetic (§1). The data-quality argument is real and
measurable (§3) but the specific mechanism — lc0's KLD gain on the root visit
distribution — measures a quantity that is *deterministic* under Gumbel
sequential halving and therefore cannot work (§5). A reformulated version is
coherent but is gated behind a change to the selfplay root value-transform that
is a bigger experiment than the one being proposed (§6).

---

## 1. The binding resource — answer first, because it reframes the question

**Short answer: the selfplay threads are search-bound, but the *box* is
Stockfish-bound, and the box wins. A free search buys ≲10% more games.**

This needs stating carefully because **the regime has flipped since the last
time it was measured**, and quoting the old number would be wrong in one
direction while quoting the new one naively would be wrong in the other.

### 1a. What changed

`docs/sf_cpu_cost_split.md` (2026-07-27) measured Stockfish at **70.4% of all
box CPU**, with selfplay game threads spending 2200–3090% of a 3200% thread
budget parked in `sf_block_starved`. That was measured at `sf_nodes = 698,289`,
`sf_multipv = 40`, uncapped labels.

The 08-06 reduced-SF relaunch bundle is live. From the published manifest
`runs/pbt2_small/server/trials/379f6_00000/publish/manifest.json` →
`recommended_worker`:

| key | 2026-07-27 | live 2026-08-09 |
|---|---|---|
| `sf_nodes` | 698,289 | **75,000** |
| `sf_multipv` | 40 | **6** |
| `sf_label_nodes_floor` / `_cap` | — / 0 (uncapped) | **150,000 / 200,000** |
| `selfplay_fraction` | ~0.49 realized | **0.8** |

Stockfish work per game fell by roughly 4–9x. The bottleneck moved.

### 1b. The selfplay threads are now search-bound (MEASURED)

`selfplay phase stats`, all four live workers. `sf_block_starved` reads
**exactly `0.0%` in all 368 of worker_00's phase-stat lines** — its entire
~6.4-hour life since the 05:29 start, no exceptions:

```
network=3084.8%  sf_block_starved=0.0%  total_thread=3138.3%   (worker_00 11:30:38)
network=3195.5%  sf_block_starved=0.0%  total_thread=3284.6%   (worker_01 11:30:18)
network=3150.0%  sf_block_starved=0.0%  total_thread=3204.8%   (worker_02 11:31:20)
network=3093.2%  sf_block_starved=0.0%  total_thread=3148.3%   (worker_03 11:30:45)
```

Traced back hourly from 05:41 to 11:32: `network` 2996–3294%,
`sf_block_starved` **0.0% in every single sample**. `pending_excluded_avg` is
5.3–6.7 against `in_flight_avg` 24.0, versus 23.5–23.7 on 07-27 — the standing
population of games parked on a pending SF query fell ~4x.

`network` wraps exactly one call — `run_network_turn(state, combined_net_idxs)`
at `chess_anti_engine/selfplay/manager.py:131-135` — i.e. our MCTS, including
its GPU waits. So **~98% of selfplay thread-time is now inside our own search,
and none of it is blocked on Stockfish.** Throughput confirms the effect:
6,161 games/h over the last 2h (12,145 games / 1.97h from compacted-shard
filename counters) against the 2,561 games/h baseline in
`docs/sf_cpu_cost_split.md` §3b — **2.4x**.

Taken alone this says "search is now the bottleneck, go optimise it."

### 1c. …but the box has no headroom to convert that into games

> **Provenance: NOT independently verified.** The CPU/GPU figures in this
> subsection are single-sample author measurements of a live box whose load
> composition changes between iterations; they were not re-sampled by a second
> reader and no raw capture was banked. They are load-bearing for the §1
> "binding resource" conclusion, so re-measure before quoting them — and note
> the standing warning that "SF ≈ 95% of loop cost" went stale exactly this way.

`/proc/stat` over a 20-second sample: **92.0% busy, 8.0% idle** across 32 cores.
Per-class CPU (three `ps` samples, 4s apart, stable to ±10%):

| | CPU |
|---|---|
| Stockfish (32 engines) | **1617%** |
| everything else (trainer, workers, broker, Ray) | 963–968% |
| box | 3200% |

GPU utilization sampled over 5s: 56–78%, shared with the trainer.

The arithmetic that matters: **games/h and Stockfish CPU are proportional** —
every extra game issues its label and curriculum-move queries. With 8% CPU idle,
games/h can rise at most ~8–10% before the CPU saturates and `sf_block_starved`
returns. And the "everything else" bucket also scales with games, so the real
ceiling is below that.

**Conclusion: a variable-node search that made our MCTS *free* would buy under
10% more games/h.** Against the standing bar (§12) that is a rounding error.
The throughput framing is dead — not because search is a small minority of
selfplay time (it isn't, any more), but because the machine cannot cash the
saving. The idea must stand or fall on data quality.

> **Caveat, stated plainly.** §1b and §1c were measured over minutes-to-hours on
> one day, at one point in the PID's trajectory. `sf_nodes` is PID-owned and
> floats; if the controller walks it back up toward `sf_pid_max_nodes`, SF cost
> rises and the loop returns to the 07-27 regime. **Re-measure `network` vs
> `sf_block_starved` before acting on any conclusion in this document.** The
> instrument is one `grep "selfplay phase stats"` on a worker log.

---

## 2. How lc0 actually does it (ESTABLISHED, verbatim source)

From `src/mcts/stoppers/stoppers.cc`, `KldGainStopper::ShouldStop`
([lc0 v0.30.0](https://github.com/LeelaChessZero/lc0)):

```cpp
bool KldGainStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  Mutex::Lock lock(mutex_);
  const auto new_child_nodes = stats.total_nodes - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto new_visits = stats.edge_n;
  if (!prev_visits_.empty()) {
    double kldgain = 0.0;
    for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = new_visits[i] / new_child_nodes;
      if (prev_visits_[i] != 0) kldgain += o_p * log(o_p / n_p);
    }
    if (kldgain / (new_child_nodes - prev_child_nodes_) < min_gain_) {
      LOGFILE << "Stopping search: KLDGain per node too small.";
      return true;
    }
  }
  prev_visits_ = new_visits;
  prev_child_nodes_ = new_child_nodes;
  return false;
}
```

Established facts:

- **The distribution is `stats.edge_n` — the ROOT CHILD VISIT COUNTS**,
  normalized by total child nodes. Not policy, not Q.
- The divergence is `KL(previous ‖ current)` between two snapshots of that
  distribution.
- Snapshots are taken every `--kldgain-average-interval` nodes (**default 100**).
- The threshold is **per node**: `kldgain / (nodes since last snapshot)`. So
  `--minimum-kldgain-per-node=0.000050` at the default interval means *stop once
  the root visit distribution moves less than 100 × 5e-5 = **0.005 nats** between
  consecutive 100-node checkpoints*.
- Firing stops the entire search.

**NOT established:** how often it actually triggers in T80, and the realized
node-count distribution it produces. No lc0 selfplay telemetry was available to
this analysis and none is quoted. The threshold value `0.000050` and the flag
semantics are sourced; everything about its *realized* behaviour is not.

---

## 3. The reallocation case, and it is genuinely strong (MEASURED)

> **Provenance: NOT independently verified.** The 111k-row `priority_policy_kl`
> table below is a single author-run aggregation and the dump was not banked, so
> no one has re-derived it from the shards. The *mechanism* citations
> (`_mcts_tree.c:4834-4839`, the per-row storage) have been checked by a second
> reader; the *numbers* have not. Treat the distribution as indicative until a
> banked dump exists — per the ledger's "bank the dump, not just the number".

At fixed total budget, the question is whether some positions are absorbing sims
that buy nothing. **Every shard row stores the instrument that answers this.**

`priority_policy_kl` is computed in `_mcts_tree.c:4834-4839` inside
`batch_process_ply`:

```c
float kl = 0.0f;
for (int j = 0; j < n_legal; j++) {
    float p  = uniform ? inv_sum : (legal_logits[j] * inv_sum);   /* net prior, softmax over legal */
    float mp = mprobs[legal_indices[j]];                          /* the search target */
    if (p > 1e-12f && mp > 1e-12f)
        kl += p * (logf(p) - logf(mp));
}
```

That is `KL(net prior ‖ search target)` — **exactly "how far did 256 sims move
the policy away from what the network already believed"**, stored per row, no
GPU needed to read it.

Measured over **111,163 policy-bearing rows** from 60 randomly sampled shards of
the live replay window
(`runs/pbt2_small/replay/train_trial_379f6_00000_.../replay_shards/`):

| percentile | p5 | p25 | **p50** | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| KL (nats) | 0.0004 | 0.0067 | **0.0243** | 0.0872 | 0.2681 | 0.5479 | 1.4834 |

mean 0.1135 — i.e. **the mean is 4.7x the median.** The tail carries everything:

| threshold | share of positions | share of total KL mass |
|---|---|---|
| KL < 0.005 | 20.6% | 0.25% |
| KL < 0.01 | 32.1% | 0.99% |
| **KL < 0.02** | **45.9%** | **2.77%** |
| KL < 0.05 | 64.7% | 8.16% |
| KL < 0.1 | 77.3% | 16.12% |

**Nearly half of all positions absorb a full 256-sim search and account for
2.8% of all the policy movement search produces.** Two-thirds account for 8%.
That is a large, real, measured reallocation headroom, and it is the single
strongest argument in favour of this idea.

It also has structure — search matters more later in the game:

| ply bucket | n | mean KL | median KL |
|---|---|---|---|
| 10–20 | 3,761 | 0.0220 | 0.0085 |
| 20–40 | 19,951 | 0.0424 | 0.0153 |
| 40–80 | 39,852 | 0.0801 | 0.0219 |
| 80+ | 47,332 | **0.1794** | 0.0428 |

**Three honest limits on this measurement, and the third is serious:**

1. `p1 = 0.0000` and `p0 = -0.368` — KL cannot be negative. The C loop skips
   terms where either side underflows `1e-12`, and the target is stored as fp16,
   so a small negative is a truncation artifact. It affects <1% of rows and does
   not move any conclusion, but the column is not a clean KL at the extreme low
   end.
2. This is the KL *at the end of a 256-sim search*, not a convergence trajectory.
   It bounds what a stopping rule could have saved; it does not show the search
   had converged early.
3. **It is partly circular, and §4 explains why.** The target is
   `softmax(log prior + σ·Q̃)` with σ growing in the sim count. A shorter search
   produces a smaller σ, hence a target *mechanically* closer to the prior, hence
   a *smaller* KL — regardless of whether the position was resolved. So "low KL"
   and "few sims" are coupled by construction. A naive stopping rule keyed on
   this quantity is partly self-fulfilling.

---

## 4. What our search actually emits, and why it is not a visit distribution

This is the load-bearing fact for the rest of the document.

`_build_improved_policy_for_board` (`mcts/gumbel.py:864-909`) builds the stored
training target as:

```
probs = softmax( log(prior) + σ · normalize(completed_Q) )      over ALL legal moves
```

Visit counts enter in exactly two places, and neither is as a distribution:

1. They decide which children's Q is *real* versus filled with the mixed value
   (`_completed_q_transform`, `mcts/gumbel.py:435-447`).
2. They set `max_visit`, which scales σ.

With the live config (`gumbel_c_scale: 0.025`, `c_visit` at its 50.0 default,
root overrides unset so the **linear** transform applies at both sites):

```
σ = c_scale · (c_visit + max_visit) = 0.025 · (50 + max_visit)
```

**The visit counts themselves are never normalized into a probability and never
appear in the target.** Any port of lc0's rule must reckon with this.

---

## 5. ⚑ The mechanism cannot be ported: under Gumbel the visit distribution is a deterministic staircase

Sequential halving allocates **uniformly within each round**. From
`gss_begin_round` (`_mcts_tree.c:1392-1423`) and its Python mirror
(`gumbel.py:1040-1051`):

```c
int32_t vpa = g->budget_remaining[i] / (rem_count * rounds_left);
if (vpa < 1) vpa = 1;
```

Every surviving candidate gets `vpa` visits — one per `rep`, and
`_collect_forced_leaves_round` (`gumbel.py:803-816`) gives *every* remaining
candidate exactly one visit per rep. So the realized schedule at the live
settings (256 sims, topk 32, ~30 legal moves) is fully determined before the
search starts:

| round | candidates | visits each |
|---|---|---|
| 1 | 30 | 1 |
| 2 | 15 | 3 |
| 3 | 8 | 7 |
| 4 | 4 | 15 |
| 5 | 2 | 32 |
| 6 | 1 | 1 |

Survivor `max_visit` = 59; 256 sims spent exactly.

> **⚑ One exception, and it is the only position-dependent spend that already
> exists.** `gss_begin_round` zeroes `budget_remaining[i]` outright when the
> root is **solved** — `t->solved[rid] != SOLVED_UNKNOWN`, i.e. terminal or
> tablebase-resolved (`_mcts_tree.c:1396-1401`) — and the comment says why:
> "the result of search is known, every additional sim is wasted GPU work."
> Such a root spends **fewer than 256 sims**, and whether it does is a property
> of the position. So the staircase below is the schedule for an *unsolved*
> root, which is the overwhelming majority of selfplay roots but not all of
> them.
>
> This does not rescue the proposal — a solved root is exactly the case where
> no KL signal is needed, and the saving is already taken — but it does mean
> "spend is position-independent" is false as an unqualified statement, and
> §11 already leans on this same mechanism as the precedent for mid-search
> budget zeroing.
>
> Note it is **C-path only, and the disagreement is total**: `grep solved` on
> `gumbel.py` returns **zero hits** — the Python search has no solved-root
> concept at all, not merely a weaker one. Production selfplay runs the C path,
> so the exception is live in production and absent from the reference
> implementation.
>
> **A second C/Python parity break sits in the same loop:** `halving_div` is
> never *read* on the Python path. It exists as a `GumbelConfig` field
> (`gumbel.py:186`) but the Python round counter hardcodes
> `rounds_left = ceil(log2(len(rem)))` (`gumbel.py:1048`), while the C walks
> `while (tmp > 1) { rounds_left++; tmp = (tmp + div - 1) / div; }`
> (`_mcts_tree.c:1411-1413`). The two agree only at `div == 2` — the shipped
> value — so any experiment that moves `HalvingDiv` and compares the paths is
> comparing two different schedules. Worth knowing before anyone uses the
> Python path as the oracle for a C change.

**The consequence is fatal to the proposal as stated.** For an unsolved root the
visit vector at any point in the search is a function of
`(budget, m, halving_div, round, rep)` and nothing else. The *position*
influences only **which** candidates survive, never **how many visits** any of
them gets. Therefore:

- **Within a round**, all survivors accrue visits in lockstep. The normalized
  visit distribution stays flat over the survivors, and lc0's
  `KL(prev ‖ new)` decays as `O(1/n)` purely from the counts growing. It would
  fire on the schedule, on every position, at the same point.
- **At a halving boundary**, half the candidates stop receiving visits, so the
  distribution jumps. The KL spikes — again identically on every position.

A visit-KLD stopping rule under Gumbel would therefore measure the halving
schedule and report it as position difficulty. It would produce a plausible
number, a plausible histogram, and a completely uninformative signal. This is
precisely the failure mode CLAUDE.md names as this codebase's signature defect:
**a metric that does not mean what its name says.** In lc0's PUCT the visit
counts *are* the search's accumulated judgement; under Gumbel they are the
schedule. The quantity is not transferable.

**Any stopping rule must instead watch the improved policy
`softmax(log prior + σ·Q̃)`** — the thing we actually store. That has its own
problem, in §6.

---

## 6. The recommended adaptation, and what it costs

### 6a. What survives truncation, and what does not

Gumbel MuZero has two distinct guarantees and they degrade differently:

- **The improved-policy target** `π' = softmax(log π + σ(q̃))` is a policy
  improvement over the prior for *any* monotone σ and any completed-Q estimate.
  It does **not** require the schedule to finish. Truncating produces a
  *weaker* improvement (smaller σ, noisier Q̃ on fewer visited actions), not an
  invalid one.
- **The action-selection guarantee** — that the played move is the argmax of the
  Gumbel-perturbed improved score over the top-m — *does* rest on sequential
  halving completing its elimination. Truncating leaves >1 candidate and the
  played move becomes the leader of an unfinished tournament.

**So mid-schedule truncation damages the PLAYED MOVE more than it invalidates
the TARGET.** That matters more than it sounds: the played move drives the game
trajectory (the data distribution), the winrate against Stockfish, and hence the
PID (§8).

One thing that is *not* a problem: because each rep gives every survivor exactly
one visit, **stopping between reps leaves the allocation balanced**. A partially
completed *round* is not a biased comparison — all remaining candidates still
have equal visits. Rep boundaries are also already the GPU batch boundaries. So
the natural stopping granularity is the rep, and it is free.

### 6b. The recommendation: vary `n` per position, complete the schedule

Of the four options in the brief:

| option | verdict |
|---|---|
| Stop mid-schedule on a KLD rule | **No** — see §5 (signal is uninformative) and §6a (breaks action selection). |
| Stop only at phase/round boundaries | **No** — at 256 sims there are only 6 rounds, so the instrument has ≤5 observations, and σ more than doubles across them (§7), so consecutive policies differ by construction. |
| Vary `m` (topk) rather than `n` | **No** — `m` is already bounded by `m_cap` (§6c) and by legal-move count; varying it changes the target's *support*, which is a bigger and less controllable intervention than changing its sharpness. |
| **Vary `n` per position, schedule complete** | **Yes** — and it is already built. |

`per_game_simulations` is already a first-class parameter of both
`run_gumbel_root_many` (`gumbel.py:924`) and the C entry point
(`gumbel_c.py:355/383/410`), and production already uses it: `network_turn.py:814`
and `:837` pass `sub_sims` to give playout-capped fast plies 32 sims and full
plies 256 (`fast_simulations: 32`, `mcts_simulations: 256`,
`playout_cap_fraction: 0.25`).

**We already run a variable-node search.** The allocation rule is "roll a die;
25% of plies get the big budget, and only those produce a training row." The
honest description of this proposal is therefore not "add variable-node search"
but **"replace the random playout-cap selector with an informed one."** That is
a much smaller and much better-posed change than the brief's framing, and it
sidesteps §5 entirely because the budget is chosen *before* the search, so no
mid-search convergence signal is needed.

The trigger would be a cheap pre-search predictor of "will search move the
policy here" — the root prior's entropy, the value head's uncertainty, or the
volatility head (`volatility_target` is already stored per row). `priority_policy_kl`
gives a labelled training set for such a predictor, free, on banked data.

### 6c. ⚑ `m_cap` — what happens when the budget goes low

`gumbel.py:707-711`:

```python
if sim_budget <= 1:
    m = 1
else:
    m_cap = max(2, (sim_budget + 1) // 2)
    m = max(2, int(min(int(topk), int(legal.size), int(m_cap))))
```

**The C driver production actually runs applies the identical cap** —
`gumbel_c.py:851-852`, `m_cap = max(2, (_game_budget + 1) // 2)` then
`m = min(topk, legal_idx.size, m_cap)`. Citing only the Python line would leave
the floor below looking like a property of the reference implementation rather
than of the shipped one.

With `gumbel_topk: 32`, `m_cap` binds at **sims ≤ 63**, and at sims = 64 exactly
`m_cap == topk == 32` — the coincidence that has already hidden a search variant
from a test regime once. Realized breadth against a 30-legal-move position:

| sims | 16 | 32 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|
| `m` | 8 | 16 | 30 | 30 | 30 | 30 |
| `max_visit` | 4 | 5 | 8 | 18 | 28 | 59 |

**So a variable-node scheme must declare a floor at 64 sims.** Below it, the
candidate set silently narrows and the target's *support* shrinks — a second,
distinct, position-correlated distortion stacked on the sharpness one in §7, and
one that is invisible in any pooled entropy statistic. Any budget below 64 is
not "a shorter search", it is a different search.

---

## 7. ⚑⚑ Target sharpness becomes position-dependent — and it is a BUG

### 7a. The size of the effect, under the live config

σ scales with `max_visit`, and `max_visit` scales with the budget. Computed
exactly from the schedule above at `c_scale = 0.025`, `c_visit = 50`:

| sims | `max_visit` | σ (live LINEAR root) | relative to 256 |
|---|---|---|---|
| 32 | 5 | 1.375 | **0.50x** |
| 64 | 8 | 1.450 | 0.53x |
| 128 | 28 | 1.950 | 0.72x |
| **256** | **59** | **2.725** | **1.00x** |
| 512 | 104 | 3.850 | 1.41x |
| 1024 | 206 | 6.400 | **2.35x** |

σ is the number of nats of logit range the completed-Q term is allowed to
contribute on top of `log(prior)`. **Over a 32x budget range it varies 4.7x.**
A variable-node scheme under the live config would inject exactly that spread
into the training corpus, correlated with position type.

### 7b. Why this is a bug and not a feature

The "feature" reading — *targets are appropriately less confident where search
was less certain* — is intuitive and **factually wrong about the direction**.

Shrinking σ does not move the target toward uniform. It moves it toward
`softmax(log prior)` — **the network's own prior**. A cheap search does not
produce a humble target; it produces *the net's own opinion, restated as a
training label*. That is zero-information self-distillation, and it would be
applied preferentially to the ~46% of positions where the prior and the search
already agree (§3) — making the targets we already suspect of being
self-referential *more* so. Given that the loop is currently at a
self-referential fixed point (`net ≈ search(net)`), a mechanism whose failure
mode is "emit more of the prior" is pointed in the worst available direction.

The "feature" argument fails on its own terms: it does not buy calibrated
uncertainty, it buys self-agreement.

**And the obvious fix — renormalise the target for the realized sim count — does
not work.** σ is not a post-hoc temperature on the output. It is used *inside*
the halving loop (`gss_score_and_halve`, `_mcts_tree.c:1425+`) to eliminate
candidates. A search run at σ=1.45 eliminated a different set of candidates than
one run at σ=2.73, and no rescaling of the output policy can undo the different
tree that was actually built. The correction must happen at the search, not the
target.

### 7c. The correction that does work, and it already exists

The codebase already contains a sim-invariant root transform, built for the
UCI/play path: `q_visit_exp_root = -1` (LOG root) with `c_scale_root = 7.0`,
`c_visit_root = 900.0`, in `PLAY_SEARCH_DEFAULTS`. Its docstring says exactly
why it exists — it "keeps q_scale ~100 from 256 to millions of nodes instead of
exploding." Recomputing the table under it:

| sims | σ (live LINEAR) | rel | σ (LOG root) | rel |
|---|---|---|---|---|
| 16 | 1.350 | 0.50x | 47.66 | 0.991x |
| 64 | 1.450 | 0.53x | 47.69 | 0.992x |
| 256 | 2.725 | 1.00x | 48.07 | 1.000x |
| 1024 | 6.400 | **2.35x** | 49.07 | **1.021x** |

**Under the log root, σ varies by 3% across a 64x budget range instead of
4.7x.** That is the prerequisite for variable-node search, and it is the real
finding of this section:

> **Variable-node search is not deployable on the selfplay path until the
> selfplay root transform is sim-invariant.** The log root is the mechanism, it
> is already implemented and already tuned, but it currently ships only on the
> PLAY path.

⚑ **And it cannot currently reach the selfplay path at all.** `network_turn.py:784-796`
constructs its `GumbelConfig` with an explicit field list — `simulations`, `topk`,
`temperature`, `c_scale`, `add_noise`, `gumbel_scale`, the encodings, and the three
volatility knobs. `c_visit_root`, `c_scale_root` and `q_visit_exp_root` are **never
passed**, so they sit at their `GumbelConfig` sentinels (`-1`, `-1`, `99.0`) and the
linear transform is selected at both sites. Setting them in the yaml today would
change nothing: there is no plumbing from `SearchConfig` to the search. That is the
"knob that never reaches the worker" pattern, and closing it is a prerequisite task
in its own right — roughly `SearchConfig` → reco → worker → `network_turn.py`, plus
the yaml validator.

### 7d. …and that prerequisite is itself a bigger experiment

Two reasons it cannot be waved through:

1. **`c_scale` is measured to be sim-specific and the project has a test
   enforcing it.** `gumbel.py:56-67`: *"the optimal q_scale keys off the TOTAL
   sim budget, so no single value is sim-invariant and each deployment gets its
   own measured optimum"*, with the measurement attached — selfplay @256 sims:
   `c_scale` 0.1 → 0.688 puzzle accuracy, 0.05 → 0.652, 0.025 → 0.598.
   `tests/test_selfplay_gumbel_c_scale.py` fails if anyone unifies the two
   values. **A ~9 percentage-point spread from mis-tuning σ is larger than
   anything the reallocation plausibly buys.** A variable-node scheme runs most
   positions at a `c_scale` tuned for a budget they are not using.
2. **The ledger has already been burned by exactly this.** Ledger §"Confound 1"
   records a sims ladder invalidated because `c_scale` was held fixed across
   rungs, and a −52.5 Elo result that did not survive the shape correction
   (corrected to +5.8 [−56.3, +68.3]). The same ledger records that the C17
   virtual-loss fix is **dose-dependent on sim count** — +4.5cp at 256 sims,
   −0.4cp and −1.4 top-1 at 32. A corpus with heterogeneous sim counts inherits
   that dose-dependence on every tuned search knob simultaneously.

**Second-order consequence, and it belongs in any entry that proposes this:** a
variable-sim corpus would confound every pooled shape statistic we currently
publish — `data_policy_entropy`, `gumbel_policy_entropy_sum`,
`gumbel_policy_top_prob_sum`, and the in-flight comparison against lc0's target
shape. Those become mixtures over an uncontrolled σ distribution. They would
have to be stratified by realized sim count, which means **the realized sim
count must be stored per row** — it currently is not (§10).

---

## 8. PID interaction

The controller (`chess_anti_engine/stockfish/pid.py`) drives two levers, both
Stockfish-side — `wdl_regret` (direction −1) and `sf_nodes` (direction +1) — and
its **input is our net's raw winrate against SF**, regulated to
`target_winrate = 0.50`.

> **⚑ An earlier revision of this section said 0.60. That number is the CODE
> DEFAULT, never the realized setpoint** — the recurring trap of reading a
> default as a live value (`docs/rl_loop_audit.md` method rules; memory
> `params_json_is_the_launch_config`, `dead_config_keys_pin_to_realized`).
> Its provenance is exactly two places:
> `pid.py:442` (`target_winrate: float = 0.60`) and `pid.py:962`
> (`config.get("sf_pid_target_winrate", 0.60)`). It *was* the yaml value, set
> in `00d1197` (2026-04-15), and was lowered later — 0.58 → 0.575 (2026-05-29)
> → **0.50** (2026-06-29).
>
> Realized, verified read-only on the live trial `train_trial_379f6_00000`
> (iteration 677): `params.json` → `sf_pid_target_winrate = 0.5`,
> `pid_ema_winrate = 0.4918`, `pid_raw_winrate = 0.500`. Config side:
> `configs/pbt2_small.yaml:95` on `main` → `0.50`; the live tree's copy → `0.5`.
> The controller is sitting on its setpoint, and that setpoint is 0.50.
>
> **§8's conclusion survives the correction, but only one of the two supporting
> arguments does. Both are stated here, because an earlier revision claimed the
> stronger one and it is false.**
>
> **Holds — the control law is setpoint-invariant.** The setpoint enters the
> loop only as a subtrahend: `err = self.ema_winrate - self.target`
> (`pid.py:800`). The negative feedback that erases the outcome signal exists
> for *any* target, so §8's conclusion — winrate is a *controlled* variable and
> therefore cannot serve as a yardstick — holds at 0.50, at 0.60, and at
> whatever the yaml carries next. That is the argument the section needs, and it
> is sufficient on its own.
>
> **⚑ Does NOT hold — "nothing downstream divides by the setpoint".** That claim
> was made here and is **false**. `tune/trainable_metrics.py:179-181` divides by
> it:
>
> ```python
> target = max(0.01, float(pid_target_winrate))
> winrate_factor = max(0.5, min(1.0, max(0.0, float(ema_winrate)) / target))
> return difficulty * winrate_factor
> ```
>
> The `max(0.01, ...)` clamp is the tell — you only guard a divisor. It is wired
> to the real setpoint at `tune/trainable_phases.py:756`
> (`pid_target_winrate=tc.sf_pid_target_winrate`).
>
> **Consequence: the reported difficulty metric is not setpoint-invariant.** At
> the live `pid_ema_winrate` (0.4976, iteration 686) the factor is 0.9952 at
> target 0.50 and 0.8293 at 0.60 — the **same run scores 16.7% lower** purely
> from the setpoint. So a difficulty/strength number is comparable across
> setpoints only if the setpoint is quoted with it; a cross-era comparison that
> spans the 2026-06-29 change is measuring partly the change itself.
>
> **Why the original grep missed it, so the next reader does not repeat it:**
> the divisor is a *renamed local* (`target`) fed by a *differently named*
> parameter (`pid_target_winrate`). Grepping `target_winrate`, `self.target`, or
> `0.6` finds nothing at that site. The lesson generalises — a value's
> reachability cannot be established by grepping the name it has at its source.
>
> Other setpoint reads, for completeness: `ema_winrate` is seeded to the target
> at construction (`pid.py:527`), an initial condition rather than a yardstick,
> and **four** `_seed_lever_history(..., target_wr=self.target)` sites
> (`pid.py:586`, `:587`, `:745`, `:747`) — an earlier revision cited only one.

Variable sims never touches a PID lever, so there is no direct coupling. The
coupling is through the winrate channel and it is worse than a direct one:

- A scheme that degrades the played move (§6a) lowers winrate. The PID reads
  that as "Stockfish is too strong" and **raises regret / lowers nodes** — a
  weaker opponent, which pushes winrate back to the setpoint (0.50 today; the
  mechanism is the feedback, not the number).
- **So the controller actively erases the outcome signal.** Winrate is not a
  yardstick for this experiment; it is the controlled variable. Any readout that
  watches winrate is watching a quantity the loop is regulating, and it cannot
  fail. That is the "gate that cannot fail" pattern.
- The damage instead shows up displaced onto the levers: `opponent_wdl_regret_limit`
  drifting up and `sf_nodes` drifting down. **Those are the observables**, and
  they must be logged per iteration in any entry.
- Secondary: an easier opponent is also a *cheaper* opponent, which lowers SF
  CPU, which changes the §1 resource balance mid-readout. The experiment would
  perturb its own binding-resource measurement.

---

## 9. C-path implementation cost

Production selfplay runs the C path — `network_turn.py:800-825` calls
`run_gumbel_root_many_c` whenever `_HAS_GUMBEL_C` and volatility search is off
(it is). A Python-only scheme is not deployable.

The good news: **the expensive part is already done.**

- `per_game_simulations` is plumbed through `gumbel_c.py:355/383/410` to
  `budget_remaining` at `gumbel_c.py:717-718`, and the C schedule is *recomputed
  every round* from `budget_remaining` (`gss_begin_round`) rather than
  precomputed from `n`. A per-position budget needs **no C change at all.**
- Mid-search budget zeroing already exists in C — `gss_begin_round` zeroes
  `budget_remaining[i]` when the root is solved (terminal/TB), so the
  "truncate a search in flight and still emit a target" path is exercised in
  production today.

What a *pre-search* variable-budget scheme (§6b, the recommended one) needs:

| change | where | size |
|---|---|---|
| compute a per-position budget | `network_turn.py` `sub_sims` construction | Python, ~20 lines |
| store the realized sim count per row | `_NetRecord` → `finalize.py` → shard schema + `resume.py` arrays | Python, ~5 files, schema bump |
| **C changes** | — | **none** |

What a *mid-search* stopping scheme would additionally need — and this is the
cost of the rejected option:

| change | where | size |
|---|---|---|
| snapshot the improved policy at rep boundaries | new C function alongside `gss_score_and_halve` | C, and it must recompute the full `softmax(log prior + σQ̃)` per board per rep — not currently computed until the search ends |
| KL between snapshots + threshold + per-board early exit | `gss_begin_round` / `continue_gumbel_sims` | C |
| expose threshold + interval knobs | `gumbel_c.py` arg tuple, `GumbelConfig`, `SearchConfig`, worker reco, yaml validator | ~6 files |
| Python-path parity | `gumbel.py` | mirror of the above |

⚑ **The dispatch guard matters more than the CLI.** The known failure mode is a
`GumbelConfig` field that the Python path honours and the C path silently drops
— a knob that returns a flawless null. Note `gumbel.py` already documents a
class of fields as **C path only** (`q_global_scale`, `q_visit_floor`,
`halving_div`, `c_visit_root`, `c_scale_root`, `q_visit_exp_root`). Any new
stopping knob must be added to whatever inert/py-only registry guards this, and
**tested at the dispatch site in `network_turn.py`, not at the CLI** — the
production path never goes through a CLI.

Also required regardless of scheme: **the live-yaml validator is
all-or-nothing.** New config keys need a restart onto code that defines them, or
the whole reload is rejected.

---

## 10. What the banked shards can and cannot support

**Can (done, §3):** the distribution of `KL(net prior ‖ search target)` at the
production budget, per position, by ply, over 111k rows, with no GPU. That is a
real measurement and it establishes the reallocation headroom.

**Cannot — and this is stated rather than modelled around:**

- **A KLD-gain trajectory cannot be reconstructed from stored data.** Shards
  store `policy_target` (the final improved policy) and no visit counts, no
  per-round snapshots, and no realized sim count. There is no intermediate state
  to difference.
- Even if visits *were* stored, §5 says the trajectory would be the schedule.
- **The realized sim count is not stored.** The corpus is 100% full-ply
  (playout-capped rows are dropped at `finalize.py`), so every stored row is a
  256-sim row and there is no within-corpus sim contrast to exploit. A
  variable-node deploy would need this field added *before* launch, or it would
  be unmeasurable after the fact.

Any number purporting to be "the fraction of our searches that would stop early"
would therefore be **modelled, not measured**, and this document does not
produce one.

---

## 11. The screen that would settle it — and it is fully offline

The whole proposal reduces to one question:

> **Is target quality CONVEX in sim count over [64, 1024]?**

At fixed total budget, reallocating sims from cheap positions to expensive ones
pays **iff** the marginal value of a sim is increasing in the region you are
moving sims *to*. If the sims→quality curve is concave — the usual
diminishing-returns shape — then taking 192 sims from a 256-sim position and
giving them to another 256-sim position **loses on both ends**, and no trigger,
however clever, can rescue it.

This is decidable offline, with no training compute, and the audit-first rule in
`docs/eval_protocol.md` requires it before any training-target candidate spends
compute. Sketch (do not run concurrently with a 256+ sim arena; use
`--gpu-mem-fraction`):

```
PYTHONPATH=. python3 scripts/audit_targets.py \
    --checkpoint <banked ckpt> --max-positions 2000 \
    --sims {64,128,256,512,1024} --gumbel-topk 32 \
    --gpu-mem-fraction 0.15 \
    --dump-per-position <dump_sims_N.jsonl>
```

Three things must be true of the run for it to mean anything:

1. **`--dump-per-position` is mandatory.** The pooled mean cannot answer a
   convexity question and cannot test the reallocation hypothesis. What is
   needed is the *per-position* curve, paired across rungs
   (`scripts/paired_compare.py --join-key key`).
2. **Pin the live shape explicitly.** `arena_standard.py`'s `--search-shape
   training` reads `production_selfplay_search_config()` from the yaml, so it
   tracks live (`c_scale` 0.025, `topk` 32) — but the *comments* in
   `arena_standard.py:72-74` and `docs/eval_protocol.md:35-37` still say
   `c_scale 0.1, topk 16`, which is stale. Read the realized knobs the script
   prints at startup; do not trust the docstring.
3. **The `c_scale` confound is unavoidable and must be labelled.** Holding
   `c_scale` fixed across a sims ladder is the exact confound the ledger records
   ("Confound 1"). Under the live *linear* root it cannot be avoided — σ is
   supposed to move with the budget. **So the ladder should be run twice: once
   with the live linear root, once with the log root of §7c.** The linear run
   measures what a naive deploy would do; the log run measures the sims→quality
   curve with σ held ~constant, which is the one that actually answers the
   convexity question. Running only the linear ladder would repeat a known
   mistake.

**Pre-committed decision rule:** if the paired per-position `cand.search.exp`
curve is concave over [64, 1024] under the log root — i.e. the 256→1024 gain is
no larger than the 64→256 loss — **the idea is dead and this document is closed
with a FAILED verdict.** No search-side code, no ledger entry, no live change.

**Cost:** four extra `audit_targets` runs at ~4.4 min each. That is the entire
price of a permanent answer.

---

## 12. Recommendation against the standing bar

The bar: the loop gains ~0.02 Elo/iter against a best instrument resolving
~2.74 Elo/DAY, so an experiment is worth running only if it is worth
**≥5–8 Elo/day** or can be **screened offline**.

| framing | verdict |
|---|---|
| Throughput | **DEAD.** ≲10% more games/h at best (§1c), and the loop's own measurement says throughput and strength are only loosely coupled anyway. Nowhere near 5–8 Elo/day. |
| Data quality, as proposed (KLD gain on visits) | **DEAD.** The signal is the halving schedule, not the position (§5). Not a tuning problem — a category error. |
| Data quality, reformulated (informed budget allocation, complete schedule) | **NOT ESTABLISHED, but offline-screenable — which is the escape hatch the bar allows.** |

**Recommendation: NO-GO on implementation. DEFER the concept behind the §11
screen.**

Reasoning, stated so it can be argued with:

- The reallocation headroom is real and large (46% of positions → 2.8% of the
  policy movement). This is the best argument and it should not be dismissed.
- But it is *not* evidence that reallocation pays. Those positions are cheap
  precisely because the prior already agrees with the search. Taking sims from
  them makes their targets **more** prior-like (§7b), which amplifies the
  self-referential fixed point rather than relieving it. The headroom points at
  the right positions for the wrong reason.
- The payoff therefore hinges entirely on the *other* end — whether 1024 sims on
  a hard position produces a materially better target than 256. That is the
  convexity question, and the prior for it is negative. Diminishing returns is
  the default shape; more pointedly, the ledger's own **shape-corrected** sims
  ladder found **256 v 32 = +5.8 Elo [−56.3, +68.3]** on the training shape with
  breadth pinned — an 8x sim increase with no detectable strength gain and a CI
  that comfortably contains zero. If eight times the search buys nothing
  measurable, redistributing sims within a factor of four is unlikely to.
  Settling it costs ~20 minutes of offline GPU.
- Deploying without that screen would mean shipping a change that (a) needs the
  selfplay root transform swapped first (§7c), (b) runs most positions at a
  mis-tuned `c_scale` with a measured 9pp cost (§7d), (c) confounds every pooled
  shape statistic we publish including an in-flight lc0 comparison, and (d)
  cannot be judged on winrate because the PID regulates it (§8). That is four
  compounding confounds bought with an unmeasured hypothesis.

If the screen shows convexity, the follow-up is **not** this design. It is the
much narrower "replace the random playout-cap selector with an informed one"
(§6b) — which needs zero C changes, reuses machinery that is already in
production, and can be pre-registered as a single-knob experiment.

---

## 13. What could NOT be established

- **How often lc0's KLD stopper actually fires in T80**, and the realized node
  distribution it produces. The flag semantics and the exact computation are
  sourced verbatim (§2); its realized behaviour is not, and nothing in this
  document depends on it.
- **What our own KLD-gain trajectory would look like.** Not derivable from
  banked shards — no visit counts, no per-round snapshots, no realized sim count
  (§10). No modelled number is offered in place of the measurement.
- **Whether target quality is convex in sim count** — the question the whole
  proposal turns on. It is not answered here; §11 is the design for answering it.
- **Whether the log root is safe on the selfplay path.** §7c establishes it makes
  σ sim-invariant (arithmetic, from the code). It does **not** establish that a
  log-root selfplay search produces better targets at 256 sims than the live
  linear root — that is an untested change with its own `c_scale` retuning, and
  it is a prerequisite, not a free precondition.
- **Whether §1's regime holds tomorrow.** `sf_nodes` is PID-owned and floating.
  The search-bound finding is 6 hours old at the time of writing and the July
  measurement found the opposite. Re-measure before acting.
- **Where inside `run_network_turn` the time goes** — GPU wait vs GIL vs C tree
  work. `network` is one undifferentiated counter. The GIL probe (mean 6.2ms,
  83% of samples over 1ms) says contention is material, and rep count falls
  super-linearly with budget (59 reps at 256 sims vs 8 at 64), so a budget cut
  would help more than proportionally — but that is an inference, not a
  measurement, and §1c makes it moot regardless.
