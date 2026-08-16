# Handoff — CAST / issue #425 / the MultiPV-6 tail

**Written 2026-08-16.** For whoever picks this up next (Claude, Codex, or Grok).
**All offline analysis that can answer these research questions from existing data is
finished.** ⚑ That is deliberately narrower than "everything without a GPU is done":
the replay-schema plumbing in §7 is CPU-only engineering that remains open. What it
cannot do without waiting is *produce* the data — that needs production selfplay to
accumulate, and any Elo claim needs a scheduled arena.

⚑ Read `docs/experiment_ledger.md` entries dated 2026-08-14 → 2026-08-16 for the
full record. This file is the map, not the territory.

---

## 1. The one-paragraph version

We investigated whether CAST-style solver credit and adaptive SF label allocation
could improve our training targets. **Three branches closed as negatives, one
correctness defect was found and quantified, and one capability was built.** No
training change was made; no config was touched; nothing here has moved Elo,
and nothing here *can* until someone turns on a loss.

---

## 2. What is SETTLED — do not re-litigate

| result | verdict | evidence |
|---|---|---|
| CAST's headline coverage claim | **NULL** | pairs reach 18.4% of rows vs `sf_p0_regret`'s 21.0%; both gated by "is the previous ply also stored", not MultiPV width |
| Relational SF6 supervision (pairwise / listwise / indifference) | **NEGATIVE, structurally** | 0.9049 vs 0.9158 held-out; and the top-6-only variant is *geometrically forbidden* before any loss design |
| The live imputed tail (`α = 1`, the midpoint rule) | **DECISIVELY WRONG** | overstates the tail **2.1–2.9×** across every arm, population and instrument |
| A rank-shaped tail | **NOT IMPLEMENTABLE** | hidden SF rank does not exist at live MultiPV 6 |

### The algebra that closes the relational branch

For `L = Σ_a p_a r_a` with `p = softmax(z)`:

```
dL/dz_i = p_i (r_i − E_p[r]) = Σ_j p_i p_j (r_i − r_j)     (exact, 2.8e-17)
```

**The reference gradient is already a pairwise object whose pair term is the regret
difference.** So relational supervision is not a different *kind* of information —
it is the same object with one block deleted. Block shares of ‖g_true‖:
surfaced×tail **0.7769** (needs a magnitude), tail×tail **0.2137** (zeroed by any
constant tail). That is what caps every constant tail near cos ≈ 0.905.

⇒ This is *why* ΔQ is the right successor: the algebra names the missing block.

---

## 3. What is RETRACTED — these will be rediscovered, don't believe them

This project has repeatedly had superseded numbers dug out of old commits and
treated as current. Five things in this thread are **wrong**:

1. **"The tail is ~5× too harsh."** Withdrawn. It proxied the played move by
   `argmax(policy_target)`, which under Gumbel at final temperature 0 is not the
   sequential-halving survivor (they agree 75.8%). Different move graded on both sides.
2. **"α = 0.176 transfers" / "does not transfer."** Both withdrawn. The primary arm's
   estimand is ill-posed — you cannot vary MultiPV width while holding search strength.
   The PRIMARY (B) *covers* 0.1779; MATCHED misses by 0.0026, a coin flip.
3. **"MATCHED is an upper bound."** Retracted — "a weaker search sees bigger gaps"
   plus one paired observation is not a monotonicity argument.
4. **"We own 60% of the ΔQ dataset."** Overclaimed → it is an availability *proxy*.
5. **"P(query | confidence) is monotone, 4.2×, a usable gate."** **RETRACTED** — binned
   over the wrong population. The real curve is flat above confidence 0.4.

**Quote instead:** *the midpoint is 2–3× too harsh; α is O(0.1–0.25); the
training-relevant number is α_grad ≈ 0.14 [0.13, 0.16].*

---

## 4. Live state — what is and is not in effect

⚑⚑ **READ THESE OFF THE LIVE TREE, NOT OFF THIS BRANCH.** `configs/pbt2_small.yaml`
differs between the live branch and `main` on ~55 keys, and this one is among them:

| | `w_sf_own_regret` |
|---|---|
| **LIVE** — `/home/josh/projects/chess` on `ops/live-20260725`, line 486 | **`0.0`** |
| this branch / `main`, line 862 | `0.7` ⚑ **not what production runs** |

An earlier revision of this file stated `0.0` unqualified, which is *false on the
branch the file lives on* — a reader checking the in-tree config would have found
`0.7` and concluded the doc was wrong. This is the stale-config trap PR #443 exists
to eliminate; do not make production claims from the in-tree config.

- ⇒ **the tail defect costs us nothing today**, because the LIVE weight is `0.0`.
- **`record_sf_p0_regret: true`** — so the (wrong) vector is banked into *every shard*.
  Latent, not inert. This is what re-opens the old `w_sf_own_regret: 0.7` verdict,
  which ran at MultiPV **40**.
- Production labels are MultiPV 6, realized nodes median **150,071** (pinned at the
  floor; the 200k cap is not what runs).

---

## 5. Open PRs from this thread

| PR | contents | state |
|---|---|---|
| **#428** | `tail_censor_screen.py`, `mpv6_tail_panel.py`, `dq_free_dataset_screen.py`, ledger | reviewed, 15/15 threads resolved, CI green |
| **#444** | `searchmoves` in `uci.py` + forwarded through `StockfishPool` | independently reviewed **and** delta-reviewed; **both rounds produced findings, all subsequently fixed**; current head `74a30cd1`, CI green on that head; 1/1 threads resolved |
| branch `relational/sf6-screen` | `relational_sf6_screen.py` — backs the relational negative | **pushed, no PR.** 100-file diff off an older base; its verdict is banked. Retrievable for verification, not proposed for merge. |

⚑ **Do not read #444 as carrying a post-fix approval.** The delta review returned
APPROVE-WITH-CHANGES — it found the 43-test property battery vacuous for the very
`$`→`\Z` mutation it claimed to catch, and the source-grep ordering assertion to be
theatre. Both were fixed in `74a30cd1` (three previously-surviving mutants now die;
the theatre test deleted), and the closure is posted on the PR. **No third review has
happened.** If one is wanted, it should be a fresh delta on `74a30cd1`.

⚑ **CI green means less than it looks**: this repo's CI does not re-run when the base
advances. Re-check before merging.

---

## 6. What is BUILT and ready to use

**`searchmoves` (PR #444)** — restrict a Stockfish search to a listed set of root
moves, so the whole node budget separates the candidates you care about. Now
forwarded through `StockfishPool`, so it is reachable from pooled callers.

Two traps documented at the call site:
- **`searchmoves` MUST be last on the `go` line.** Per UCI it consumes every
  remaining token: `go nodes 1000 searchmoves e2e4 movetime 50` restricts the root to
  `{e2e4, movetime, 50}` and *the time limit silently vanishes*.
- **`[]` and `None` both mean "no restriction"** and emit the byte-identical
  pre-existing `go` line. A caller that computes a filtered list which comes out
  empty gets a **full-width** search, not an error.

⚑ **"Both candidates outside SF6" is ONE query, not two** — `searchmoves a_P a_M` at
MultiPV 2 returns both evaluations.

---

## 7. The next experiment, and the cheap way to unblock it

**ΔQ phase 1 is pre-registered and not started.** Its target is
`P(buying the missing evaluation changes the training decision | pre-query diagnostics)`.
Kill rule: **below 10% material flips on ≥300 disagreement rows, the branch dies.**

### The blocker, and the fix worth proposing

The live question pairs `a_P = argmax π_θ(s)` against `a_M = MCTS_θ(s)` for **one θ**.
Offline we can only pair a *current* checkpoint's prior against a *historical* net's
played move. **This is not fixable with better historical data** — the replay schema
never persists the generating prior (`_NetRecord.policy_probs` exists only in
selfplay memory; only the improved `policy_target` reaches a shard).

**Proposed fix — cheap, and it makes the dataset accumulate for free.** Persist two
small fields per row at selfplay time: `prior_top1_index` (int16) and
`prior_top1_prob` (float16), ≈4 bytes/row. At selfplay time the prior and the MCTS
move are the *same* θ by construction, so every future shard yields same-model pairs,
on every future checkpoint, with no dedicated run.

⚑ This is a **data-affecting change**: it needs its own ledger entry first, and the
code must be deployed **before** any yaml key referencing it — an unknown key is
survivable mid-run but **fatal at launch**.

### Sequencing — do these in this order

1. **Drain/merge the instrumentation you need to trust the experiment first** (#423 is
   the most mature; #442 and #443 explicitly still want independent review; #432 and
   #438 are substantial instrument changes). An experiment run on unmerged, unreviewed
   instruments cannot be judged.
2. **Then add the two `prior_top1_*` replay fields and let them accumulate passively.**
3. ⚑ **Do NOT schedule a dedicated ΔQ GPU run merely to manufacture same-model pairs.**
   Normal selfplay creates that dataset for ~4 bytes/row. Spend GPU on the arena that
   judges a training change, not on generating rows production is already generating.

### Do NOT build a confidence gate

The graded gate is dead (§3.5). Collect the full dataset first, then ask whether *any*
pre-query diagnostic — possibly multivariate — predicts material decision changes.
Do not try to rescue the gate from the historical proxy after it has already reported
its limitations. Repeat on a second checkpoint before promoting any allocator:
this project has already demonstrated checkpoint-dependent **sign reversals**.

---

## 8. If the goal is Elo, this is not the lane

Nothing in this thread is an Elo lever. It is instrument repair. The ledger names
better candidates:

1. **"Our gap is TRAINING, not capacity"** — half our params beats us. Largest
   unexplained result on the board.
2. **The policy-target bad-tail repair arm that already passes its gate.** ⚑ It must
   move **bad-tail Q −0.278 vs −0.304**, *not* the 21.5cp mean.

Both are ahead of ΔQ for Elo. Either needs a scheduled paired arena with the **same
search on both sides** — the only instrument here that has survived scrutiny.
⚑ Pause-and-run beats concurrent GPU work by **1.89×**, and concurrent arenas have
OOMed training twice.

---

## 9. Method lessons that cost us real time here

These generalise beyond this thread and are the most transferable output.

- **⚑⚑ A property test can be vacuous for its own regression.** A 43-test battery
  written to catch a `$`→`\Z` bug passed **15/15 with the bug reintroduced**: the
  validator runs syntax *then* legality, `chess.Move.from_uci` raises a `ValueError`
  too, so a boolean oracle over the pipeline could not attribute the rejection.
  ⇒ **When testing one stage, find the input regime that disables the others**, or you
  are testing the disjunction. Mutation-verify each claim, not the suite.
- **⚑ A source-grep guard is theatre.** `inspect.getsource`-based assertions stayed
  green through the exact refactor they claimed to forbid, because the ordering lived
  in the *caller*. Assert on observed behaviour. A guard that cannot fail is worse
  than none — it reads as assurance.
- **⚑ Read the lint EXIT CODE.** Grepping lint output for `error:` misses ruff's
  channel entirely; a hard red was reported as green. And a worktree without built C
  extensions emits ~140 masking errors — build them first, and never copy the live
  tree's `.so` (the C source differs between branches).
- **⚑ When two framings of one question coexist, make them different TYPES.** A
  partial fix that leaves one consumer on the old framing is *worse* than no fix,
  because the correction reads as proof everything was checked.
- **⚑ Freeze the observations, iterate the estimator.** Now CLAUDE.md protocol rule 6.
  Because the panel banked its raw PVs, swapping a row bootstrap for a game-cluster
  bootstrap was a re-analysis of the same 500 positions — every point estimate came
  back byte-identical, which is itself proof the fix changed only what it claimed.
- **⚑ Check rows-per-cluster before believing a clustered CI.** Clustering barely moved
  the panel because its sampler already drew ~1 position per game (1.26 rows/game).
  "I clustered and nothing moved" can mean there were no clusters.
- **⚑ Never condition a split on the quantity being compared.** The ΔQ-magnitude split
  looked like a finding and was algebraically forced.

---

## 10. Cross-agent note

`docs/experiment_ledger.md` and `CLAUDE.md` are the **shared** surface — Claude, Codex
and Grok all read them. Claude's `memory/` directory is **not** visible to the others,
which is why every durable result and retraction above was written into the ledger and
CLAUDE.md rather than only into memory.

The practical bottleneck for anyone picking this up is not this thread: it is that
**10 PRs are open against `main`**, several of them instrument-correctness fixes in the
same family (#423, #432, #438, #442, #443). Draining that queue is probably worth more
than starting a new experiment.
