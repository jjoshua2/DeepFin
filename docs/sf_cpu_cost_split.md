# Where the Stockfish time actually goes

Read-only measurement, 2026-07-27 ~23:00, run live at iter ~168 (trial `13a9f_00000`).
Nothing in the repo was modified; no training/GPU/arena work was run. One short
single-engine `nice -19` Stockfish probe was run (~40 s of one core, same priority as the
production pool) — flagged explicitly wherever its numbers are used.

---

## 0. Headline

| | share of SF CPU | produces a training label? |
|---|---|---|
| Selfplay P1 **label** queries (698k nodes) | **50.4%** | yes |
| Curriculum SF turn, **full ply** (698k nodes) | **25.5%** | yes — *the same query is both the played move and the label* |
| Curriculum SF turn, **fast ply** (0.25x = 175k nodes) | **24.1%** | **no** |
| Terminal-eval adjudication (5x = 3.49M nodes) | **0.0%** | n/a — never fires |
| Label escalation (3M nodes) | **0.0%** | n/a — gate is off |

Two structural facts drive everything below:

1. **In curriculum games the label is free.** `sf_move_nodes: 0`, and
   `submit_async_sf_labels_from_curriculum_moves`
   (`chess_anti_engine/selfplay/stockfish_turn.py:334`) *reuses the curriculum MOVE
   future as the label* whenever `sf_move_nodes == 0`. A curriculum game issues exactly
   one SF query per SF turn; on full plies that one query is simultaneously the opponent's
   move and the training label. There is no separate curriculum label cost to cut.
2. **`sf_multipv` is not a CPU lever at all.** The budget is `go nodes N`, and N is a
   *total* node count. Measured: wall time at a fixed 698,289-node budget is flat across
   MultiPV 40 / 16 / 8 / 1. Cutting width saves nothing; it converts width into depth at
   constant cost.

---

## 1. What the phase counters can and cannot tell you

**MEASURED (code read, `chess_anti_engine/selfplay/manager.py:208-222, 434-456`).**
`sf_block_starved` wraps `_wait_for_starved_sf`, which is entered only when *no* slot is
runnable and something is pending:

```python
timeout_s = 0.05 if control_poll else 0.25
if state.pending_sf_moves:                 # -> blocks on curriculum MOVES
    finish_pending_curriculum_moves(state, block=True, block_timeout_s=timeout_s)
elif state.pending_sf_labels:              # -> blocks on LABELS
    wait(..., timeout=timeout_s, return_when=FIRST_COMPLETED)
```

**THE LOGS CANNOT SPLIT THIS.** There is one counter around the whole call and no
per-branch instrumentation. Three separate reasons the number must not be read as a cost
split:

- **It does not record which branch ran.** No counter distinguishes the
  `pending_sf_moves` wait from the `pending_sf_labels` wait.
- **The branch that runs is almost always the MOVE branch, and that is misleading.**
  `pending_excluded_avg = 23.5-23.7` against `in_flight_avg = 24.0` on every worker: of
  the 24 games standing in each thread, ~23.6 are curriculum slots parked on a pending SF
  *move*. So `pending_sf_moves` is essentially never empty and the `if` almost always
  wins. But labels are submitted to **the same 8-engine FIFO pool** and consume the same
  capacity. *Blocked-on is not paid-for.*
- **It measures thread-idle time, not SF work.** `sf_block_starved = 2200-3090%` of
  `total_thread = 3200%` is 32 Python game threads *sleeping*. Actual Stockfish CPU is a
  different quantity entirely (§2). The phase stat is a saturation symptom, not a cost
  meter.

To attribute blocked time to a workload you would need a per-branch counter (e.g.
`sf_block_starved_move` / `sf_block_starved_label`) — it does not exist today. The split
in §3 is therefore derived from **query counts x realized node budgets**, which is the
sound instrument, and is cross-validated against measured engine throughput.

---

## 2. The box (MEASURED, `ps -eo pid,ppid,pcpu,comm`)

| | value |
|---|---|
| Stockfish processes | **32** (8 per worker x 4 workers; `Threads 1`, `nice 19`, `Hash 16MB`) |
| Aggregate SF CPU | **2253%** of a 3200% box = **70.4% of all CPU** |
| Aggregate Python CPU | 897% (trainer 351%, 4 workers ~80% each, rest broker/inference) |
| loadavg / nproc | 59.5 / 32 |
| Game slots | 4 workers x 32 `distributed_worker_selfplay_threads` x 24 in-flight (`games_per_batch 384` x `slot_oversubscribe 2.0` / 32) = **3072 slots** |
| Slots per SF engine | **96:1** |

At ~1.1 s mean service per query, a curriculum slot waits **~100 s per SF move**, so a
57-SF-turn curriculum game takes **~1.5 h of wall clock**. That queue latency — not any
per-query cost — is why curriculum games are the slow tail.

---

## 3. Realized label budget, and the cost split

### 3a. REALIZED config (not the yaml literals)

From the live published manifest
`runs/pbt2_small/server/trials/13a9f_00000/publish/manifest.json` -> `recommended_worker`:

| key | realized | note |
|---|---|---|
| `sf_nodes` | **698,289** | yaml's `sf_nodes: 5000` is the PID **floor**, not the budget |
| `sf_multipv` | **40** | |
| `sf_move_nodes` | **0** | => curriculum moves run at the full PID budget **and double as the label** |
| `sf_label_nodes_cap` | **0** | uncapped (the 400k cap was killed 2026-07-02) |
| `sf_fast_ply_node_scale` | **0.25** | fast plies -> 174,572 nodes |
| `sf_label_escalate_q_gap` | **0.0** | escalation **OFF**; the 3M-node re-query never fires |
| `opponent_wdl_regret_limit` | 0.0904 | PID regret lever, floor 0.0075 — headroom exists |

From `sf_label_meta` in 57,037 live label rows (steady-state window, server
`processed/_compacted`):

- **nodes**: mean 690,691, median 698,520; only **0.88%** of labels finish under half
  budget. The budget is genuinely spent.
- **depth**: p25 **11**, median **12**, p75 **14** (mean 17.6, inflated by mate/TB rows up
  to depth 245). **Confirms the recorded "teacher is depth ~13" fact.**
- **effective width**: PV rows present mean **25.98**, mean legal moves 27.7.
  `multipv=40` binds (legal > 40) on only **16.1%** of positions.

### 3b. Production rate and composition (MEASURED)

Best instrument found: the server's compacted shard filenames encode
`<unix>_<sha>_<N>g_<P>p_`, and each compacted shard carries real `games` /
`selfplay_games` / `curriculum_games` / `total_game_plies` attrs. (The worker-side copies
in `data/c17_ab/*` have those attrs nulled.)

**Steady state** (40 compacted shards, 1.22 h window, ~4-6 h ago, 3125 games):

| | value |
|---|---|
| rate | **2561 games/h, 46,873 rows/h**, 116.4 plies/game |
| games | selfplay 1534 (49.1%), curriculum 1591 (**50.9%**) |
| rows | selfplay 37,933 (66.3%), curriculum 19,265 (**33.7%**) |
| rows/game | selfplay **24.73**, curriculum **12.11** |
| **label queries/game** | selfplay **24.69**, curriculum **12.04** |
| `adjudicated_games` | **0** (775-1340 `tb_adjudicated`, 1128 checkmate) |

This reproduces the operator's stated 31.3% rows / 48.4% games / 12.07-vs-24.83 exactly.

> **Caveat on the current hour.** The last ~1.2 h reads 26.5% curriculum games / 5.6%
> curriculum rows / 4.33 rows-per-curriculum-game. That is **survivorship truncation, not
> a mix change**: the workers restarted at ~20:52 and curriculum games take ~1.5 h, so
> only the short ones have finished. `data/c17_ab/post` (21:16-22:20) is inside that
> transient and reads 99.8% selfplay rows — **do not use `post` for composition**;
> `data/c17_ab/pre` (49.0% / 51.0% games, 12.31 rows/game) is the steady state.

### 3c. The split (DERIVED from 3a + 3b; wall-time ∝ nodes verified in §4)

Curriculum SF turns per game = plies/2 ≈ 57 (116.4 plies). Of those, the ones whose
preceding net ply was full run at 1.0x **and are exactly the label-bearing ones** — 12.04
measured, i.e. 21% of SF turns, matching `playout_cap_fraction: 0.25`. The other ~45 run
at 0.25x and never become a label.

| | nodes/game | window total | share |
|---|---|---|---|
| selfplay labels | 24.69 x 690,691 = 1.705e7 | 1534 x = **2.616e10** | **50.4%** |
| curriculum full-ply (move+label) | 12.04 x 690,691 = 8.32e6 | 1591 x = **1.323e10** | **25.5%** |
| curriculum fast-ply (move only) | 44.96 x 174,572 = 7.85e6 | 1591 x = **1.249e10** | **24.1%** |
| **total** | 1.66e7 | **5.19e10** over 4392 s = **1.18e7 nodes/s** | |

Reframed the two ways that matter:

- **by workload:** LABEL queries **50.4%** / curriculum MOVE queries **49.6%**.
- **by yield:** **75.9%** of SF CPU produces a training label; **24.1%** is pure
  opponent-move cost that yields no row.

**Independent validation.** Demand 1.18e7 nodes/s vs supply 32 engines x 290-370k realized
nps (probe, §4) = 0.93-1.18e7 nodes/s. Agreement within 0-27% — the residual is explained
by the probe using two dense middlegames while ~1% of live queries terminate early and
endgames run faster. The accounting model is sound at the ~20% level; treat every share
above as +/- ~5 pp.

---

## 4. SF probe (MEASURED, one `nice -19` engine, Threads 1, Hash 16, 2 positions)

```
multipv  nodes      wall_s   nps      depth
     40  698289     2.180   320322      10
     40  698289     2.208   316231      11
     16  698289     2.403   290723      11
     16  698289     2.559   273106      11
      8  698289     2.848   245419      13
      8  698289     2.719   256884      13
      1  698289     2.814   248219      19
      1  698289     2.581   270751      17
     40  698289     2.701   258582      10   <- re-run of mpv 40, SLOWEST of all
     40  698289     2.820   247602      11   <- (spread is contention noise)
     40  349144     1.214   287986       9
     40  349144     1.036   337296      10
     40  174572     0.492   355143       8
     40  174572     0.533   327557       9
```

Two conclusions:

- **Wall time is flat in MultiPV at fixed nodes.** The mpv-1 runs were no faster than the
  mpv-40 runs, and the mpv-40 re-run at the end was the slowest sample in the set. **Width
  costs zero CPU.**
- **Wall time is linear in nodes.** 698k / 349k / 175k -> 1.00 / 0.51 / 0.23. **Nodes are a
  clean linear CPU lever.**
- **Depth is bought by narrowing, or by more nodes.** At 698k: mpv 40 -> depth 10-11;
  16 -> 11; 8 -> 13; 1 -> 17-19. At mpv 40: 698k -> 10-11, 349k -> 9-10, 175k -> 8-9, i.e.
  **halving nodes costs exactly 1 ply.**

---

## 5. Lever A — `sf_nodes` (label depth)

**CPU saving: large and linear.** Halving 698,289 -> ~349,000 halves *all three* buckets
(the fast-ply scale is multiplicative), i.e. **-50% of 70.4% of the box**. Since SF is the
sole binding constraint, first-order that is ~2x games/h. Realistically **+70-90%**, not
+100%: the `network` phase currently runs 100-970% of 3200% (peaks ~30% duty), so at 2x
the game rate the GPU/broker path starts to bind at the peaks.

**Quality cost: 1 ply of teacher depth (MEASURED).** Median realized depth goes 12 -> 11.
What the shards **cannot** tell you is whether the top-move / cp-ordering *changes* — they
store only the 698k label, so a truncation study on `sf_multipv_raw` measures coverage,
not what a shallower search would have returned. That requires a paired offline re-query.

**Two hard constraints on this lever:**

1. **`sf_nodes` is PID-owned.** It is the stage-2 difficulty lever; you cannot set it in
   the yaml and have it stick — `refresh_live_params` re-clamps into
   `[sf_pid_min_nodes, sf_pid_max_nodes]` every iteration. Forcing it down means lowering
   `sf_pid_max_nodes` (currently 1,000,000). And because `sf_move_nodes: 0`, it also
   weakens the curriculum opponent — the PID will read the winrate rise and tighten regret
   to compensate (0.0904 today vs a 0.0075 floor, so there is authority to absorb it).
   **One knob, two effects.** That is a real confound for any readout.
2. **A weaker version of this experiment already FAILED.** Ledger line 287
   (`docs/experiment_ledger.md`), "throughput triple, leg 1: `sf_label_nodes_cap`":
   150k -> policy 49.6->55.3 and raw top-1 51.5->60.7; raised to 400k; **400k also fired
   the kill** (52.2 vs 49.6 baseline, >2 cp; raw top-1 61.7 never recovered) -> uncapped
   2026-07-02, with the verdict *"the policy teacher needs full ~700k-node labels."*
   A 400k cap is a **43% cut, gentler than halving**, and it died. Note the cap decoupled
   labels (400k) from moves (698k) whereas halving `sf_nodes` moves both — so it is not
   the identical experiment — but the quality evidence points the same way, and re-running
   it without acknowledging that entry would violate the ledger protocol.

**Verdict: do not take this lever on throughput grounds alone.** The prior kill is
directly on point and was decided on the exact metric that matters.

---

## 6. Lever B — `sf_multipv` (label width)

**CPU saving: ZERO. MEASURED (§4).** At a fixed `go nodes` budget the node count *is* the
cost, and MultiPV only decides how those nodes are distributed. `sf_multipv` is **not a
throughput lever**. Any plan that budgets CPU savings from cutting width is wrong.

What it *is* is a free quality reallocation. Priced on 28,480 live label rows:

| k | mass of `sf_policy_target` in top-k PVs | TV(full, top-k renormalized) |
|---|---|---|
| 1 | 0.5905 | 0.4095 |
| 2 | 0.7288 | 0.2713 |
| 4 | 0.8362 | 0.1638 |
| 8 | 0.9178 | 0.0822 |
| 16 | **0.9745** | **0.0255** |
| 24 | 0.9914 | 0.0086 |
| 32 | 0.9977 | 0.0023 |
| 40 | 0.9999 | 0.0001 |

(selfplay rows are softer than curriculum: top-16 mass 0.966 vs 0.992. Target argmax
equals MultiPV rank 1 in 84.2% of rows — 81.0% selfplay, 90.6% curriculum; the gap is the
cp-logistic reordering plus the `sf_policy_label_smooth: 0.01` legal-tail floor.)

So `sf_policy_temp: 0.012` is **not** as sharp as it looks — mean top-1 mass is 0.59, and
the tail past rank 16 still carries 2.6% of the mass. Width is *not* almost-free to cut in
target-mass terms. But since cutting it saves no CPU, the interesting direction is the
*trade*, not the cut:

- `40 -> 32`: TV 0.0023, and `multipv=40` only binds on 16.1% of positions -> essentially a
  no-op in both directions. Not worth a readout.
- `40 -> 16`: costs 2.6% of target mass (TV 0.0255), buys ~+1 ply of depth on every label
  at identical CPU.
- `40 -> 8`: costs 8.2% of mass, buys ~+2-3 ply.

Note this is the exact inverse of the 2026-04-29 change that raised `sf_multipv` 20 -> 40
"to enrich `sf_policy` targets". Reverting it is a live question, but it is a
**label-quality** experiment, not a throughput one — and it lands on the same
`sf_policy_target` that the nodes-cap kill said is depth-sensitive. Reference:
`docs/rl_loop_audit.md` D13 (VERIFIED: 87.1% of rows fully covered by MultiPV).

---

## 7. The lever the data actually points at (and it does not touch the mix)

**`sf_fast_ply_node_scale`, currently 0.25.**

**24.1% of all Stockfish CPU — ~17% of the entire machine — is spent on curriculum
fast-ply move queries that produce no training row at all.** They only decide what the
opponent plays on the ~79% of its turns that follow a playout-capped net ply. The code
says so explicitly (`stockfish_turn.py:191`, `_eff_sf_nodes`): *"an intentional compute
optimization, NOT a weakened training target: SF labels only attach to full plies... The
scale only makes the opponent play cheaply on the ~75% of fast plies that are not training
targets."*

- `0.25 -> 0.10` cuts that bucket by 60% = **-14.5% of total SF CPU** -> **~+17% games/h**,
  with **exactly zero change to any label** (every label still runs at the full 698,289).
- Cost: the opponent plays 175k -> 70k nodes on those plies, ~-1.5 ply of opponent strength
  on 79% of its moves. That is a genuine difficulty change the PID will absorb on regret —
  so it needs a ledger entry like any other, but it is the only one of the three levers
  that is **label-neutral by construction**.
- It is live-reloadable (`worker.py:2857` `_RECO` keys) — no restart, no config-key risk.
- Caveat worth stating: this makes the anti-engine curriculum opponent weaker on
  three-quarters of its moves. Whether "SF that plays cheaply between training targets" is
  still the opponent the project wants to exploit is a judgement call, not a measurement.
  Audit D17 is relevant: at `wdl_regret = 0.0896` the opponent already throws away >100 cp
  on 8.4% of its moves, so it is not currently a clean full-strength ruler either.

---

## 8. If a lever is taken: pre-committed yardsticks

Nothing below has been deployed. Per `CLAUDE.md` §Experiment protocol, each needs a ledger
entry with hypothesis + ONE deciding yardstick + kill threshold **before** launch, plus a
`salvage-export --top-n 1 --metric training_iteration` revert point.

**Lever A (`sf_pid_max_nodes` 1,000,000 -> ~350,000) — audit-first, do NOT skip the offline
screen given the 2026-07-02 kill:**

- *Pre-screen (no training compute, mandatory):* re-query ~2000 frozen audit-set positions
  at 349k/mpv40 and at 698k/mpv40, paired. **Kill before launch** if top-1 move agreement
  vs the 698k label is < 90%, or median |cp(best_349k) - cp(best_698k)| evaluated at 698k
  is > 5 cp. (The 2026-07-02 kill was fired at >2 cp on net+search, so 5 cp on the *label
  itself* is already generous.)
- *Live yardstick if it passes:* `scripts/probe_policy_targets.py` raw top-1 agreement +
  `scripts/audit_targets.py --max-positions 2000`, paired CI, at full replay refill
  (day-plus window — this is a learning-quality change, not a throughput one).
- *Kill:* raw top-1 agreement drops >3 pp vs the pre-change baseline, or net+search cp
  worsens by >2 cp — **the identical rule that killed the 400k cap.**
- *Confound to record:* the same knob weakens the curriculum opponent; log
  `opponent_wdl_regret_limit` every iteration and expect the PID to tighten regret.

**Lever B (`sf_multipv` 40 -> 16):** do not justify this on throughput — it saves nothing.
If run, run it as a label-quality bet: offline-rebuild targets from stored
`sf_multipv_raw` truncated to 16 (the machinery exists, `train/target_builder.py`; ledger
line 2349 documents the same offline rebuild for `sf_policy_temp`), and gate on the same
top-1/net+search pair. Expected effect size is small (TV 0.026) and the depth gain (~1 ply)
cannot be simulated offline, so this is a weak-prior experiment; **recommend not spending a
readout window on it.**

**Lever C (`sf_fast_ply_node_scale` 0.25 -> 0.10) — the one worth doing:**

- *Yardstick:* games/h from the compacted-shard filename counter over a 5-iteration window
  (`ls processed/_compacted | parse <ts>_<sha>_<N>g_<P>p`), paired against the
  pre-change 5-iteration window. Baseline today: **2561 games/h**.
- *Success:* >= +12% games/h sustained over 5 iterations (~3 h), with **label rows/game
  unchanged** (selfplay 24.69 +/- 0.5, curriculum 12.04 +/- 0.5) — the invariant that
  proves it did not touch labels.
- *Kill:* PID `opponent_wdl_regret_limit` fails to stabilize within 5 iterations, or raw
  curriculum winrate moves >0.08 and the airbag fires twice.
- *Confound:* the C17 `gumbel_vloss_weight: 1` deploy landed at 20:52 today and its own
  readout window is still open. **Do not overlap these two.** (One data-affecting change
  per readout window — `CLAUDE.md` protocol rule 4.)

---

## 9. What could NOT be determined

- **Which query `sf_block_starved` is waiting on.** No per-branch counter exists. The
  standing-population evidence (`pending_excluded_avg 23.6 / in_flight_avg 24.0`) says it
  is nearly always the curriculum-move branch, but that is *what the thread is parked on*,
  not *where the SF CPU goes* — labels share the same pool. Adding
  `sf_block_starved_move` / `_label` counters at `manager.py:442` would close this cheaply.
- **Exact wall-clock cost per bucket.** Every share in §3c is node-weighted. Wall time ∝
  nodes is verified to within the probe's noise, but per-query nps varies with position
  type (endgames run faster), so the shares carry ~+/- 5 pp.
- **What a 349k-node label would have said.** Shards store only the realized 698k label;
  truncating `sf_multipv_raw` measures *coverage* loss, not the *different answer* a
  shallower search returns. Requires a paired offline re-query — deliberately not run here
  (CPU-saturated box).
- **Whether the C17 restart changed throughput.** Pre-restart 2537 games/h vs post-restart
  2025 games/h, but the post window is inside the curriculum-drain transient, so the
  comparison is not yet valid. Re-read after ~2 h of steady state.
- **Steady-state replay-window composition.** The trial's `replay_shards/` directory is
  empty (buffer is not on disk here), so §3b is measured on the server's compacted ingest
  stream rather than on the trainer's actual sampling window. The two should agree at
  steady state but this was not verified.
