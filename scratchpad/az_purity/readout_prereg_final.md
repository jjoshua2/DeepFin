# PREREG (FINAL) — task #273: native-NNUE 2-arm × sims-ladder LABEL-QUALITY readout

Status: **FINAL — thresholds written, noise floor MEASURED, no ladder cell run yet.**
Supersedes the "PREREG SKELETON" section of `scratchpad/az_purity/stage1_launch_package.md`.
Ledgering is the coordinator's; this file is the pre-committed document the readout is
judged against. Nothing here launches anything.

Everything in §2–§5 was measured **2026-08-25** before any threshold below was written,
and the two calibration cells that set the noise floor are **not** cells of the readout
(§4 gives the base rung fresh seeds of its own) — so no threshold is estimated from data
that also appears in a comparison.

Paths below use shell variables so this file carries no machine-local path:

```bash
REPO=<this repository>                 # e.g. the checkout you are reading this in
WT=<a worktree pinned to the code sha in §1>
ART=<a scratch directory for cells and scoring outputs>
PACK=$ART/big.pack                     # built in §1; sha256 pinned there
SF=$REPO/e2e_server/publish/stockfish  # the production Stockfish symlink
```

---

## 1. Frozen state — ONE code sha for every cell

| thing | value |
|---|---|
| generator + ruler code | `main` @ **`dc0abecf23cb996eb68cd942cfea8a73869ed261`** (PR #468 merge) |
| worktree cleanliness | every run's sidecar must read `provenance.code_dirty == false` **and** `code_dirty_at_finish == false` |
| C extensions | rebuilt with `python3 scripts/build_production_extensions.py` (GCC 15.3, `-march=native -flto`), exit 0 |
| NNUE kernel selected at runtime | `avx2` (echoed on every realized line) |
| Stockfish binary (labels **and** ruler) | `$SF`, md5 **`a740cb57dc24e34edcfa683e4efa6602`** |
| NNUE source net | sha256 **`f68ec79f0fe38df7faf58e10404f719cf33e92548f4e71d37daccc93c0b92835`** (canonical `nn-f68ec79f0fe3.nnue`, 89,277,198 B) |
| NNUE weight pack | sha256 **`7dd38b92043b5ea97109142e7cd9009e600dca830364aae8b74a19f1d464eb05`** (111,261,760 B), arch big L1=1024 L2=31 L3=32 |

Building the pack (deterministic — verify the two sha256s above, do not ship the binary):

```bash
printf 'export_net big.nnue\nquit\n' | $SF && \
PYTHONPATH=. python3 scripts/nnue_pack.py big.nnue $PACK
```

**ANY change to the sha, the extensions, the binary or the pack VOIDS every completed
cell.** The sidecar records all of it per run; the verdict must quote the sidecar, not
this table.

---

## 2. The instrument — deep-SF LABEL ruler

`scratchpad/az_purity/score_shard_labels.py` (written for this readout; the existing
rulers score a NET against the frozen audit set, which cannot answer a question about
labels on a generator's own positions).

It rebuilds each sampled shard row's position from its stored input planes, deep-SF
labels it, and scores the row's stored `policy_target` against those labels.

**Ruler settings — IDENTICAL for every scored set, native and banked-UCI alike:**

| knob | value |
|---|---|
| nodes | **1,000,000** (the project's audit-set standard; the script refuses below it) |
| MultiPV | **10** (refuses below it) |
| Hash | 256 MB · Threads **1** per engine · 8 engines · `nice 10` |
| transposition table | **cleared (`ucinewgame`) before EVERY position** |
| Syzygy | **none** |
| regret cap | 1000 cp (`eval/audit.AUDIT_REGRET_CAP_CP`) |
| positions per cell | **5,000** |
| sample | uniform **without replacement over the cell's rows**, shards in NAME order, `--sample-seed 20260825` |
| exclusions | rows with `has_policy == 0` (≤1 legal move) only |
| label cache | keyed by position key, **shared across every cell in one invocation** |

⚑ The fresh-TT choice is deliberate and it is a **ruler change**: `scripts/build_audit_set.py`
labels with a WARM TT, so numbers from this ruler must never share a table with
frozen-audit-set numbers. Fresh is required here because sample order differs between
cells and a warm TT would make a label depend on which cell happened to ask first.

### 2.1 Same population, verified — not assumed

Every scored cell is drawn by the *same* function, the *same* seed and the *same*
shard ordering, and the rebuild is cross-checked against each row's own `legal_mask`:
the compact-1858 indices of the rebuilt board's legal moves must equal the stored mask.

Measured 2026-08-25 — **100% agreement on all four scored cells**:

| cell | sampled | decoded | legal-mask agree | `has_policy==0` | undecodable | terminal |
|---|---|---|---|---|---|---|
| nf_a (native static, seed 273001) | 5000 | 4964 | **4964/4964** | 36 | 0 | 0 |
| nf_b (native static, seed 273002) | 5000 | 4951 | **4951/4951** | 49 | 0 | 0 |
| UCI@512 (banked anchor) | 5000 | 4978 | **4978/4978** | 22 | 0 | 0 |
| UCI@32 (banked anchor) | 5000 | 4980 | **4980/4980** | 20 | 0 | 0 |

19,873 row-positions → 19,656 unique (1.1% duplicates), 1,902 s at 8 workers (10.3 pos/s).
SF depth reached: 27.7 / 29.1 / 22.1 / 21.5 mean.

### 2.2 The ruler's positive control

`UCI@512 − UCI@32 = −14.38 cp raw [−24.77, −4.25]` and `−14.64 cp standardised
[−24.67, −4.82]`. A 16× deeper leaf search produces better labels and this ruler resolves
it in both readings. An instrument that could not separate those two would not be allowed
to judge the arms.

### 2.3 The ruler's own knobs reach the engine — receipts, not assumptions

- **MultiPV.** Predicted before running: with `MultiPV 10` every row should come back with
  exactly `min(n_legal, 10)` scored lines. Measured over all 19,873 scored rows:
  `n_pvs == min(n_legal, 10)` on **100.0%** of rows in **every** cell (means 8.46–8.96,
  min 2, max 10). MultiPV is live at the engine and never silently collapsed to 1.
- **`--nodes`.** Same 197 positions of UCI@512 at 1,000,000 vs 4,000,000 nodes:
  `sf_depth_mean` **21.7 → 29.7**. The knob is consumed.
- **…and the ruler's depth is NOT a lever on the verdict.** That 4× node increase moves the
  anchor's primary by **1.0 cp** (103.32 → 102.29 on that subset) — an order of magnitude
  inside the 6.75 cp noise floor. The choice of 1M nodes is therefore not load-bearing,
  which is exactly what a ruler needs to be before it can arbitrate a 17 cp threshold.

---

## 3. PRIMARY metric — pre-committed BEFORE any ladder cell exists

**PRIMARY: phase-standardised mean TOP-1 deep-SF regret, in cp. Lower is better.**

- *top-1 regret* = the deep-SF regret of the move the stored `policy_target` ranks first.
- *phase-standardised* = the cell's per-phase means reweighted onto ONE fixed reference
  phase distribution, pre-committed now as the **UCI@512 anchor's** mix:
  **endgame 0.5944 / middlegame 0.1527 / opening 0.2529**.
  The reference is an EXTERNAL, frozen, banked cell — no arm under test can move it.
- CIs are **cluster bootstraps over GAMES** (`game_id`), 20,000 resamples, seed 12345.
  Never over rows: rows inside a game are heavily correlated.

**REQUIRED CO-REPORT (not decisive): the RAW mean top-1 regret**, i.e. the same statistic
at the cell's own phase mix. Raw is the property of the *corpus*; standardised is the
property of the *labelling*. A raw/standardised disagreement is a **finding about the
arm's trajectories** (its phase mix moved) and must be named, never absorbed.

### Why top-1, and why not the alternatives

- **Insensitive to target sharpness**, which is exactly what the pinned `--nnue-cp-per-unit`
  confound (§8) moves first. Expected regret is not.
- **Expected regret is linear in cp**, so its fixed-entropy minimiser is the Gibbs
  distribution in cp — it rewards a cp-softmax target by construction
  (`eval/audit.py::expected_blunder_rates` documents the 2026-08-04 measurement where two
  distributions swapped places under a change of units).
- ⚑ **This is not hypothetical here. On the calibration cells the primary and the
  secondaries already point in OPPOSITE directions** (§5.4): the native arm is *worse* on
  top-1 and *better* on expected regret and blunder mass, because its target is flatter
  (TV-to-uniform 0.468 vs the anchor's 0.497; sharp rows 94.2% vs 97.4%). Choosing the
  metric after seeing that would decide the readout by choosing a ruler.

**DISAGREEMENT RULE (pre-committed):** if primary and secondaries disagree, the verdict
follows the PRIMARY and the disagreement is recorded as a finding — it is evidence about
target SHAPE and it revives the `--nnue-cp-per-unit` / sharpness sweep as a **named
separate experiment**, never as a re-read of this one.

**Secondaries reported for every cell, always, decisive for nothing:** raw top-1 mean;
median / P90 / >100cp% / >300cp%; expected regret; `blunder@100`, `blunder@300`
(`E_p[1{regret > τ}]`); `top1_is_sf_best`; `value_agrees`; per-phase and per-criticality
breakdowns; in-check leaf fraction; resolver expansion factor; realized root budget.

⚑ **`wdl_target` is the GAME OUTCOME, not an arm-derived value label** (generator
docstring; `selfplay.game._result_to_wdl`). With `has_sf_wdl` and `has_search_wdl`
identically 0 the value target is `game_oh` bit-identically for any fracs. So the arms'
NNUE values reach the corpus through (a) the improved `policy_target` and (b) which moves
get played — and **G2 is therefore a POLICY-label gate**. The raw arm values are recoverable
only from the leaf bank, which is why §8 makes `--bank-leaf-observations` mandatory.
`value_agrees` (stored outcome class vs deep-SF WDL argmax) is reported as a property of
the GAMES, not of the labelling.

---

## 4. Cells

**Arms (fixed, both carry the mandatory recursive check-resolution — arm-fairness
invariant; the qsearch arm is that SAME infrastructure plus a tactical quiescence):**

- `--value-source nnue-static`
- `--value-source nnue-qsearch`

**Ladder — rungs chosen so the REALIZED per-root budget genuinely doubles.**
Under `--all-root-moves` (default ON for the native arms) `--sims` is a FLOOR: the realized
budget is `max(--sims, 2 * n_legal)`. Measured 2026-08-25 (20-game probes, static arm):

| `--sims` | realized mean root sims |
|---|---|
| 32 | 44.4 |
| 64 | 65.3 |
| 90 | **90.0** |
| 128 | 128.0 |
| 180 | **180.0** |

⇒ **rungs are `--sims` 32 / 90 / 180** = realized **1× / 2.03× / 4.05×**. A naive
32/64/128 ladder would have been 1× / 1.47× / 2.88×. Base = 32 because it is the
anchor-identical setting (banked UCI anchors realized 46.8 and 47.3 mean root sims; the
native base realizes 44.4–44.9 — the residual is the legal-move distribution and is
reported per cell, not assumed away).

**Every preregistered cell is a SEED PAIR**: two independent runs, same config, different
seeds, 500 games each, 5,000 scored rows each, **pooled by row** into one cell estimate
(≈10,000 rows, ≈980 games). This is the construction whose noise the calibration measured,
and pooling two independent game pools is what buys the resolution in §7.

**Seeds — pre-listed, no substitutions:**

| cell | arm | `--sims` | seeds |
|---|---|---|---|
| calibration (NOT a readout cell) | nnue-static | 32 | 273001, 273002 |
| S-base | nnue-static | 32 | **273011, 273012** |
| S-2x | nnue-static | 90 | **273021, 273022** |
| S-4x | nnue-static | 180 | **273031, 273032** |
| Q-base | nnue-qsearch | 32 | **273111, 273112** |
| Q-2x | nnue-qsearch | 90 | **273121, 273122** |
| Q-4x | nnue-qsearch | 180 | **273131, 273132** |

Ruler `--sample-seed` is **20260825** for every cell and for both anchors, forever.

**Anchors** (already banked, `data/gen0/bench_anchors/`, scored 2026-08-25 on the ruler
above — never re-scored on a different ruler version): UCI@512 = the G2 bar; UCI@32 =
context.

---

## 5. MEASURED — the noise floor and the anchor, on the instrument above

Calibration pair: `nnue-static`, `--sims 32`, 500 games, 4 workers,
`nice -n 10 … --nice 10` (realized nice 19 — read back from `os.getpriority`),
`--bank-leaf-observations`, `--nnue-cp-per-unit 0.28`, seeds 273001 / 273002. **Identical
in every respect except the seed.**

### 5.1 The cells

| cell | rows | games | wall s | rows/h | realized root sims | in-check leaf frac | resolver expansion |
|---|---|---|---|---|---|---|---|
| nf_a (273001) | 109,769 | 500 | 76.3 | **5,180,078** | 44.9 | 0.0655 | 1.07 |
| nf_b (273002) | 119,616 | 500 | 83.2 | **5,177,742** | 44.5 | 0.0670 | 1.07 |

### 5.2 THE NOISE FLOOR

| reading | nf_a | nf_b | **Δ (the floor)** | 95% CI on Δ |
|---|---|---|---|---|
| **PRIMARY (phase-standardised)** | 120.74 [113.97, 127.59] | 127.49 [120.38, 134.64] | **6.75 cp** | [−16.52, +3.04] — includes 0 |
| raw (co-report) | 126.77 [119.24, 134.33] | 138.32 [130.14, 146.44] | **11.56 cp** | [−22.52, −0.52] — **excludes 0** |

⚑ **The raw metric's two same-config cells are statistically DIFFERENT.** The within-cell
cluster bootstrap (SE ≈ 5.7 cp) does not contain the between-cell variance, because a new
seed draws a new game pool and therefore a new position mix. Standardising removes ~42% of
that noise and puts the two cells back inside each other's CI — which is the expected,
healthy result and the direct evidence for the §3 choice of primary.

⇒ **Take the noise floor as the OBSERVED between-cell difference, never as the within-cell
CI.** Anyone reading only the bootstrap SE would have called an 11.6 cp seed artifact a
real effect.

### 5.3 The G2 anchor, same ruler, same sample size

| set | raw mean top-1 cp | 95% CI | standardised | 95% CI |
|---|---|---|---|---|
| **UCI@512 (G2 bar)** | **98.09** | [90.77, 105.30] | **98.09** (it is the reference mix) | [91.08, 105.07] |
| UCI@32 (context) | 112.47 | [105.30, 119.73] | 112.72 | [105.67, 119.93] |

### 5.4 Secondaries on the calibration run — and the metric disagreement

| cell | median | P90 | >100cp | >300cp | expected regret | blunder@100 | blunder@300 | top1_is_sf_best | value_agrees |
|---|---|---|---|---|---|---|---|---|---|
| nf_a | 24.0 | 318.0 | 28.8% | 10.4% | 174.85 | 0.3923 | 0.1657 | 0.3040 | 0.1944 |
| nf_b | 26.0 | 400.0 | 29.3% | 11.6% | 183.93 | 0.3937 | 0.1697 | 0.2999 | 0.1818 |
| UCI@512 | 3.0 | 287.6 | 21.5% | 8.5% | 194.19 | 0.4096 | 0.2069 | 0.3935 | 0.2973 |
| UCI@32 | 10.0 | 300.0 | 24.3% | 9.8% | 196.13 | 0.4047 | 0.2067 | 0.3355 | 0.3088 |

The native arm is **worse on top-1 and better on expected regret and blunder mass** than
both anchors. §3's disagreement rule governs. (The `value_agrees` column tracks the games,
not the labels: the native cells draw 82% of their games — insufficient-material 236/207 of
500 — against the anchor's 48%, so their outcome labels are draw-heavy while deep SF calls
most of these ruined positions decided.)

### 5.5 Throughput, and what it is and is not comparable to

- Static base **with the leaf bank ON**: 5,180,078 / 5,177,742 rows/h =
  **6.37× the ≥813,000 rows/h G1 gate**, 31.8× UCI@32 (162,667), 42.8× UCI@512 (120,961).
- **Leaf-bank cost, measured on a BIT-IDENTICAL corpus** (seed 273001 re-run, only the flag
  differs; rows 109,769, shards 53, W/D/L 52/409/39, leaves 4,802,713 all reproduced
  exactly): 6,103,660 → 5,180,078 rows/h = **−15.1%**. That re-run also proves the
  generator is **seed-deterministic**, so a cell can be reproduced exactly and the leaf
  bank does not perturb the search.
- ⚑ **Cell length changes the number by 2.7×.** The 20-game probes read ~1.8 M rows/h
  against 5.2 M for the 500-game cells — pure startup amortisation.
  **G1 is read ONLY off a ≥500-game run's `gen0_summary_*.json → rows_per_hour`.**
- Higher rungs (20-game probes, ratios only, to be re-measured per cell):
  `--sims 90` = 0.936× and `--sims 180` = 0.610× of the `--sims 32` rate ⇒ S-4x projected
  ≈ 3.16 M rows/h ≈ 3.9× the gate. **A projection is not a G1 pass.**
- **qsearch, measured first-hand** (8 games, 4 workers, `nice 12`, contending with the 8
  ruler engines): 255.1 s, 1,604 rows, **22,632 rows/h = 0.028× the gate — 36× BELOW it**.
  Consistent with the PR #468 ledger's 26.6k. `qnodes` 57,430,997 for 8 games; mate-band
  leaf fraction 1.30% vs the static arm's 0.0005%; resolver expansion 1.28 vs 1.07.

### 5.6 What the calibration already implies — DISCLOSED, and not a verdict

The calibration cells share the S-base configuration (they differ only in seed), so this
prereg is being written by someone who can already see roughly where the base rung will
land. Saying so is the point of writing it down:

| comparison (calibration cells, NOT readout cells) | standardised Δ | 95% CI |
|---|---|---|
| nf_a − UCI@512 | **+22.65 cp** (worse) | [+12.97, +32.62] |
| nf_b − UCI@512 | **+29.40 cp** (worse) | [+19.35, +39.33] |
| nf_a − UCI@32 | +8.02 | [−1.92, +17.81] |
| nf_b − UCI@32 | +14.77 | [+4.66, +24.66] |

⇒ **the native static arm at base budget is behind UCI@512 by roughly 23–29 cp on the
primary, and behind UCI@32 as well.** The ladder therefore has ~23 cp to close before G2
can pass, and G2 is `≤ +6.75` — the base rung is nowhere near it.

Why this does not corrupt the prereg: the thresholds in §6–§7 come from the **DIFFERENCE**
between the two calibration cells, which is statistically orthogonal to their **MEAN**;
G2 is a fixed non-inferiority margin against a frozen external anchor; and the bend rule
compares rungs to each other, not to anything above. The base rung is re-run with fresh
seeds (§4) so no readout cell is a calibration cell. **These four rows are context. They
are not the readout, and no verdict may be built on them.**

⚑ **REGIME CONFOUND, STATED.** The calibration cells ran with **live PBT training STOPPED**
(`./scripts/train.sh status` → `Not running`); the box carried one offline GPU job
(`scripts/lc0_control_train.py`, ~1.3 CPU cores, GPU 64%) and load ≈ 2. The banked UCI
anchors ran **concurrent with live GPU training**. Every ladder cell must record its regime
in the readout; a cross-regime rows/h comparison is reported as such and never quietly
pooled.

---

## 6. GATES — pre-committed thresholds

### G1 — throughput

A cell **passes G1** iff **both** of its seed runs report
`rows_per_hour ≥ 813,000` in `gen0_summary_*.json` (5× the banked UCI@32 anchor of
162,667), each from a **≥500-game** run, with `partial == false`,
`--bank-leaf-observations` ON, 4 workers, `nice -n 10 … --nice 10`.
The regime is recorded with the number.

### G2 — label quality (a NON-INFERIORITY test, margin = one noise floor)

Let `D = standardised_mean(cell) − standardised_mean(UCI@512)` (cp; positive = worse).

> **G2 PASS iff the 95% cluster-bootstrap CI on `D` has an UPPER bound ≤ +6.75 cp**
> (= R, the measured single-run noise floor of §5.2).

- A cell that passes G1 but fails G2 ⇒ the **"no SF search" pre-commitment FAILS** and the
  hybrid lane (native rollouts + SF labelling) goes **back on the table**. That is a
  finding, recorded as a finding, not a reason to re-tune, re-seed or re-read.
- G2 is judged against **UCI@512**, not UCI@32, because deep labels are nearly free on the
  UCI path (−26% rows/h for 16× nodes) — beating only the shallow anchor would not justify
  dropping SF search. UCI@32 is reported for context and decides nothing.
- The margin is the measured **single-run** floor `R₁`, not the pooled one, because the
  UCI@512 anchor side of this comparison is a single 5,000-row run rather than a seed pair.
- The raw-metric version of the same comparison is co-reported with margin 11.56 cp.

---

## 7. Operating point — the bend rule, with the multiple stated

**Resolution constants, derived from §5.2 and written down here BEFORE any cross-cell
number is read:**

| symbol | value | where it comes from |
|---|---|---|
| `R₁` | **6.75 cp** | **MEASURED** single-run noise floor = the observed `\|Δ\|` between the two calibration cells, primary metric (§5.2) |
| `sd(Δ₁)` | 8.46 cp | `R₁ / √(2/π)` = `R₁ / 0.798` — `R₁` is ONE draw of `\|Δ\|` from a mean-zero law |
| `τ_cell` | 5.98 cp | `sd(Δ₁) / √2` — sd of a single run's estimate |
| `sd(Δ_pooled)` | 5.98 cp | two SEED-PAIR cells: `(τ_cell/√2) × √2` = `τ_cell`. DERIVED, assuming seeds independent |
| `z` | 2.734 | Bonferroni over the **8 preregistered comparisons** (2 bend steps per arm, 3 cross-arm rungs, 1 G2), α = 0.05 |
| **`Δ*` (decision threshold)** | **17 cp** (primary) | `z × sd(Δ_pooled)` = 2.734 × 5.98 = 16.4, rounded up. **The multiple is `Δ* = 2.5 × R₁`.** |
| `Δ*` raw co-report | **28 cp** | same construction on the raw floor `R₁ = 11.56` cp (`sd(Δ₁)` 14.49 → `sd(Δ_pooled)` 10.25) |

**BEND RULE (pre-committed).** Starting at the base rung, a doubling BUYS quality iff
**both**:

1. `standardised_mean(rung r+1) − standardised_mean(rung r) ≤ −Δ*` (≤ −17 cp), **and**
2. the 95% cluster-bootstrap CI on that difference excludes 0.

The **operating point is the LAST rung before the marginal gain fails this test**, subject
to that rung also passing G1 and G2. Never read off a plot; never chosen post hoc.

⚑ **Named limitation, with its remedy pre-committed.** `Δ* = 17 cp` is coarse relative to
the ~23 cp gap the native arm currently has to close (§5.6). If the ladder produces
movement that is real but sub-threshold, the pre-committed response is to **re-measure the
noise floor at a LARGER cell size and re-run the affected rungs at that size** — cost ≈ 5
min of generation plus ≈ 25 min of ruler per added pair. It is **never** to lower `Δ*`
after seeing the numbers. Every quantity below `R₁` in the table rests on a single observed
`|Δ|` (1 degree of freedom), which is the other reason not to shave it: dropping the
multiplicity correction (`z = 1.96`) would put the multiple at 1.75 and `Δ*` at 12 cp, and
a 1-dof variance estimate does not support that kind of precision.

---

## 8. Pinned knobs, confounds, kill rules

**PINNED FOR EVERY CELL OF BOTH ARMS — no exceptions, no per-cell variation:**

| knob | value | note |
|---|---|---|
| `--nnue-cp-per-unit` | **0.28** | see the confound below |
| `--nnue-cp-slope` | 0.006 | generator constant `NNUE_CP_SLOPE`, documented there as live `selfplay.sf_wdl_cp_slope` — verify against the LIVE yaml, not `main`'s, if it ever matters |
| `--nnue-cp-draw-width` | 120.0 | generator constant `NNUE_CP_DRAW_WIDTH`, same caveat |
| `--bank-leaf-observations` | **ON** | mandatory; costs 15.1% throughput (measured) and ≈1.7 GB per 500-game cell |
| `--all-root-moves` | ON (arm default) | anchors were generated with it |
| `--nnue-resolver-max-depth` | extension default 32 | `ctx_depth_cutoffs` must read 0 |
| games / workers / nice | 500 / 4 / `nice -n 10 … --nice 10` (realized 19) | anchor-identical |
| `--c-scale` / `--policy-temp` / `--temperature` / `--gumbel-scale` | 0.1 / 1.5 / 0.0 / 1.0 | anchor-identical |
| `--target-max-visit-cap` / untempered prior / `--vloss-weight` / `--target-batch` | 5 / on / 1 / 0 | anchor-identical |
| `--max-plies` / `--shard-size` | 450 / 2000 | anchor-identical |
| openings / `--random-start-plies` | none / 0 | anchor-identical |

Most of these are the tool's DEFAULTS and §10.1 does not restate them on the command line —
deliberately, because a flag typed at its default and a flag not typed are the same run and
the second cannot drift from the constant. The **receipt is the realized line and the
sidecar**, which echo every one of them from the consumer's own config snapshot; that is
what the verdict quotes.

**⚑ CONFOUND 1 — `--nnue-cp-per-unit 0.28` is a FREE PARAMETER, pinned, not neutralised.**
Provenance: least squares against Stockfish `go nodes 512` search cp, **R² 0.877** (PR #468).
It is pinned at one value for both arms and all six cells; a sweep is a **separate later
experiment** and is not part of this readout. Stating its reach precisely, because the
short version understates it: per leaf, `q = W − L` of `cp_to_wdl(value × cp_per_unit)`,
which is monotone in the raw value — so for a FIXED tree it would only re-sharpen the
target and the top-1 primary would be blind to it. But `Qbar` is an **average of** those
per-leaf `q`s and the map is non-linear, so averaging and rescaling do not commute: the
knob changes sequential-halving's own decisions, hence the trajectories and hence the
target's argmax. **It is a genuine confound the primary metric reduces but does not remove.**

**⚑ CONFOUND 2 — box regime.** §5.5. Anchors: concurrent with live GPU training. Calibration
cells: live training stopped. Recorded per cell; never pooled across regimes silently.

**⚑ CONFOUND 3 — ruler TT hygiene.** Fresh TT here, warm TT in `build_audit_set.py`. These
numbers do not belong in a table with frozen-audit-set numbers.

**⚑ CONFOUND 4 — each cell scores its OWN positions.** Two cells never see the same
position set. That is inherent to the question (a corpus's label quality includes which
positions it visits); the noise floor was measured under exactly this condition and the
standardisation of §3 removes the phase-mix part of it. The native arm's mix already
differs from the anchor's (endgame share 0.659 / 0.684 vs 0.594) — which is why the
standardised reading is the primary and the raw reading is co-reported.

**KILL / DISCIPLINE RULES**

1. One code state for all cells (§1). Any change voids every completed cell.
2. Seeds are those in §4. **No cell is re-run on a "weird" number without a NAMED defect**
   recorded in the readout; a re-run for any other reason makes the cell an exploratory
   extra, recorded as such, never swapped in.
3. The readout is judged on the pre-listed cells only. Exploratory cells are labelled
   exploratory and never enter a gate.
4. `partial == true`, a non-zero `failed_workers`, `orphan_shards`, or
   `ctx_depth_cutoffs > 0` **invalidates a run** — regenerate that seed, do not score it.
5. Metric, standardisation reference, ruler settings and thresholds are frozen by this
   document. Post-hoc metric or threshold changes are a new experiment with a new prereg.
6. No rolling reads: a cell is scored once, complete, and its number stands.

---

## 9. The qsearch arm — EXPECTED FAIL, stated up front

**Q-base, Q-2x and Q-4x are expected to FAIL G1 and the failure is structural, not a
tuning miss.** Measured first-hand 2026-08-25: **22,632 rows/h**, i.e. **0.028× the
813,000 gate (36× below)** and ~229× slower than the static arm. Nothing in the ladder
recovers a factor of 229.

⚑ **SPEC DEVIATION, forced by that measurement.** The skeleton specified the two arms at
**MATCHED WALL-CLOCK per move** ("the qsearch arm spends ~5–20 evals/leaf so it runs
proportionally fewer sims"). At a 229× cost ratio, equalising wall-clock would put the
qsearch arm below one simulation per root — the design is degenerate. **Matched wall-clock
is therefore retained only as a REPORTED quantity** (each cell's measured `rows_per_hour`
and wall seconds), **not as an equalised design parameter**: both arms run the SAME nominal
sims ladder, so the arm contrast is *leaf quality at equal search shape* and the cost is
read off the manifest separately. This is a deviation from the skeleton and is recorded as
one.

**Budget:** a Q cell (seed pair, 500 games each) costs ≈ **8.9 h** at base, ≈ 9.5 h at 2×
and ≈ 14.5 h at 4× — **≈ 33 h serial for the arm**, against ≈ 10 min of generation for the
entire static ladder.

**STAGING GATE (pre-committed, not post-hoc):** run S-base, S-2x, S-4x and **Q-base**
first. Q-2x and Q-4x run **only if** Q-base is better than S-base by at least `Δ*`:

> `standardised_mean(Q-base) ≤ standardised_mean(S-base) − 17 cp`

Rationale: qsearch cannot be the operating point at any rung (G1), so its only value in
this readout is the QUALITY question — and an arm that is not better at the cheapest budget
has no path to justifying 33 h.

⚑ **The risk this accepts, named:** a qsearch arm that wins ONLY at high budget would be
missed. Accepted deliberately, cost stated (24 h to close it). Cells not run are reported
**NOT RUN** — never as a null, never as a fail.

The static arm's ladder runs in FULL regardless, so the readout's main product — the
operating point — is never conditional on anything.

---

## 10. Exact commands

### 10.1 One generator run (repeat per seed in §4)

```bash
cd $WT && PYTHONPATH=. nice -n 10 python3 scripts/gen_random_selfplay_shards.py \
  --out-dir $ART/cell_<label>_s<seed> \
  --games 500 --workers 4 --nice 10 \
  --value-source <nnue-static|nnue-qsearch> --nnue-pack $PACK \
  --nnue-cp-per-unit 0.28 --bank-leaf-observations \
  --sims <32|90|180> --seed <seed>
```

Check on every run before scoring it: `provenance.code_sha` == §1's sha and
`code_dirty`/`code_dirty_at_finish` false; `partial == false`; `failed_workers` empty;
`nnue.ctx_depth_cutoffs == 0`; the realized line shows
`nnue_cp_per_internal_unit=0.28` **from the consumer's own config snapshot** (and, on the
static arm, `nnue_qsearch_max_ply=None` — the arm does not accept knobs it cannot read).

### 10.2 Scoring — ONE invocation over every cell and both anchors

One pass, one shared label cache, so a position two cells share is scored against the
identical ruler row:

```bash
cd $WT && PYTHONPATH=. nice -n 10 python3 $REPO/scratchpad/az_purity/score_shard_labels.py \
  --cell S-base-a=$ART/cell_S-base_s273011 --cell S-base-b=$ART/cell_S-base_s273012 \
  --cell S-2x-a=$ART/cell_S-2x_s273021   --cell S-2x-b=$ART/cell_S-2x_s273022 \
  --cell S-4x-a=$ART/cell_S-4x_s273031   --cell S-4x-b=$ART/cell_S-4x_s273032 \
  --cell Q-base-a=$ART/cell_Q-base_s273111 --cell Q-base-b=$ART/cell_Q-base_s273112 \
  --cell uci512=$REPO/data/gen0/bench_anchors/gen0_bench_512 \
  --cell uci32=$REPO/data/gen0/bench_anchors/gen0_bench_32 \
  --positions 5000 --sample-seed 20260825 \
  --stockfish $SF --nodes 1000000 --multipv 10 --hash-mb 256 \
  --sf-workers 8 --sf-nice 10 --bootstrap 20000 --bootstrap-seed 12345 \
  --out-json $ART/ruler_ladder.json --dump-rows $ART/ruler_ladder_rows.jsonl
```

### 10.3 Primary metric, standardisation and the pooled cell estimates

```bash
cd $WT && PYTHONPATH=. python3 $REPO/scratchpad/az_purity/noise_floor_analysis.py \
  --rows $ART/ruler_ladder_rows.jsonl --reference-cell uci512 \
  --metric top1_regret_cp --n-boot 20000 --seed 12345 \
  --out-json $ART/ladder_standardised.json
```

A seed pair is pooled by concatenating its two cells' rows (`cell` field) before this
step; `game_id` stays the cluster key and is unique within a run, so pooled runs must be
namespaced by run before clustering.

### 10.4 Reproducing the calibration in §5

Identical to 10.1 with `--value-source nnue-static --sims 32` and seeds 273001 / 273002,
then 10.2 with those two cells plus both anchors, then 10.3. Banked outputs:
`scratchpad/az_purity/readout_artifacts/` (`ruler_noisefloor.json`,
`noise_floor_analysis.json`, `ruler_rows.jsonl.gz`, per-cell logs and sidecars; local paths
in those files were replaced with `$REPO`/`$WT`/`$ART` — this repository is public).

---

## 11. What this readout does NOT decide

- **Value-label quality.** The stored value target is the game outcome (§3), so no cell
  here scores an arm's value labels. The leaf bank makes that a later REANALYSIS rather
  than a regeneration — it is not part of this prereg.
- **`--nnue-cp-per-unit`.** Pinned, confound, separate experiment (§8).
- **Small-net labels.** The skeleton's small-net side-read is not part of this gate.
- **Whether the corpus TRAINS well.** This is an audit-first label gate. Downstream training
  value is a separate arm with its own prereg.
- **Any Elo claim.** No arena is run here and none should be quoted from these numbers.
