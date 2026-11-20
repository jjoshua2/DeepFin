# RL run 1: a measured multi-generation self-play run from the SL base (plan)

**Status:** DRAFT for owner approval. Nothing has been launched. No stage below
starts until the owner has approved this record and the
[decisions](#decisions-for-the-owner) are recorded in a follow-up commit
(CLAUDE.md "Experiments" rule 1).

**Why this record exists.** The owner set a new direction on 2026-11-20. In his
words:
- "i do want the RL loop to be the main path to strength";
- "i expect to be working on this project for years but i would hope to have a
  very strong engine within weeks";
- on 4,000 vs 16,000 games vs "1 hour": "they both seem equally made up to me … i
  wish we could be more rigorous instead of guessing";
- on search headroom: the SL base "gained at each more step and it should
  continue to do so as we increase the model; even lc0 and ceres do this at like
  1500 elo stronger base."

So the one-generation experiment is no longer the goal. It becomes the
calibration step (Stage 1) of a multi-generation run (Stage 2), and doses come
from measurements. Every number that is still a guess is listed under
[What is still a guess](#what-is-still-a-guess), together with the measurement
that would replace it.

**Relation to the loop-exp1 record.** The
[loop-exp1 record](2026-11-20-loop-exp1-one-generation-from-sl-base.md) is
superseded as a standalone experiment. This plan reuses it as the source of:
- the commands (the collection, `--collect-from` arms, planned updates);
- the collect gates G-C1–G-C11 and their threshold file;
- the model gates G-T1–G-T4 and the `gt1` / `gt2` verdict commands;
- the arena command and the seat-balanced pooling (`scripts/arena_seat_balanced_pool.py`);
- the net write-protection and sha256 provenance rules.

"As loop-exp1" means that record's text with #1437's additions:
- openings split step 0b and SL base copy step 0a;
- `gt1`, `gt2` and the pooling script's `--z`;
- the recalibrated G-C5 and G-T4 bounds.

#1437 must merge before launch.

Naming: "SL base" is the supervised starting net, sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. The driver is
`scripts/simple_rl_loop.py` ([docs](../simple_rl_loop.md)).

## Decisions for the owner

Only these three are the owner's to make. Everything else is set either by a
measurement or by a rule written below before any data.

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| O1 | **Time cap for Stage 1's collection** | **One day of collection in total,** half at 64 sims and half at 256 sims. At the measured 3,284 games/h, that is about 39,000 games at 64 sims, plus about a quarter as many at 256 sims (that rate is measured before launch). Stage 1 as a whole is then about 2.9–3.9 GPU days ([compute](#stage-1-compute)). | Two days in total. That doubles the data at every rung's time point, and roughly doubles the training time; arena time is unchanged. |
| O2 | **Primary recipe: P (policy only) or V (search value)** | **Run both as arms through Stage 1, and let the pre-written rule choose** ([Stage 1 output](#stage-1-output-the-pre-written-rule)) | Name one recipe. Stage 1 then runs only that recipe's arms, which halves training and arena cost. |
| O3 | **Sims per move: include the 256-sim factor** | **Include it.** A 256-sim collection costs the same GPU time as the 64-sim one, and the two are compared at equal GPU time. Stage 0 shows the 256-sim teacher is about +272 Elo stronger. | 64 sims only. This drops half the arms and puts the whole collection budget into 64-sim games. |

## Stage 0: search ceiling — answered by banked matches

No new matches are run. The SL base's search gain is already banked in
`~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`
(takeover record D1). The rows, verbatim:

| Label | Search vs search | Games | Score | Elo (95% CI) |
|---|---|---:|---:|---|
| `a39_s64_v_s1` | 64 vs 1 sim | 200 | 0.9925 | +848.7 (+690.7 to undefined) |
| `a39_s256_v_s64` | 256 vs 64 sims | 400 | 0.8275 | +272.4 (+243.1 to +305.3) |
| `a39_s800_v_s64` | 800 vs 64 sims | 200 | 0.9400 | +478.0 (+410.2 to +582.3) |

These matches were played in one orientation only. A candidate-seat offset of up to
about 15 Elo (D4 round 2's pooled same-net read was +6.4, −4.2 to +17.1) is
negligible at these sizes.

**What follows from it:**
1. **The headroom is large.** A 4× search is worth about +270 Elo to the SL base.
   So the question is not whether there is anything to gain. It is whether training
   TRANSFERS the search's strength into the raw net.
   - **The one measured transfer point is P1:** policy-only training at about 950
     updates on F12's rows read −10.9 Elo (−27.7 to +5.9) against the SL base. Its
     upper bound is about 2% of the +272.
   - **A detectable gain needs only a small transfer.** A 2,000-game seat-balanced
     match detects +15.6 Elo with 80% power. That is 5.7% of +272. At 5% of +272,
     about +14 Elo, it is near the edge of one such match.
2. **A working generation should show tens of Elo.** Stage 1's matches can
   therefore be small (400 games per rung, [below](#power)). The time saved is spent
   on more dose rungs, more learning rates and the 256-sim factor.
3. **Sims per move is a real lever, not a nuisance.** Collecting at 256 sims gives
   targets from a teacher about +272 Elo stronger, at about 4× the cost per game.
   The old record's D3 measured only target fit against deep search, never Elo.
   Stage 1 therefore compares 64-sim and 256-sim collections at equal GPU time: fewer
   games with stronger targets, against more games with weaker ones.

## Stage 1: calibration sweep

### Question

On collections from the SL base, does Elo against the SL base rise with the GPU
time spent collecting (dose)? And which combination does that best: recipe P or V,
which learning rate, 64 or 256 sims? Finally, what is the smallest dose that
measurably beats the SL base?

The learning rate is a factor because 3e-6 was copied from earlier Codex runs, and
is as much a guess as the dose.

### Collections

There are two collections from the SL base. Each uses loop-exp1's `$COLLECT` with
these changes:

| Collection | Sims | `--seed` | `--games` |
|---|---:|---|---|
| `c64/` | 64 | 2026112100 | half of O1 × the 64-sim rate measured on the day: about 39,000 at 3,284 games/h |
| `c256/` | 256 | 2026112100 | half of O1 × the 256-sim rate from pre-launch check P6: about 9,800 if the rate is a quarter |

Each game count is rounded down to a multiple of 100. Both collections take the
same GPU time. That makes a fraction f of either collection the same GPU time, and
so the dose axis is GPU time.

- **Shared seed, so shared openings.** Both collections walk the same TRAIN
  permutation from the same cursor. The 256-sim collection's lines are therefore
  the first lines the 64-sim collection plays too, so the two are paired on
  openings.
- **Rows:** 96.8 rows per game (the mean of three clean 128-game collects: 96.7,
  93.1, 100.5). After the 5% holdout, at batch 256, that gives N ≈ 14,200 updates
  for `c64/` and N ≈ 3,550 for `c256/`. Each N is computed exactly from its sealed
  collect by loop-exp1's step-2 snippet, and recorded.
- **Disk:** about 0.26 MB per game (33 MB per 128 games), so about 13 GB for both
  collections, plus about 15 GB of models for 16 arms. Free space is checked before
  launch.
- **Gates:** loop-exp1's collect gates G-C1–G-C11 and threshold file, on each
  collection. Its SL-base bands were measured at 64 sims, so on `c256/` only the
  machinery gates are gated: G-C1, G-C2, G-C3 ≥ +0.50, G-C7 and G-C10. Bands G-C4,
  G-C5, G-C6, G-C8 and G-C9 are recorded only. A failed gate voids that
  collection's arms.

### Arms

There are 16 arms: 2 collections × 2 recipes × 4 learning rates. Each arm
references its collection with `--collect-from`, so all arms on one collection
share the seed, the holdout and the batch plan (loop-exp1 step 3). The arm flags,
beyond loop-exp1's `$TRAIN`:

| Recipe | Flags (X = 3e-6, 1e-5, 3e-5 or 1e-4) |
|---|---|
| P | `--w-wdl 0 --lr X --lr-floor X` |
| V | `--w-wdl 1 --value-outcome-weight 0 --value-search-weight 1 --value-search-source survivor_q --value-draw-source net --lr X --lr-floor X` |

- **Optimizer:** fresh NAdamW, warmup 0, flat LR (`--lr-floor` = `--lr`), as
  loop-exp1.
- **LR grid:** steps of about ×3, from the copied 3e-6 to 1e-4. The driver's own
  default is 3e-5. The SL base was trained at 3e-6 on its non-matrix parameters and
  at 6e-5 on its matrices (Aurora). 1e-4 is above both; if it trips the G-T2 gross
  guard early, that is a measured edge, not a lost arm.
- **Arm S is not in Stage 1.** It is a third recipe; it can be a later factor.
- **Movement:** probe KL and |ΔQ| against the SL base are recorded at every
  snapshot and on the final model, as descriptors. They never choose an LR or a
  dose ([why](#why-no-sizing-from-loss-curves)).

### Dose rungs

Each arm writes snapshots at each halving of its data:

```bash
N=<planned updates from the arm's sealed collect>
SNAPS=$((N/64)),$((N/32)),$((N/16)),$((N/8)),$((N/4)),$((N/2))   # doses 1/64 … 1/2; the final model is dose 1
```

That gives 7 rungs, x = log₂(fraction) ∈ {−6, …, 0}. On `c64/` at N ≈ 14,200 the
snapshots fall at about 221, 443, 887, 1,775, 3,550 and 7,100 updates. On
`c256/` at N ≈ 3,550 they fall at about 55 … 1,775. Rung x is the same GPU time on
both collections.

**A snapshot counts as a dose rung only because the schedule is exact-once,
shuffled and at a flat LR.** The rows seen by update N/64 are a uniform random
subset of the collection, trained at the same LR a shorter run would use. With a
decaying LR schedule this would not hold.

**Gates per rung:** loop-exp1's G-T1–G-T4 on every snapshot and on the final
model, with its bands per recipe. A snapshot that fails a gate ends that arm there:
its later rungs are not played. The arm's trend test then uses only the rungs
before the failure. It needs at least three; with fewer, the arm reads "gate
failed".

### Openings: disjoint blocks

The EVAL split (4,000 lines) is cut once, in `eval.jsonl` order. Stage 1 gets 7
rung blocks of 100 lines; Stage 2 gets four blocks of 500:

```bash
mkdir -p $R/openings/blocks
python3 -c 'import json, sys
d = sys.argv[1]; lines = [json.loads(l)["line"] for l in open(d + "/eval.jsonl")]
assert len(lines) == 4000
cuts = [(f"rung_{k}", 100 * k, 100 * (k + 1)) for k in range(7)] + [
    ("readout_base", 1000, 1500), ("readout_prev", 1500, 2000), ("confirm", 2000, 2500), ("anchor", 2500, 3000)]
for name, a, b in cuts:
    open(f"{d}/blocks/{name}.fen", "w").write("".join(x + "\n" for x in lines[a:b]))
print(len(cuts), "blocks")' $R/openings
```

- Rung x uses `rung_{x+6}.fen` in every arm.
- Lines 700–999 and 3000–3999 are left unused, in reserve.

This block was run verbatim on the banked split. It wrote 11 files; in the
arena's FEN loader, each rung block has 100 unique rows and each Stage 2 block has
500. An earlier cut of the same split into 8 blocks of 500 loaded with every row
carrying its move history.

**Why disjoint blocks, not shared openings.** With one shared opening set, the rung
estimates would be positively correlated (correlation ρ through the openings).
- **What sharing would buy.** The slope is a contrast whose weights sum to zero, so
  sharing would cut its variance by about (1 − ρ).
- **What it would cost.** It would need an opening-level slope estimator: per
  opening, a slope across rungs, with the SE taken over openings. That estimator is
  not written. The plain weighted fit below assumes independent rungs, so with
  shared openings its SE would be overstated.
- **What disjoint blocks give.** The rungs are independent, so the registered
  estimator is exactly the pooling script's per-rung output plus a weighted fit. That
  is conservative: the possible (1 − ρ) gain is given up, and ρ is unmeasured.

Arms share the block at each rung. Comparisons between arms at one rung are
therefore paired on openings, and the output rule uses that pairing.

### Matches

Each rung of each arm is played seat-balanced against the SL base: 400 games, as
two runs of 200 games on the rung's 100 openings. With `--openings-fen`, a file of
100 rows and `--games 200`, every row is played once, with no subsample. This is
loop-exp1's arena command, with `--openings-fen` in place of `--openings` and
`--opening-plies`:

```bash
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
COMMON="--sims 64 --games 200 --mode matched_sims --search-shape training --no-gumbel-noise \
  --openings-fen $R/openings/blocks/rung_$((X+6)).fen --max-plies 300 --syzygy $TB \
  --compile off --seed 2026112101"
# $M = the rung's model (snapshot or final), $T = a tag such as c64_P_3e-6_x-6
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON --candidate $M --reference $R/sl_base.pt \
  --label s1_${T}_A --games-out $R/arenas/s1_${T}_A.games.jsonl --pgn-out $R/arenas/s1_${T}_A.pgn \
  --out $R/arenas/results.jsonl
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON --candidate $R/sl_base.pt --reference $M \
  --label s1_${T}_B --games-out $R/arenas/s1_${T}_B.games.jsonl --pgn-out $R/arenas/s1_${T}_B.pgn \
  --out $R/arenas/results.jsonl
PYTHONPATH=. python -u scripts/arena_seat_balanced_pool.py --arm $M \
  --a-games $R/arenas/s1_${T}_A.games.jsonl --b-games $R/arenas/s1_${T}_B.games.jsonl \
  --a-result $R/arenas/results.jsonl --b-result $R/arenas/results.jsonl \
  --a-label s1_${T}_A --b-label s1_${T}_B --out $R/arenas/s1_${T}_pooled.json
```

- The arena plays at 64 sims for every arm. It measures the raw net plus a fixed
  search, whatever sims made the targets.
- Every arena record must meet loop-exp1's "Required in every arena record" list:
  the shas, `game_log_provenance`, `eval_cuda_graphs`, zero flushes, and a complete
  run.
- Every net that plays is write-protected from the end of its arm's training
  until its pooling is recorded.

### The registered test: trend of Elo on log₂(dose)

For each arm, every rung k gives three numbers:
- the dose x_k = log₂(fraction) ∈ {−6, …, 0};
- y_k = `result.pooled.excess`: the arm's seat-balanced per-game score excess over
  0.5;
- s_k = `result.pooled.se`: its SE, clustered by opening (100 clusters).

**Estimator.** Weighted least squares with weights w_k = 1 / s_k²:
- x̄ = Σ w_k x_k / Σ w_k, and ȳ the same way;
- slope b = Σ w_k (x_k − x̄)(y_k − ȳ) / Σ w_k (x_k − x̄)²;
- SE(b) = 1 / √(Σ w_k (x_k − x̄)²);
- z = b / SE(b);
- the fitted excess at dose 1 is ŷ₀ = ȳ − b x̄, with
  SE(ŷ₀) = √(1 / Σ w_k + x̄² SE(b)²).

On the Elo scale, the slope is about 694.9 × b Elo per doubling (the derivative of
−400 log₁₀(1/(0.5 + x) − 1) at x = 0 is 1600 / ln 10).

**Test.** One-sided at 2.5%, family-wise over the m arms whose trend is tested
(Bonferroni). An arm passes when z > z_crit = Φ⁻¹(1 − 0.025 / m). With 16 arms,
z_crit = 2.955; with 8, 2.734.

**What a pass means.**
- The null hypothesis is "Elo does not change with dose", at any level, including a
  constant offset from the SL base. A pass says more collection time measurably
  helps.
- A pass does not by itself show the net beats the SL base. That is the per-rung
  bound in the [output rule](#stage-1-output-the-pre-written-rule).
- A non-monotone response (the dip and recover seen in the ft595 fine-tune) is read
  as its average trend.
- If the response is not linear in log₂ dose, the test loses power, not validity.

```bash
trend() {  # $1 = m (arms in the family); then x=pooled.json for each rung played
  python3 -c 'import json, math, sys
from statistics import NormalDist
m = int(sys.argv[1]); pts = []
for a in sys.argv[2:]:
    x, f = a.split("=", 1); p = json.load(open(f))["result"]["pooled"]
    pts.append((float(x), p["excess"], p["se"]))
if len(pts) < 3 or any(s is None or not s > 0 for _, _, s in pts):
    print("fewer than 3 rungs, or an unavailable SE: no trend test"); sys.exit(4)
w = [1 / s ** 2 for _, _, s in pts]; sw = sum(w)
xb = sum(wi * x for wi, (x, _, _) in zip(w, pts)) / sw
yb = sum(wi * y for wi, (_, y, _) in zip(w, pts)) / sw
sxx = sum(wi * (x - xb) ** 2 for wi, (x, _, _) in zip(w, pts))
b = sum(wi * (x - xb) * (y - yb) for wi, (x, y, _) in zip(w, pts)) / sxx
se = 1 / math.sqrt(sxx); zc = NormalDist().inv_cdf(1 - 0.025 / m); k = 1600 / math.log(10)
y0 = yb - b * xb; se0 = math.sqrt(1 / sw + xb ** 2 * se ** 2)
out = {"rungs": len(pts), "slope_excess": b, "se": se, "z": b / se, "z_crit": zc,
       "slope_elo_per_doubling": k * b, "se_elo_per_doubling": k * se,
       "fitted_excess_dose1": y0, "fitted_se_dose1": se0, "passed": b / se > zc}
print(json.dumps(out)); sys.exit(0 if out["passed"] else 3)' "$@"
  echo "exit=$?"
}
T=c64_P_3e-6
trend 16 -6=$R/arenas/s1_${T}_x-6_pooled.json -5=$R/arenas/s1_${T}_x-5_pooled.json \
  -4=$R/arenas/s1_${T}_x-4_pooled.json -3=$R/arenas/s1_${T}_x-3_pooled.json \
  -2=$R/arenas/s1_${T}_x-2_pooled.json -1=$R/arenas/s1_${T}_x-1_pooled.json 0=$R/arenas/s1_${T}_x0_pooled.json
```

Exit codes: 0 = pass, 3 = not passed, 4 = no test (fewer than three rungs).
The function above was run verbatim on seven synthetic rungs with a slope of
0.0072 excess per doubling and an SE of 0.01796 per rung (a 400-game rung). It
printed an SE of 2.36 Elo per doubling and z = 2.12, not passed at m = 16
(`exit=3`), and recovered the fitted dose-1 excess of 0.0100 (SE 0.0122).

The function's slope arithmetic was checked on five synthetic pooled files, with a
slope of 0.0072 excess per doubling and an SE of 0.008032 per rung:
- it printed z = 2.835, a slope of +5.00 Elo per doubling and an SE of 1.76;
- that is a pass at m = 6 (z_crit 2.638) and at m = 1;
- given only two rungs, it printed `exit=4`.

### Power

The per-pair SD is 0.254, from the banked matches. A 400-game rung has P = 200
pairs, so its pooled SE is s = 0.254 / √200 = 0.0180 excess, or 12.5 Elo. That
assumes the two orientations of an opening are uncorrelated. A correlation r > 0
inflates s by √(1 + r); the test uses the realised clustered SE.

With seven equally weighted rungs, Σ (x − x̄)² = 28, so SE(b) = s / √28 = 2.36 Elo
per doubling. The effect the test detects with 80% power is (z_crit + 0.842) ×
SE(b): 6.6 Elo per doubling for one arm, and 9.0 with all 16 arms in the family.

The games per rung needed for 80% power are
2 × ((z_crit + 0.842) × 0.254 × 694.9 / (δ × √R_x))², where R_x = Σ (x − x̄)²:

| Rungs (R_x) | Arms tested (m) | δ = +10 Elo per doubling | δ = +25 Elo per doubling |
|---|---:|---:|---:|
| 5 (10) | 1 | 489 | 78 |
| 5 (10) | 16 | 898 | 144 |
| 7 (28) | 1 | 175 | 28 |
| 7 (28) | 8 | 285 | 46 |
| 7 (28) | 16 | 321 | 51 |

**Design: 7 rungs × 400 games.** With all 16 arms in the family, this has 80% power
for a true trend of +9.0 Elo per doubling, which is +54 Elo across the six
doublings.
- 400 rather than the minimum 321: the per-rung bound used by the output rule needs
  it, and 100 opening clusters keep the clustered SE stable.
- At +25 Elo per doubling, 51 games per rung would do. 400 is a floor for the normal
  approximation and for opening coverage, not a power requirement.
- A smaller true trend is likely to read "not passed". That means unread, not "no
  effect".

### Stage 1 output: the pre-written rule

The rule is applied in this order, with no other discretion:

1. **Eligible arms.** Those whose trend test passes at the family-wise z_crit
   (m = the number of arms tested). Only rungs that passed every gate count.
2. **Recipe, LR and sims.** Among the eligible arms, take the highest fitted excess
   at dose 1 (`fitted_excess_dose1`): the most Elo for the same collection GPU time.
   Every arm's dose 1 is the same time.
   - A tie is two fitted values within 5 Elo of each other.
   - Ties go first to the steeper slope, then to the lower LR, then to 64 sims (the
     cheaper target, and the one every banked read used), then to P over V. P has
     no named failure mechanism; V carries the flatter-value risk (loop-exp1,
     "Prior and choice of primary").
3. **Rows per generation.** The smallest dose rung of the chosen arm whose pooled
   95% lower bound (the low end of `excess_ci95`) is above 0. Ties go to the smaller
   dose.
   - If no rung's lower bound is above 0 but the fitted dose-1 excess is positive,
     use dose 1.
   - Otherwise the outcome is "trend without a gain": stop and report (rule 5).
4. **Stage 2 starts** from the chosen arm's model at the chosen rung (snapshot or
   final). It uses that arm's recipe, LR and sims. The generation size is the chosen
   dose's games (the fraction × that collection's games).
5. **Kill / stop.** If no arm passes, Stage 2 does not launch, and the record says
   which reading applies:
   - **KILL:** every tested arm's one-sided 97.5% upper bound on the slope is below
     +5 Elo per doubling;
   - **"unread, insufficient evidence"** otherwise.

   The next lever is named in either case. Stage 0 shows the search headroom, so the
   failure is in transfer. The candidates are the target construction (value-target
   mix, policy softening such as Arm S, sharper or deeper targets beyond 256 sims)
   and the optimizer (Aurora with the SL base's own state,
   `--init-optimizer-state checkpoint`). Each is a new record with its own
   calibration.

Rule 3 selects among 7 rungs. Its per-rung 95% bound is a decision threshold, not a
registered claim; the claim Stage 1 registers is the trend test. Stage 2's readouts
re-measure every claim, on openings Stage 1 never used.

### Stage 1 compute

**Rates:**
- collect: 3,284 games/h at 64 sims (128-game runs); 256 sims to be measured
  (pre-launch check P6);
- training: 0.47–0.84 s per update (F12 record: 512 updates in 239–432 s);
- arenas: 1,274–1,800 games/h (the 800-game D4 arms took 1,600–2,260 s).

One job runs on the GPU at a time. Nothing else runs during an arena.

| Item | 16 arms (default) | 8 arms (64 sims only, or one recipe) |
|---|---:|---:|
| Collect (O1 = one day in total) | 24 h | 24 h |
| Train `c64/` arms, ≈ 14,200 updates each | 15–27 h (8 arms) | 15–27 h |
| Train `c256/` arms, ≈ 3,550 updates each | 3.7–6.6 h (8 arms) | — |
| Model gates (7 models per arm) | ≈ 2 h | ≈ 1 h |
| Arenas, 7 rungs × 400 games per arm | 25–35 h (44,800 games) | 12–18 h |
| **Total** | **≈ 70–95 h (2.9–3.9 days)** | **≈ 52–70 h** |

Collect gates run on CPU, an estimated 10–15 min per collection. With O1 = two days,
collection and training time double (about 112–151 h in total), and arena time does
not change.

## Stage 2: the multi-generation run

Stage 2 launches only on a Stage 1 pass. It takes the recipe, LR, sims, starting
model and generation size from the [rule](#stage-1-output-the-pre-written-rule).

### Driver settings

The run is one run dir driven by `run --generations G`.

**Start:**
- `--init-checkpoint` is the Stage 1 model the rule chose. It is copied into the run
  dir, sha-verified and write-protected, the same way as the SL base copy.
- That model is a weights-only snapshot or a final model. Either way, the first
  phase starts a fresh optimizer.
- Stage 2's generation 1 is the loop's second generation.

**Generation size:** the chosen dose's game count, at the chosen sims and
concurrency 128, with Stage 1's collect flags.

**Phase window: `--presentations 1`.** Each phase trains once on its own
generation's rows, which is the regime Stage 1 measured. These alternatives were
not measured and are not registered:
- `--presentations 2`: this generation's rows plus the previous generation's, each
  row twice in its lifetime;
- `--sample-with-replacement --window`.

**Optimizer: `--optimizer-state fresh`** every generation, at the Stage 1 LR, with
warmup 0 and a flat schedule. Each phase then starts the optimizer the way Stage 1
measured.
- `--optimizer-state continue` would carry Adam moments across generations. The
  driver supports it: it restores the previous phase's optimizer, scheduler and
  step, and refuses a changed `--lr` unless `--reapply-optimizer-hparams` is given.
- It was not measured, and is not registered.

**Openings:** `--opening-sampling cycle` with `--seed 2026112200`, an independent
permutation of TRAIN.
- At small generation sizes, the cursor walks the 153,842 lines without reuse
  within Stage 2.
- Stage 1's collections played up to about 25% of TRAIN, with a different seed.
  That share of Stage 2's lines was also played in Stage 1, by the SL base.
- This is counted, not avoided: the driver cannot continue another run dir's
  cursor.

**Holdout and probe:** `--holdout-frac 0.05` and `--probe-rows 2048`. The probe
measures drift against the previous generation, the driver's definition.

**Evaluation:** `--no-eval`. Readouts are run separately (below), so the driver's
own per-generation arena does not spend GPU time every generation.

### Banked per generation (kept for the whole run)

- `gen_NNNN/collect/`: rows, games, stats and manifest;
- `gen_NNNN/train/`: the Trainer checkpoint with its optimizer state,
  `presented.npz`, `metrics.json` and manifest;
- gate outputs in `$R2/gates/`: collect gates, `gt1`, `gt2` and the G-T4 reads;
- each collect's content digest (`simple_rl_collect_gates.py --content-digest`).

Disk use is about 0.26 MB per 64-sim game, plus about 0.7 GB per Trainer
checkpoint. Nothing is pruned during the run.

### Per-generation gates

Loop-exp1's bands were set on SL-base collects. Later generations are different
nets, so in Stage 2 the gates are split into three groups.

**Gated, machinery.** A failure here means the collect cannot be trusted:
- G-C1 = 0;
- G-C2 equal;
- G-C3 ≥ +0.30 on both root classes (the positive control's bound);
- G-C7 ≥ 80 rows per game;
- G-C10 ≤ 0.05.

**Recorded, with a drift alarm:** G-C4, G-C5, G-C6, G-C8 and G-C9, read against
loop-exp1's bands. If two consecutive generations fall outside a band, the run
pauses for a report. A pause voids nothing.

**Model gates:**
- **G-T2** against the previous generation: KL above 1.5 or |ΔQ| above 0.4 fails
  (`gt2`).
- **G-T1** on the audit rows: mean |Q| ≥ 0.33 and P(draw) in 0.45–0.60. These
  bands are guesses, taken from V's lower edge and the control's band.
- **G-T4** realised settings, as loop-exp1.

**On a gate failure mid-run:**
- **Collect gate:** redo that collect once (`collect --redo --generation g`). The
  redo starts from the sealed cursor, so it replays the same lines. If the redo
  also fails, the run stops at generation g − 1, the last passing generation.
- **Model gate:** the run stops at g − 1. The failed model is kept for diagnosis.
- Restarting after either stop needs an addendum to this record.

### Readouts

Every k generations, the newest net plays two seat-balanced matches of 2,000 games
each. Each is two runs of 1,000 games on a block of 500 openings, pooled as in
Stage 1:
- **vs the SL base,** on `readout_base.fen`, at every readout. The openings are the
  same each time, so changes from one readout to the next are paired on openings.
- **vs the previous readout net,** on `readout_prev.fen`. At the first readout, the
  previous readout net is Stage 2's starting model.

**Size.** A 2,000-game match has an SE of 5.58 Elo, so it detects
(1.96 + 0.84) × 5.58 = 15.6 Elo with 80% power (one-sided, 2.5%). After Stage 1,
its realised per-pair SD replaces 0.254. If Stage 1 shows per-generation gains of
tens of Elo, the readouts shrink to the size that keeps 80% power for half the
measured gain.

**Cadence.** k is the smallest integer with k × g₁ ≥ 15.6 Elo, clamped to 1–4:
- g₁ is Stage 1's point estimate of the chosen rung's Elo against the SL base,
  taken as the per-generation gain;
- a vs-previous readout then has about 80% power for the expected gain;
- if g₁ ≤ 3.9 Elo, k = 4.

**Two anchors that do not move.** Anchored Elo depends on the anchor here: one
recorded three-contrast design found the anchors differing by up to 82 Elo
(experiment ledger). So the run uses two anchors.
1. **The SL base**, through the vs-SL-base match above.
2. **Stockfish at fixed nodes**, through `scripts/match_vs_uci.py`, with our engine
   run as `python -m chess_anti_engine.uci --checkpoint <net>`.
   - Before Stage 2, the SL base plays Stockfish at a few node counts. The count at
     which it scores 25–75% is fixed for the whole run and recorded.
   - Each readout plays 1,000 games against it (`--nodes-b <fixed>
     --enforce-nodes-b --openings $R/openings/blocks/anchor.fen`).
   - The exact UCI command and our engine's per-move budget are checked before
     Stage 2.

Each readout records both anchored readings. The transitivity check is that
(net vs SF) − (SL base vs SF) agrees with (net vs SL base) within the combined
CI. A disagreement is reported, not averaged away.

### Stop rule (pre-committed)

The run stops at the first of these:
1. **Regression:** two consecutive readouts whose vs-previous pooled upper 95%
   bound is below 0.
2. **Plateau:** three consecutive readouts that are unread against the previous
   readout net (the 95% CI contains 0).
3. **Gates:** a gate failure that one re-collect does not clear.
4. **Time:** the owner's time cap.

**The deliverable net** on a stop is the readout net with the highest pooled
estimate against the SL base, among those whose lower bound is above 0. That choice
is a selection, with winner's curse, so the delivered claim is re-measured once: a
2,000-game seat-balanced match against the SL base on `confirm.fen`, openings no
earlier match used, read at 95%.

### Recovery

- Every generation's model and Trainer state are kept, so the run can roll back to
  any generation.
- To resume from the last passing generation g, use `--redo --redo-downstream` on
  generation g + 1's collect. That moves the later stages to `superseded/`, where
  they are kept and never read. The alternative is a new run dir with
  `--init-checkpoint` set to generation g's model.
- The SL base, the live checkout, the production recipe and every earlier bank are
  left untouched. Nothing is promoted to production by this run.

### Compute per generation and per readout

With C = the generation size in games, at the chosen sims:

- **A generation** costs the collect, C / rate hours, plus training,
  (C × 96.8 × 0.95 / 256) updates × 0.47–0.84 s, plus a few minutes of gates.
  - At 64 sims and C = 1,200 (a 1/32 dose of `c64/`): about 0.4 h of collect plus
    0.1 h of training, so about 40 generations a day before readouts.
  - At 64 sims and C = 39,000 (dose 1): about 12 h plus 1.9–3.3 h, so about 1.6
    generations a day.
- **A readout** is 4,000 arena games, 2.2–3.1 h, plus the Stockfish anchor (1,000
  games, at a rate not yet measured).

### What success means at one week

**The one-week target, pre-committed:** within 7 days of Stage 2's start, the
deliverable net's confirmation match against the SL base reads a 95% lower bound
above 0, with a point estimate of at least +50 Elo.
- Short of +50, with the lower bound above 0: "gain confirmed, below target".
- With the lower bound not above 0: "not shown at one week".

+50 Elo is about a fifth of what a 4× search is worth to the SL base. It is still a
guess ([below](#what-is-still-a-guess)). Before Stage 2 launches, Stage 1 replaces
it with a projection: generations in a week × g₁, discounted by half. The
projection is recorded at the Stage 1 readout.

## Why no sizing from loss curves

The experiment ledger (2026-08-23) records three cases where a fit or audit gain did
not transfer to Elo:
- the F-only arm: held-out CE fell 0.043 nats, for 0 cp;
- absorption: 11.6× the fit, for 0 Elo;
- v020: +3.1 cp on the deep-SF audit, for −9.56 Elo (−29.59 to +10.41) at 800
  games.

Separately, the ft595 fine-tune traded Elo for fit: −41.0 Elo (−71.3 to −11.4).

So every dose, LR, sims and recipe choice here is made on arena Elo against a fixed
opponent. Loss, CE, probe KL and |ΔQ| are recorded, and serve only as gross-defect
gates.

## What is still a guess

| Number | Where it is used | The measurement that replaces it |
|---|---|---|
| 3,284 games/h at 64 sims | collection size, every compute line | Stage 1's own timed collect (only 128-game runs are banked) |
| The 256-sim rate (a quarter of 64-sim assumed) | `c256/`'s game count, compute | pre-launch check P6 (a 128-game 256-sim collect) |
| 96.8 rows per game | N, updates, training time | the sealed collects (N is computed exactly before any arm trains) |
| 0.47–0.84 s per update | training time | Stage 1's first arm (the train manifest's wall time) |
| 1,274–1,800 arena games/h | arena time | Stage 1's first rung match |
| Per-pair SD 0.254, zero correlation between orientations | all match sizing and power | Stage 1's realised clustered SEs; Stage 2's sizing is redone with them |
| +9 Elo per doubling as the effect worth detecting | games per rung | none: it states what is worth detecting, and the owner can adjust it. A smaller real effect reads "unread" |
| LR grid 3e-6 / 1e-5 / 3e-5 / 1e-4 | Stage 1 arms | Stage 1 measures inside it; an optimum at an edge points the next sweep outward |
| Dose grid 1/64 … 1, linear in log₂ dose | the trend test | Stage 1's rung estimates, where curvature is visible |
| That a Stage 1 rung's gain (g₁) repeats each Stage 2 generation | readout cadence k, the one-week projection | Stage 2's first readouts |
| `--presentations 1`, a fresh optimizer each generation | Stage 2 | a separate later comparison; not measured here |
| +50 Elo at one week | the success line | Stage 1's projection, recorded before Stage 2 |
| The Stockfish node count for the anchor | the second anchor | the pre-Stage-2 calibration match (25–75% for the SL base) |
| Stage 2's G-T1 bands (\|Q\| ≥ 0.33, P(draw) 0.45–0.60) | the per-generation model gate | Stage 1's audit reads at every rung (the range of the moving nets) |
| G-T2 bounds KL 1.5 / \|ΔQ\| 0.4 | the gross gate | Stage 1's per-rung probe reads (the realised range across LRs) |
| The arena's 64 sims as the yardstick | every match | not replaced: fixed so that matches stay comparable |
| The Gumbel noise and move-temperature schedules | collection | not measured here; inherited from the driver's defaults |
| The one-week horizon and the one-day Stage 1 cap | budgets | the owner's choices (O1) |

## Pre-launch checks

- **P1.** #1437 is merged. The run executes from a detached worktree at a pinned
  main sha, recorded here, with the extensions rebuilt.
- **P2.** The openings split is reproduced and checked (loop-exp1 step 0b), and the
  blocks are cut, with their sha256 recorded.
- **P3.** The SL base is copied into the run dir and write-protected (loop-exp1 step
  0a).
- **P4.** At least about 30 GB of disk is free for Stage 1, plus Stage 2's estimate.
- **P5.** The `trend`, `gt1` and `gt2` functions are re-run on the dry run's banked
  files, and on synthetic files, and print the outputs stated in their sections.
- **P6.** A 128-game collect at 256 sims (otherwise `$COLLECT`, `--games 128`, a
  scratch run dir) measures the 256-sim games/h and passes the machinery gates. The
  `c256/` game count is computed from that rate and recorded.

## Confounds

- Each collection's arms share one sample, so every arm verdict is conditional on
  its collection.
- Stage 1's rungs are snapshots along one trajectory per arm. The models at
  successive rungs are not independent fits, though their matches use disjoint
  openings. The trend describes this trajectory.
- `c64/` and `c256/` share their first openings (same seed and cursor). This pairs
  the comparison between them; it does not bias it.
- Stage 2's first generation starts from a net trained on part of a Stage 1
  collection.
- The yardstick (64 sims, no root noise, EVAL openings, a 300-ply cap) is not F12's
  instrument.
- loop-exp1's confounds (twofold-repetition draws, the same-coloured-bishops gap)
  apply unchanged.
