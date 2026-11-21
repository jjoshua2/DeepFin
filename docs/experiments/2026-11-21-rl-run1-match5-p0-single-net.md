# 2026-11-21: RL run 1, three follow-up matches — P0 as a single net (5); S's value on the base policy, raw and recalibrated (6); Zq recalibrated vs Zq (7) (pre-registration)

**Status: registered, not launched.** Nothing launches before the statistics review and
this PR's merge. The three matches are diagnostic: they adopt nothing and change nothing
live.

**Order of play: 5, 7, 6b, 6a.**
- The two plain-speed matches go first.
- In match 6, the control (6b) goes before the recal graft (6a).
- No rule depends on the order.

**Total ≈ 6.8 GPU-h** at measured rates (see "Cost").

**Multiplicity.** There are three post-hoc diagnostic readings, each at ≈ 2.5% one-sided.
Any "adoptable" or "shape" reading only NOMINATES a candidate. That candidate must be
confirmed on fresh games in its own record.

## Common to all three matches

**Commit.** The pilot's `$PIN` `0c795fb7a`, from the existing worktree
`~/projects/chess-worktrees/rl-run1-pilot-0c795fb7a`. This is the commit of the pilot's
matches, at 64 sims, with CUDA graphs and `--compile off`. So the launch record's
within-run comparison rule holds: every result row's `git_sha` must equal `$PIN`.

**Nets.** All read by absolute path from the frozen pilot bank
`~/chess-artifacts/operations/rl-run1-pilot/`, which is read-only:

| Net | Path | sha256 |
|---|---|---|
| SL base | `sl_base.pt` | `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0` |
| P0 | `arm_P0/gen_0001/train/model.pt` | `4503bd4dc0a1d2e3161f65778ab2414407f5297b911a2f63b901156cc244f9b6` |
| S | `arm_S/gen_0001/train/model.pt` | `66ceb271beee37206afca097904156a0f36f23fab5aecda1439aca7bf23f455b` |
| Zq | `arm_Zq/gen_0001/train/model.pt` | `ad16d2f177036a3246d092856e99477a1e91dddec7f0229801688c2cf5980b0a` |

**Output root.** A new root, `~/chess-artifacts/operations/rl-run1-followup/`:
- real runs in `arenas/<match>/`;
- smokes in `smoke/arenas/<match>/`.

The pilot bank is never written.

**Balance and pooling.**
- Seat-balanced A/B: run B swaps the seats, and both runs use the same seed and openings.
- One look per match.
- `arena_seat_balanced_pool.py` at its default z = 1.96.
- Every rule reads `pool.json` → `result.pooled.elo_ci95`.

**SEs.** Predicted SEs scale from the gate's measured **6.74 Elo at 1,600 games**. The graft
measured 3.86 at 4,800, which agrees. At 6,400 games the SE is ≈ 3.37 and the 95% interval
is ≈ 13.2 Elo wide. **The rules are read on each match's MEASURED clustered SE, and the
realised SE is reported.**

**Kill / void.**
- The launch record's [max-plies rule](2026-11-21-rl-run1-pilot-launch-record.md#the-max-plies-rule-registered)
  applies as registered: an aborted look is void and is reported with its sidecar; it is
  never resumed or re-rolled; it is rerun whole only under that section's backup-seed rule.
- An arena exit other than 0 voids the run.
- A failed seat readback (below) voids the run.

**Recovery.** Nothing to recover: no training, no live change.

**Opening overlap.** I reproduced the arena's own draw read-only: `load_paired_openings` with
`np.random.default_rng(seed)`. That method reproduces the gate's 400 openings exactly. Each
seed below draws 1,600 distinct openings from the same 4,000-line `eval.pgn`. The overlaps
are what independent seeds give, and none biases a match: the opening is a covariate.

| Match (seed) | vs the pilot's matches (1,496 distinct) | vs match 5 | vs match 6 |
|---|---|---|---|
| 5 (2026112151) | 620, of which 448 with the graft | — | — |
| 6a and 6b (2026112161) | 595, of which 431 with the graft | 654 | — |
| 7 (2026112171) | 588, of which 446 with the graft | 640 | 615 |

**Seeds.** None of the three appears in any pilot match:
- the pilot matches used 2026112101, 2026112103 and 2026112141/42/43;
- the S extension reserved 2026112131/32/33;
- match 2 reserved 2026112102;
- the collection used 2026112100;
- none of the three appears anywhere under `docs/`.

### The launcher (new; flags byte-identical to the pilot's `$ARENA`)

The pilot's `run_match.sh` hard-codes the pilot bank as its output root, so it cannot be
used. This copy takes the output root as its first argument and refuses the frozen bank.
`$ARENA` expands to exactly the pilot's string: the same flags, with the openings and
Syzygy paths read from the pilot bank. It still checks the PIN and refuses a look that
already has a game log. It is saved as `~/chess-artifacts/operations/rl-run1-followup/run_followup_match.sh`:

```bash
#!/bin/bash
# RL run 1 follow-up matches 5/6/7: one arena run at the pilot's PIN, flags identical to the pilot's run_match.sh.
# usage: run_followup_match.sh <out_root> <match> <look> <A|B> <seed> <games-per-run> <seat args...>
P=$HOME/chess-artifacts/operations/rl-run1-pilot            # frozen pilot bank: READ ONLY
PIN=$(cat $P/PIN.txt)
PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
PY=$HOME/projects/chess/.venv/bin/python
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
ARENA="--mode matched_sims --sims 64 --search-shape training --no-gumbel-noise --compile off \
  --openings $P/openings/eval.pgn --opening-plies 16 --max-plies 600 --on-max-plies abort \
  --syzygy $TB --max-concurrent-games 128"
O=$(realpath -m "$1") m=$2 k=$3 s=$4 seed=$5 n=$6; shift 6   # absolute before the cd below
case "$O" in "$(realpath "$P")"|"$(realpath "$P")"/*) echo "refusing to write into the frozen pilot bank"; exit 94;; esac
d=$O/arenas/$m; mkdir -p $d
cd $PINW || exit 97
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "worktree not at PIN"; exit 96; }
[ ! -e $d/${m}_${k}_$s.games.jsonl ] || { echo "look ${m}_${k}_$s already has a game log: never resume or re-roll a registered look"; exit 95; }
PYTHONPATH=. $PY -u scripts/arena_standard.py $ARENA --seed $seed --games $n "$@" \
  --label ${m}_${k}_$s --games-out $d/${m}_${k}_$s.games.jsonl --pgn-out $d/${m}_${k}_$s.pgn \
  --out $d/results.jsonl > $d/${m}_${k}_$s.log 2>&1
rc=$?
echo "$rc $(date -Is)" > $d/${m}_${k}_$s.exit
exit $rc
```

### Seat readback: how the hybrid seats are proven to take effect (matches 6 and 7)

This check is saved as `rl-run1-followup/check_seat.py`. It reads three things, all written
by the consumer, and exits nonzero on any mismatch:
- **the built module:** the arena log line from the module itself;
- **the result row:** what the arena recorded for each seat;
- **the PGN:** the engine names.

```python
#!/usr/bin/env python3
# usage: check_seat.py <arena dir> <label> <hybrid role: candidate|reference> <value-from sha256 or -> <T,b or ->
# Explicit exits, never assert: `python -O` cannot disable a check.
import json, re, sys

def fail(msg):
    sys.exit(f"seat readback FAILED: {msg}")

d, label, role, vsha, recal = sys.argv[1:6]
if role not in ("candidate", "reference"):
    fail(f"role must be candidate or reference, got {role!r}")
other = "reference" if role == "candidate" else "candidate"
t_b = None if recal == "-" else [float(v) for v in recal.split(",")]
log = open(f"{d}/{label}.log").read()
m = re.search(rf"\[arena\] {role} HYBRID built: value_net=(\w+) recal_T=(\S+) recal_bias=\[([^\]]*)\]", log)
if not m:
    fail(f"no '{role} HYBRID built' line in {label}.log")
if m.group(1) != ("own" if vsha == "-" else "loaded"):
    fail(f"value_net={m.group(1)}, expected {'own' if vsha == '-' else 'loaded'}: {m.group(0)}")
if t_b is None:
    if m.group(2) != "None":
        fail(f"a recalibration was applied but none was expected: {m.group(0)}")
else:
    if m.group(2) == "None":
        fail(f"no recalibration was applied, expected T,b = {t_b}: {m.group(0)}")
    bias = [float(v) for v in m.group(3).split(",")]
    if abs(float(m.group(2)) - t_b[0]) > 1e-9 or abs(bias[1] - t_b[1]) > 1e-6 or bias[0] != 0.0 or bias[2] != 0.0:
        fail(f"built recalibration differs from T,b = {t_b}: {m.group(0)}")
if f"[arena] {other} HYBRID built" in log:
    fail(f"the {other} seat was built as a hybrid")
rows = [r for r in map(json.loads, open(f"{d}/results.jsonl")) if r.get("label") == label]
if not rows:
    fail(f"no result row labelled {label}")
row = rows[-1]
v = row.get(f"{role}_value")
if v is None or v.get("recal") != t_b:
    fail(f"result row {role}_value {v} does not carry recal {t_b}")
if (v.get("value_from") or {}).get("sha256") != (None if vsha == "-" else vsha):
    fail(f"result row {role}_value value_from {v.get('value_from')} is not sha256 {vsha}")
if row.get(f"{other}_value") is not None:
    fail(f"result row carries a {other}_value block: {row.get(f'{other}_value')}")
names = re.findall(r'\[(White|Black) "([^"]*)"\]', open(f"{d}/{label}.pgn").read())
pairs = list(zip(names[::2], names[1::2]))
if not pairs or any(w[1] == b[1] for w, b in pairs):
    fail("engine names coincide in some game (or the PGN has no games)")
print(f"seat readback OK: {label} {role} value_from={vsha[:12]} recal={t_b}")
```

**When it runs, and how a failure stops a run before any game counts:**
1. **Smoke first.** Before each real run of 6a, 6b and 7, run a smoke (4 games per run) with the SAME
   seat arguments and the same seed into `smoke/`, then `check_seat.py` on it. The real run
   starts only if the smoke passes. That proves the configuration takes effect before a
   real game is played.
2. **Smoke pool.** The smoke's A and B are also pooled. The pool must accept them: its
   `check_seat_values` requires the arm seat's (value-from sha256, [T, b]) to be equal in
   both runs.
3. **After each real run,** `check_seat.py` runs on the real log, row and PGN. A mismatch
   voids that run. No game counts before then, because a match is read only after this
   check.

For the record, the pool checks the arm seat's value signature. It does not print it, so
`check_seat.py` reads it from the result rows instead.

---

## Match 5 — P0 as a single net vs the SL base

**Question.**
- The pilot's match 4 graft paired P0's policy with the SL base's value, using two
  forwards per seat. It read +7.7 [−0.1, +15.4]: "unread".
- Match 5 asks whether P0 as a SINGLE net is at least not worse than the base.
- P0's own value head got no value loss, but it sits on a trunk that the policy training
  moved.
- A policy-only generation would ship P0 as one net.

**Design.**
- Plain seats: P0 vs the SL base, with no `--*-value-*`.
- Seed 2026112151.
- **6,400 games: 3,200 per run, 1,600 openings.**

**Size and power.** Power here is P(adoptable | true 0); it equals P(damage | true −10).

| Games | SE | Power | GPU-h |
|---|---|---|---|
| 1,600 | 6.74 | 0.32 | 0.38 |
| 3,200 | 4.77 | 0.55 | 0.76 |
| 4,800 | 3.89 | 0.73 | 1.14 |
| **6,400** | **3.37** | **0.84** | **1.52** |

**Rules.**
- **"damage"** iff UB95 < 0;
- **"single-net policy-only step is adoptable"** iff LB95 > −10;
- otherwise **"unread"**.

At the predicted SE (3.37) the interval is ≈ 13.2 Elo wide, so "damage" and "adoptable"
cannot both hold.

**Tie-break, if the MEASURED clustered SE is below ≈ 2.55 Elo.** The interval is then under
10 Elo wide, and both can hold, for example [−8, −1]. Both facts are reported, and the
reading is **"adoptable within the margin, with a measured loss"**:
- the step stays within the amendment's margin, which tolerates up to 10 Elo;
- like any "adoptable", it only nominates.

**Secondary, report-only.**
- The cost of P0's value drift = match 5's estimate − 7.66 (the graft), with
  SE ≈ √(3.37² + 3.86²) ≈ 5.1.
- That SE treats the matches as independent. They share 448 openings, which likely
  correlates them positively, so the SE is conservative.
- The graft's 7.66 is its third-look estimate from a group-sequential design that never
  stopped.

**Licenses.**

| Reading | Licenses |
|---|---|
| adoptable | NOMINATES a second policy-only generation for the owner, under its own record, confirmed on fresh games. Nothing more |
| damage | Policy-only training needs the value kept out of the moving trunk before any generation 2: for example a frozen value tower, or shipping the hybrid seat |
| unread | Nothing; the pilot's "stop for the owner" stands |

**Commands.**

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup; P=$HOME/chess-artifacts/operations/rl-run1-pilot
SL=$P/sl_base.pt; P0=$P/arm_P0/gen_0001/train/model.pt
PY=$HOME/projects/chess/.venv/bin/python; PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-$(cut -c1-9 $P/PIN.txt)
pool() { d=$1/arenas/$2; (cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $3 \
  --a-games $d/${2}_1_A.games.jsonl --b-games $d/${2}_1_B.games.jsonl --a-result $d/results.jsonl \
  --b-result $d/results.jsonl --a-label ${2}_1_A --b-label ${2}_1_B --out $d/pool.json > $d/pool.log 2>&1); }
$F/run_followup_match.sh $F p0_single 1 A 2026112151 3200 --candidate $P0 --reference $SL || exit 1
$F/run_followup_match.sh $F p0_single 1 B 2026112151 3200 --candidate $SL --reference $P0 || exit 1
pool $F p0_single $P0
```

---

## Match 7 — Zq recalibrated vs Zq, head-to-head (a NEW registration)

**Why, and what it is not.**
- **The pilot skipped its match 2 by rule.** Match 2 was this head-to-head, conditional on
  the gate reading damage. The gate read −10.0 [−23.2, +3.2].
- **Match 7 is a NEW registration, motivated by post-hoc offline evidence.** It is Addendum 1
  of the pilot value diagnosis (`rl-claude-20261119/pilot-value-diagnosis/`):
  `PREREG.addendum1.md`, sha256 `2a07543c…`, sealed before its run; results in
  `out/sibling_order.json`, sha256 `bb8e5b16…`.
- **The finding.** It measured sibling-ordering accuracy against deep SF on clear pairs
  (gap ≥ 50 cp, 38,705 pairs; the base scores 0.8554). Each net vs the base:
  - S: −0.0096 [−0.0129, −0.0064]; after recalibration −0.0093, so not repaired;
  - Zq: −0.0007 [−0.0032, +0.0017];
  - P0's own head: −0.0020 [−0.0040, +0.0001].

  So broken ranking does not explain Zq's −10. For Zq the shape account stays open: its
  (T, b) = (0.704, −0.227) removes 60% of its KL to the base (main README of the
  diagnosis, sha256 `0bdac6a5…`).
- **Match 7 cannot be read as the pilot's match 2, and it changes no pilot verdict.**

**Design.**
- Both seats are Zq's file. The arm seat is Zq + recal, a single forward per seat at plain
  speed.
- Recal: `--candidate-value-recal 0.7036189636786957,-0.22711530923843357` in run A, and
  `--reference-value-recal …` with the same value in run B.
- Seed: a fresh **2026112171**.
- **6,400 games: 3,200 per run, 1,600 openings.**

**The fit: the banked one is the launch record's Step 3 fit.**
- The fit is `pilot-value-diagnosis/out/recal_Zq.json`, field `cli`. It was produced at
  `0c795fb7a` (`scripts/run_fits.sh`) on the same Zq and SL base files, the same collect
  (manifest sha256 `61fc049f…`), and its hold-out games only: 416 games, 39,237 rows. It ran
  on CUDA with bf16 logits and batch 256, which is the tool's default.
- **The one difference from Step 3:** `--holdout-frac 0.05` was passed as a flag
  (`holdout_frac_source: "flag, unpinned"`). The record expected that value pinned in
  `$R/collect/run_config.json`, but the file is absent from the collect run dir. The value
  is the same, so the rows are identical.
- **No refit.** A refit by Step 3's exact command would need the same flag, and it would be
  a second fit after the gate and Addendum 1 were seen.

**Seed choice.** A fresh seed rather than the reserved 2026112102.
- The reserved seed was never played, so the two are statistically equivalent.
- A game log under 2026112102 would read as the pilot's registered match 2.

**Size and power.**
- At 6,400 games and the gate's SE, P(LB95 > 0 | true +10) = 0.84.
- **The gate's SE probably overstates the SE of a head-to-head between two nearly identical
  nets,** because more games are decided by the same move. The rule is therefore read on
  the MEASURED clustered SE, and the realised SE is reported.

**Rule.**
- **"the loss is shape"** iff LB95 > 0, with the arm being the recalibrated seat;
- otherwise **"unread"**.

**Report-only: the recovered fraction.** This is match 7's estimate divided by 10.0. The
gate's interval [−3.2, +23.2] puts the denominator near 0, so the fraction is poorly
determined: it is reported beside its numerator and never read.

**What it can and cannot establish.**
- It can establish that recalibration improves Zq.
- An "unread" leaves the shape account open.
- It says nothing about Zq vs the base.

**Licenses.**
- "shape" NOMINATES a recalibrated-Zq vs base adoption match, as a later and separate
  registration.
- "unread" licenses nothing.

**Seat readback (launch check).**
- Role: candidate in run A, reference in run B.
- Value-from: `-` (the arena log line reads `value_net=own`).
- Recal: the banked `cli` above.

**Commands** (continuing the shell from match 5):

```bash
Zq=$P/arm_Zq/gen_0001/train/model.pt; RC7=0.7036189636786957,-0.22711530923843357
for O in $F/smoke $F; do n=$([ $O = $F ] && echo 3200 || echo 4)
  $F/run_followup_match.sh $O zq_recal_h2h 1 A 2026112171 $n --candidate $Zq --reference $Zq --candidate-value-recal $RC7 || exit 1
  $PY $F/check_seat.py $O/arenas/zq_recal_h2h zq_recal_h2h_1_A candidate - $RC7 || exit 1
  $F/run_followup_match.sh $O zq_recal_h2h 1 B 2026112171 $n --candidate $Zq --reference $Zq --reference-value-recal $RC7 || exit 1
  $PY $F/check_seat.py $O/arenas/zq_recal_h2h zq_recal_h2h_1_B reference - $RC7 || exit 1
  pool $O zq_recal_h2h $Zq || exit 1
done
```

The first pass (`$F/smoke`, 4 games per run: 2 openings, so the pool has two clusters) is the smoke. The second pass plays the
real match, and runs only if every smoke step exits 0.

---

## Match 6 — S's value on the base policy, raw (6b) and (T, b)-recalibrated (6a)

**Question and source.**
- The pilot value diagnosis is at `rl-claude-20261119/pilot-value-diagnosis/` (README sha256
  `0bdac6a50852fa83d4f3cbcc7bfa067696440abcc027e2118eb5aa0ec1c00dbe`).
- It concludes that the value loss changed CONTENT, not shape:
  - S's (T, b) = (0.853, −0.120) removes 24% of its KL to the base;
  - 1-ply value regret: S +6.8 cp [+4.6, +9.2] vs the base on the frozen audit set;
  - (T, b) changes 1–2% of choices and none of the regret.
- It names its falsifier: S's value head, recalibrated and grafted onto the base policy,
  recovering most of the −20 at 64 sims.
- **Offline prediction on record (Addendum 1):** recalibration does NOT repair S's sibling
  ordering (−0.0096 → −0.0093), so it should NOT recover S's loss.

**The two seat options compose at `$PIN` (verified in the code).**
- `scripts/arena_standard.py` builds one `SeatValue(value_from=…, recal=…)` per seat.
- `chess_anti_engine/eval/hybrid_value.py` → `HybridValueNet.forward` takes the WDL from the
  VALUE net, then applies `wdl / T + [0, b, 0]` to it, at the root and at every leaf.
- That is the banked fit's formula: `softmax(logits / T + [0, b, 0])` on W, D, L logits,
  side to move.
- The fit is `pilot-value-diagnosis/out/recal_S.json`, field `cli` =
  `0.853159495428655,-0.12023510932922338`. It was fit on the pilot's 416 hold-out games,
  which are TRAIN-split openings and so disjoint from `eval.pgn`.

**Design.**

| | 6b (control): raw graft | 6a: recal graft |
|---|---|---|
| Arm seat | SL base policy + S's value: `--candidate-value-from $S` | the same, plus `--candidate-value-recal $RC6` |
| Base seat | plain SL base | plain SL base |
| Seed / openings | 2026112161, 1,600 openings | the same seed, so **exactly the same openings** |
| Size | **6,400 games** (3,200 per run) | **6,400 games** (3,200 per run) |

**The raw graft is the control; match 3 does not suffice.** Match 3 played S as a single net,
so its −20 mixes S's policy change with its value change. The predicted raw-graft result is
≈ −20 − 7.7 ≈ −28. That is an estimate, not a rule input.

**The never-run combination.** Both seats load the same checkpoint (the base). The PGN names
differ by the hybrid suffix, and `check_seat.py` and the pool prove which seat was the
hybrid. Both smokes must pass, or match 6 is not played. No code changes under this record.

**The recovery estimator, fixed now.**
- R = full 6a estimate − full 6b estimate (Elo, each from its own `pool.json`).
- SE_R = √(SE_a² + SE_b²) ≈ √(3.37² + 3.37²) ≈ **4.8**.
- The two matches share every opening, so their estimates are likely positively
  correlated, which makes this SE conservative.
- A paired-on-openings estimate may be reported as a report-only secondary. No rule reads
  it.

**Power.**
- P(R's LB95 > 0) is 0.55 at true R = +10, and 0.84 at R = +14 (half of the predicted −28).
- P(6a's UB95 < 0 | true −10) = 0.84.

**Rules: a decision order, read on the measured SEs.** The steps are mutually exclusive
and exhaustive; the first step that applies gives the reading.

1. **6b's UB95 ≥ 0 → "unread (value-graft damage not established)".**
   - R cannot be read as a recovery.
   - 6a, 6b and R are reported, each with its interval.
   - Whatever steps 2–4 would say is reported as report-only.

   Steps 2–5 apply only when 6b reads damage (UB95 < 0).
2. **Both rules hold: "partial recovery".** That is, 6a's UB95 < 0 AND R's UB95 < +10 AND
   R's LB95 > 0 AND 6a's LB95 > −10.
   - This needs R's interval under 10 wide AND 6a's interval under 10 wide. That means a
     measured SE_R below ≈ 2.55 and a measured SE_a below ≈ 2.55; the predicted values are
     4.8 and 3.37.
   - So the R-based clauses do not by themselves keep the two rules apart, and this step
     defines the reading.
   - Reported as: recalibration recovers a measurable part, 0 < R < +10, and the recal
     graft still loses, but within the margin.
   - It licenses what "shape" licenses (a nomination only). The diagnosis's content account
     is recorded for the unrecovered part.
3. **"content"** iff 6a's UB95 < 0 AND R's UB95 < +10.
4. **"shape"** iff R's LB95 > 0 AND 6a's LB95 > −10.
5. **Otherwise "unread".** The cells this step covers, all with 6b reading damage:
   - **6a reads damage, but R's UB95 ≥ +10:** partial recovery is not excluded.
   - **R's LB95 > 0, but 6a's LB95 ≤ −10:** reported as "recovery without adoption-level
     strength".
   - **6a's UB95 ≥ 0 and R's LB95 ≤ 0:** neither rule fires.

**Licenses.**

| Reading | Licenses |
|---|---|
| shape | NOMINATES a recalibrated S-value cell for the owner, under its own record, confirmed on fresh games. Nothing is adopted |
| content | The diagnosis's mechanism stands: a value target built from the 64-sim search value harms ranking that recalibration cannot fix. Value-target work needs a different target, or a held-out value, before generation 2 |
| partial recovery (step 2) | As "shape": a nomination only, with the content account recorded for the unrecovered part |
| unread | Nothing |

**Commands** (continuing the shell; 6b first, then 6a; each preceded by its 4-game smoke):

```bash
S=$P/arm_S/gen_0001/train/model.pt; SSHA=66ceb271beee37206afca097904156a0f36f23fab5aecda1439aca7bf23f455b
RC6=0.853159495428655,-0.12023510932922338        # pilot-value-diagnosis/out/recal_S.json "cli"
for spec in "s_value_raw -" "s_value_recal $RC6"; do set -- $spec; m=$1; rc=$2
  ra=(); rb=(); [ $rc != - ] && ra=(--candidate-value-recal $rc) && rb=(--reference-value-recal $rc)
  for O in $F/smoke $F; do n=$([ $O = $F ] && echo 3200 || echo 4)
    $F/run_followup_match.sh $O $m 1 A 2026112161 $n --candidate $SL --reference $SL --candidate-value-from $S "${ra[@]}" || exit 1
    $PY $F/check_seat.py $O/arenas/$m ${m}_1_A candidate $SSHA $rc || exit 1
    $F/run_followup_match.sh $O $m 1 B 2026112161 $n --candidate $SL --reference $SL --reference-value-from $S "${rb[@]}" || exit 1
    $PY $F/check_seat.py $O/arenas/$m ${m}_1_B reference $SSHA $rc || exit 1
    pool $O $m $SL || exit 1
  done
done
```

In each pool, the "arm" is the hybrid seat: run A's candidate and run B's reference.

---

## Cost (measured rates)

- **Rates.** Plain and recal-only seats cost 0.19 GPU-h per 800 games (launch record).
  Hybrid value-from seats cost 0.23–0.24 GPU-h per 800 games (measured, `plan-review-1473/`).
- **Smokes.** 4 games per run, 8 per configuration; negligible.

| Match | Games | Seats | GPU-h |
|---|---|---|---|
| 5 | 6,400 | plain | 1.52 |
| 7 | 6,400 | recal only | 1.52 |
| 6b | 6,400 | hybrid | 1.84–1.92 |
| 6a | 6,400 | hybrid | 1.84–1.92 |
| **Total** | 25,600 | | **≈ 6.7–6.9** |
