# 2026-11-21: RL run 1, matches 5, 6 and 7 — P0 as a single net; S's value recalibrated on the base policy; Zq recalibrated vs Zq (pre-registration)

**Status: registered, not launched.** Nothing launches before the statistics review and
this PR's merge. Both matches are diagnostic: they adopt nothing and change nothing live.
**Order of play: match 5, then match 7, then match 6.** Total ≈ 4.6 GPU-h (see the end).

## Question

The pilot's match 4 graft paired P0's policy with the SL base's value, so each seat ran
two forwards. That graft read +7.7 Elo, interval [−0.1, +15.4]: "unread". The pilot's
results record holds the full pilot readout.

Match 5 asks a different question. **As a SINGLE net, is P0 at least not worse than the
base?**
- P0's value head got no value loss.
- But it shares a trunk that P0's policy training moved.
- A future policy-only generation would ship P0 as one net, not as a graft.

## Design

| Item | Value |
|---|---|
| Candidate | P0: `arm_P0/gen_0001/train/model.pt`, sha256 `4503bd4dc0a1d2e3161f65778ab2414407f5297b911a2f63b901156cc244f9b6` |
| Reference | SL base: `sl_base.pt`, sha256 `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0` |
| Seats | plain: no `--*-value-from`, no recalibration |
| Arena flags | the pilot's `$ARENA` verbatim, from the bank's `run_match.sh`: `--mode matched_sims --sims 64 --search-shape training --no-gumbel-noise --compile off --openings $R/openings/eval.pgn --opening-plies 16 --max-plies 600 --on-max-plies abort --syzygy <production pair> --max-concurrent-games 128` |
| Commit | the pilot's `$PIN` `0c795fb7a`, from the existing worktree `~/projects/chess-worktrees/rl-run1-pilot-0c795fb7a`; `run_match.sh` refuses any other HEAD |
| Balance | seat-balanced A/B. Run A has P0 as candidate; run B swaps the seats. Both runs use the same seed, so they play the same openings |
| Seed | **2026112151**. Not used by any pilot match (2026112101, 2026112103, 2026112141/42/43), by the reserved S extension (2026112131/32/33) or by the collection (2026112100), and not found anywhere under `docs/` |
| Size | **ONE look of 6,400 games: 3,200 per run, 1,600 openings** |
| Pool | `arena_seat_balanced_pool.py` at its default z = 1.96 (one look) |

**Overlap with the pilot's openings.** I reproduced the arena's own draw read-only:
`load_paired_openings` with `np.random.default_rng(seed)`. That method reproduces the
gate's 400 openings exactly. Seed 2026112151 at 1,600 pairs draws 1,600 distinct openings:

| Pilot match | Its distinct openings | Shared with match 5 |
|---|---|---|
| match 4 (graft) | 1,087 | 448 |
| match 1 (gate) | 400 | 167 |
| match 3 (S) | 200 | 92 |
| any pilot match | 1,496 | 620 |

Both draws come from the same 4,000-line `eval.pgn`, so this is the overlap independent
seeds give. It does not bias match 5: the opening is a covariate, and the seed is new.

## Size and power

All SEs below scale from the gate's measured SE: **6.74 Elo at 1,600 games**. The graft
measured 3.86 at 4,800, which agrees.

"Power" is the chance of reading "adoptable" (LB95 > −10) when the true effect is 0. By
symmetry, it is also the chance of reading "damage" (UB95 < 0) when the true effect is −10.

| Games | SE (Elo) | Power | GPU-h |
|---|---|---|---|
| 1,600 | 6.74 | 0.32 | 0.38 |
| 3,200 | 4.77 | 0.55 | 0.76 |
| 4,800 | 3.89 | 0.73 | 1.14 |
| **6,400** | **3.37** | **0.84** | **1.52** |

At 1,600 games, the likeliest reading for a P0 that is truly no worse is "unread". 6,400
games is the smallest size in steps of 1,600 with power above 0.8.

## Reading rules (fixed in advance)

On the pooled estimate and its 95% interval:
- **"damage"** iff UB95 < 0;
- **"single-net policy-only step is adoptable"** iff LB95 > −10 (the amendment's adoption margin);
- otherwise **"unread, insufficient evidence"**.

The first two can both hold, for example [−9, −1]. If so, both are reported, and "damage"
governs what follows.

**Secondary, pre-declared, report-only.** The cost of P0's value drift, defined as match 5's
estimate minus the graft's +7.66, with SE ≈ √(3.37² + 3.86²) ≈ 5.1 Elo. The SE treats the
two matches as independent: separate seeds and separate runs. They share 448 openings,
which most likely correlates the two estimates positively, so this SE is likely
conservative. Two caveats:
- The graft's +7.66 is its third-look estimate from a group-sequential design that never
  stopped early.
- No rule reads the secondary.

## What each outcome licenses

| Outcome | Licenses |
|---|---|
| adoptable | A second policy-only generation becomes a candidate for the owner, under its own record. Nothing more |
| damage | Policy-only training needs the value kept out of the moving trunk before any generation 2: for example a frozen value tower, or shipping the hybrid seat |
| unread | Nothing. The pilot's "stop for the owner" stands |

## Control, confounds, recovery, cost

- **Control.** The SL base, in the same arena invocation, with seats swapped between runs A and B.
- **Confounds.** None beyond the pilot's. Match 5 runs at 64 sims, with CUDA graphs and `--compile off`, on the same commit (`0c795fb7a`) as the pilot's matches, so the launch record's within-run comparison rule holds: every result row's `git_sha` must equal `$PIN`.
- **Recovery.** Nothing to recover. Match 5 involves no training and no live change.
- **Cost.** ≈ 1.5 GPU-h at the measured 0.19 GPU-h per 800 plain games. An earlier ≈ 0.4 GPU-h estimate was for 1,600 games, which has power 0.32.
- **Kill / void.** The launch record's [max-plies rule](2026-11-21-rl-run1-pilot-launch-record.md#the-max-plies-rule-registered) applies as registered:
  - an aborted look is void and is reported with its sidecar;
  - it is never resumed or re-rolled; `run_match.sh` refuses an existing game log;
  - it is rerun whole only under that section's backup-seed rule.

  An arena exit other than 0 voids the match.

## Commands

The bank's `run_match.sh`, verbatim (`~/chess-artifacts/operations/rl-run1-pilot/run_match.sh`):

```bash
#!/bin/bash
# RL run 1 pilot, Step 4: one arena run at $PIN. Verbatim common flags from the launch record.
# usage: run_match.sh <match> <look> <A|B> <seed> <games-per-run> <seat args...>
R=$HOME/chess-artifacts/operations/rl-run1-pilot
PIN=$(cat $R/PIN.txt)
PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
PY=$HOME/projects/chess/.venv/bin/python
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
ARENA="--mode matched_sims --sims 64 --search-shape training --no-gumbel-noise --compile off \
  --openings $R/openings/eval.pgn --opening-plies 16 --max-plies 600 --on-max-plies abort \
  --syzygy $TB --max-concurrent-games 128"
m=$1 k=$2 s=$3 seed=$4 n=$5; shift 5
d=$R/arenas/$m; mkdir -p $d
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

Match 5. The bank is frozen, so the operator decides where the new `arenas/p0_single`
directory and the start / exit markers live:

```bash
R=$HOME/chess-artifacts/operations/rl-run1-pilot
SL=$R/sl_base.pt; P0=$R/arm_P0/gen_0001/train/model.pt
PY=$HOME/projects/chess/.venv/bin/python
PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-$(cut -c1-9 $R/PIN.txt)
$R/run_match.sh p0_single 1 A 2026112151 3200 --candidate $P0 --reference $SL || exit 1
$R/run_match.sh p0_single 1 B 2026112151 3200 --candidate $SL --reference $P0 || exit 1
d=$R/arenas/p0_single
cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $P0 \
  --a-games $d/p0_single_1_A.games.jsonl --b-games $d/p0_single_1_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl \
  --a-label p0_single_1_A --b-label p0_single_1_B --out $d/pool.json > $d/pool.log 2>&1
```

The readout is `pool.json` → `result.pooled` (`elo`, `elo_ci95`). It is read by the rules
above, in the same session, and recorded beside the pilot's results record.


---

# Match 6 — is S's value damage shape or content? (diagnostic, adopts nothing)

## Question and source

The pilot value diagnosis is banked at
`~/chess-artifacts/operations/rl-claude-20261119/pilot-value-diagnosis/`
(README sha256 `0bdac6a50852fa83d4f3cbcc7bfa067696440abcc027e2118eb5aa0ec1c00dbe`). It
concludes that the value loss changed **content**, not shape:
- **Recalibration removes little of the change.** S's fitted (T, b) = (0.853, −0.120)
  removes 24% of its KL to the base; Zq's (0.704, −0.227) removes 60%.
- **The value got worse at ranking moves.** 1-ply value regret on the frozen audit set,
  each vs the base:

  | Net | Regret vs base |
  |---|---|
  | S | +6.8 cp [+4.6, +9.2] |
  | Zq | +2.3 cp [+0.6, +4.0] |
  | P0's own head | +1.9 cp [+0.2, +3.6] |

  (T, b) changes 1–2% of choices and none of the regret.
- **Calibration to outcomes improved.** Both trained heads are better calibrated to
  outcomes on hold-out games.

The diagnosis names its own falsifier: S's value head, recalibrated by (T, b) and grafted
onto the base policy, recovering most of the −20 in a 64-sim arena. Match 6 is that test.

**Offline prediction now on record (Addendum 1).** Recalibration should NOT recover S's
loss.
- Source: `pilot-value-diagnosis/PREREG.addendum1.md`, sha256 `2a07543c…`, sealed before
  its run; results in `out/sibling_order.json`, sha256 `bb8e5b16…`.
- Measure: pairwise sibling-ordering accuracy against deep SF on clear pairs (gap ≥ 50 cp,
  38,705 pairs), where the base scores 0.8554.
- S scores −0.0096 [−0.0129, −0.0064] against the base, and −0.0093 after recalibration.
  The ordering is not repaired.

Match 6 tests that prediction; it does not change its rules.

## The seat composes at `$PIN` with no new code (verified)

Read at `0c795fb7a`:
- **Both options apply to one seat.** `scripts/arena_standard.py` passes both
  `--candidate-value-from` and `--candidate-value-recal` into one `SeatValue`, and
  `chess_anti_engine/eval/hybrid_value.py:build_seat_model` builds one
  `HybridValueNet(policy_net, value_net, recal=...)`.
- **The recalibration acts on the grafted head.** `HybridValueNet.forward` takes `wdl`
  from the VALUE net, then applies `wdl / T + [0, b, 0]` to it, at the root and at every
  leaf. The `--*-value-recal` help says the same: "applied to `--<seat>-value-from`'s net
  when both are given".
- **The formula matches the banked fit.** The fit's `apply` field is
  `softmax(logits / T + [0, b, 0])` on W, D, L logits, side to move.
- **Consumer-side readback.** At start the arena logs
  `[arena] candidate HYBRID built: value_net=loaded recal_T=… recal_bias=[0, b, 0]` from the
  built module. The launch checks that line against the CLI values below.
- **Tests at the PIN.** `tests/test_arena_hybrid_value.py` exercises value-from plus recal
  on one seat: provenance, the PGN suffix, and resume refusal.

**Banked fit:** `pilot-value-diagnosis/out/recal_S.json`, field `cli` =
`0.853159495428655,-0.12023510932922338`.
- Net: S, sha256 `66ceb271…`. Reference: the SL base.
- Rows: 39,237 hold-out rows from 416 hold-out games of the pilot collect.
- Those rows are TRAIN-split openings, disjoint from `eval.pgn` by the split.

**One untested combination: the same checkpoint in both seats.** Both seats are the base
net, and only the candidate is a hybrid. Read at the PIN:
- The arena refuses only identical PGN player names. The hybrid seat's name carries a
  `+V…+recal(…)` suffix, so the names differ.
- The pool's sha256 seat-swap check passes trivially when the same checkpoint sits in both
  seats. Its `check_seat_values` is what proves the hybrid sat on the arm seat in both runs.

That combination has not been run. **Launch precondition:** a 4-game A/B smoke of match 6b
in a scratch directory, outside the bank, must play and pool without error. If it does not,
match 6 is not played, and the failure is recorded here; no code is changed under this
record.

## Design

Every row not listed below is match 5's: the pin, the worktree, `run_match.sh`, the
`$ARENA` flags and the kill / void rules.

| | 6a: recal graft | 6b: raw graft (control) |
|---|---|---|
| Arm seat | SL base policy + S's value, recalibrated: `--candidate-value-from $S --candidate-value-recal 0.853159495428655,-0.12023510932922338` | SL base policy + S's value, raw: `--candidate-value-from $S` |
| Base seat | plain SL base | plain SL base |
| Run B | seats mirrored: `--reference-value-from $S` (plus `--reference-value-recal …` in 6a) | the same mirroring |
| Seed | **2026112161** (shared) | **2026112161** (shared) |
| Size, one look | **6,400 games**: 3,200 per run, 1,600 openings | **1,600 games**: 800 per run, 400 openings |
| Openings | the 1,600 that seed 2026112161 draws | its first 400, checked to be an exact prefix of 6a's draw |

**Seed checks.**
- 2026112161 appears in no pilot match, no reservation, nowhere in `docs/`, and is not
  match 5's seed.
- Overlap of 6a's 1,600 openings: 595 with the pilot's matches, 654 with match 5.
- Overlap of 6b's 400: 135 with the pilot, 154 with match 5.

**The raw graft is played as the control; match 3 does not suffice.** Match 3 played S as a
single net, so its −20 mixes S's policy change with its value change. The "shape" reading
needs the value damage alone, on the same base policy the recal graft uses.
- The predicted raw-graft result is ≈ S's −20 minus the policy's share, i.e. ≈ −28 if the
  graft's +7.7 transfers. That prediction is an estimate, not a rule input.
- 6b shares 6a's seed and its first 400 openings, so the recalibration's effect is a paired
  contrast on those openings.
- Each match's game RNG still differs after the opening draw, so the games themselves are
  not identical.

## Size and power

SEs scale from the gate's measured 6.74 Elo at 1,600 games. Hybrid seats measured the same
SE as plain seats in match 4 (3.86 at 4,800).

| Match | Games | SE | Power |
|---|---|---|---|
| 6a | 6,400 | 3.37 | P(LB95 > −10 \| true 0) = 0.84; P(UB95 < 0 \| true −10) = 0.84; P(UB95 < 0 \| true −28) ≈ 1.00 |
| 6b | 1,600 | 6.74 | P(UB95 < 0 \| true −28) = 0.99; P(UB95 < 0 \| true −15) = 0.61 |

The diagnosis bank suggested "the single 800-game recal-graft arena". At 800 games
(SE ≈ 9.5) the "shape" reading would have power ≈ 0.18, so it is not used.

## Reading rules (fixed in advance)

On each match's pooled 95% interval (`pool.json` → `result.pooled.elo_ci95`):
- **"the damage is shape after all"** iff 6a's LB95 > −10 AND 6b reads damage (UB95 < 0);
- **"content"** iff 6a's UB95 < 0;
- otherwise **"unread, insufficient evidence"**.

If 6b does not read damage, "shape" cannot be read. "Content" depends on 6a alone.

**Secondary, report-only.** The recalibration's recovery, 6a − 6b, with
SE ≈ √(3.37² + 6.74²) ≈ 7.5. That SE treats the two matches as independent. They share
400 openings, which most likely correlates them positively, so the SE is likely
conservative. No rule reads the secondary.

## What each outcome licenses

| Outcome | Licenses |
|---|---|
| shape | The value damage is recoverable by a 2-parameter post-fit. A recalibrated S-value cell becomes a candidate for the owner, under its own record. Nothing is adopted here |
| content | The diagnosis's mechanism stands. A value target built from the 64-sim search value harms ranking that recalibration cannot fix. Value-target work needs a different target or a held-out value before generation 2 |
| unread | Nothing. |

## Cost and commands

- **Cost.** Hybrid seats measured 0.27 GPU-h per 800 games. 6a ≈ 2.16 GPU-h; 6b ≈ 0.54
  GPU-h; match 6 ≈ 2.7 GPU-h.
- **Total GPU cost of this record:** see the end of match 7.
- **Control, confounds, recovery.** As for match 5: the same commit, 64 sims, CUDA graphs,
  `--compile off`. Every row's `git_sha` must equal `$PIN`. Nothing is trained and nothing
  live changes.

```bash
R=$HOME/chess-artifacts/operations/rl-run1-pilot
SL=$R/sl_base.pt; S=$R/arm_S/gen_0001/train/model.pt
RC=0.853159495428655,-0.12023510932922338        # pilot-value-diagnosis/out/recal_S.json "cli"
PY=$HOME/projects/chess/.venv/bin/python
PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-$(cut -c1-9 $R/PIN.txt)
# 6b first (cheap control), then 6a; both seed 2026112161
$R/run_match.sh s_value_raw 1 A 2026112161 800 --candidate $SL --reference $SL --candidate-value-from $S || exit 1
$R/run_match.sh s_value_raw 1 B 2026112161 800 --candidate $SL --reference $SL --reference-value-from $S || exit 1
$R/run_match.sh s_value_recal 1 A 2026112161 3200 --candidate $SL --reference $SL \
  --candidate-value-from $S --candidate-value-recal $RC || exit 1
$R/run_match.sh s_value_recal 1 B 2026112161 3200 --candidate $SL --reference $SL \
  --reference-value-from $S --reference-value-recal $RC || exit 1
for m in s_value_raw s_value_recal; do d=$R/arenas/$m
  cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $SL \
    --a-games $d/${m}_1_A.games.jsonl --b-games $d/${m}_1_B.games.jsonl \
    --a-result $d/results.jsonl --b-result $d/results.jsonl \
    --a-label ${m}_1_A --b-label ${m}_1_B --out $d/pool.json > $d/pool.log 2>&1
done
```

In each pool, the "arm" is the hybrid seat: run A's candidate and run B's reference. The pool
proves this through `check_seat_values` (the value-from sha256 and [T, b] must match across
the runs).


---

# Match 7 — Zq recalibrated vs Zq, head-to-head (a NEW registration; diagnostic, adopts nothing)

## Why, and what it is not

**The pilot skipped its match 2 by rule.** Match 2 was this same head-to-head, and it was
conditional on the gate reading damage. The gate read −10.0 [−23.2, +3.2], so match 2 was
skipped.

**Match 7 is a NEW registration, motivated by post-hoc offline evidence.** Addendum 1's
sibling-ordering test (above) found S's ranking broken, but not Zq's:
- Zq: −0.0007 [−0.0032, +0.0017] vs the base;
- P0's own head: −0.0020 [−0.0040, +0.0001].

So "ranking content broke" does not explain Zq's −10. For Zq the shape account is still
open: its (T, b) = (0.704, −0.227) removes 60% of its KL to the base.

**Match 7 therefore cannot be read as the pilot's match 2, and it changes no pilot
verdict.** The gate's "not reproduced, not adoptable" and the pilot's "stop for the owner"
stand whatever match 7 reads.

## Design: the launch record's match 2, as designed there

| Item | Value |
|---|---|
| Arm seat | Zq + recal: `--candidate Zq --candidate-value-recal $RC7`, one forward per seat, plain speed |
| Base seat | plain Zq |
| Run B | mirrored: `--reference-value-recal $RC7` |
| Both seats | Zq's file, `arm_Zq/gen_0001/train/model.pt`, sha256 `ad16d2f177036a3246d092856e99477a1e91dddec7f0229801688c2cf5980b0a` |
| Size | 1,600 games, one look: 800 per run, 400 openings |
| Seed | **2026112171** (fresh; see below) |
| Pin, flags, worktree, kill / void | match 5's |

**The fit: the banked one is the record's fit.** `pilot-value-diagnosis/out/recal_Zq.json`,
`cli` = `0.7036189636786957,-0.22711530923843357`. Compared with the launch record's Step 3
command, `fit_value_recal.py --net $Zq --reference $SL --collect-dir $R/collect/gen_0001/collect --device cuda`:

| | Banked fit | Launch record's Step 3 |
|---|---|---|
| Code | `0c795fb7a` (`run_fits.sh`) | `$PIN` (`0c795fb7a`) |
| Zq | sha256 `ad16d2f1…` | the same file |
| Reference | SL base, sha256 `a39f900c…` | the same file |
| Collect | collect manifest sha256 `61fc049f…` | the same collect |
| Rows | hold-out games only: 416 games, 39,237 rows | hold-out games only |
| Device | CUDA, bf16 logits | CUDA, bf16 logits |
| `--batch` | 256, given explicitly | the tool's default, also 256 |
| `holdout_frac` | **`--holdout-frac 0.05` passed as a flag** (`holdout_frac_source: "flag, unpinned"`) | read from `$R/collect/run_config.json` |

The only real difference is the last row. The record expected the run's pinned 0.05 in
`$R/collect/run_config.json`, but that file is absent from the collect run dir. The fit
therefore needed the flag, with the same value 0.05, the value the cells trained by. The
split and the rows are identical.

**Decision: use the banked fit, no refit.**
- A refit by the record's exact command would refuse for the missing pin, or need the same
  flag.
- A refit would also be a second fit of the same quantity after the gate and Addendum 1
  were seen.

## Seed: fresh, not the reserved 2026112102

- **Statistically the two are the same:** 2026112102 was reserved but never played, so it
  carries no information either way.
- **The fresh seed is cleaner for bookkeeping:** a game log or pool under the reserved
  seed would read as the pilot's registered match 2, which match 7 must not be mistaken
  for.

2026112171 appears nowhere in `docs/`. Its 400 openings overlap 135 with the pilot's
matches, 162 with match 5 and 158 with match 6.

## Reading rule (as the launch record had it)

- **"the loss is shape"** iff LB95 > 0, read on match 7's pooled interval, with the arm
  being the recalibrated seat;
- otherwise **"unread"**.

**Report-only: the recovered fraction.** This is match 7's estimate divided by −Elo(Zq)
= 10.0. The denominator is itself 10.0 [−3.2, +23.2] (the gate), so the fraction is poorly
determined: unbounded when the gate's true value is near 0. It is reported beside its
numerator and never read.

**What 1,600 games can and cannot establish.** At SE ≈ 6.74 Elo:
- "shape" reads with 80% power only if recalibration gains ≈ 19 Elo (2.80 × SE) over raw
  Zq. That is more than Zq's whole measured loss.
- If recalibration recovers exactly Zq's −10, the power is 0.32.

So the match can establish that recalibration helps Zq. It cannot establish that
recalibration helps by less than about 19 Elo, and it cannot establish "shape is not the
cause": an "unread" result leaves the shape account open.

**What it licenses:** at most, a later and separate registration of a recalibrated-Zq vs
base adoption match. Nothing is adopted here.

## Cost and commands

- **Cost.** Recal-only seats run at plain speed (launch record: 0.19 GPU-h per 800 games),
  so match 7 ≈ **0.38 GPU-h**.
- **Total GPU cost of this record:** match 5 ≈ 1.52 + match 7 ≈ 0.38 + match 6 ≈ 2.70 =
  **≈ 4.6 GPU-h**.

```bash
R=$HOME/chess-artifacts/operations/rl-run1-pilot
Zq=$R/arm_Zq/gen_0001/train/model.pt
RC7=0.7036189636786957,-0.22711530923843357      # pilot-value-diagnosis/out/recal_Zq.json "cli"
PY=$HOME/projects/chess/.venv/bin/python
PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-$(cut -c1-9 $R/PIN.txt)
$R/run_match.sh zq_recal_h2h 1 A 2026112171 800 --candidate $Zq --reference $Zq --candidate-value-recal $RC7 || exit 1
$R/run_match.sh zq_recal_h2h 1 B 2026112171 800 --candidate $Zq --reference $Zq --reference-value-recal $RC7 || exit 1
d=$R/arenas/zq_recal_h2h
cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $Zq \
  --a-games $d/zq_recal_h2h_1_A.games.jsonl --b-games $d/zq_recal_h2h_1_B.games.jsonl \
  --a-result $d/results.jsonl --b-result $d/results.jsonl \
  --a-label zq_recal_h2h_1_A --b-label zq_recal_h2h_1_B --out $d/pool.json > $d/pool.log 2>&1
```

As the launch record notes for match 2, both seats are Zq's file. The pool refuses unless
A's `candidate_value` equals B's `reference_value`, and the engine names carry the
`+recal(T,b)` suffix.
