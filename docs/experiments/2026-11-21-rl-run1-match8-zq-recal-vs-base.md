# 2026-11-21: RL run 1, match 8 — recalibrated Zq vs the SL base (pre-registration)

**Status: registered; launched after the merge and completed 2026-11-21 (22:41).** Result: +9.6 [+3.3, +16.0], **"adoptable" and "improves on the base"** ([follow-up results](2026-11-21-rl-run1-followup-results.md#match-8-recalibrated-zq-vs-the-sl-base-seed-2026112191)). Launch required both:
- this record is reviewed and merged;
- the current follow-up chain (6b, then 6a) has released the GPU.

**Order: match 8 runs first**, as soon as the GPU is free, and before the value-anchor
screen (PR #1475, `2026-11-21-rl-run1-value-anchor-screen.md`). No training, no live change.

## Why

Match 7 is registered in [the follow-up record](2026-11-21-rl-run1-match5-p0-single-net.md).
It played Zq + recal against Zq: 6,400 games, seed 2026112171, PIN `0c795fb7a`. Its result
(`rl-run1-followup/arenas/zq_recal_h2h/pool.json`):
- **+21.4 Elo [+15.1, +27.6]**;
- measured clustered SE 3.18 Elo (the value the CI implies), 1,600 clusters;
- as candidate +25.7, as reference +17.1;
- 0 unresolved games; the smoke and both seat readbacks passed.

By its registered rule that reads **"the loss is shape"** (LB95 > 0). It licenses, at most,
"a recalibrated-Zq vs base adoption match as a later, separate registration". This is that
registration.

**Match 8 is not the pilot's match 2, and it changes no pilot verdict.** It adopts nothing.

## Design

- **Seats.**
  - Run A: `--candidate $Zq --candidate-value-recal $RC7 --reference $SL`.
  - Run B, the mirror: `--candidate $SL --reference $Zq --reference-value-recal $RC7`.
  - The arm seat is Zq's own file with the recalibration applied to its WDL output: a single
    forward, with no value-from net.
- **Recalibration:** (T, b) = **(0.7036189636786957, −0.22711530923843357)**, the banked
  `pilot-value-diagnosis/out/recal_Zq.json` field `cli`. It is the same fit match 7 used.
  - **When it was made.** `recal_Zq.json` was written at 13:09:31. That is AFTER the pilot gate
    (finished 11:10:45) and the graft looks (finished by 13:01:32), and BEFORE match 7 (started
    15:52:52).
  - **No refit.** The fit's target is the base's WDL on the pilot's hold-out rows (416 games),
    not any game outcome, and it precedes match 7. A refit now would be a second fit made after
    match 7 was seen. Match 7 tested this exact (T, b), so match 8 must use it too.
- **Games:** 6,400 (3,200 per run, 1,600 openings). **Seed 2026112191**, which appears in no
  pilot, follow-up or anchor-screen match nor anywhere under `docs/`.
- **Pin and launcher.**
  - PIN `0c795fb7a`; every result row's `git_sha` must equal it.
  - The follow-up launcher `rl-run1-followup/run_followup_match.sh`, sha256 `7c8fd065…`, with
    output root `rl-run1-followup/`.
  - The kill/void rules of the follow-up record's "Common" section apply unchanged.
- **Seat proof.** A 4-game-per-run smoke goes into `rl-run1-followup/smoke/` first, then
  `check_seat.py` (sha256 `8f4ce9eb…`) and the smoke pool. The real run starts only if all of
  them pass, and `check_seat.py` runs again on the real logs.
  - Expected readback: run A role `candidate`, run B role `reference`; value-from `-` (the log
    reads `value_net=own`); recal = the `cli` above.
- **Opening overlap**, computed with the arena's own draw (`load_paired_openings`,
  `default_rng(seed)`). The method reproduces match 7 vs match 5 = 640, as the follow-up
  record states.

  | Compared with | Shared openings |
  |---|---|
  | match 1, the gate (2026112101, 400 openings) | 167 |
  | match 7 (2026112171) | 633 |
  | match 5 | 640 |

  These overlaps are what independent seeds give. The opening is a covariate, not a bias.

## Rules

Read on `pool.json` → `result.pooled.elo_ci95`, with the arm being the recalibrated seat, on
the measured clustered SE (reported).

| Reading | Condition |
|---|---|
| damage | UB95 < 0 |
| adoptable | LB95 > −10 |
| unread | neither |
| **improves on the base** (pre-declared, stronger; reported alongside adoptable) | LB95 > 0 |

Match 5's tie-break is kept as written: below SE ≈ 2.55, damage and adoptable can both hold,
and both are reported.

**Power,** at the gate-scaled SE 3.37:
- **For a true +11:** P(improves on the base) ≈ 0.90, and P(adoptable) ≈ 1.00.
- **For a true 0:** P(adoptable) ≈ 0.84.

The +11 is the naive sum of Zq's gate (−10) and match 7 (+21). **Elo is not transitive across
different opponent pairs**, so +11 is a planning figure, not a prediction.

## What each outcome licenses

| Reading | Licenses |
|---|---|
| adoptable (with or without "improves") | NOMINATES **"Zq + per-generation recalibration"** as the value recipe for a 2-generation run with the policy, under its own record: the amendment's adoption step 3, where the recalibration is refitted and folded into the WDL head each generation. It adopts nothing here |
| damage | Recalibration does not rescue Zq against the base; the shape account explains match 7 but not the gate |
| unread | Nothing |

## Interaction with the value-anchor screen (#1475)

If match 8 reads adoptable, the anchor screen's Zq cells should ALSO be read after
recalibration. A follow-up registration may then be needed.
- #1475's design is not changed for this. Its Stage A already reports each cell's (T, b) fit.
- #1475 carries one line noting that match 7 exists.

## Cost and order

- About **1.52 GPU-h**: recal-only seats measured at 0.19 GPU-h per 800 games.
- Queue: after 6b and 6a, before the value-anchor screen.

## Commands

```bash
F=$HOME/chess-artifacts/operations/rl-run1-followup; P=$HOME/chess-artifacts/operations/rl-run1-pilot
SL=$P/sl_base.pt; Zq=$P/arm_Zq/gen_0001/train/model.pt; RC7=0.7036189636786957,-0.22711530923843357
PY=$HOME/projects/chess/.venv/bin/python; PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-$(cut -c1-9 $P/PIN.txt)
printf '%s  %s\n' 7c8fd065865f3d5abcd8e9eb6f7fc1d48e882dfcfb88bbaa92d8696ce4a9510d $F/run_followup_match.sh \
  8f4ce9ebcf232073e7580d6929c2555c53ea17611ca640d2a529a1998821db92 $F/check_seat.py | sha256sum -c --quiet || exit 93
pool() { d=$1/arenas/$2; (cd $PINW && PYTHONPATH=. $PY scripts/arena_seat_balanced_pool.py --arm $3 \
  --a-games $d/${2}_1_A.games.jsonl --b-games $d/${2}_1_B.games.jsonl --a-result $d/results.jsonl \
  --b-result $d/results.jsonl --a-label ${2}_1_A --b-label ${2}_1_B --out $d/pool.json > $d/pool.log 2>&1); }
for O in $F/smoke $F; do n=$([ $O = $F ] && echo 3200 || echo 4)
  $F/run_followup_match.sh $O zq_recal_vs_base 1 A 2026112191 $n --candidate $Zq --candidate-value-recal $RC7 --reference $SL || exit 1
  $PY $F/check_seat.py $O/arenas/zq_recal_vs_base zq_recal_vs_base_1_A candidate - $RC7 || exit 1
  $F/run_followup_match.sh $O zq_recal_vs_base 1 B 2026112191 $n --candidate $SL --reference $Zq --reference-value-recal $RC7 || exit 1
  $PY $F/check_seat.py $O/arenas/zq_recal_vs_base zq_recal_vs_base_1_B reference - $RC7 || exit 1
  pool $O zq_recal_vs_base $Zq || exit 1
done
```

The first pass is the smoke. The second pass is the real match, and it runs only if every
smoke step exits 0.
