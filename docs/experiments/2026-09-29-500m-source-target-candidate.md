# 500M source and target candidate after the direct E–D replay

The current **candidate**, not a production prescription, is to keep some
Stockfish-generated **positions**, train their ordinary policy and value targets
from BT4/Ceres, and test sparse Stockfish **policy corrections** as a separate
intervention. Position source, policy target and value target are distinct
factors. The fractions, correction selector and exact target route are not yet
selected by a matched-strength result.

The corrected strict six-man factorial's average policy-recipe contrast was
+6.79 Elo [−16.30,+29.25]; its average value-recipe contrast was +25.83
[+4.07,+47.11]. The value contrast adds Ceres while reducing both SF and BT4,
so it is not an isolated SF effect. The direct original E-versus-D replay on
the same 576-opening bank used for the two Selected-E matches found **E−D
−12.37 Elo [−27.18,+2.39]** over 1,152 color-swapped games. Its registered
zero-Elo decision is unresolved; E and D have identical 50/50 BT4/Ceres policy
targets, while E removes SF from value and reweights BT4/Ceres. The indirect
three-edge discrepancy was +0.52 score percentage points [−3.17,+4.30],
secondary and also unresolved. See [the direct record](https://github.com/jjoshua2/DeepFin/pull/929)
and [the factorial readout](2026-09-27-factorial-strict-rule50-readout.md).
These are one-seed, fixed-bank comparisons. The direct interval's lower
endpoint happens to exceed −30 Elo, but the preregistered gate was against
zero; it does not establish a population 30-Elo guarantee or an intrinsic
benefit or harm from SF value labels.

The cheaper Selected-E one-teacher route scored −3.02 Elo
[−17.74,+11.69] against E and passed its predeclared −20-Elo gate **to
measure label cost**. Against D it scored −19.02 [−33.43,−4.68], inconclusive
under a separate −30-Elo gate. Neither match measured full end-to-end
annotation saving or a 500M transfer. A same-row four-arm full-wall comparison
is still needed; the older S→D two-arm small-receipt complete-arm screen showed an 11.96%
wall difference on 12,288 rows, below its >15% promotion threshold;
independent full-byte target reread is still pending. See the [Selected-E versus E](2026-09-27-selected-e-vs-e-arena.md)
and [Selected-E versus D](2026-09-28-selected-e-vs-d-direct-strict-readout.md)
records.

## Capacity boundary

Assume, optimistically, that all 58,090,688 legacy E rows qualify, leaving
441,909,312 new rows. For illustrative **equal source thirds** this means
147,303,104 SF-origin, BT4-origin and Ceres-origin rows each. At the measured
selected-before-global-dedup source/readback rates of 172.079 BT4 rows/s and
127.197 Ceres rows/s, BT4/Ceres source-plus-readback projects to 9.908 and
13.404 ideal serial GPU days. Half BT4/half Ceres selected targets
would project to 10.954 more days at 800.19/329.64 rows/s. At the cache-affected
run12 external-ZIP rate of 2,081.983 rows/s, one 500M-row epoch would
project to 2.780 ideal days. The sum is **37.045 ideal GPU-stage days**
without generating-owner target reuse. Exact fair-route reuse would reduce
that to 33.393 days; source-matched reuse to 29.742 days, but neither route
has byte-qualified reuse and the latter changes the conditional
source-by-teacher distribution. A 30-day month at 95% GPU duty supplies 28.5
ideal days, before dedup loss, retries, uncached-read penalties or sparse SF search. These
are capacity sensitivity calculations, not an admitted-unique forecast or a
promise to finish in one month. They exclude SF-origin CPU generation/search
and strict replay, assuming those CPU stages overlap GPU work completely.
Physical dedup, pack qualification and any uncached-read penalty
beyond the cache-affected trainer measurement are also excluded. The trainer's projected 2.780 days is smaller than the projected
neural source and label terms; its rate is cache-affected and no cold 500M
epoch was measured. See the [external trainer screen](2026-09-29-zip128-external-trainer-screen.md)
and the [capacity arithmetic](evidence/2026-09-29-500m-candidate-capacity.json).

A bounded SF-origin source pilot has filled its 256 opening roots × 8 selected
games quota: 2,048 selected games and 390,764 selected rows before global
dedup. This is a source-roster result, not a target, packed-corpus or trainer
result; [the source update](https://github.com/jjoshua2/DeepFin/pull/920#issuecomment-5891439382)
records its strict replay. The first frozen three-source slice is 128 SF games
plus 512 neural games, 83,991 gross rows. Its metadata/source packet passed,
but exact cross-source history dedup, selected policy/value target bytes,
pack readback and real trainer consumption remain **zero-credit gates**.

## Next decisions

1. Finish a bounded physical admission of the frozen three-source slice.
   Require complete game history, rule50-aware six-man Syzygy WDL+DTZ
   source-terminal replay and physical-call reconciliation, then cross-source
   dedup, selected target-byte readback, pack and the first external-drive
   trainer read. Retain the same strict rule for later arenas. Reserve one GPU job and one heavy-I/O job at a time. Do not extrapolate pre-dedup counts.
2. Compare selected versus dual BT4/Ceres labeling on exactly the same rows
   with fresh sessions, matched charged warmups, four balanced arms, separate
   full-byte output/readback and a predeclared cost gate. Generating-owner
   reuse additionally needs input/history, legal-map, model/head,
   calibration, precision and target-byte equality to fresh labeling.
3. Test **position source** while holding the target recipe fixed. The
   proposed first contrast changes 25% of a matched 1M-row bank to SF-origin
   positions, followed by a conditional 10M-row confirmation if the bounded
   screen is complete and nonrejecting. The 1M screen can reject gross harm,
   not prove noninferiority; the 10M lower-bound gate addresses that question.
   The current neural-source bank and physical admission are too small to
   launch either strength arm.
4. For sparse SF corrections, first finish the source/game-disjoint cheap
   selector holdout. The seven-piece late-endgame candidate searches about
   9.85% of old G10 rows and captures about 28.42% of retrospective
   Tactical300 moved-target mass. Complete held-out d9 >300-cp deficit
   capture by material (even in the old run06 bank), SF wall and student
   strength are unmeasured. Then compare a
   frozen **policy-only** correction (S) with unchanged neural targets (B) and an equal-search-count
   source/game-stratified random roster (R), holding source, value, schedule
   and initial weights fixed. Predeclare a new strict six-man opening bank and
   full SF wall charge. Do not use an SF answer to select whether to ask SF.
   A value-only SF correction would need its own separate matched test.

This order retains the possibility that a small, targeted SF contribution
helps without assuming that the expensive shallow-to-deep ladder should label
hundreds of millions of positions. It also keeps a non-SF-only source mix as
a testable hypothesis rather than relying on an untested claim that generating
engine never matters.
