# Effective branching factor: ours vs BT4 on the frozen audit set

**Executed 2026-08-14. Report written 2026-08-14 by a second agent from the banked
artifacts, with every headline number RECOMPUTED from `rows.jsonl` rather than
transcribed.** The measuring agent could not write this file; the raw dump, both
aggregate texts, and both scripts are banked beside it.

## Verdict in one line

**We are SHARP AND WRONG.** Our policy's effective branching factor at the root is
**3** where BT4's is **11**, and we buy that narrowness by being wrong more often:
top-1 **0.4273 vs 0.5683**, top-1 deep-SF regret **47.40 vs 20.07 cp**. Sharpness is
not, by itself, an asset -- a beam of 3 that excludes the best move is worth less
than a beam of 11 that contains it.

---

## 1. Provenance -- pinned by CONTENT, not by name

| | value |
|---|---|
| positions | `data/audit_set_v1.jsonl`, **all 4000 rows** (frozen audit set v1) |
| `ours` | `scratchpad/tier13/banked/arm_A_iter100/checkpoint_000099/trainer.pt`, **step 91999**, weights md5 **`4cfe18420774f039795d8457261c61cb`** |
| `bt4` | `data/lc0/onnx/BT4-it332-vanilla-winner.onnx`, md5 **`a4fbd2bfda0375d931f04ea6dcb57920`** |
| `unif` | 1/`n_legal` over the IDENTICAL legal list -- negative control |
| encoding (ours) | `lc0_root_legacy_meta` / `v2_threats`, policy `lc0_1858`, relations off (read off the checkpoint at run time, `run.log`) |
| encoding (BT4) | `lc0_root` planes, history-fill = **repeat**, gathered in canonical Leela 1858 space |
| batch size | 128 (both nets) |

⚑ **The checkpoint identity was NOT recorded in any banked artifact** -- `branching.py`
takes it as a defaulted argparse flag and `run.log` prints only the encoding. It was
recovered here by re-scoring the first 128 positions with each candidate net and
comparing against `rows.jsonl`:

- `arm_A_iter100` at batch 128: **max |Δp_top1| = 0.000e+00, argmax 128/128, N95 128/128 -- bit-exact.**
- `ck_2026-08-12_5ce02_iter218` (the net #206 used): max |Δp_top1| **0.496**, argmax 56/128 -- a **different net**.

So these numbers are on **Tier-13 arm A @ iter 100**, the Tier-13 control arm, NOT on
`ckpt218`. That matters for one cross-reference: the #206 prereg's banked `fen_only`
argmax baseline for `ckpt218` is **47.34 cp** and this run reads **47.4047 cp** on
arm A. The agreement to 0.06 cp is a coincidence of two nearby nets, **not** a
reproduction -- do not cite one as confirming the other. (Tier-13 was a three-arm NULL,
so arm A is a legitimate stand-in for the current net; it is just not the same file.)

**Adapter sanity, from `run.log` (all three read zero):** legal moves with no Leela-1858
slot **0**; legal moves hitting our -1e9 pad slot **0**; positions where SF's best is
absent from our legal list **0**. `n_legal` equals python-chess's own count on
**4000/4000**.

## 2. Negative control -- PASSES EXACTLY

The uniform arm is not decorative; it is the proof that `dist_stats` computes what its
names claim.

| quantity | measured | analytic | agreement |
|---|---|---|---|
| mean N95 | 27.1148 | mean ceil(0.95*n_legal) = 27.1148 | **element-wise identical** |
| mean perplexity | 28.043000 | mean n_legal = 28.043000 | max abs err **1.28e-13** |
| mean top-1 acc | 0.0703 | mean 1/n_legal = 0.0619 | differs by construction |

Note the third row is *not* a failure: mean top-1 **accuracy** under a uniform tie-break
is not mean 1/n_legal, because `argsort` breaks the all-equal tie deterministically on
move order. The two quantities a broken width metric would corrupt -- N95 and
perplexity -- reproduce their closed forms exactly.

## 3. Headline numbers -- all recomputed, all matched

Recomputed with an independent script and a **different bootstrap seed** (11223344 vs
`analyze.py`'s 20260814); point estimates are exact, CI endpoints move in the third or
fourth decimal, which is resampling noise.

### Width

| statistic | ours | BT4 | paired diff [95% CI] |
|---|---|---|---|
| **median N95** | **3.0** | **11.0** | **-8.00 [-8.00, -7.00]** |
| mean N95 | 3.2760 | 11.3407 | -8.0648 [-8.2647, -7.8665] |
| median N90 | 2.0 | 5.0 | -3.00 [-4.00, -3.00] |
| mean N90 | 2.6035 | 6.9165 | -4.3130 |
| median perplexity | 2.0637 | 5.2643 | -3.2006 [-3.3374, -3.0838] |
| mean perplexity | 2.6201 | 6.7010 | -4.0810 |

Paired bootstrap, 10,000 resamples, over the SAME 4000 positions.

### Placement

| statistic | ours | BT4 | paired diff [95% CI] |
|---|---|---|---|
| **top-1 acc** | **0.4273** | **0.5683** | **-0.1410 [-0.1578, -0.1238]** |
| top-3 acc | 0.7127 | 0.8535 | -0.1408 |
| top-5 acc | 0.8337 | 0.9323 | -- |
| top-10 acc | 0.9423 | 0.9848 | -- |
| top-16 acc | 0.9830 | 0.9980 | -0.0150 [-0.0192, -0.0110] |
| median rank(SF best) | 2.0 | 1.0 | +1.00 [+1.00, +1.00] |
| **mean top-1 regret (cp)** | **47.4047** | **20.0745** | **+27.3302 [+23.5130, +31.2664]** |
| mean expected regret (cp) | 52.5282 | 50.9255 | (see the `E|listed` warning below) |

Uniform control for scale: top-1 0.0703, mean top-1 regret **175.80 cp**.

⚑ The **expected**-regret row reads near-parity (52.53 vs 50.93) and **must not be
quoted**: `move_regrets` floors every move deep-SF's MultiPV did not list at the worst
listed move's regret, which systematically under-penalises a diffuse net. That is the
ruler artifact already established in the 2026-08-13 "our mass is competitive is DEAD"
entry, and this run reproduces it. Quote `top1_regret`, or `E|listed`. Not `E`.

### The state decomposition -- the actual finding

Sharp = N95 <= 3. Right = argmax is SF's best.

| net | sharp | sharp & right | **sharp & WRONG** | flat & right | flat & wrong |
|---|---|---|---|---|---|
| **ours** | 0.6630 | 0.3405 | **0.3225** | 0.0867 | 0.2502 |
| BT4 | 0.1108 | 0.0910 | **0.0198** | 0.4773 | 0.4120 |

**16.3x.** And our sharp-and-wrong states are worse-shaped than BT4's: when we are
sharp and wrong, SF's best sits at median rank 3 (p90 **9**) with mean p(SF best)
**0.0846**, and is **outside our own 95% beam 57.8%** of the time. BT4's sharp-and-wrong
states put SF's best at median rank 2 (p90 **3**), mean p 0.2677, outside its beam only
**10.1%** of the time. BT4 being wrong is a near-miss; ours is a miss.

### Mass -- and the one place we are genuinely level

| statistic | ours | BT4 | paired diff [95% CI] |
|---|---|---|---|
| mean p(own top move) | **0.7209** | 0.4956 | +0.2253 [+0.2175, +0.2330] |
| **mean p(SF's BEST move)** | 0.3905 | 0.3995 | **-0.0090 [-0.0190, +0.0011]** |

We put **45% more mass on our own top move** and *the same* mass on the right move. The
extra confidence buys nothing; it is spent on the move we picked, not on the move that
was correct. (The mass-on-SF's-best CI straddles 0 -- that is a genuine null, and it is
the one honest "parity" in this table.)

### SF's best inside the net's own 95% beam

| net | P(rank(SF best) <= own N95) |
|---|---|
| ours (T=1.0) | **0.7285** |
| ours at `gumbel_policy_temp` 1.5 | 0.8490 |
| BT4 (T=1.0) | **0.9945** |
| BT4 at T=1.5 | 0.9998 |
| uniform | 0.9748 |

Our beam misses the best move **27% of the time** (15% at the temperature search
actually applies). BT4's misses it **0.55%** of the time. ⚑ Note the uniform control
scores **0.9748** here -- a metric where a *random* policy beats us by 25 points is
measuring beam width more than beam quality, which is exactly the point of section 4.

### Implied search depth -- EXTRAPOLATION, NOT MEASUREMENT

d = log(N)/log(b) at N = 1e6 nodes.

| b from | ours b | ours d | BT4 b | BT4 d | **ratio** |
|---|---|---|---|---|---|
| **median N95** | 3.000 | **12.58** | 11.000 | **5.76** | **2.183** [2.096, 2.183] |
| mean N95 | 3.276 | 11.64 | 11.341 | 5.69 | 2.046 |
| median perplexity | 2.064 | 19.07 | 5.264 | 8.32 | 2.293 [2.214, 2.368] |
| mean perplexity | 2.620 | 14.34 | 6.701 | 7.26 | 1.975 |
| geo-mean perplexity | 2.185 | 17.68 | 5.176 | 8.40 | 2.103 |

**At the temperature production search actually consumes** (`gumbel_policy_temp: 1.5`,
applied to our logits before the root rule) our median N95 rises 3 -> 4 and d falls
12.58 -> **9.97**, giving **ratio 1.730 [1.661, 1.730]** against BT4's untempered prior
(lc0's own default policy softmax temperature is 1.0, so `bt4` is BT4's production
prior). The symmetric what-if -- both at T=1.5 -- puts BT4's median N95 at 24 and returns
the ratio to 2.29; that arm is reported for completeness and is not what either engine
runs.

### By criticality bucket (best-vs-2nd cp gap)

top-1 acc / median N95 / median perplexity:

| bucket | n | ours | BT4 |
|---|---|---|---|
| quiet (<20) | 2135 | 0.329 / 3 / 2.38 | 0.417 / 11 / 6.38 |
| soft (20-50) | 774 | 0.429 / 3 / 2.02 | 0.633 / 10 / 4.97 |
| sharp (50-100) | 446 | 0.545 / 2 / 1.73 | 0.715 / 10 / 3.72 |
| decisive (>=100) | 645 | 0.670 / 2 / 1.30 | 0.890 / 7 / 2.31 |

We narrow *fastest* exactly where being wrong is most expensive: in the decisive bucket
our median beam is **2** and our top-1 is 22 points behind BT4's.

### Castling sanity (the known adapter defect class)

256 positions have a legal castle; SF's best IS the castle in 40. On that subset top-1
is ours **0.7000** vs BT4 **0.7250** -- no collapse on either side, so the castling
remap is not driving the headline. (The audit set carries castling *rights* on 13.9% of
rows.)

---

## 4. ⚑⚑ WHAT THIS RETIRES: the inference behind "97.9% top-16", NOT the number

The standing claim on record is: *"P(deep-SF's best is among our 16 root candidates) =
0.979, vs BT4 0.974 => support is NOT the differentiator; our search already holds the
right move 98% of the time."* (`docs/experiment_ledger.md`, 2026-08-13 "THE ARGMAX GAP
DIAGNOSED".)

**The number is TRUE and correctly computed.** This run reproduces its deterministic
analogue: our top-16 recall is **0.9830**, and restricting to the 3207 positions with
**more than 16 legal moves** -- where the metric can actually fail -- it is **0.9788**,
i.e. 97.9%. The restriction has teeth: the uniform control falls **0.6362 -> 0.5463**
under it, so the 3207-row subset is a real test and our number survives it.

**The INFERENCE is retired.** At k = 16 the metric is **saturated for both nets**:

| k | ours | BT4 | **spread** |
|---|---|---|---|
| **1** | 0.4273 | 0.5683 | **14.10 pp** |
| 3 | 0.7127 | 0.8535 | 14.08 pp |
| 5 | 0.8337 | 0.9323 | 9.86 pp |
| 10 | 0.9423 | 0.9848 | 4.25 pp |
| **16** | 0.9830 | 0.9980 | **1.50 pp** |

Top-16 recall separates two nets that differ by 14 points at k=1 by **1.5 points**. It
has almost no discriminative power, and a metric that cannot distinguish a healthy
policy from a sick one **cannot be evidence that ours is healthy**. "SF's best is in
our top 16" and "our policy is fine" are different propositions, and only the first one
was measured.

⚑ **A direction reversal worth stating explicitly, and it is an instrument difference,
not an error.** The banked 0.979-vs-0.974 has us marginally AHEAD of BT4; the
deterministic top-16 measured here has us **behind** (0.9830 vs 0.9980). Both are
right. The banked figure is a Monte-Carlo over the production root rule -- Gumbel
top-k *sampling* of 16 candidates from the tempered prior -- and stochastic sampling
penalises a diffuse prior, so BT4 loses ground there that it does not lose under a
deterministic rank cut. Two different questions ("does the sampler draw it" vs "is it
in the top 16") that a shared 16 makes look like one. This is
[[same_name_different_population]] again. Neither ordering supports "support is not the
differentiator", because at 1.5 pp and 0.5 pp respectively **neither has the resolution
to support anything**.

**What replaces it.** The discriminative statistics on this axis are `top-1`, `median
N95`, and `P(rank <= own N95)`. Cite those.

---

## 5. Caveats -- load-bearing, not boilerplate

1. **`d = log N / log b` assumes uniform branching at EVERY ply, and b was measured at
   the ROOT only.** The depth table is an **extrapolation, not a measurement**. Nothing
   in this run touches a non-root node. A real branching factor would come from the
   search tree, and the entire implied-depth section collapses to "our root prior is
   3.5x narrower" if that assumption fails -- which it certainly does to some degree.

2. **`gumbel_topk: 16` caps the root candidate set independently of the prior.** For us
   the cap is essentially non-binding: **P(N95 <= 16) = 0.9975** (0.9862 at T=1.5). For
   BT4 it would bind **17.8%** of the time (P(N95 <= 16) = 0.8223), and 71.6% at T=1.5.
   So a straight "BT4 searches 11 wide" reading over-credits BT4 relative to what our
   root rule would let it do -- the comparison is of PRIORS, not of realised root sets.

3. **Depth into a subtree that excludes the best move is not worth the same as depth
   into one that contains it.** A 2.18x depth ratio purchased by a beam that drops SF's
   best 27% of the time is not 2.18x more search value, and this run measures no play
   strength whatsoever. Regret cp is not Elo.

4. **BT4 history-fill was `repeat`.** Both known side effects **disadvantage BT4**:
   (a) the plain `lc0_root` encoding carries no en-passant plane -- measured here at
   **5 of 4000 rows = 0.125%** with an EP square set (⚑ a summary of this work circulated
   as "~1%"; the measured figure is 0.125%, an order of magnitude smaller, and it moves
   in the safe direction); and (b) eight identical history positions look like a 2-fold
   repetition to a net that reads history. Both handicap BT4, so **neither can
   manufacture the direction of this result** -- if anything BT4's true margin is larger.
   ⚑ Repeat-fill is the FAIR convention, not the TRUE one; the audit set structurally
   cannot carry real history [[audit_set_has_no_history_or_castling]], and our net is
   out-of-distribution under it too.

5. **Ties.** **400 of 4000 rows (10.0%)** have >=2 moves at the best cp. Recomputing
   top-1 tie-tolerantly -- argmax counts as correct if its cp equals the best listed cp --
   gives ours **0.4592**, BT4 **0.6065**, paired **-0.1473 [-0.1640, -0.1300]**
   (uniform control 0.0930). **Conclusion unchanged; the gap widens slightly.**
   ⚑ Method note for anyone re-deriving this: the obvious shortcut `top1_regret <= 0`
   is WRONG. On the **91 positions** where every listed move scores the same cp, the
   unlisted-move floor is itself 0, so unlisted moves get zero regret and the shortcut
   counts them as correct. It reads 0.4622 / 0.6090 -- off by ~0.3 pp. Use `move_cp`
   directly.

6. **Not measured at all here:** play strength, Elo, the search's *chosen* move (this is
   the raw prior only -- the #206 decider is the search-side instrument), any non-root
   node, real game history, and the ONNX-vs-torch numeric path.

---

## Reproduce

```bash
PYTHONPATH=. python3 scratchpad/branching_AGENT_D/branching.py \
  --audit-set data/audit_set_v1.jsonl \
  --checkpoint scratchpad/tier13/banked/arm_A_iter100/checkpoint_000099/trainer.pt \
  --onnx data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
  --device cuda --batch-size 128 \
  --out scratchpad/branching_AGENT_D/rows.jsonl

PYTHONPATH=. python3 scratchpad/branching_AGENT_D/analyze.py \
  scratchpad/branching_AGENT_D/rows.jsonl
```

⚑ **Batch size 128 is load-bearing for bit-exactness** -- at batch 64 the same
checkpoint reproduces `p_top1` only to ~3.5e-2 and disagrees on 2 of 64 argmaxes. That
is ordinary GPU batch-shape nondeterminism, but it means "I reran it and got slightly
different numbers" is expected, not a defect, unless the batch shape matches.

Banked beside this file: `rows.jsonl` (4000 rows, per-position, all five distributions
including the two T=1.5 arms), `analysis.txt` (T=1.0 aggregate), `analysis_temp.txt`
(the search-consumed T=1.5 aggregate and the root-cap table), `branching.py`,
`analyze.py`, `run.log`, `rows_smoke.jsonl`.
