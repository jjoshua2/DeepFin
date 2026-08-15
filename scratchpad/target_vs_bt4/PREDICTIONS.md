# PRE-REGISTERED PREDICTIONS — target argmax vs SF best, judged by BT4

Written **before** any BT4 forward pass was run (2026-08-15).
Namespaced dir: `scratchpad/target_vs_bt4/`. All files here are prefixed `tb4_`.

## The question

The 2026-08-15 TARGET AUDIT (`666d20561`) measured, on live shards:

    target argmax == shard-SF best move   0.4193 [0.4075, 0.4313]
    target top-1 mass                     0.734
    cp regret of target argmax (LISTED)   21.5 cp mean, frac_eq_0 0.42

and read it as "the target is sharp and WRONG". The ledger itself flags the step
as INFERRED: we deliberately train to exploit SF weaknesses, so disagreement
with SF is not automatically error.

- **H_wrong** — the target's argmax is genuinely a worse move. An independent,
  SF-agnostic strong evaluator also prefers SF's pick.
- **H_exploit** — the target's argmax is a deliberate anti-SF choice. The
  independent evaluator finds it comparable to or better than SF's pick.

## Ruler

BT4 (`data/lc0/onnx/BT4-it332-vanilla-winner.onnx`, 191.3M, lc0 selfplay-trained,
never sees an SF label). CPU-only ONNX via the venv's CPU-only onnxruntime
(`providers=["CPUExecutionProvider"]` — structurally cannot touch the GPU).

Encode path is the banked one from `scripts/foreign_net_audit.py`:
`encode_position(board, input_history_encoding="lc0_root")` +
`fill_lc0_history_repeat`, 1858 policy gathered at the legal moves via
`chess_anti_engine.moves.leela_index.leela_index_for_move` (the board-aware
remap; the castling defect fix is `c49b89937`, PR #376, **merged and on the live
branch HEAD** — verified by `git merge-base --is-ancestor`).

Two BT4 readings per candidate move `m` from parent position `P`:

1. **1-ply value (PRIMARY).** `Q(m) = L_child - W_child` where the child WDL is
   BT4's value head at `P.push(m)` (child is opponent-to-move, so negate).
   Same construction for every candidate, so the repeat-history fill is a
   SHARED treatment of a PAIRED comparison.
2. **Root policy (SECONDARY).** BT4's softmax prior over `P`'s legal moves.

## Sample

- Era E only: `runs/pbt2_small/replay/train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53`
  (live era, Aug 14-15, sims 100 / topk 16 / c_scale 0.1) — the same era the
  0.4193 was measured on.
- Newest 16 `replay_shards/*.zarr` (same `N_SHARDS` as `tgt_probe2.py`).
- Rows with `has_policy & has_sf_p0_regret`.
- Uniform random sample, seed 20260815, **n_target = 2000 rows**.
- `sf_best` = argmin of `sf_p0_regret` over the legal set (same as `tgt_probe2`).
- `covered` / "SF-listed" = `legal & (sf_p0_regret != fill)`, where `fill` is the
  value `sf_p0_regret` takes on an ILLEGAL slot of that row — the fabricated
  constant. Identical identification to `tgt_probe2.py`. Every cp statistic is
  quoted `|listed` only.

## Exact-count predictions

- Rows read from 16 shards = **16 x 2000 = 32000** exactly.
- Rows surviving `has_policy & has_sf_p0_regret`: **6000-12000** (era-E
  `n_sf/n_rows` was ~0.21 at N_SHARDS=16 in the prior audit; 16x2000 x 0.21 ~ 6.7k).
  If fewer than 2000 survive, n = all of them and I say so.
- Sampled rows n = **2000** exactly (or the survivor count if smaller).
- Rows where `target argmax == sf_best`: **0.40-0.44** of n (replicates 0.4193).
  ⚑ This is a REPLICATION check on my row selection, not a new finding. A value
  outside that band means my population is not the audited population and every
  number below is suspect.
- Rows where the target argmax is NOT SF-listed: **0.12-0.18** of n
  (audit: "not in SF's top-6 on 14.9% of rows").

## Alignment control (A) — can I even reconstruct the position?

Shards store **no FEN**. The board is decoded from `x[:112]`
(`input_history_encoding: lc0_root_legacy_meta`, from the shard attrs):
pieces from planes 0..11 in the side-to-move frame, castling from planes
104..107 (`us-Q, us-K, them-Q, them-K` per `_write_metadata_planes_root`), side
flag plane 108, rule50 from plane 109, EP file from plane 110.

> **AMENDMENT (written pre-run, before any BT4 forward pass, after reading
> `encoding/lc0.py` more carefully).** An earlier draft of this line asserted
> that plane 109 carries a RAW halfmove clock in the root layout and that
> `scripts/diagnose_sf_search_disagreements.py::_decode_current_board` is
> therefore 100x off. That is true of pure `lc0_root`
> (`_write_metadata_planes_root` writes `float(board.halfmove_clock)`) and
> **FALSE of the encoding these shards actually use**:
> `apply_lc0_root_legacy_meta_planes` overwrites plane 109 with
> `min(halfmove, 100)/100` and plane 110 with the EP file. The shard attrs say
> `input_history_encoding: lc0_root_legacy_meta`, so the decode is
> `halfmove = round(x[109,0,0] * 100)` and the existing helper is correct. This
> is a correction to a claim about the CODE, made before seeing any measurement,
> not a moved threshold. It also means the two layouts disagree on plane 109's
> SCALE, which is exactly why I re-encode for BT4 with `lc0_root` (raw count,
> lc0-canonical) rather than shipping the stored planes.

- **A1.** For every sampled row, the decoded board's legal-move set, mapped to
  our compact `lc0_1858` ids, equals the row's stored `legal_mask` **EXACTLY**.
  Predict **100.0%** of rows, 0 mismatches. This is the load-bearing check: it
  proves the position I hand BT4 is the position the row encodes, over ~22-27
  moves per row, and it would fail on a wrong side-to-move, a wrong orientation,
  wrong castling rights, or a wrong EP square.
  Rows that fail A1 are EXCLUDED and reported; predict 0 exclusions.
- **A2.** Re-encoding the decoded board with `lc0_root_legacy_meta` reproduces
  the row's planes 0..11 and 104..111 **exactly** (predict 100%). Plane 12 (the
  slot-0 repetition plane) may differ because the decoded board has no move
  stack — predict a mismatch rate **0-6%** there, reported separately, and it
  is inert for BT4 (which gets a fresh `lc0_root` encode with repeat fill).
- **A3.** Cross-row legal-mask control: `policy_target[i]` mass under
  `legal_mask[perm[i]]` collapses from 1.000 to **< 0.10** (the prior audit read
  0.049).
- **A4 (what I CANNOT reconstruct, stated up front).** The parent's real move
  history and repetition state are NOT reconstructed — BT4 gets lc0's own
  empty-history fill (`fill_lc0_history_repeat`), which is exactly what every
  banked BT4 number here used. `fullmove_number` is not stored and is set to 1.
  Consequence: BT4 cannot see repetition, and en-passant availability at the
  CHILD is only as good as the EP plane at the parent. A robustness arm feeds
  BT4 the row's OWN 112 planes (real 8-slot history, plane 110 zeroed to match
  canonical `lc0_root`) for the ROOT policy reading on the full sample; predict
  the root-policy agreement statistics move by **< 0.05 absolute**. If they move
  more, the history fill is load-bearing and I say so.

## Positive control (P) — is the ruler any good?

- **P1.** `mean[ Q(sf_best) - Q(random legal move) ]` over all rows.
  Predict **>= +0.25** in Q units (Q in [-1, 1]), CI far from 0.
- **P2.** `P( Q(sf_best) > Q(random legal) )` predict **>= 0.85**.
- **P3.** BT4's own top-1 policy move equals `sf_best`: predict **0.45-0.65**
  (banked: BT4 agrees with the >=1M-node deep-SF best on 0.57 of audit
  positions; here SF is the shard's 150-200k-node MultiPV-6 label, so a little
  lower is expected).
- **KILL:** if P1 < +0.10 or P2 < 0.70 or P3 < 0.30, the ruler is broken and
  **no verdict is reported in either direction.**

## Shuffle / permuted-label control (S)

Break the row correspondence between the TARGET and the POSITION, keeping the
position fixed: for row `i`, pick the move that is the argmax of
`policy_target[perm[i]]` **restricted to row i's own legal set** ("foreign
target argmax"). Then re-run the same statistics.

- **S1.** `P( foreign-target argmax == sf_best )` collapses to chance,
  `mean(1/n_legal)` ~ **0.036**; predict **0.02-0.06**.
- **S2.** `mean[ Q(foreign argmax) - Q(sf_best) ]` collapses to the random-legal
  baseline, i.e. within **+/- 0.05** of `mean[ Q(random legal) - Q(sf_best) ]`.
- **KILL:** if S1 > 0.10, the agreement statistic is measuring the pipeline and
  not the row; if S2 is materially better than the random baseline, the ΔQ
  statistic is not row-specific. Either voids the verdict.

## THE DECIDING MEASUREMENT

Population: rows where `target argmax != sf_best` (predicted ~58% of n, so
~1160 rows). On those rows, paired per row:

    dQ_i = Q_BT4(target argmax_i) - Q_BT4(sf_best_i)

**Primary statistic (scale-free): CORROBORATION RATE**

    C = P( Q_BT4(sf_best) > Q_BT4(target argmax) )   on disagreement rows

- **H_wrong predicts C in [0.62, 0.85]** — BT4 independently prefers SF's move
  on a clear majority.
- **H_exploit predicts C in [0.40, 0.55]** — BT4 is indifferent between them, or
  mildly prefers the target's pick.
- Chance = 0.50.

**DECISION RULE, pre-committed:**
- 95% Wilson CI for C lies entirely **above 0.58** → **H_wrong**.
- 95% CI lies entirely **below 0.58** → **H_exploit** (BT4 does not corroborate
  SF's preference at a rate that would explain a 21.5 cp deficit).
- CI straddles 0.58 → **UNDECIDED**, reported as such, no re-thresholding.

**Secondary statistic (magnitude): mean dQ**, with a paired bootstrap 95% CI
(10,000 resamples over rows).

- **H_wrong predicts mean dQ in [-0.30, -0.08]** Q units. Derivation, so the
  number is falsifiable rather than a band I can slide: the audit's listed-set
  mean regret of the target argmax is 21.5 cp with `frac_eq_0` 0.42, so on
  DISAGREEMENT rows it is ~21.5/(1-0.42) = **37 cp**. lc0's Q->cp map
  `cp = 111.71 * tan(1.5621 * Q)` has slope 174.5 cp per Q at Q=0, so full
  corroboration by BT4 is **dQ ~ -0.21**.
- **H_exploit predicts mean dQ in [-0.03, +0.10]**.
- I also report `mean[ cp(Q_target) - cp(Q_sfbest) ]` per row through that same
  transform, clipped to +/-1500 cp, for readability. Predict under H_wrong:
  **-60 to -15 cp**.

**Pre-committed splits (reported whether or not they help):**
1. target argmax **SF-listed** vs **NOT SF-listed**. The unlisted subset
   (~15% of rows) is where the SF ruler is FABRICATED and literally cannot
   speak, so it is where H_exploit has the most room. Predict, under H_wrong,
   the unlisted subset is **worse** (more negative dQ, higher C) than the
   listed subset — a move SF never listed is usually just bad. Under H_exploit
   the unlisted subset should be the one where BT4 sides with the target.
2. by the target's top-1 mass, bins `[0,0.5) [0.5,0.9) [0.9,0.99) [0.99,1.01)`.
   The audit's "sharp and wrong" claim predicts the deficit does NOT shrink as
   the target gets more confident. Predict `C` roughly FLAT (within 0.10)
   across bins under H_wrong.

**Third statistic (policy ruler, no pushes): BT4's own preference.**

    A_sf  = P( BT4 root-policy top-1 == sf_best )        over ALL rows
    A_tgt = P( BT4 root-policy top-1 == target argmax )  over ALL rows

- H_wrong predicts **A_sf - A_tgt >= +0.10**.
- H_exploit predicts **|A_sf - A_tgt| < 0.05**.
Also reported: mean BT4 log-prob and mean BT4 rank of each of the two moves.

## Ruler-robustness arm (not a decider)

Re-run the deciding measurement on the first **600** sampled rows with
`BT4-it332-vanilla-q.onnx` (the search-Q value tower, which per `docs/bt4.md`
is the q-like head and is the closer analogue of a move-quality evaluator than
`winner`). Predict `C` within **0.08** of the `vanilla-winner` value. A larger
gap means the verdict is head-dependent and I report it as such rather than
picking the head I like.

## What stays INFERRED no matter how this reads

BT4 is SF-agnostic but it is not ground truth. A move can be objectively
slightly worse AND practically better against a specific handicapped Stockfish
(that is the whole anti-engine thesis), and no static evaluator can see
"practically better against SF". So a corroborated deficit shows the target is
objectively worse; it does **not** by itself prove the target is not exploiting
SF. The observation that WOULD close that gap is a paired arena of a net trained
on this target versus one trained on an SF-corroborated target, which is not
what this measurement is.
