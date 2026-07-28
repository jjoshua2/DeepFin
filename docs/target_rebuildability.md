# Which training targets can be rebuilt from a stored shard row

A target change normally only affects **newly generated** data, so it cannot be
read until the ~1.5M-row replay window turns over — measured 2026-07-27 at ~99
iterations, **~18 h** (50 % at ~9 h). That latency is the rate limiter on the
whole target-experiment loop.

Some targets escape it. Shards (schema v2+) store the raw Stockfish MultiPV
rows the live pipeline built its SF targets from, so those targets can be
recomputed from scratch at sample time under new parameters —
`train.rebuild_sf_targets` (default OFF), implemented in
`chess_anti_engine/train/target_builder.py`. When it is on, an
`SfTargetParams` change applies to **the entire existing window on the next
iteration**.

Most targets do not escape it. This file says exactly which, why, and what
extra storage would move a row from the second table to the first. Every claim
below was verified against 10 live shards (18 307 rows) from the iter-168
window on 2026-07-27; the "verified" column gives rows checked and the worst
disagreement observed, which for the rebuildable ones is fp16 storage noise.

## Rebuildable from the row itself

| target | source on the same row | param-dependent? | verified |
|---|---|---|---|
| `sf_policy_target` | `sf_multipv_raw` + `sf_legal_mask` | yes — `sf_policy_temp`, `sf_policy_label_smooth`, `sf_wdl_use_cp_logistic`, `sf_wdl_cp_slope`, `sf_wdl_cp_draw_width` | 16 822 / 16 822 rows, max TV **2.2e-4** at the live params |
| `sf_wdl` | `sf_label_meta` | yes — the three `sf_wdl_cp_*` knobs | 17 778 / 17 778 rows, max abs **2.4e-4** |
| `categorical_target` | `wdl_target` + `sf_wdl` (+ `search_wdl`) | yes — `categorical_blend_frac`, `categorical_search_blend_frac`, `hlgauss_sigma` | already shipped as the separate `train.rebuild_categorical_target` flag |
| `priority_sf_search_gap` | `sf_wdl` + `search_wdl` | yes, via `sf_wdl` | 17 778 / 17 778 rows, max abs **9.2e-4** — **but see the note below: rebuilding it in a batch is useless** |

`priority_sf_search_gap` is arithmetically rebuildable and practically not: it
is a **sampling** priority, consumed once at ingest to weight the row in the
buffer index. Recomputing it inside a sampled batch changes nothing about which
rows were sampled. Moving a sampling knob still requires re-ingest, i.e. it is
still an 18-hour question (or an offline `retarget_retrain.py` run over a copied
shard dir).

## Stored, but nothing to rebuild

These are SF-derived yet carry **no** `SfTargetParams` dependence, so they never
go stale under a rebuild and must be left alone.

| target | why it is parameter-free |
|---|---|
| `sf_move_index` | SF's `bestmove` from the UCI reply, not a function of our scoring. A naive "argmax of `sf_multipv_raw`" rebuild reproduces it on only **95.6 %** of rows (16 822 checked) — so a rebuild would *corrupt* 4.4 % of them. Don't. |
| `sf_p0_regret`, `sf_played_regret`, `sf_played_rank`, `sf_played_move_index`, `future_sf_regret_*` | normalized centipawn regret / rank orderings (`finalize._sf_move_score`, cap 1000 cp). No temperature, no cp→WDL mapping. |
| `sf_legal_mask`, `legal_mask`, `future_legal_mask`, `moves_left` | positional or game-length facts. |

## NOT rebuildable — the source is on a different row

Both of these **are** functions of what the rebuild moves, but the source row is
not in a randomly sampled batch. `rebuild_sf_targets_in_arrays` therefore
**masks** them (clears the presence flag) rather than leaving them pointing at
capture-time values while their source moves underneath.

| target | is exactly | verified | consequence |
|---|---|---|---|
| `sf_p0_policy_target` | ply **t-1**'s `sf_policy_target` | 2 276 / 2 276 rows, max TV **0.0** (bit-exact) | masked ⇒ the `w_sf_own` own-move teacher (0.1) is inert while the rebuild is on. `w_sf_own_regret` (0.7) is NOT masked. |
| `sf_volatility_target` | `abs(sf_wdl[t+6] - sf_wdl[t])` | 2 374 / 2 374 rows, max abs **4.9e-4** | masked ⇒ `w_sf_volatility` (0.05) is inert while the rebuild is on. |

Masking is **unconditional** — not "only when the params actually moved". A
control run (flag on, capture-identical params) and a treatment run must mask
the *same* rows, or the paired comparison is confounded by the own-move teacher
switching on and off between arms.

Two rejected alternatives, for the record:

* **Leave them stale.** This is the defect the masking exists to prevent: the
  loop's only external move-teacher would train on capture-time targets while
  its sibling `sf_policy_target` moved, and nothing would report it.
* **Re-temper the stored `sf_p0_policy_target`** (`p' ∝ p^(T0/T')`). Measured on
  800 live rows against a true rebuild: sharpening 0.012 → 0.006 is fine (mean
  TV 0.0013), but **softening is not** — 0.012 → 0.05 gives mean TV **0.066**,
  p90 0.25, max 0.84, against a true rebuild's 7e-5. The stored target is fp16
  and at temp 0.012 keeps only ~18 non-zero entries; softening has to resurrect
  a tail that fp16 already flushed to zero. Softening is the direction any
  `sf_policy_temp` experiment goes, so this is not a substitute.

## NOT rebuildable — the source was never stored

| target | built from | what the worker would have to store |
|---|---|---|
| `policy_target` | MCTS visit counts at the root, discarded after `finalize` | sparse root visits: `(K=64) × (move_idx int16, visits int32)` ≈ **384 B/row** |
| `policy_soft_target` | the same visit counts under `soft_policy_temp` | same 384 B/row — one field unlocks both |
| `future_policy_target` | ply **t+2**'s MCTS distribution | the visits above, *and* a cross-ply copy |
| `search_wdl` | the net's own MCTS root value average | nothing: it *is* the raw datum, there is no parameter to re-point |
| `volatility_target` | `abs(search/net wdl[t+6] - [t])` | cross-ply *and* search-derived |
| `wdl_target` | the game result (+ Syzygy / adjudication rescoring) | the raw datum |
| `priority`, `priority_policy_kl`, `priority_q_delta` | the generation-time model's surprise (KL of search vs prior, Q delta) | these are properties of *the model that played the game*; no storage makes them re-derivable for a different model. Re-weighting requires re-ingest. |

### Storage that would close the two masked rows

* `sf_p0_multipv_raw` — `(48, 5) int16` = **480 B/row uncompressed**. On the
  measured shards `sf_multipv_raw` compresses to 153 B/row at ~92 % coverage;
  `sf_p0` coverage is ~22 %, so expect ≈ **+35-40 B/row, about +2.5 %** on a
  1 588 B/row shard. `finalize._build_replay_samples` already computes
  `prepare_multipv(prev_idx)` for `sf_p0_regret`, so it is a one-line reuse of a
  value that is already in hand.
* `sf_label_meta_t6` — `(6,) int32` = **24 B/row**, unlocks
  `sf_volatility_target`.

Both only help data generated *after* they ship, so they do not shorten the
first readout — they shorten every readout after it.

## What this does and does not buy

**Does**: any experiment whose only lever is `SfTargetParams` —
`sf_policy_temp`, `sf_policy_label_smooth`, `sf_wdl_use_cp_logistic`,
`sf_wdl_cp_slope`, `sf_wdl_cp_draw_width` — plus the categorical-blend family
under its own flag. Those move from "wait ~18 h for window turnover" to
"applies to the entire existing window at the next iteration".

Note the useful interaction: the SF param keys are also worker reco keys, so one
yaml edit retunes the capture params for new data *and* re-points the rebuild
for the old data. The window stays homogeneous instead of spending 18 h as a
mixture of two target regimes.

**Does not**: anything search-derived. `policy_target`, `policy_soft_target`,
`search_wdl`, `volatility_target` and every `priority_*` come from the search
that was actually run, and no re-parametrisation of stored bytes recovers a
search that did not happen. **C17 (duplicate leaves / virtual loss) is squarely
in this category** — it changes which nodes get visited, so its readout still
costs a full window turnover. Sampling/priority knobs are also excluded: they
act at ingest, not at sample time.
