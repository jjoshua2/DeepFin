# Model heads — outputs, targets, losses

Reference detail for `ChessNet`'s multi-task heads (moved out of CLAUDE.md;
source of truth for the loss wiring is `train/losses.py`).

Policy heads emit policy-size logits — **lc0_1858 compact in production**,
legacy az_4672 (see the Move Encoding section of CLAUDE.md).

| Head output | Shape | Training target | Target source | Loss | Weight knob |
|---|---|---|---|---|---|
| `policy` / `policy_own` | policy logits | `policy_t` (soft) | Gumbel completed-Q **improved policy** over all legal moves (`rec.policy_probs` = softmax(log prior + σ(completed Q)) at the searched root — the paper's recommended target, NOT raw visit counts). Move-selection temperature affects only the played move, not this target. | CE, legal-masked | `w_policy` |
| `policy_soft` | policy logits | `policy_soft_t` (soft) | Same improved policy as `policy_t`, retempered via `apply_policy_temperature(soft_policy_temp)` (typically softer) | CE, legal-masked | `w_soft` |
| `policy_future` | policy logits | `future_policy_t` (soft) | The t+2 record's `policy_probs` — improved policy at position t+2 (predict-own-reply) | CE, **no** mask | `w_future` |
| `policy_sf` | policy logits | `sf_policy_t` (soft) | Softmax over SF's MultiPV candidate WDL scores + label smoothing. SF labels are queried at **P1** (after the net's move), so this is the **opponent's reply distribution**, NOT a move-teacher for the sample's own position — the `sf_p0_*` fields (one-ply shift, selfplay rows) provide that. `sf_move_index` is the stored bestmove pointer, used only by the `sf_move_acc` metric. | CE (soft), no mask | `w_sf_move` |
| `wdl` | 3 logits | three-way soft blend | `game_frac`·game outcome (one-hot) + `sf_wdl_frac`·SF eval + `search_wdl_frac`·own MCTS root WDL (fracs live-tunable). `sf_wdl` is a **cp-logistic** label (`sf_wdl_use_cp_logistic`, slope/draw-width in config), soft by construction, not SF's native near-one-hot WDL. See the blend note below. Within `wdl_terminal_outcome_plies` of the game's end part of the search share moves to the outcome (default 0 = off) | soft CE on the blend | `w_wdl` + the three `*_frac` knobs |
| `sf_eval` | 3 logits | `sf_wdl` (soft) | SF's WDL eval only (auxiliary, **not** used in MCTS) | Soft CE | `w_sf_eval` |
| `categorical` | 32 logits | `categorical_t` (HL-Gauss) | Game outcome as 32-bin Gaussian distribution (distributional value) | CE | `w_categorical` |
| `volatility` | N scalars | `volatility_t` | Net-derived position volatility signal | Huber (δ=0.1) | `w_volatility` |
| `sf_volatility` | N scalars | `sf_volatility_t` | SF-derived position volatility signal | Huber (δ=0.1) | `w_sf_volatility` |
| `moves_left` | 1 scalar | `moves_left` | Plies remaining in the game | smooth L1 | `w_moves_left` |

Implementation details:

- Each of the 4 policy heads is a separate `AttentionPolicyHead` (Q/K/underpromo
  projections) that shares the trunk. Uses `Q@K^T` with `1/√d` scaling plus a
  learnable `log_temp` scalar per head (added Apr 2026 — the `1/√d` scale alone
  squashed output sharpness below what MCTS targets required).
- `wdl` is the ONLY value head used in MCTS search. `sf_eval` and `categorical`
  are auxiliary supervision signals that share the trunk but don't feed the
  search.
- Rows are recorded per net turn with `has_policy` ⇔ full-sim ply. Only the
  MAIN policy CE masks by `has_policy`; the aux policy heads mask by their own
  presence flags, so value-only rows must have those targets cleared at
  finalize (they are — see `_build_replay_samples`).

## Reported value-loss names (read before quoting one)

Three names, two quantities. `losses.py` returns:

| key | metric column | what it is |
|---|---|---|
| `wdl_ce` | `wdl_loss`, `test_wdl_loss`, `wdl_loss_{selfplay,curriculum,open,mid,end}` | **the trained loss** — soft CE against the three-way blend, the term `w_wdl` multiplies in `total` |
| `blended_wdl_ce` | `blended_wdl_loss` | the same tensor under its explicit name, kept for existing readers |
| `wdl_onehot_ce` | `wdl_onehot_loss`, `test_wdl_onehot_loss` | **diagnostic only** — hard one-hot CE against the recorded game result. No gradient reaches the update through it |

Until 2026-07-26 the `wdl_loss` family reported the one-hot diagnostic, so the
holdout's value column tracked a number no gradient came from (they differ:
0.7599 vs 0.7859 at iter 42). The columns therefore have a **one-time step at
that deploy** — a change of quantity, not of the net. Anything computing a
gradient share, a head weight, or a value verdict must use the trained loss;
`wdl_onehot_loss` belongs only in "how does the head score against raw results"
readings, and never in a denominator of gradient contribution
(docs/rl_loop_audit.md I3/I7).

## The WDL blend note (load-bearing)

**The blend's SF component must not be zeroed.** The main `value_wdl` head
trains on one soft-CE against a three-way blend built in `losses.py`:
`game_frac`·outcome + `sf_wdl_frac`·SF cp-logistic label +
`search_wdl_frac`·own search-root WDL (no separate `w_sf_wdl` knob exists —
the blend replaced the old dual loss). Removing SF value supervision crashed
winrate 0.64 → 0.40 in 4 iters (2026-04-17, reverted in 52ab9c0). The
cp-logistic label is deliberately soft and ~calibrated to actual selfplay
outcomes (verified 2026-07-01), so the head's "hedged" predictions are an
accurate fit to its target, not under-fitting — the head sharpens only as the
net's real conversion ability improves. Don't chase value-head sharpness
against a deep-SF ruler; that comparison is a category error. `value_sf_eval`
is a weak auxiliary channel, not a substitute.

**Terminal-proximal outcome share (`wdl_terminal_outcome_plies`, default 0 =
OFF).** The global `game_frac` stays 0 and deep-game outcome labels stay out of
the target — they previously carried a value collapse. But the outcome's noise
as a value label RAMPS with distance to the terminal: measured offline over
149k rows (2026-08-07) sd ≈ 0.10 at d = 1–2 plies, 0.22 at 5–6, crossing the
search component's own noise at d ≈ 6–7, 0.77 deep. Within a few plies of the
end the outcome is the cleanest of the three estimators — crisper even than the
deliberately-soft cp-logistic SF label. So for rows with d ≤ N the blend hands
part of the **search** share to the one-hot outcome, on a linear taper:

```
d      = round(moves_left × max_plies)          # moves_left is plies-remaining / ply cap
taper  = clamp((plies − d) / (plies − full_plies), 0, 1)
w_out  = realized_search_frac × taper  ( + realized_sf_frac × wdl_terminal_outcome_sf_frac × taper )
```

with `wdl_terminal_outcome_plies` (the distance at which the taper reaches
zero; planned production 7) and `wdl_terminal_outcome_full_plies` (the distance
at or below which the whole share moves; default 2). At 7/2: d ≤ 2 gives the
outcome the entire search share, d = 6 gives it 20 % of it, d ≥ 7 is unchanged.
Both fracs are the **realized** ones at that step — `sf_wdl_frac` is
PID-recomputed every iteration, so nothing here may hard-code 0.69/0.31. Rows
with `has_moves_left = 0` keep the unchanged blend (the field defaults to 0.0,
which would otherwise decode as d = 0, the maximum transfer); rows with no
search label already fall back to the outcome, so the transfer is a no-op for
them by construction. Off (`plies: 0`) the blend is **bit-identical** to the
pre-feature one.

`wdl_terminal_outcome_sf_frac` (default **0.0** — the SF share is then never
touched at any d) is the fraction of the SF share the outcome may additionally
take on the same taper. It exists for the offline screen's aggressive arm; the
SF component is load-bearing (see above), so raising it is a training-target
experiment of its own. At sf_frac 0.5 with the production floor sf 0.69 /
search 0.31: d ≤ 2 → SF 0.345 / search 0.0 / outcome 0.655; d = 6 → SF 0.621 /
search 0.248 / outcome 0.131.

Proof of effect is the `wdl_terminal_outcome_frac` column in progress.csv (the
mean weight moved onto the outcome across all batch rows, exactly 0.0 while the
knob is off), read with its `wdl_terminal_outcome_rows` count — not the echoed
config value. Pinned by `tests/test_wdl_terminal_outcome.py`.

**`sf_wdl_conf_power` and `sf_wdl_draw_scale` are NOT blend knobs.** Despite the
`sf_wdl` prefix, both feed exactly one mask (`_compute_sf_wdl_mask` in
`losses.py`) whose only consumer is the auxiliary `sf_eval` head. The blend's SF
component is weighted by `sf_available * keep`, and `keep` carries only the
`sf_search_dampen_sf_low/high` terms (both 0.0 in production). Change either
knob and `wdl_ce` is bit-identical; only `sf_eval_ce` moves (~0.006 % of total
loss). There is **no** knob that damps the SF component of the value target by
SF's own draw confidence — adding one is a training-affecting change and needs a
ledger entry. Pinned by `tests/test_sf_wdl_conf_knobs_are_aux_only.py`.
