# PREREG DRAFT — Tier-14 arm (a): `search_wdl_draw_mode: parametric_q`

**Status: DRAFT, NOT LAUNCHED, NOT A LEDGER ENTRY.** This file is the text to paste into
`docs/experiment_ledger.md` when Josh decides to launch. Nothing here is a verdict, and no
GPU compute has been spent on it. Ships alongside the PR
"selfplay: build search_wdl's draw channel from the searched q (Tier-14 arm (a), default off)".

## Hypothesis

The stored `search_wdl` target's DRAW axis is the net's raw root draw output
(`chess_anti_engine/mcts/_mcts_tree.c`, `float d_raw = wdl_net[1]`), untouched by search, and
that same number CLAMPS the searched q to `+-(1 - d_raw)`. Two consequences, both code facts:

1. `search_wdl_frac` of the trained value target's draw mass is the net grading itself
   (live yaml 0.31; `main`'s committed yaml still reads 0.20 — main's yaml is a stale
   reference, so read the share off the live file, not off the repo).
2. The clamp caps the target's win probability on DECISIVE rows — measured
   (`scratchpad/tier14_clamp_rate_20260812.txt`, 22,725 rows) at 12.27% of the lowest-d_raw
   quartile, 3.97% overall, 24.0% within 5% of the bound.

**H:** replacing the whole `search_wdl` triple with a function of the searched q alone —
`D(q) = coth(w) - sqrt(csch(w)^2 + q^2)`, `w = sf_wdl_cp_slope * sf_wdl_cp_draw_width`, the
cp-logistic family's OWN implied draw curve, i.e. the same family and the same two knobs the
SF component of the blend already uses — removes both defects and improves the value head
against its designated deep-SF ruler.

**What the mechanism does NOT claim.** The +0.0500 search-vs-SF optimism measured in
`scratchpad/audit/target_construction_20260812.md` lives on the W-L axis, which this arm does
not touch (q is preserved exactly, and on clamped rows it is preserved BETTER). The claim
"part of the 41.6cp head-vs-label gap is target error" stays INTERPRETATION; this experiment
is what would establish it.

## THE ONE DECIDING YARDSTICK (exact command)

Value head, deep-SF 1-ply regret. ⚑ Brier/ECE are BANNED — they are fooled by calibration,
and this arm changes the target's calibration by construction, which is precisely the case
where they would read as a free win.

Per arm, then the paired compare (the B1-corrected frozen form from the Tier-12 amendment;
`--paired-against` is NOT a flag of `value_regret.py`):

```
PYTHONPATH=. python3 scripts/value_regret.py --checkpoint <arm.pt> \
  --audit-set data/audit_set_v1.jsonl --max-positions 2000 --min-pieces 8 \
  --batch-size 128 --input-encoding fen_only --gpu-mem-fraction 0.35 \
  --dump-per-position scratchpad/tier14/vr_<arm>_dump.jsonl

PYTHONPATH=. python3 scripts/paired_compare.py \
  scratchpad/tier14/vr_parametric_dump.jsonl scratchpad/tier14/vr_netraw_dump.jsonl \
  --label-a parametric_q --label-b net_raw
```

The scorer flags are pinned because the ruler is batch/encoding-dependent and
`require_same_ruler` refuses a mismatch. The control is a FROZEN anchor exported the same
session, never a rolling read.

## PRE-COMMITTED THRESHOLDS

n ~ 1723 gives a paired CI of about +/-3.8 cp.

- **WORKED**: paired `value_regret` improves by **>= 5.0 cp** with the 95% CI EXCLUDING 0.
  → keep `parametric_q`, promote into the live yaml, record the verdict the same session.
- **FAILED**: the CI does not exclude 0 in the improving direction, OR the point estimate is
  negative. → revert to `net_raw`. Arm (a) is then closed, and the raw-D mechanism was not
  the binding defect; arm (b) (tree-backed-up D) inherits nothing from this — it addresses
  only the D channel and not the confidence cap that arm (a) also removes.
- **KILL EARLY (stability, not quality)**: curriculum winrate outside [0.40, 0.60] for 5
  consecutive iterations, or `replay_has_search_wdl_frac` below 0.99. Both read out in ~3h.
- Anything between WORKED and FAILED is FAILED. "Promising" is not a verdict.

Secondary (NOT deciding; recorded for the mechanism story and judged only if the primary
fires): the draw-position value error from the `value_head_overoptimistic_on_loss_positions`
scorer, and `E[search_sig - sf_sig]` on fresh shards (the ratchet the audit's S1 describes).

## Deploy gating

- **DO NOT deploy during Tier-13.** One data-affecting change per readout window; Tier-13's
  arm contrast owns the current window.
- **RESTART REQUIRED.** `search_wdl_draw_mode` is baked into the frozen `GameConfig` at
  selfplay-session start (`WorkerSession._RECO_RESTART_KEYS`); a mid-run yaml edit changes
  nothing until workers restart.
- **C EXTENSION REBUILD REQUIRED** on every machine that runs selfplay:
  `python3 scripts/build_production_extensions.py`. The PR bumps `ABI_VERSION` to 4, so a
  stale `.so` refuses to start with the rebuild command instead of failing mid-game.
- **MERGE BEFORE ADDING THE KEY TO THE LIVE YAML.** Category (a): a yaml key absent from the
  target branch's schema is fatal AT LAUNCH (`run.py` calls `flatten_run_config_defaults`
  outside any try). Order: merge → restart onto the merged code → then add
  `selfplay.search_wdl_draw_mode: parametric_q`.
- **SNAPSHOT FIRST** (protocol step 2):
  `./scripts/train.sh salvage-export --top-n 1 --metric training_iteration --out data/salvage/pre_tier14a`

## Confounds

- **Replay-window mixing.** The window holds ~a day of rows built under `net_raw`. The
  contrast is not clean until the window has turned over; judge no earlier and state the
  turnover fraction in the readout. A yaml revert is likewise NOT a rollback.
- **Restart transients.** Post-restart winrate reads ~+0.110 too high (sampling bias) and the
  diff-focus normalizer re-warms (one warmup of unnormalized, unclipped priorities enters the
  window). Neither is arm signal.
- **The C rebuild** ships alongside the flag. The default-off path is pinned bit-identical by
  test, so the rebuild is not a confound for the `net_raw` control — but that control must be
  run on the SAME build, not read off a banked pre-rebuild number.
- **Downstream consumers of `search_wdl_est` that are LIVE in production.** Exactly one:
  `selfplay/blindspot_harvest.py:304` (production sets `blindspot_harvest_out_path`), and it
  consumes only `W - L`, which this arm preserves exactly — except on the ~4% of rows where
  `net_raw`'s clamp truncated it, where the harvester's `nq` becomes MORE accurate. The other
  three consumers are off at production settings: `selfplay/finalize.py:636` (needs
  `volatility_source: search`, production `raw`), `selfplay/finalize.py:938` (needs
  `categorical_search_blend_frac > 0`, production 0.0), `selfplay/stockfish_turn.py:1124`
  (needs `sf_label_escalate_q_gap > 0`, production 0.0). Re-check all three before launch; if
  any has been turned on since, it becomes a second changed quantity in the same window.
- **⚑ VOIDS NOTHING on the `wdl_regret` series.** That series is a net signal only while the
  SEARCH config is frozen. This flag touches none of `gumbel_c_scale`, `gumbel_policy_temp`,
  `mcts_simulations`, `gumbel_topk` or the root transform, and does not touch move selection
  at all (proof below), so the series survives this deploy. It still must not be the verdict
  instrument — the deciding yardstick above is.

## Proof that the flag changes only the STORED target, never move selection

- The played action is an INPUT to the function that builds `search_wdl`:
  `chess_anti_engine/mcts/_mcts_tree.c` `py_batch_process_ply` receives `actions`
  (`int32_t action = actions[i]`) and only later writes `wdl_search_out[i*3 + ...]`. Move
  selection happened in `run_gumbel_root_many_c` before this call; nothing in the mode branch
  feeds back into it.
- `wdl_search_out` has four write sites in the whole C file, all inside that one block
  (`grep -n wdl_search_out chess_anti_engine/mcts/_mcts_tree.c`).
- Python fallback, same shape: `chess_anti_engine/selfplay/network_turn.py` computes
  `search_wdl_est` AFTER `board_before.push(move)` and stores it on the `_NetRecord`; the
  action came from `actions[j]`.
- The record field is read only by the four consumers listed under Confounds, none of which
  chooses a move.
- Independent of all the above: `search_wdl_draw_mode` is a `GameConfig` field, not a
  `SearchConfig` field (`chess_anti_engine/selfplay/config.py`), so no search object can see
  it.

## Rollback

Set `selfplay.search_wdl_draw_mode: net_raw` (or delete the key) and restart. Bit-identity of
the `net_raw` path is pinned by
`tests/test_search_wdl_draw_mode.py::test_omitting_the_new_args_is_bit_identical_to_passing_net_raw`
and `::test_net_raw_branch_is_bit_for_bit_the_original_formula`, so the revert restores the
exact prior arithmetic. The replay window still needs a full turnover before the loop is back
on pure `net_raw` data.
