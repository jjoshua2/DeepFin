# PREREG DRAFT — Tier-14 arm (a): `search_wdl_draw_mode: parametric_q`

**Status: DRAFT, NOT LAUNCHED, NOT A LEDGER ENTRY.** This file is the text to paste into
`docs/experiment_ledger.md` when the maintainer decides to launch. Nothing here is a
verdict, and no GPU compute has been spent on it. Ships alongside the PR
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

**⚑ THE COST BEING TRADED, NAMED UP FRONT — this is the falsifier if the arm reads negative.**
`cp_to_wdl`'s draw mass is that SAME function of its own q (agreement measured at 3.3e-8), so
after this change BOTH components of the blended value target carry `D = D(q_component)` under
ONE shared curve. The draw axis of the trained target then holds **no information independent
of the two W-L signals** — every position with the same q gets the same draw mass. That is
precisely the self-reference this arm removes, and it is also the removal of the net's
POSITION-SPECIFIC draw knowledge: fortresses, opposite-coloured bishops, and the closed
positions this project exists to exploit are exactly where a hand-drawn `D(q)` is wrong and
`d_raw` might not be. If the primary reads negative, this is the first hypothesis, not a
surprise — and it is what arm (b) (tree-backed-up D) would address instead.

**⚑ THE ARM IS TWO MECHANISMS, AND THEY HAVE VERY DIFFERENT REACH.** A null does not kill both
at once. From `scratchpad/tier14_clamp_rate_20260812.txt`: the confidence CLAMP binds on
**3.97%** of rows and, where it binds, truncates an already near-decisive q by a few
hundredths (ceiling mean 0.929 / median 0.962). The D-AXIS replacement touches **100%** of
rows. So a readout is overwhelmingly attributable to the D axis; the clamp removal is a rider
too small for this instrument to resolve and must not be reported as tested by it.

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
- **⚑ DEPLOYMENT SKEW IS A MANUAL GATE HERE — there is no in-band guard, and the obvious one
  is a trap.** A worker on pre-#409 code drops the unknown reco key, is not restart-keyed by
  it, and keeps uploading `net_raw` rows into the SAME replay window as `parametric_q` rows
  with **no column distinguishing them** — unrecoverable, in exactly the window this
  experiment reads.
  - **Do NOT reach for `PROTOCOL_VERSION`.** It was bumped 2 → 3 in this PR and then
    REVERTED: `_check_worker_compat` requires EXACT protocol equality (`got_p != req_p` →
    426), so a bump takes the fleet to zero in BOTH directions of a rolling deploy. `main`
    documents this in `chess_anti_engine/version.py` and names `min_worker_version` (a `>=`
    comparison over `PACKAGE_VERSION`) as the cutover mechanism instead. Only the local merge
    check against a moved `main` caught this; CI would not have.
  - **The gate that works: bump `PACKAGE_VERSION` in the same deploy.** It is published on
    every manifest as `min_worker_version`, compared with `version_lt`, so it excludes stale
    workers without breaking the transition. ⚑ It only fires if the package version is
    ACTUALLY bumped — an in-tree `git pull` leaves it identical, so this is a required deploy
    step, not a nicety, and it means re-running the install on every worker machine.
  - **Manual confirmation regardless**, because our workers are same-machine and restart with
    the trainer: before setting the key, confirm every selfplay process is on merged code
    with an ABI-4 `.so`; after the first restart, confirm the stored-row property below on
    fresh shards. That check reads the DATA, so it cannot be faked by a config echo — it is
    the only instrument that would notice a mixed window.
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
- **Downstream consumers of `search_wdl_est`. SIX read sites, not four** — the first draft of
  this list was short by two, both found by review. RE-CHECK EVERY ROW BELOW BEFORE LAUNCH; a
  gate that has been turned on since makes its site a second changed quantity in the window.

  | site | live in production? | what it reads |
  |---|---|---|
  | `selfplay/blindspot_harvest.py:304` | YES (`blindspot_harvest_out_path` set) | `_q()` = `W - L` only |
  | `selfplay/blindspot_harvest.py:369` | YES (same path, feeds `value_blind_candidates`) | `_q()` = `W - L` only |
  | `selfplay/finalize.py:985` | YES — **always computed** | `W - L` only, into `sf_search_gap` |
  | `selfplay/finalize.py:636` | no — needs `volatility_source: search`, production `raw` | full triple |
  | `selfplay/finalize.py:938` | no — needs `categorical_search_blend_frac > 0`, production 0.0 | full triple |
  | `selfplay/stockfish_turn.py:1124` | no — needs `sf_label_escalate_q_gap > 0`, production 0.0 | full triple |

  Every LIVE consumer reads only `W - L`, which this arm preserves exactly — except on the
  ~4% of rows where `net_raw`'s clamp truncated it, where they become MORE accurate.
  ⚑ `finalize.py:985` is the one to watch anyway: its `sf_search_gap` is written to every
  shard as the `priority_sf_search_gap` COLUMN regardless of any flag. Its own consumer
  (`replay/disk_buffer.py::_gap_boost_and_mask`) is gated off — `replay_sf_gap_priority_weight`
  is 0/absent (experiment #104 killed) and `replay_fast_low_surprise_priority` is unset — so
  the column is inert today, but re-confirm both before launch, because turning either on
  would make the priority distribution a second changed quantity.
  (`selfplay/finalize.py:1117` and `selfplay/resume.py` also touch the field; the first IS the
  stored target and the second is serialization, so neither is a consumer.)
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
