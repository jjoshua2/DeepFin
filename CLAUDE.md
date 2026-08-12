# CLAUDE.md

The single source of project instructions for every agent (Claude Code, Codex, Grok).
`AGENTS.md` is a pointer to this file — keep it that way: a second copy of these rules
drifts, and deleting it would silently strip Codex of its instructions.

Everything here is either a hard rule or a fact you cannot get by reading the code.
Module-level detail is deliberately absent — read the source.

## Project

Chess anti-engine training framework — trains a transformer to exploit Stockfish
weaknesses (fortress blindness, horizon effects, closed-position overconfidence).
CUDA, primarily one RTX 5090.

Per-iteration data flow: distributed selfplay (MCTS vs Stockfish) → shard upload →
ingest into disk-backed replay buffer → training step → checkpoint → publish model to
workers.

Layout: `chess_anti_engine/` is the package — `encoding/` + `moves/` (input and policy
encoding), `model/` + `mcts/` + `selfplay/` + `train/` + `replay/` (the training loop),
`stockfish/` + `server/` + `worker.py` + `tune/` (the distributed pipeline). Plus
`tests/`, `configs/`, `scripts/`. `runs/`, `tb/`, `server/`, `data/` and large
model/book artifacts are runtime output — they stay uncommitted.

## Commands

```bash
pip install -e ".[dev]"     # full env; the test suite hard-requires every extra (no skips)
pip install -e ".[worker]"  # lite selfplay-client install
python -m pytest

PYTHONPATH=. python3 -m chess_anti_engine.run --config configs/pbt2_small.yaml --mode tune [--resume]
```

Scripts need `PYTHONPATH=.`. CLI modes are `train`, `tune`, `salvage` — there is no
`single` mode. `--mode train` is a single distributed trial (no PBT), not a local one:
it still boots the server and at least one worker, because there is no non-distributed
selfplay path. Drive live training with `scripts/train.sh`, not the module directly;
see `docs/operations.md` for that and for salvage, blind-spot seeding, and lint detail.

## Configs

- `configs/pbt2_small.yaml` — **production**, and the only config active training uses.
  512-dim × 16-layer × 16-head, **63.08M trainable params** (63,084,128, counted
  2026-07-26 by unique storage on `checkpoint_000042`). **An earlier revision of
  this line "corrected" 63M up to 78.8M. That was the error, not the fix.**
  78.81M is the sum of `numel()` over the 496 `state_dict` entries, which
  double-counts weight tying: the 16 `layer_smolgens.N.gen_weight.weight` keys
  are **one shared tensor** (16 keys, 1 distinct storage, 1,048,576 params), so
  15 × 1,048,576 = 15,728,640 is counted 15 times too many — exactly the gap.
  Count unique `v.untyped_storage().data_ptr()`, never `sum(v.numel())`.
  Per-layer Smolgen still dominates, but by less than advertised:
  `layer_smolgens` **26.7M of 63.08M (42.3%)**, not 42.4M of 78.8M (54%).
  `ffn_mult` is per-layer and non-uniform (1.5 rising to ~1.9 in the upper
  blocks), so the count is not reproducible by assuming a flat multiplier.
  Related: only 28.6% of trainable params are in the Aurora matrix group
  (`matrix_optimizer_scope: mlp_out`) — see `docs/rl_loop_audit.md` I12.
  `tests/test_param_count.py` rebuilds both configs and fails if any number in
  this section drifts from the measurement.
- `configs/default.yaml` — reference BT3-scale model (768-dim × 15-layer × 24-head,
  **73,700,885 params**), unused. The "~105M" this line used to claim was never
  measured; the config has no tied tensors, so its `state_dict` sum agrees.
- `configs/exp_*.yaml` — flag-gated research bets, ALL default off. A flag enters the
  production config only once promoted; promotion status lives in the ledger, not here.

All tunables live in the yaml — grep it rather than assuming a value.

## Working on a live run

The run is usually live. These break production:

- **Never `git checkout` in this working tree while a run is live.** The live yaml is
  part of the tree and is re-read every iteration, so a branch switch silently reverts
  live experiments (2026-07-02: rolled back three experiments for 3 iterations). Use
  `git worktree` for all branch work, and merge PRs touching the live yaml promptly.
- **⚑ A bad live-yaml edit has three outcomes, not one. Establish which CATEGORY the key
  is in BEFORE you edit** — (a) not in the yaml schema, (b) in the schema, read by
  `TrialConfig.from_dict`, and validated, (c) in the schema but never validated. All
  measured 2026-08-09, re-verified against `main` 2026-08-10.
  - **(a) Not in the schema.** MID-RUN: `_reload_yaml_into_config`
    (`tune/trainable_config_ops.py`) catches the `ValueError`, logs `YAML reload failed`,
    and keeps the OLD config — the WHOLE reload is rejected, **the trial survives**. AT
    LAUNCH the same `ValueError` is fatal: `run.py` calls `flatten_run_config_defaults`
    before the main argument parser is built and outside any `try`, so **the process never
    starts** and there is no old config to fall back to. ⇒ **restarting onto code that
    predates a live yaml key does not silently revert, it fails to boot.** Before ANY
    restart, diff the live yaml's KEYS against the target branch's SCHEMA
    (`utils/config_yaml.py`), not against the target branch's yaml — and add a new key to
    the live yaml only AFTER restarting onto code that defines it. This is why a PR that
    adds a live key must be merged before the branch it is missing from can be restarted
    onto.
  - **(b) In the schema, validated by `from_dict`, value out of range.** The reload
    SUCCEEDS, then `TrialConfig.from_dict` (`tune/trial_config.py`) raises inside
    `train_trial`'s iteration-loop `try:` — which has a `finally:` and **zero `except`** —
    so `_cleanup_trial_resources` runs and **the trial dies mid-iteration**. Measured on
    `gumbel_policy_temp` (band `[0.05, 20.0]`, endpoints inclusive): `0.0`, `0.02` and
    `200` each kill it, `2.0` applies cleanly. The realistic trigger is a decimal typo.
    This belongs to EVERY `TrialConfig` validator — a band does not create the hazard, it
    only widens the trigger set.
  - **(c) In the schema, NOT validated** — the schema accepts it, nothing range-checks it,
    the overlay applies it, and its consumer gets it raw: `zclip_max_norm: 1e-9` lands
    silently. Neither a crash nor a rejection — **silent wrongness**, and the slowest of
    the three to notice. Two sub-shapes, and the second is the dangerous one:
    - never read by `from_dict` at all — `w_wdl`, `zclip_max_norm`, and most of the live
      yaml's `train:` section. These reach their consumer through the raw `config` dict
      instead.
    - **read by `from_dict`, but with no validator** — most of the constructor, including
      **`lr`** (`lr=float(config["lr"]) if "lr" in config else 0.0003`). ⚑ Do not read
      "not validated" as "inert": `lr` is a `TrialConfig` field, it IS read, and a live
      `lr: 0.3` is accepted and applied to the running trainer (GPBT: LR > 0.003 destroys
      the model). Only a NON-NUMERIC value trips it, and then `float()` raises inside
      `from_dict` and it dies as category (b).

    ⚑ Do not test membership by `.get("<key>", ...)`: `from_dict` reads `lr` by
    SUBSCRIPT, so a `.get`-only scan reports the single most consequential key in the
    file as unread. `sf_pid_*` splits: 4 of its 40 schema keys (`sf_pid_enabled`,
    `sf_pid_ema_alpha`, `sf_pid_target_winrate`, `sf_pid_wdl_regret_max`) are read by
    `from_dict`; the other 36 are not.

  ⇒ "an unknown key rejects the whole reload" is category (a) MID-RUN only; it is wrong
  about (b) and about (c). **A live edit to a validated key is not a soft operation.**
  Dry-run on a COPY first —
  `TrialConfig.from_dict(flatten_run_config_defaults(yaml.safe_load(open(<copy>))))` —
  and only then write the live file. Note the dry-run proves only that the value SURVIVES
  category (b); it says nothing about category (c), which by construction cannot fail it.
- **Never run a 256+ sim arena concurrent with training** — GPU OOM crashed the run
  2026-06-18. sims-1/32 arenas and `audit_targets`/`value_regret` at small batch with
  `--gpu-mem-fraction` are safe.
- Ray prunes live checkpoints. Copy one out of the tune dir before using it as a
  long-lived arena/audit baseline.
- After pulling `.c`/`.h` changes, rebuild with
  `python3 scripts/build_production_extensions.py` (GCC15 + native + LTO, forced
  rebuild) — NOT `pip install -e .`, the .venv setuptools lacks PEP 660. Keep
  distributed wheels portable unless every target CPU matches.

## Experiment protocol — MANDATORY

**`docs/experiment_ledger.md` is the canonical experiment record** (WORKED / FAILED /
LIVE-UNREAD verdicts, yardstick anchors, revert points). Read it before proposing,
launching, judging, or reverting any experiment.

**`docs/rl_loop_audit.md` is the companion invariant record** — per-stage checks with
the exact instrument and current status. The ledger says whether an experiment worked;
the audit says whether the pipeline that produced the number was sound. Read its
"Method rules" before deriving any metric by hand: several confident findings there
turned out to be artifacts of the measurement, not the loop. A verdict read off a stage
that FAILS its invariant is not a verdict.

1. **Before a training-affecting change goes live** (config key, loss weight,
   data-pipeline or selfplay change, PR merge that alters training): add a ledger entry
   with the hypothesis, ONE deciding yardstick as an exact command, and a pre-committed
   kill/success threshold. No entry → don't launch.
2. **Before big changes**, snapshot weights + optimizer + PID + replay window and record
   it in the ledger's Revert points table:
   `./scripts/train.sh salvage-export --top-n 1 --metric training_iteration --out data/salvage/<label>`
   (safe while training runs, ~2.3G; `--metric training_iteration` is REQUIRED — the
   default picks the best-metric row, not current state). A yaml revert is NOT a
   rollback: the replay window holds ~a day of data made under the old settings.
3. **After a readout**, record the verdict the same session, judged by the pre-committed
   rule rather than post-hoc reading. "Deferred" is not a verdict. Stability/throughput
   changes read out in ~5 iterations (~3h, spanning ≥2 of the failure's cadence periods);
   learning-quality changes need day-plus windows and paired CIs.
4. **One data-affecting change per readout window.** Unavoidable overlaps go in each
   entry's Confounds line.
5. **Before running any yardstick**, read the ledger's "Protocol gotchas".

## Evaluation

`docs/eval_protocol.md` is the decision protocol. The audit-first rule: every
training-target candidate is scored against the frozen deep-SF audit set BEFORE any
training compute, and one that loses the direct audit is killed without training.

- `scripts/arena_standard.py` — paired-opening arena, pentanomial Elo + 95% CI.
  `matched_sims` for search/target changes; `matched_time` only when the change ships in
  the fast C path, since Python-path features are under-credited.
- `scripts/build_audit_set.py` / `scripts/audit_targets.py` — the frozen audit set
  (≥1M nodes, MultiPV ≥10, side-to-move canonical) and its scorer. The set FREEZES after
  generation; new sampling = new version.
- `scripts/value_regret.py` — value-head 1-ply deep-SF regret, the VALUE yardstick.
  Brier/ECE are fooled by calibration and must not be used to judge value strength.
- `scripts/probe_policy_targets.py`, `scripts/retarget_retrain.py`,
  `scripts/convert_shards_v2_threats.py` — policy/soft-policy divergence, offline
  SF-target retuning, offline v1→v2_threats shard conversion.
- **Production Syzygy is the colon-separated pair**
  `/home/josh/projects/chess/data/syzygy_3-4-5:/home/josh/projects/chess/data/syzygy_6` —
  `configs/pbt2_small.yaml`'s `syzygy_path`, and **both halves are local**. The directory
  names lie, so read them off the contents rather than the name: `syzygy_3-4-5` holds 3–6
  man WDL (`.rtbw`, 510 files) plus 3–5 man DTZ (`.rtbz`, 145); the 6-man DTZ that supplies
  root ranking and 50-move-exact conversion is the separate local `data/syzygy_6` (365
  `.rtbw` + 365 `.rtbz`, 151G). Pass the full pair as `SyzygyPath` to BOTH engines for a
  production-equivalent match; `data/syzygy_3-4man` is a smoke-test set only.
  `tests/test_param_count.py` pins this pair to the config, because the claim above is
  exactly the kind that drifts: production moved off the external drive on 2026-07-14 and
  15 research configs (14 `configs/exp_*.yaml` plus `configs/bt4_aurora_asha.yaml`, all
  default-off) still point their second half at `/mnt/e/chess/syzygy_6_dtz` — same tables,
  external drive, **not what production reads**.

## Non-obvious training facts

Consequential and not apparent from the code:

- **The WDL blend's SF component is load-bearing — do not zero it.** Removing it crashed
  winrate 0.64 → 0.40. The cp-logistic label is deliberately soft; don't chase value
  sharpness against a deep-SF ruler. Blend spec and the full head/target/loss table are
  in `docs/model_heads.md` — read it before touching `train/losses.py`, loss weights, or
  replay-sample target building.
- **`policy_sf` trains on the OPPONENT's reply distribution**, labels queried at P1 after
  the net's move and POV-flipped. It is not a move-teacher, which is why upweighting it
  hurt.
- **`wdl` is the only value head used in MCTS**; `sf_eval` and `categorical` are
  auxiliary.
- **PID regret runs backwards from intuition**: SF picks randomly among moves within
  `wdl_regret` of best, so higher regret = weaker SF = model wins more. The controller
  LOWERS regret to raise difficulty and RAISES it as an airbag on low winrate. Training
  targets are always best-move based and never depend on which handicapped move SF
  actually played.
- **⚑⚑ `wdl_regret` MEASURES THE AGENT (net + search), NEVER THE NET. Do not read a regret
  move as a strength verdict without first checking what changed in the search.** The PID
  drives regret to hold curriculum winrate at `sf_pid_target_winrate` (0.5), and that
  winrate is produced by the net PLUS its search. So a search-config change injects a STEP
  into the regret series that is arithmetically indistinguishable from the net improving.
  - **MEASURED, not argued.** At iter 736 `gumbel_c_scale` went 0.025 → 0.1 (`7f4304db9`,
    2026-08-09 20:58) — worth **+245 Elo with the SAME weights on both sides**. Regret fell
    **0.0538 → 0.0334 (−38%) in ~25 iterations**. Over that SAME window the paired arena
    with the same search both sides and only the WEIGHTS differing read **−51.6 Elo**. ⇒
    **regret reported strong progress while the net measurably degraded.** The ruler gives a
    FALSE POSITIVE for exactly the failure mode it would be needed to catch: search tuned up
    while the net rots reads as success.
  - ⇒ **regret is a net signal ONLY while the search config is FROZEN.** Frozen since
    2026-08-09 20:58. Touching `gumbel_c_scale`, `gumbel_policy_temp`, `mcts_simulations`,
    `gumbel_topk` or the root transform VOIDS the series and needs a fresh baseline.
  - **A restart that restores PID state RE-INJECTS the step.** `salvage_restore_pid_state:
    true` from `pre_search_authority_20260809` restored regret **0.0550** (calibrated at
    `c_scale` 0.025; old run iter 672 read 0.0549 — exact fingerprint) into a run using 0.1.
    The 2026-08-11 run's **0.055 → 0.025 over iters 1-39 is the SAME +271 Elo search gain
    being absorbed a SECOND time**, not learning.
  - **DYNAMIC EQUILIBRIUM: a continuously improving net NEVER settles at setpoint.** The
    controller lands a persistent positive winrate OFFSET (379f6: mean **0.5035**, last-200
    **0.5057** against a 0.500 target) and regret ramps at the rate that matches. So "wait
    for it to equalize" is wrong — the offset and the ramp ARE the signal, and a winrate
    sitting exactly at 0.500 with flat regret means the net is FLAT.
  - **Difficulty is 2-D: regret AND `sf_nodes`.** Check nodes before reading regret — a
    regret drop paid for by a node drop is not a difficulty rise. (379f6: `sf_nodes` pinned
    at 75000 for all 862 iterations, so that run's ramp was pure difficulty.)
  - **PROVISIONAL scale, NOT a calibration: Δregret 0.0095 ≈ 51.6 Elo ⇒ ~5.4 Elo per 0.001
    regret**, from two confounded endpoints (neither equilibrated; the eras also differ in
    the diff_focus clip regime). Task #170 owes the real one. It is stated here as a
    FALSIFIER for that experiment, not as a number to quote.
  - **"Better than last time" is cleared by ZERO improvement**, because last time was
    degrading. A regret floor below 379f6's 0.033–0.036 means "we did not lose the 51.6
    Elo", not "we gained".
  - ⚑ **Do NOT tune search toward the play optimum for arena numbers.** `c_scale` play
    optimum is ~0.2, the target-quality `better_in` peak is **0.1**, and production runs 0.1
    deliberately. Moving to 0.2 would raise agent strength, lower regret, look like progress
    on BOTH instruments, and tell us nothing about the net — while giving up target quality.
- `selfplay.record_fast_ply_value` is OFF in production — tried and REVERTED for trunk
  dilution. Check the ledger before re-enabling.
- Production input is 175 planes (`v2_threats`); `v1`'s 146 planes are legacy. Production
  policy output is the compact `lc0_1858` encoding, though search still uses 4672 action
  ids. Use the shared device-cached lookups in `moves/torch_maps.py` — don't add
  per-module `lru_cache` copies.
- Production optimizer is `aurora` with `matrix_optimizer_scope: mlp_out`.
- Step budget uses views-targeting (`train_views_per_position`), so steps scale with
  ingest volume; `train_window_fraction` is the legacy mode and the ingest-drought floor.
- GPBT is wired up but effectively off — the production config pins everything.

## Code conventions

Python 3.10+ with `from __future__ import annotations`; type hints on functions and
dataclasses; tests in `tests/`. 4-space indent; `snake_case` functions and modules,
`PascalCase` classes, `test_*` tests; imports grouped stdlib / third-party / local.
Write code that reads like the code around it.

Add or update tests with every behaviour change, and prefer deterministic units around
encoding, replay, MCTS and training targets. For distributed or selfplay-path changes
also run `tests/test_e2e_smoke.py` (`-k gumbel_selfplay_smoke` for the search path) — it
boots the real selfplay → replay → train → checkpoint chain, so it catches wiring that
unit tests mock away. The `scripts/e2e_distributed_smoke_gumbel.sh` the old `AGENTS.md`
named alongside it has not existed since `dcb31fdf2` (2026-04-18); the instruction
outlived the script by four months, which is the drift this file exists to stop.

Run `./scripts/lint.sh <paths>` after editing, **and `./scripts/lint.sh` with no
arguments before committing**; the gate is kept at zero findings repo-wide with no
baseline. Naming paths narrows basedpyright to those files, so a path-scoped run
structurally cannot see breakage the change caused in a file it did not open — that is
how `main` went red. Every invocation *without* paths (bare, `--changed`, `--fast`,
`--deep`, `--slop`, `--all`) uses CI's whole-repo scope.
Fix a new finding in the same commit or disable the rule in
`pyrightconfig.json` if the whole category isn't worth the ceremony — there is no
deferral queue, don't recreate one. basedpyright ignores mypy-style
`# type: ignore[...]`; use `# pyright: ignore[reportRuleName]`. Never write a suppression
whose validity depends on the installed numpy/torch version — rewrite the code
version-proof instead.

## Reviews and pull requests

Optimize for end-state quality, not the cheapest diff. When a review surfaces an
improvement, decide it now: either make the change or say why it isn't worth making.
"Deferred to later" is an unresolved decision rotting in a comment. "Premature
abstraction" is a valid reason to skip; "it touches more files than I expected" is not.
State the call and the reasoning explicitly.

Open PRs ready for review, not draft, unless asked. Every PR gets a manual correctness
review before it counts as done, with the verdict recorded in the PR conversation or
session summary.

**⚑⚑ `reviewDecision` IS A FALSE NEGATIVE — DO NOT USE IT TO ASK "HAS THIS BEEN REVIEWED".**
It is set only by a FORMAL submission (`APPROVED`/`CHANGES_REQUESTED`), and reviews here are
almost never submitted that way: **agent reviewers post their verdict as an ordinary PR
COMMENT**, and Codex submits as `COMMENTED`. Neither sets the field. Measured 2026-08-10:
all five open PRs read `reviewDecision: ""` while three carried full independent reviews,
one of them an APPROVE-WITH-CHANGES whose four blocking findings were already closed — and
that PR was wrongly reported as unreviewed and held. Read the bodies instead:

```bash
gh pr view <N> --json comments,reviews          # human/agent verdicts, in comment bodies
gh api repos/{owner}/{repo}/pulls/<N>/comments  # ⚑ INLINE review threads — NOT in the above
gh pr view <N> --json commits                   # "close the review's ..." = findings addressed
```

**⚑⚑ THE SECOND COMMAND IS NOT OPTIONAL, AND IT IS THE SAME BUG ONE LEVEL DOWN.**
`--json comments` returns ISSUE comments and `--json reviews` returns review BODIES; neither
returns the inline, line-anchored review threads, which is where Codex puts **every** finding.
Measured on PR #381: `gh pr view 381 --json comments,reviews` yields 5 issue comments and one
review whose body is 621 characters of pure boilerplate — while the REST comments endpoint
returns **10 P2 findings** on `scripts/search_gain_probe.py`. Reading only the two `gh pr view`
fields there reports "reviewed, zero findings" about a review with ten. Same shape as
`reviewDecision`: the field you queried is populated and truthful, and it is not the field the
question was about.

Judge three axes separately, because they move independently: **reviewed?** (comment bodies
*and* inline threads) · **findings closed?** (later commits) · **CI meaningful?** (green is
worthless if the base advanced — CI does not re-run on base changes, `strict: false`).

**The Codex bot is INSTALLED and answers `@codex review` on demand — it does NOT review
automatically.** (The line that used to sit here said it "was disabled 2026-07-11"; that went
stale. On 2026-08-10 it reviewed six PRs — #373/#374/#377/#379/#381/#382 — each 7-21 min after
an explicit `@codex review` comment, and did NOT review #388, opened later the same day with no
trigger.) So: still don't wait for a bot review, but do summon it —
`gh pr comment <N> --body "@codex review"` — as a second reviewer from a different model family,
alongside a Claude review agent rather than instead of one. ⚑ It is credit-metered, and its
review BODY is always the same boilerplate template regardless of what it found: per that
template it comments when it has suggestions and reacts 👍 when it does not. So the body tells
you nothing at all — a Codex review's content is exactly its inline threads, and a template
with no threads under it is a genuine zero-finding pass. A presence check is not a value read,
here as everywhere else.

**THE REVIEWER MUST NOT BE THE AUTHOR.** A subagent that wrote a change does not review
it — spawn a SEPARATE agent whose only job is the review, and give it the PR, not the
authoring agent's summary. Self-review reliably passes: the author re-reads the reasoning
that produced the code, so a wrong premise gets confirmed rather than caught. The failure
this rule exists for is real and recent — PR #267's J9 "fix" replaced a working
`str.replace(...)` with `str.removeprefix(...)`, which silently cannot strip the NESTED
`module._orig_mod.*` key `AveragedModel` produces. Its own tests passed, because they only
inspected the output where the prefix happened to be leading. A separate reviewer caught
it by patching `main`'s helper to the PR's semantics and watching `main`'s own test fail.

Give the reviewing agent the standing bias: **this codebase's signature defect is a value
that is accepted and then silently ignored** — a knob that never reaches the worker, a
metric that does not mean what its name says, a gate that cannot fail. So the review
question is not "is this code correct" but "does this take effect on the production path,
and what observation would prove it did".

When the author is the main session rather than a subagent, the same rule applies: spawn a
review agent. If that is genuinely impractical, say in the PR that the review was
self-performed — an unlabelled self-review is the thing to avoid, not the occasional
justified one.
