# Operations

Detailed procedures. `CLAUDE.md` carries the always-on rules and points here.

## Driving training

`scripts/train.sh` manages the PID file, log, and Ray cleanup.

```bash
./scripts/train.sh start                   # auto-resumes if $WORK_DIR/tune state exists; else fresh
./scripts/train.sh start --fresh           # force a fresh run (ignore prior tune state)
./scripts/train.sh stop                    # SIGTERM + ray stop + orphan worker sweep
./scripts/train.sh restart                 # stop + start (auto-resume same as start)
./scripts/train.sh status | log            # status / tail -f log
```

`start` auto-passes `--resume` when `$WORK_DIR/tune/experiment_state-*.json` exists.
Without that, restarting after a stop silently drops the running trial and spawns a
random-init one. To abandon the current trial's state, pass `--fresh` or use
`salvage-restart` from a good pool; never `rm` the tune dir while a run is live.

**Graceful pause before killing PBT**: `python3 scripts/graceful_restart.py` creates
`pause.txt` in the tune dir; active trials finish the current iteration, hold, then the
script restarts cleanly. Use it before a restart that would otherwise orphan a
mid-iteration trial.

**Observers** (auto-started by `train.sh start`, one instance each):
`scripts/watchdog_loop.sh` — stall detection with auto-recovery (confirmed stall →
`scripts/recover_stall.sh`; log `scratchpad/watchdog.log`); `scripts/monitor_fen.sh` —
cadenced FEN-panel reads + the seed retire/probation step (log
`scratchpad/live_read/monitor/`).

## The worker credential

The distributed selfplay server authenticates workers, and it binds `0.0.0.0`. The
password is **not** in any tracked file — `configs/*.yaml` carries only the username.
Resolution order (`chess_anti_engine/server/secrets.py`, first hit wins):

1. the environment variable **named by** `distributed_worker_password_env`, if the
   config sets one — explicit beats ambient, so a stray export cannot outrank it;
2. `$CAE_WORKER_PASSWORD` — the one-line step for everyone else;
3. `$CAE_WORKER_PASSWORD_FILE`, else `.secrets/worker_password` (gitignored, must not
   be group/world readable).

With all of them empty the driver **refuses to start**. That is deliberate: it
provisions an account, so an empty-password fallback would create a real login on a
public listener rather than a broken server. The boot log always names the source it
used:

```
[run_tune] worker credential source: $CAE_WORKER_PASSWORD
```

Restarting the live run onto this code is one line:

```bash
export CAE_WORKER_PASSWORD='...'   # in the shell that runs scripts/train.sh
```

The value that used to sit in the yaml is **disclosed** — it is in the git history and
stays there. Rotating it is a separate operator-gated step (task #161).

### Password policy

Passwords must be **at least 8 characters** (user policy, 2026-08-05). Enforced where a
password is *chosen* — `manage_users add` / `set-password` on all three input routes,
and volunteer self-registration — never where one is *checked*: `verify_password` is
untouched, so **existing accounts with shorter passwords keep working**, and raising the
bar cannot lock the fleet out mid-rotation. The boot path is the deliberate exception:
a short credential in the environment **warns and still boots**, because a length rule
that stops production from starting is a worse outage than the weak password it was
guarding.

```bash
export WORKER_PW='...'
PYTHONPATH=. python3 -m chess_anti_engine.server.manage_users add volunteer --password-env WORKER_PW
```

Use `--password-env`, not `--password`: an inline password is visible in `ps auxww` to
every account on the box and lands in shell history. `users.json` holds PBKDF2 hashes
only and is written `0600` regardless of umask — it is writable-equals-auth-bypass, not
merely readable-equals-disclosure.

## Salvage — warm-start fresh trials from past checkpoints + replay

```bash
./scripts/train.sh salvage-export --top-n 3 [--out DIR] [--metric KEY]
#   → data/salvage/<run-id>_<ts>/{manifest.json, seeds/slot_NNN/{trainer.pt,pid_state.json,replay_shards/}}

./scripts/train.sh salvage-restart data/salvage_iter37
#   stops, then starts with the pool activated via CLI flags.
#   Defaults: restore PID + full trainer state, keep GPBT-sampled LR, don't reinit volatility.
#   Toggles: --no-pid, --no-optimizer, --reinit-volatility, --donor-config.
```

Salvage is driven entirely by CLI flags (`--salvage-seed-pool-dir`,
`--salvage-restore-*`) — no yaml edit needed to activate or disable it. When to salvage:
after a bad exploit, a run that regressed, or to rebase onto a better-regret checkpoint.
A pool is a one-shot seed; once trials are past startup it plays no further role.
`./scripts/train.sh best-save | best-list` manages named good-checkpoint pools under
`data/best_pools/`.

## Blind-spot FEN seeding — the active data lever

Seeds selfplay openings from positions the net misplays. The active list is
`selfplay.opening_fen_list_path` in the live yaml, delivered via the server dole
(`opening_fen_dole_per_iter`), selfplay-only and PID-safe. The path is live-reloaded —
no restart to change lists.

**Removal is automatic.** `scripts/blindspot_retire_step.py` (run on cadence by the
monitor_fen loop) retires seeds after 2 consecutive AWARE reads, runs probation re-feed
on retirees that regress, writes a new versioned list, and repoints the yaml itself.
Don't hand-edit the active list for removals.

**Feeding new seeds is manual and ledger-gated** — it is a data-affecting change, so one
per readout window with a ledger entry.

```bash
PYTHONPATH=. python3 scripts/blindspot_feed_step.py --batch <vetted-file> --tag <label>
```

It dedupes against the active pool and the retired store, writes a fresh versioned file,
validates, and repoints the yaml. `--dry-run` first. Loader caches are path-keyed —
**never edit a list in place**. Per-seed dose control via `# weight=N` line markers.

New seed material comes from mining external-match losses
(`scripts/mine_blindspot_seeds.py --pgn`; default bakes blunder + deep-SF first-refute
ply into the seed — `--no-append-refute-ply` for bare blind-spot terminals). `--existing`
is for incremental *new-hole* growth only: dedup keys the pre-blunder blind-spot for both
bare and post-refute lines, so a bare list there will not re-emit upgraded seeds for those
holes. To upgrade known bare holes, re-mine from PGN without them in `--existing`, write a
new versioned list, and feed it as a replacement — vetting/gating still applies.

## Static analysis

`./scripts/lint.sh <paths>` — default gate is ruff + basedpyright + vulture, a few
seconds, kept at **zero findings repo-wide with no baseline**. CI gates basedpyright on
the whole repo.

```bash
./scripts/lint.sh                    # ruff + basedpyright at CI's scope — run before committing
./scripts/lint.sh --changed          # changed/untracked .py (basedpyright: whole repo)
./scripts/lint.sh --deep [paths...]  # + skylos + ruff cleanup report (advisory, ~40s)
```

**Naming paths is the only thing that narrows basedpyright** — 2 files analyzed in ~1.4s
versus 544 in ~32s whole-repo. Every invocation without paths (bare, `--changed`,
`--fast`, `--deep`, `--slop`, `--all`) runs it at `pyrightconfig.json`'s whole-repo scope,
which is what CI runs. A path-scoped run cannot see breakage the change caused in a file
it never opened; that is how PR #295 turned `main` red. Run the no-argument form before
committing.

`--changed` narrows **ruff and vulture** to the changed `.py` files, falling back to the
full default list when the change collected none. It is a scope *swap* for basedpyright,
not purely a widening: a changed `.py` outside `pyrightconfig.json`'s include set is no
longer type-checked by it. `setup.py` was the only tracked file in that position and is
now in the include set, so there is no in-tree instance left — but a new top-level `.py`
would need adding there too.

In a **fresh worktree the whole-repo form is not clean until the C extensions are
built**: without the three in-place `.so` files it emits 120 spurious
`reportMissingModuleSource` errors (`chess_anti_engine.encoding._lc0_ext`,
`chess_anti_engine.mcts._mcts_tree`, `chess_anti_engine.encoding._features_ext`), which
all disappear after `python3 scripts/build_production_extensions.py`.

`--deep` sweeps the not-yet-gated ruff groups (B, NPY) as a cleanup shopping list; only
the needs-judgment tail remains there (B905 `zip strict=`, B008 call-in-default, NPY002
legacy-RNG — never autofix, NPY002 changes RNG streams). Promote a rule into the
`pyproject.toml` gate once it reaches zero findings.

Config lives in `pyproject.toml` (`[tool.ruff.lint]`, `[tool.vulture]`) and
`pyrightconfig.json` (rules we'll never fix are disabled outright — ML-typing noise like
`reportAny`, `reportMissingTypeArgument`).

Lint tool versions are exact-pinned in the dev extras so local and CI agree; the weekly
`lint-canary` workflow runs latest tools non-blocking to surface upcoming breakage. To
bump: upgrade local, run the gate, commit new pins.
