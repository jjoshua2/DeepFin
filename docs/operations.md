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
`scratchpad/live_read/monitor/`); `scripts/ratchet_loop.sh` — the daily strength
ratchet (log `scratchpad/ratchet_loop.log`).

## The nightly pause window

`scripts/pause_window.sh <cmd>` runs a job with the trial parked and selfplay drained,
and `scripts/ratchet_loop.sh` wraps the whole daily ratchet in one. Measured 2026-08-09
(ledger: "the ack-gated pause window"): 1.89× the arena games in 0.46× the wall time,
and it costs the loop ~7 iterations against ~15 for the same ratchet run beside
training. Default ON; `CAE_RATCHET_PAUSE_WINDOW=0` opts out.

The **order is the mechanism**: snapshot each worker log's byte offset → touch
`pause.txt` → **wait for `.paused_<trial>.ack`** → SIGTERM the workers → run → clear the
marker. `_revive_fleet` runs inside the ingest phase, so it is inert only while the
trial is parked; draining before the ack just gets the fleet relaunched and the window
silently measures a contended machine.

### ⚑ Merging does not deploy it, and the ORDER of the two restarts matters

`train.sh:156-159` starts `ratchet_loop.sh` **once**, only if `pgrep` finds none, and
`stop()` never stops it (`watchdog_loop.sh` is started the same way, and likewise never
stopped). A long-running bash keeps the file it was launched with, so after a merge both
loops keep running the OLD scripts until they are replaced. `train.sh restart` does
**not** help: they are still running, so the `pgrep` guards skip them.

⚑ **THE HAZARD IS A MIXED GENERATION, NOT A CLOBBERED FILE.** Bash has already parsed
each `while true` loop as one compound command, so editing the files under the running
pids does not corrupt them — they simply keep the old behaviour forever. But everything
they *invoke* — `train_watchdog.py`, `recover_stall.sh`, `pause_window.sh`,
`daily_gate_ratchet.sh` — is re-exec'd fresh every poll and flips **immediately**. So the
moment the tree carries this change you have a **new `train_watchdog.py` returning exit 6
driven by an old `watchdog_loop.sh` with no exit-6 branch**: an abandoned window's marker
is alerted on and *never cleared*, which is exactly the failure the branch was added to
fix. Restart the ratchet loop before the watchdog loop and the new nightly failure mode is
armed with its only recovery disabled.

**Order (each step gated on the previous):**

1. **Reconcile the live tree's uncommitted edits first.** Commit them on the live branch,
   or diff them against `main` and discard them deliberately — **never** `git checkout --`
   or `git stash`, which CLAUDE.md bans in this tree and which would discard the watchdog
   currently running. A `git pull`/`merge` will otherwise abort with "your local changes
   would be overwritten", and the reflex fix is the forbidden one.
2. **Merge, and require `git diff HEAD...origin/main` to be EMPTY** before restarting
   anything. A merged PR is not the same as code on the live branch.
3. **Restart `watchdog_loop.sh` FIRST**, so the exit-6 consumer is live before any
   producer can create an owned marker.
4. **Then `ratchet_loop.sh`**, and only **between** polls — `tail scratchpad/ratchet_loop.log`
   for `starting daily ratchet`. Killing it mid-ratchet orphans the arena and can leave the
   pause marker held.

```bash
pgrep -af 'scripts/watchdog_loop.sh'         # step 3
kill <pid>; setsid nohup bash scripts/watchdog_loop.sh < /dev/null > /dev/null 2>&1 &

pgrep -af 'scripts/ratchet_loop.sh'          # step 4
kill <pid>; setsid nohup bash scripts/ratchet_loop.sh < /dev/null > /dev/null 2>&1 &
```

**Prove each from its own log line, never from a pid** — a new pid proves only that a
process restarted, not which file it is running.

```bash
# the watchdog half: a line only the new file can emit
grep -nE 'CLEARING ABANDONED|PAUSE-ABANDONED|AUTO-RECOVER' scratchpad/watchdog.log | tail
# the ratchet half: BOTH of these, not either
grep -n '\[pause-window\]' scratchpad/ratchet_loop.log | tail -20
grep -n 'pause marker(s) detected' /tmp/chess_training.log | tail -3
```

The `[pause-window]` lines are the wrapper announcing itself; the `pause marker(s)
detected:` line is the **trial's own** confirmation that it saw the marker and parked.
Neither alone is proof: the wrapper logs before it knows whether anything parked, and the
trial's line also appears for an operator pause.

### Env knobs

All default sane; none is set in production.

| variable | default | what it does |
|---|---|---|
| `CAE_RATCHET_PAUSE_WINDOW` | `1` | `0` runs the ratchet beside training (the pre-2026-08-09 behaviour). |
| `CAE_RATCHET_PAUSE_MAX_FAILS` | `2` | Wrapper failures (**exit 7** — 3 is the arena's no-pairs status per `ratchet_common.sh`) tolerated per calendar day before the loop stops asking for a window and takes the contended reading, so the day still gets measured. State: `data/ratchet/pause_window_fails`. |
| `CAE_PAUSE_ACK_TIMEOUT` | `1800` | Seconds to wait for `.paused_<trial>.ack` after setting the marker. On timeout it aborts **without draining** — draining with revive live is worse than not trying. |
| `CAE_PAUSE_STALE_ACK_TIMEOUT` | `180` | The shorter clock used when an ack for this trial already exists. That case provably cannot resolve (`_wait_if_paused` guards on an `announced` flag and will not re-ack), so waiting the full 1800s is 30 min of parked production per poll. |
| `CAE_PAUSE_DRAIN_TIMEOUT` | `180` | Seconds to wait for the workers to exit after SIGTERM. A survivor **aborts the job** rather than running it beside a live worker. |
| `CAE_PAUSE_POLL_SECONDS` | `5` | Poll interval for both waits. |
| `CAE_PAUSE_ALLOW_NO_WORKERS` | `0` | `1` proceeds when `pgrep` matches no workers. By default that **refuses**, before the marker: a fleet that is down and a pattern that has drifted from the workers' argv look identical here, and the second means the job runs against a full fleet and is filed as uncontended. |
| `CAE_PAUSE_JOB_KILL_TIMEOUT` | `30` | Seconds to wait after SIGTERMing the job's **process group** on an interrupt, before escalating to SIGKILL. The group, not the child: the job is a shell running the arena under `timeout`, and signalling only the child leaves the arena running. |

### A marker held by a dead window

`train_watchdog.py` reports `PAUSE-ABANDONED` (exit **6** — 5 is `CRASHED`, taken by #371) for a marker that names its owner
(`pid=`, written by `pause_window.sh`) whose owner is gone, or which has been held past
`--pause-max-minutes` (default 180). `watchdog_loop.sh` then removes that marker — and
only that kind: `graceful_restart.py`'s marker carries no `pid=` and is never touched,
however old. `WATCHDOG_AUTO_RECOVER=0` disables this along with stall recovery. To
inspect one by hand, `cat runs/pbt2_small/tune/pause.txt` — it names the pid, the start
time, and the job.

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
every account on the box and lands in shell history. The two are mutually exclusive —
passing both used to exit 0 having silently used the environment one, leaving you with
an account that does not accept the password you typed. `users.json` holds PBKDF2 hashes
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

## Volunteer workers — registration and bans

The server can accept workers from people who are not you. It is **off by
default** and stays off until someone flips it:

```yaml
tune:
  worker_self_register: true   # DEFAULT false
```

**Restart-gated, and not a live reload.** `create_app` captures the flag once,
and the driver bakes `--worker-self-register` into the server's command line at
launch, so it takes a full `run.py` restart to change — not a trial restart.
The key is in the yaml allowlist, so adding it does not break the all-or-nothing
live-reload validator; setting it on a running server simply does nothing until
the next restart. The first server log line tells you which world you are in:

```
worker self-registration is disabled (unknown usernames are refused)
worker self-registration is ENABLED (unknown usernames will create accounts)
```

**Trust on first use.** With the flag on, the first client to present an unknown
username creates that account, and every later use of the name requires the same
password. Only PBKDF2 material is stored — never the plaintext.

If two clients first-connect on the same name at once, one registers and the other is
verified against what was just stored: **losing the race is an ordinary sign-in, not a
free pass.** Two workers deployed with the same shared credential both get in; a client
racing a name with a different password gets 401 and is charged a failed sign-in.

**`users.json` is locked across processes.** `manage_users` and the server both write it
(the server only for self-registration), so every read-modify-write takes an `flock` on
`users.json.lock` first. Without it, a `manage_users disable` landing inside a
registration was written back out of the copy loaded before it — the CLI exited 0 and the
revoked worker kept uploading. If the lock is held past 10s the server answers **503
`users db busy, retry`** rather than writing unlocked, and the CLI says which pid holds
it. The kernel releases an `flock` when its holder dies, so there is no stale lock to
clear after a crash.

A corrupt or wrong-shaped `bans.json` **fails open** — nobody is banned — because a LAN
fleet must not lose its server to a bad ban file. It now logs a WARNING saying so; an
absent file stays silent, since that is the normal case.

**Minimum 8 characters** (the same policy as `manage_users`, from the same
`auth.check_new_password`). A shorter one is refused **400**, not 401 — the
credential is not wrong, it is unacceptable, and a 401 would send the client
into a retry loop with the same password. It does not count against the
failed-sign-in throttle either, so a volunteer cannot lock themselves out with a
typo. Accounts that already exist with shorter passwords keep working: the rule
is on setting, not on checking.

**There is no password reset, by design.** A volunteer who forgets theirs
registers a new name. That removes the entire account-recovery surface, which is
the part of a volunteer auth system that actually gets abused.

**Throttles.** Per IP: 5 new accounts per hour, and 20 failed sign-ins per 5
minutes (cleared by a success, so a worker that fixes its credential is not held
out by its own retries). These are in-memory and therefore depend on uvicorn
staying single-process — see the A18 note; adding uvicorn workers would silently
multiply every limit by the worker count.

**A quarantined shard's `.reason.txt`** now carries the server's machine-readable
`reason_code` after the human reason (`... (reason_code=worker_too_old)`), so a rejection
can be classified without parsing prose that is free to change.

### Banning someone

```bash
PYTHONPATH=. python3 -m chess_anti_engine.server.manage_users \
    --users-db runs/pbt2_small/server/users.json \
    ban --username spammer --ip 203.0.113.7 --reason "uploading garbage"

PYTHONPATH=. python3 -m chess_anti_engine.server.manage_users \
    --users-db runs/pbt2_small/server/users.json list-bans
```

`unban` takes the same flags. Bans live in `bans.json` beside `users.json` (so
under `runs/`, untracked), are enforced at authentication — one ban costs the
identity every authenticated route at once — and take effect on the **next
request**, with no server restart.

Ban the **IP as well as the username** when the flag is on. A username-only ban
against trust-on-first-use just means they register a new name.

`ban` is not `disable`. `disable` is for an account you created and control;
`ban` is for a volunteer identity, and it is checked before any password work,
so a banned client cannot spend the server's PBKDF2 budget.

### Quarantining a banned volunteer's shards

**A ban is not retroactive** — it stops future uploads and leaves everything
already ingested in place. Shards carry a `username` attribute, so a banned
volunteer's contributions can be found and quarantined after the fact. Use the existing
tooling rather than writing new tooling:

- `scripts/quarantine_desync_shards.py` already builds a quarantine **manifest**
  from a predicate over shards, and `scripts/build_era_probe_set.py` refuses any
  shard a manifest names — so a manifest is the mechanism a downstream consumer
  already honours. Point its predicate at the banned `username` attribute.
- The server also maintains `<server_root>/quarantine/`, with client reports
  under `quarantine/client_reports/` from `/v1/report_bad_shard`.

If the shards are already in the replay window, the window ages them out on its
own schedule — quarantining stops them being re-drawn and re-shared, it does not
un-train them.

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
