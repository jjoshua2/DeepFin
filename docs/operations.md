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

**`$CHESS_ANTI_ENGINE_LIVE_CONFIG` — which yaml the offline instruments treat as
production.** `train.sh` exports it (absolute path to `$TRAIN_CONFIG`, default
`configs/pbt2_small.yaml`), and ⚑ **that export reaches training and its observer
descendants ONLY.** `./scripts/train.sh start` runs the script in a SUBPROCESS, and a
subprocess cannot modify its parent, so the shell you launched training from still has
the variable unset. Read that as the normal case, not the exception: set it by hand in
**every** shell — including the one that started the run — that runs
`scripts/audit_targets.py`, `scripts/arena_standard.py --search-shape training`,
`scripts/value_regret.py` or `scripts/probe_policy_targets.py`:

```bash
export CHESS_ANTI_ENGINE_LIVE_CONFIG=/abs/path/to/live/tree/configs/pbt2_small.yaml
```

⚑ This is not optional hygiene. CLAUDE.md mandates a `git worktree` for branch work,
and a worktree's in-tree `configs/pbt2_small.yaml` is stale by construction — the live
tree is the only writer and its edits are routinely uncommitted (`origin/main` carries
none of `gumbel_policy_temp` / `gumbel_target_max_visit_cap` /
`gumbel_target_untempered_prior`, which the live yaml sets). With the variable unset,
`audit_targets.py` now **refuses** to emit rows (d)/(e) rather than score the wrong
search under the heading "production training target"; pass `--allow-stale-config`
only when you deliberately mean a non-production config, and note that it stamps the
per-position dump `config_authority.authoritative = false`.

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

**⚑ Ops scripts now operate on THE TREE THEY LIVE IN.** `monitor_fen.sh`,
`recover_stall.sh`, `run_bootstrap_512x16.sh` and **`bank_rolling_checkpoints.sh`** used
to `cd` to (or hardcode) an absolute root, so they drove the main checkout no matter where
they were launched from; they now derive it with `cd "$(dirname "$0")/.."`. Launched from
`train.sh` / `watchdog_loop.sh` in the repo root this is identical behaviour — but running
one **out of a `git worktree` now drives the worktree**, where `runs/` does not exist and
the script will find no trial. Given the standing "use a worktree for all branch work"
rule, run these from the main checkout, or `cd` there first. (`ratchet_common.sh` is
*sourced*, so it uses `${BASH_SOURCE[0]}`, not `$0`; `RATCHET_ROOT` / `WATCHDOG_ROOT`
still override.)

⚑⚑ **`bank_rolling_checkpoints.sh` is the one where that silently cost you something**, so
it now refuses instead: from a tree with no `runs/pbt2_small/tune` it prints `[bank] ERROR:
no tune dir at <path>` and exits **2**, where before it printed nothing, created an empty
`data/salvage/rolling`, and — with `ONCE` unset, i.e. in production — looped on that
forever. It is the rolling half of the **revert points** the experiment protocol depends
on, and Ray prunes the originals meanwhile, so a silently-empty bank is only discovered at
the moment a rollback is needed. `TUNE_DIR=<path>` overrides the location.

Engine-running tools
(`blindspot_*`) do NOT have this problem — they discover the published Stockfish through
`chess_anti_engine.utils.engine_discovery`, which falls back from the current checkout to
the **main** checkout via `git rev-parse --git-common-dir`, because
`e2e_server/publish/` is untracked runtime output that exists only where it was
published. `CAE_STOCKFISH` overrides.

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

### Ingest invariants and storage budgets

The server checks a shard's counter metadata for internal consistency before
accepting it, and **rejects terminally** on a violation (`rejected: true`, not a
retryable non-200) — an arithmetic inconsistency is a permanent property of
those bytes, so a retry would resend the same bad shard forever.

⚑ **This is corruption detection, not authentication.** A worker that wants to
forge these numbers can forge a self-consistent set, and there is no trusted
server-side source to check them against — the server never sees a game. What it
buys is that the *least*-protected channel stops being *silently* wrong: the
arrays are Blosc/zstd compressed so a corrupt block fails loudly, while `.zattrs`
is plain JSON where a flipped digit stays valid JSON with a wrong number — and
`wins`/`draws`/`losses` become the PID's curriculum winrate.

⚑ **Two predicates are deliberately NOT enforced**, and both look obvious:

- `wins + draws + losses == games` — **wrong**. `selfplay/finalize.py` counts
  W/D/L only for curriculum games (`not is_sp`) that are not blind-spot seeds
  (`not source.startswith("fenlist")`, since seeds force the net onto the losing
  seat and would drag the PID). Any shard with selfplay or seeded games has a
  strict inequality. This equality would reject nearly every real shard.
- `games <= positions` — **wrong**. diff-focus filtering can drop every position
  of a game.

Both are pinned as false-positive controls in `tests/test_shard_meta_invariants.py`
so they do not get "fixed" back in.

Storage budgets (all `create_app` parameters, all defaulted):

| knob | default | bounds |
|---|---|---|
| `quarantine_max_bytes` / `quarantine_max_entries` | 4 GiB / 200 | retained invalid uploads |
| `arena_max_body_bytes` | 1 MiB | one arena result body |
| `arena_max_bytes` / `arena_max_entries` | 256 MiB / 5000 | whole arena sink |

Each budget is **ONE GLOBAL CEILING for the whole sink**, across every user and
trial — that is the only thing bounding disk. Renamed from `arena_user_max_*` in
#407 precisely because it is no longer per user: a per-user budget's ceiling is
`users × budget`, and `worker_self_register` makes the user count caller-controlled,
so the "per-user arena inbox" this table used to describe had no bound at all.

Eviction is **largest-bucket-first, oldest-first within it** — the oldest entry of
whichever user currently holds the most, measured in whichever dimension is over
(bytes when the byte ceiling binds, entries when the count one does). Oldest-first
globally, which this said until #407, made the sweep a diagnostic-destruction
primitive: any one worker's flood evicted every other worker's evidence. Fairness
here is the choice of victim under the ceiling, **not** a second quota — with a
single bucket the behaviour is still exactly oldest-first.

Newest-worth-keeping is still the rationale within a bucket: a quarantined shard
diagnoses a defect happening *now*. Entry counts are bounded as well as bytes
because the growth mode is many small unique files — a byte budget alone lets an
account burn inodes.

⚑ **Known residual, live only if `worker_self_register` is on** (it is off by
default and absent from production's yaml): because the ceiling is a fixed entry
count, an attacker minting more buckets than that count forces someone to hold
zero, and largest-first zeroes the legitimate heavy user first. Measured at 5
accounts — the per-IP hourly registration limit — a victim goes to **0 retained**.
The fix belongs at the identity layer, and a global account cap only closes it if
`account_cap < entry_ceiling`.

### Requiring a lease for uploads

```yaml
tune:
  require_worker_lease: true   # DEFAULT false
```

⚑⚑ **Do NOT set this on the in-tree fleet — it takes ingest to ZERO.** The
driver launches every worker with `--trial-id`, which sets `fixed_trial_id`,
which skips lease negotiation entirely. A driver-launched worker therefore
*structurally* never obtains or sends a lease id and is refused **403 on every
upload, forever**. Measured on this server: 821,818 uploads, zero leases ever
issued, one `/v1/lease_trial` request and it was a 401.

It is restart-gated like `worker_self_register`, so it detonates only after a
full `run.py` restart — and then the trainer starves against
`distributed_wait_timeout_seconds`. It is for a volunteer deployment whose
workers negotiate their own leases.

**Independently of the flag**, a lease id that IS supplied must now belong to
the authenticated account, be unexpired, and match the route's trial. Today's
driver-launched workers send none, so this is inert on the in-tree fleet.

⚑ **A lease is attribution, NOT trial isolation.** `/v1/lease_trial` honours the
caller's requested trial whenever it names a *published* trial, so an
authenticated worker still chooses its own assignment and then passes the
upload check legitimately. What the lease buys is that an upload is tied to an
issued, expiring, named grant that can be revoked — not a restriction on which
trial a worker can reach. Production publishes one trial, so the set it may
choose from is a singleton; this only becomes a real question if multi-trial
PBT is ever exposed to untrusted volunteers, and closing it then means making
the server assign the trial instead of honouring the request.

### Cleartext transport

The worker refuses `http://` to a **non-loopback** server and exits, because
HTTP Basic sends a reusable credential on every request and the same channel
carries the manifest sha256 that is the only integrity check on the model
checkpoint. Loopback is exempt, which is what the driver hands in-tree workers
(`distributed_server_host: 0.0.0.0` resolves to `http://127.0.0.1:<port>`).

⚑ **An off-box LAN worker will refuse to start.** `configs/pbt2_small.yaml`
publishes `distributed_server_public_url: http://192.168.1.212:45453`, and that
is exactly the shape the guard rejects. Until the server speaks TLS, start such
workers with `--allow-cleartext-http` (or `allow_cleartext_http: true` in
`worker.yaml`).

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
already ingested in place.

**Read `contributors`, not `username`.** `username` on a shard the server
compacted is `server_compactor` — the compactor merges several uploaders, so no
single one owns the output — and reading it was the documented procedure until
2026-08-12, which meant a ban had nothing left on disk to act on. Compacted
shards now carry a `contributors` list of
`{"username", "start", "count"}` entries over the shard's own row order, so one
contributor's rows can be excised instead of the whole shard being discarded.
⚑ **Trust `username`/`contributors` only when `provenance_verified` is true.**
The server stamps both from the **authenticated** account — never from the
uploader's own claim, or one volunteer could aim your ban at another — and sets
`provenance_verified: true` at the same time. It also **clears** any
`contributors` list the uploader shipped — that field is server-owned, so on a
raw single-uploader shard anything present came out of the tarball.

⚑ **The marker alone is not the evidence — the server-side watermark is.**
`.zattrs` travels *inside* the uploaded tarball, so a shard staged by a server
from **before 2026-08-12** carries whatever the uploader wrote there, and that
includes a hand-written `provenance_verified: true` next to someone else's
name. Nothing inside such a shard can be checked, because it predates the code
doing the checking. On its first boot after that date the server therefore
writes `<server-root>/provenance_migration.json`, recording every shard already
staged; those are re-seeded with a `null` contributor no matter what their
attrs claim, and only shards that arrived afterwards — and so went through the
stamp — are attributable. A null is honest; a possibly-wrong name gets the
wrong person quarantined.

Do not delete that file. If it goes missing the server re-snapshots on the next
boot, which marks the *then-pending* shards legacy and loses their attribution
— safe, but lossy.

The read is:

```python
if not attrs.get("provenance_verified"):
    continue                      # unverifiable; do not attribute to anyone
rows = attrs.get("contributors") or [
    {"username": attrs.get("username"), "start": 0, "count": attrs["positions"]}
]
rows = [r for r in rows if r["username"] is not None]
```

`contributors` is absent on raw single-uploader shards, where `username` is
still the answer. ⚑ Note the **last** line: a re-seeded shard can produce
`[{"username": None, ...}]`, which is a *truthy* list, so `contributors or …`
will NOT fall through to `username` — filter the nulls or the predicate gets
`None` handed to it.

Use the existing tooling rather than writing new tooling:

- `scripts/quarantine_desync_shards.py` already builds a quarantine **manifest**
  from a predicate over shards, and `scripts/build_era_probe_set.py` refuses any
  shard a manifest names — so a manifest is the mechanism a downstream consumer
  already honours. Point its predicate at the banned name using the
  `contributors`-then-`username` read above. ⚑ The predicate is shard-level, so
  it quarantines whole shards; the row ranges are recorded so a future
  row-exact excision tool has what it needs, but that tool does not exist yet.
- The server also maintains `<server_root>/quarantine/`, with client reports
  under `quarantine/client_reports/` from `/v1/report_bad_shard`.

If the shards are already in the replay window, the window ages them out on its
own schedule — quarantining stops them being re-drawn and re-shared, it does not
un-train them.

## UCI search options (match play)

The engine (`python -m chess_anti_engine.uci`) exposes its search parameters as
UCI options, so a TCEC-style config, cutechess-cli, or Arena can set them
without a code change. Every option below **except `MinibatchSize`** is also a
CLI flag on the same module; the UCI name and the flag set the same field.
`MinibatchSize` is UCI-only — `run_gumbel_root_many_c(target_batch=...)` starts
at the C-side default of 0 and only a `setoption` moves it, so there is no
`--minibatch-size` and passing one is an `unrecognized arguments` error. The
map that decides this is `_SEARCH_OPTION_ARG` in `uci/__main__.py`, and a
`None` there is asserted against the real parser
(`test_every_registry_option_has_a_named_startup_source`), so this paragraph
cannot drift from the code without a test going red.

**The rule this surface is built around: an option that cannot take effect is
never silently accepted.** The engine has five search paths and most knobs are
live on only some of them — `c_puct` and the `fpu_*` family are structurally
inert in a Gumbel search (`GumbelConfig.full_tree=True` makes the PUCT descent
unreachable), and the Gumbel shape knobs are equally inert on the PUCT walker
pool. So a `setoption` that lands on a knob the *live* path cannot read is
accepted, applied to the config, and reported:

```
info string CPuct set to 2.5 — NO EFFECT on the live search: CPuct does not
reach the gumbel path (classic single-thread Gumbel ...). Use `searchconfig`
to see every realized value.
```

### Reading back what the engine actually used

```
searchconfig
```

A non-spec command (no GUI sends it) that dumps every registered parameter with
its realized value and a LIVE/INERT verdict for the path that will actually
run:

```
info string searchconfig path=gumbel (classic single-thread Gumbel (Threads=1, UseVL off, 1 device))
info string searchconfig PolicyTemperature = 1.0 [LIVE]
info string searchconfig CScale = 0.025 [LIVE]
info string searchconfig CPuct = 1.75 [INERT] — CPuct does not reach the gumbel path (...)
info string searchconfig QVisitExpRoot = -1.0 [BRANCH] — every value < 0 is the same search: ...
info string searchconfig 11 live, 1 branch-pinned, 4 inert on path=gumbel
```

The path is read off the **live worker**, not off the options: `UseVL true`
against an evaluator without the slot API is accepted and then falls through to
classic Gumbel, and an options-derived answer would report a path that never
ran. Values come from `SearchWorker.realized_search_values()` — the same
objects the search reads, which for the PUCT family means the installed pool's
or chunker's own config and **not** the shared `GumbelConfig` a `setoption`
writes. That distinction is the difference between evidence and a restatement
of the request: sourced from the shared config, the readback printed
`CPuct = 99.0 [LIVE]` byte-identically whether or not the rebuild that delivers
it had run. Neuter the rebuild today and `searchconfig` says `CPuct = 2.5
[LIVE]` — the value the threads are descending on.

### Which path is live

| path | selected by | search |
|---|---|---|
| `gumbel` | `Threads 1`, `UseVL false`, one device | classic single-thread Gumbel C |
| `rpg` | `SearchParallel gumbel` + ≥2 devices | root-parallel Gumbel; Gumbel at the root, PUCT below it |
| `walker` | `Threads >1` | PUCT walker pool |
| `pucv` | `UseVL true` (+ slot-API evaluator) | single-thread batched-VL PUCT |
| `pucv_pool` | `UseMultiGpuPUCV true` + ≥2 devices | multi-GPU shared-tree PUCT |

### The options

Floats use `type string` (UCI has no float type — lc0's `PolicyTemperature`
convention) and are parsed and range-checked by the engine; an out-of-range or
unparseable value is refused with an `info string` and the previous value is
kept. Defaults are advertised from the value the worker was actually built
with, not a retyped constant.

That holds for the **first** `uci` too, which is the only one most GUIs send.
`uci` is answered on the reader thread from `startup_options` while the model
is still loading on the build thread, so the worker → options copy at the end
of `_build_engine` lands *after* the handshake a GUI sees. Until 2026-08-09 the
first handshake therefore advertised the registry defaults: launching with
`--c-scale 0.077 --topk 9 --policy-temp 1.5 --chunk-sims 777` advertised
`0.025 / 32 / 1.0 / 2048`, and only a second `uci` reported the truth.
`_startup_engine_options` now seeds the whole surface from the parsed CLI
before the build thread starts (`_SEARCH_OPTION_ARG` maps every registry field
to its argparse `dest`, and refuses to run if one is missing), and the copy
`_build_engine` takes is deferred until after the multi-GPU / RPG pool
installs, because `realized_search_values()` reads the PUCT family off the
installed descent object. `searchconfig` remains the authority for what a
*running* search is using.

| option | type | range | default | live on | notes |
|---|---|---|---|---|---|
| `PolicyTemperature` | string | 0.5 – 5.0 | 1.0 | gumbel, rpg | prior temperature (logits/T); >1 softens. **Costs search time when ≠ 1.0 — see below.** |
| `CScale` | string | 1e-4 – 100 | 0.025 | gumbel; rpg only if `CScaleRoot < 0` | descent value-transform scale |
| `CVisit` | string | 0 – 1e5 | 50.0 | gumbel; rpg only if `CVisitRoot < 0` | descent transform floor |
| `CScaleRoot` | string | -1 – 1000 | 7.0 | gumbel, rpg | root-only c_scale; <0 = use `CScale` |
| `CVisitRoot` | string | -1 – 1e5 | 900.0 | gumbel, rpg | root-only c_visit; <0 = use `CVisit` |
| `QVisitExpRoot` | string | -10 – 99 | -1.0 | gumbel, rpg — reported **`[BRANCH]`**, not INERT; see below | <0 = log root (sim-invariant); ≥90 = LINEAR root (exponent 1.0) |
| `HalvingDiv` | spin | 2 – 8 | 2 | gumbel, rpg | sequential-halving divisor |
| `Topk` | spin | 2 – 256 | 32 | gumbel, rpg | root candidates |
| `GumbelScale` | string | 0 – 5 | 0.0 | gumbel, rpg | root Gumbel-noise strength; 0 = deterministic |
| `ChunkSims` | spin | 32 – 1048576 | 2048 | all | sims per search chunk |
| `VLossWeight` | spin | 0 – 64 | 3 | all | virtual loss on in-flight leaves |
| `MinibatchSize` | spin | 0 – 8192 | 0 | **gumbel only** | C-side leaf-flush target; 0 = C default |
| `CPuct` | string | 1e-4 – 100 | 1.75 | walker, pucv, pucv_pool, rpg | PUCT exploration constant |
| `CPuctFactor` | string | 0 – 100 | 3.89 | walker, pucv, pucv_pool, rpg | 0 = fixed CPuct |
| `CPuctBase` | string | 1 – 1e9 | 38739.0 | walker, pucv, pucv_pool, rpg | log((N+base)/base) |
| `FpuReduction` | string | -10 – 10 | 0.33 | walker, pucv, pucv_pool, rpg | first-play-urgency reduction |

**Removed 2026-08-23: `QVisitExp`, `QVisitFloor`, `QGlobalScale`.** They drove
three DESCENT value-transform knobs on `GumbelConfig` that were never promoted —
absent from every config in this repo and from `PLAY_SEARCH_DEFAULTS`, sweep
leftovers from the root/descent split work. The axis that DID pay off is the
separate ROOT family (`CScaleRoot` / `CVisitRoot` / `QVisitExpRoot`), which is
untouched and is what production PLAY runs as the log root. The descent
transform is now fixed linear, which is the search every config here was already
running. A GUI still sending one of the three gets the normal unknown-option
handling instead of an option that pretended to tune something.

Names match `scripts/arena_standard.py --cand-gumbel` / `--ref-gumbel` field
names with the underscores removed and CamelCased (`c_scale` → `CScale`,
`q_visit_exp_root` → `QVisitExpRoot`), so a shape tuned in an arena transfers to
a UCI config by transliteration. The arena keeps snake_case because those flags
address `GumbelConfig` fields directly; the UCI side follows lc0's CamelCase.

**⚑ One exception, and it is the knob you are most likely to be transferring:
`policy_temp` → `PolicyTemperature`, not `PolicyTemp`.** The name follows lc0's,
which spells it out. This matters because unknown `setoption` names are ignored
in silence (the lc0/Stockfish convention, kept deliberately — see the rule
above), so `setoption name PolicyTemp value 2.5` prints nothing and the whole
match runs at 1.0. Measured:

```
setoption name PolicyTemp value 2.5          -> (no output)   policy_temp stays 1.0
setoption name PolicyTemperature value 2.5   -> info string PolicyTemperature set to 2.5
```

Transliterate mechanically for every other field; check this one by eye, or
send the `searchconfig` command — the readback prints every registered name and
its realized value, so a knob you cannot find in that dump is a knob the engine
will never hear you set.

### `QVisitExpRoot` is branch-pinned, not inert — and the readback says which

`searchconfig` reports this one `[BRANCH]`, counted separately from `[INERT]`.
The distinction is load-bearing and was got wrong once already.

- `[LIVE]` means **a `setoption` you send right now changes the next search** —
  not merely that the path *would* read the field if it were rebuilt. See
  "A `[LIVE]` row is a promise about `setoption`" below; it was false for four
  rows on the default path, and it is a test now.
- `[INERT]` means **the path cannot read this option at all** — `CPuct` under
  classic Gumbel. No value of it matters, and no companion knob rescues it.
- `[BRANCH]` means **the option does reach the search and does change it**, but
  the current value sits inside a branch whose members are all equivalent.

#### A `[LIVE]` row is a promise about `setoption`, not about the CLI

Four of the paths run a PUCT descent that reads `CPuct` / `CPuctFactor` /
`CPuctBase` / `FpuReduction` off a config **snapshotted when the helper was
built** — `WalkerPoolConfig` at pool construction, `PucvChunker._c_puct` /
`._fpu_red` at chunker construction. Assigning the shared `GumbelConfig` field
therefore does *not* reach those two descents, so `Engine._sync_cpuct_to_worker`
rebuilds whichever helper is installed: `_reinstall_rpg_if_active` for `rpg`,
`_install_multi_gpu_pucv_pool` for `pucv_pool`, and
`SearchWorker.rebuild_puct_helpers` for `walker` and `pucv`.

That last branch did not exist until 2026-08-09, and on the **shipped default**
(`--walkers default=2` → `walker`) the engine printed `FpuReduction set to 9.0`
and `searchconfig FpuReduction = 9.0 [LIVE]` while the pool kept descending on
`1.2`. Worse than dead: the dropped value landed **later**, whenever the next
unrelated option happened to rebuild — on the `pucv` path a subsequent
`VLossWeight 4` moved the tree `7024 → 4269` because an `FpuReduction` from
three commands earlier finally took effect, with no readback showing the
difference.

The guard is behavioural, not a predicate check:
`test_the_puct_family_reaches_the_pucv_descent_through_a_setoption` requires a
`setoption` to produce the **same** search the value produces at construction
(with the construction-time delivery asserted first, so a blind ruler cannot
pass it), and deleting either rebuild branch turns it red. `rpg` cannot be
driven behaviourally without ≥2 devices and is a stated gap, covered by
inspection only.

**What a `[LIVE]` row does not promise: that the change is retroactive.** A
`CPuct` / `FpuReduction` `setoption` mid-game keeps the search tree, so until
the tree turns over the search is a hybrid of both settings rather than the one
just configured. Measured on `pucv`, two `go`s with the tree reused:

| arm | second `go` |
|---|---|
| default throughout | `(1921, 14753)` |
| `c_puct=99` from construction | `(1921, 15144)` |
| default `go`, then `setoption CPuct 99`, then `go` | `(1921, 15094)` |

The third matches neither, because the visits already banked were bought by the
old constant. Keeping the tree is the deliberate choice — discarding a game or
ponder tree to honour a `setoption` is the worse failure, and at handshake time
(the TCEC/cutechess case) the tree is empty and the question does not arise.
Note the asymmetry with the row above it in `searchconfig`: `VLossWeight` *does*
reset the tree, because vloss-adjusted `Q` is not comparable across weights.
`test_a_mid_tree_puct_change_keeps_the_tree_and_is_a_hybrid` pins the decision
in both directions.

`VLossWeight`'s delivery on the shipped default (`Threads 2` → `walker`) rests
entirely on the pool rebuild inside `SearchWorker.set_vloss_weight`. The
`_reinstall_configured_search_path()` that `_apply_search_option` calls next
*looks* like a second cover and is not: with `search_parallel="pucv"`,
multi-GPU off and not leaving Gumbel, that method falls through every branch
and does nothing. Measured with the rebuild removed — the engine prints
`VLossWeight set to 17` while `WalkerPoolConfig.vloss_weight` and
`searchconfig` both stay at 3. `MinibatchSize` has the matching one-liner:
it reshapes the leaf batch the cudagraph was captured at, so it sets
`_warmup_dirty` and the next idle `isready` re-captures **before** the clock
starts. Both are pinned behaviourally
(`test_vloss_weight_reaches_the_walker_pool_on_the_shipped_default`,
`test_minibatch_size_marks_the_captured_cudagraph_stale`).

`QVisitExpRoot` is the second kind, and its arms are exact — read off the C, no
threshold, nothing fitted:

| current value | what the C does | what is pinned |
|---|---|---|
| `< 0` (the shipped `-1.0`) | log root: `CScaleRoot·log1p(CVisitRoot + max_visit)` (`_mcts_tree.c:1498`) | the expression **does not contain the exponent**; only its sign chose the branch, so every negative value is one search |
| `>= 90` | LINEAR root sentinel: the exponent resolves to `1.0` (`:3947`, reading the literal `gumbel_c` passes for the deleted `q_visit_exp`) | every value in [90, 99] is one search |
| `0 … 90` | power root: `CScaleRoot·(CVisitRoot + max_visit^exp)` (`:1500-1504`) | the exponent is *added to* `CVisitRoot`, so at `CVisitRoot >> max_visit^exp` it moves `q_scale` very little |

**Crossing a boundary changes the search, and the engine must never call that
"no effect".** At `CScaleRoot=0.05`, `QVisitExpRoot -1.0 → 1.0` moves the best
root action 306 → 553 and the tree 8037 → 7764.

#### ⚑ A withdrawn claim, recorded because it was wrong in the dangerous direction

An earlier revision of this section reported `QVisitExpRoot` `[INERT]` using a
`q_scale`-versus-prior-span bound, and claimed the bound "under-fires rather
than over-fires" and "never calls a working knob dead". **Both claims were
false.** The engine printed `NO EFFECT on the live search` for the 306 → 553
transition above; the reverse direction printed *"Only crossing to >= 0 can
change anything"* having just crossed from `>= 0`.

The mechanism, for anyone tempted to retune rather than remove such a bound: two
root transforms differ **only** by the scalar `q_scale` — the normalized
`u = (q − min_q)/(max_q − min_q)` is identical between them. So a candidate pair
flips exactly when `−Δlog_prior/Δu` falls between the two `q_scale` values, and
`Δu` can be arbitrarily small on near-tied Q. No readback-time bound can rule
that out, which is why the heuristic was deleted rather than retuned, and why
every arm above is an exact statement about what the C reads.

The measurements that motivated the bound are still true and still useful, just
not as a predicate: at the shipped `CScaleRoot=7 / CVisitRoot=900` the crossing
is byte-identical on the harness position, and lowering `CScaleRoot` to 2.0 or
below makes it observable. Treat that as "this knob is hard to move at the
shipped root scale", not as "this knob is dead".

`tests/test_uci_search_options.py::test_a_no_effect_report_implies_the_search_did_not_move`
pins the implication on the message the operator actually sees, over cells that
**cross** arm boundaries as well as ones that stay inside them — the all-within-arm
cell list is precisely why the original defect passed its own calibration test.

**Two knobs deliberately have no UCI option:**

- `gumbel_scale_after` (and the `*_decay_*` schedule around it) is a **selfplay**
  knob. It lives on `selfplay.SearchConfig` and is applied by
  `selfplay/network_turn.py::_scheduled_gumbel_scale` from the game's move
  number. Nothing in the UCI engine reads it, and a UCI move has no "move
  number in a training game" to schedule against. Use `GumbelScale` for a flat
  match-play noise level.
- `policy_target_temp` is a **training-time** temperature on the policy TARGET.
  It never runs during play. It is not the same knob as `PolicyTemperature`,
  and exposing it would invite exactly that confusion.

### The cost of `PolicyTemperature != 1.0`

`policy_temp != 1.0` disables the compact-legal bf16 leaf transport — the C
bf16 leaf softmax has no temperature hook — so leaves fall back to dense
float32 4672-wide. `mcts/gumbel_c.py` warns once per process when this fires.

The `~1.9x end-to-end` figure in that warning comes from the 2026-08-03
play-path audit (1.63s → 3.12s, 40 searches × 8 boards × 256 sims, CPU).
**It was NOT reproduced when this option shipped, and the attempt is worth
recording because it shows what such a measurement needs.**

A naive `T=1.0` vs `T=1.5` comparison is confounded: a different temperature
searches a different tree, so the two arms do not do the same work. Comparing
`T=1.0` against `T=1.0+1e-7` removes that — numerically identical priors, but
the `!= 1.0` transport gate still trips. Measured that way against a numpy CPU
stand-in evaluator, at the same 40 × 8 × 256 shape: **0.97x, i.e. nothing.**

That 0.97x is a fact about the harness, not evidence against 1.9x. A numpy
stand-in computes the dense policy for every leaf in *both* arms, so the thing
the compact path saves — never materialising or transferring 4672 floats per
leaf — does not exist to be saved. Only an evaluator with the real bf16 legal
transport can price it.

**Update (merge with #379): it has now been priced on a real transport, and
1.9x did not survive.** The 1.9x was measured on the DIRECT evaluator, which
has no bf16 leaf transport to lose in the first place. On the broker path that
distributed selfplay actually runs, the same gate measured **0.87–1.01x,
non-monotone in `T`, inside the instrument's own ±13% noise** — see
`docs/experiment_ledger.md` "selfplay search policy temperature" (e) and the
warning text in `mcts/gumbel_c.py`, which carries both numbers. So treat 1.9x
as an upper bound observed on one transport, not as a cost to budget for, and
**measure on your own transport before paying for it**. The `--policy-temp` and
`PolicyTemperature` help strings say the same thing.

So: treat 1.9x as an unverified upper bound, and **measure on the target
evaluator before running a `PolicyTemperature` sweep at a tournament time
control.** The warning fires once per process, so a match log will tell you the
fallback engaged; what it costs is hardware-specific.

### Setting search parameters in `audit_targets.py`

`scripts/audit_targets.py` scores target candidates against the frozen audit
set. Its search shape is now overridable:

```bash
PYTHONPATH=. python3 scripts/audit_targets.py --checkpoint <ckpt> \
    --gumbel policy_temp=2.2 --gumbel topk=8,halving_div=4 \
    --dump-per-position runs/audit_T2.2.jsonl [--dump-distributions]
```

- Keys are **raw `GumbelConfig` field names**, identical to
  `arena_standard.py --cand-gumbel` / `--ref-gumbel`, so a shape moves between
  the two by copy-paste. The UCI engine CamelCases the same fields
  (`c_scale` → `CScale`); that is the only naming difference in the repo.
- `c_puct` / `cpuct_factor` / `cpuct_base` / `fpu_reduction` are **refused**,
  same as in the arena: inert in a Gumbel search.
- The override applies to the **PLAY row (b)** only. `--gumbel-training-rows`
  extends it to rows (d)/(e); it is off by default because those rows describe
  the target production actually stores.
- **Guarded at the dispatch, not the CLI.** After the `GumbelConfig` is built
  it is asserted field-by-field against what was asked for, and the run aborts
  if anything failed to plumb. This is not theoretical: `_SearchProfile` has a
  fixed field list, so before this guard an override for any field outside it
  (e.g. `halving_div`) would have parsed, printed in the header, and been
  dropped — a flawless null. The realized values are echoed as
  `[audit] b) net + Gumbel search (PLAY settings): --gumbel realized policy_temp=2.2`.

`--dump-per-position` records, per position and per candidate, the chosen move
plus two **paired per-position booleans**: `top1_agree` (chosen move is among
the deep-SF co-best set) and `out_of_top10` (chosen move is outside the deep-SF
top 10, or `null` when the MultiPV list is shorter than 10 and the question
cannot be asked). They are computed once here rather than left to each caller,
because tie handling and short lists are exactly where two reimplementations
would silently disagree. The record also carries `sf_top1` / `sf_top10` (the
reference they were judged against) and the `gumbel_overrides` in force.
`--dump-distributions` adds each candidate's full `{uci: p}` over legal moves.

**Positive control for `policy_temp`** — the knob most likely to be silently
dropped, because 1.0 is a no-op so any plumbing bug looks exactly like "no
effect". Driven through this exact path (32 positions, sims 256, PLAY shape,
`add_noise=False` as the script runs it), against a numpy stand-in evaluator:

| T | ΔH raw prior | ΔH search output | reference (real net, noise ON) |
|---|---|---|---|
| 1.36 | +0.171 | +0.061 | +0.196 |
| 1.50 | +0.210 | +0.087 | +0.266 |
| 2.20 | +0.312 | +0.215 | +0.430 |
| 3.00 | +0.356 | +0.078 | +0.612 |

**Read the prior column first.** It is monotone and unambiguous: the knob
reaches the priors. The search-output column is smaller and *non-monotone* —
it peaks at T=2.2 and falls back at T=3.0 — because the completed-Q transform
re-sharpens whatever prior it is handed, and past some flatness the search
output stops tracking the prior at all. That is the same offsetting effect
already measured in this repo ("163% of the policy gain was the net prior;
search offset it"). So **a small search-output ΔH is not evidence the knob was
dropped**, and a control that looked only at the search output would have
produced a false negative at T=3.0.

Magnitudes here run roughly half the real-net reference: a random-projection
stand-in has a much flatter prior to begin with (H = 3.26 nats over ~30 legal
moves), so the same T moves it less. Re-run on a real checkpoint before
quoting an absolute number.

ΔH ≈ 0 in the **prior** column means the knob is being dropped between the CLI
and the search — find where before reporting any number from that run.

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
