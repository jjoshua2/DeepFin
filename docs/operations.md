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

## UCI search options (match play)

The engine (`python -m chess_anti_engine.uci`) exposes its search parameters as
UCI options, so a TCEC-style config, cutechess-cli, or Arena can set them
without a code change. Every option below is also a CLI flag on the same
module; the UCI name and the flag set the same field.

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
info string searchconfig 15 live, 4 inert on path=gumbel
```

The path is read off the **live worker**, not off the options: `UseVL true`
against an evaluator without the slot API is accepted and then falls through to
classic Gumbel, and an options-derived answer would report a path that never
ran. Values come from `SearchWorker.realized_search_values()` — the same
objects the search reads.

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

| option | type | range | default | live on | notes |
|---|---|---|---|---|---|
| `PolicyTemperature` | string | 0.5 – 5.0 | 1.0 | gumbel, rpg | prior temperature (logits/T); >1 softens. **Costs search time when ≠ 1.0 — see below.** |
| `CScale` | string | 1e-4 – 100 | 0.025 | gumbel; rpg only if `CScaleRoot < 0` | descent value-transform scale |
| `CVisit` | string | 0 – 1e5 | 50.0 | gumbel; rpg only if `CVisitRoot < 0` | descent transform floor |
| `CScaleRoot` | string | -1 – 1000 | 7.0 | gumbel, rpg | root-only c_scale; <0 = use `CScale` |
| `CVisitRoot` | string | -1 – 1e5 | 900.0 | gumbel, rpg | root-only c_visit; <0 = use `CVisit` |
| `QVisitExp` | string | 0 – 2 | 1.0 | gumbel; rpg only if `QVisitExpRoot >= 90` | descent exponent on max_visit |
| `QVisitExpRoot` | string | -10 – 99 | -1.0 | gumbel, rpg — but **reported INERT at the shipped defaults**, see below | <0 = log root (sim-invariant); ≥90 = use `QVisitExp` |
| `QVisitFloor` | string | -1 – 1e4 | -1.0 | gumbel; rpg only if `QVisitExpRoot >= 0` | additive (decoupled) transform floor; <0 = legacy coupled |
| `QGlobalScale` | check | — | false | gumbel | scale descent transform by the ROOT max child-visit |
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

Names match `scripts/arena_standard.py --cand-gumbel` / `--ref-gumbel` field
names with the underscores removed and CamelCased (`c_scale` → `CScale`,
`q_visit_exp_root` → `QVisitExpRoot`), so a shape tuned in an arena transfers to
a UCI config by transliteration. The arena keeps snake_case because those flags
address `GumbelConfig` fields directly; the UCI side follows lc0's CamelCase.

### `QVisitExpRoot` is inert at the shipped play shape — and says so

`searchconfig` reports this one `INERT` out of the box. That is not a wiring
bug; it is the measurement. At the shipped `CScaleRoot=7` / `CVisitRoot=900`
the knob is **byte-identically unobservable** — same best move, same tree size,
same root visit distribution — at every value tried (0, 0.5, 1, 2, 98, −10)
against a deterministic evaluator at 2048 nodes.

The root sequential-halving cut ranks on
`gumbel + log_prior + q_scale·(q̂ − min_q)/(max_q − min_q)`
(`_mcts_tree.c:1505-1524`), so the completed-Q term spans exactly `q_scale`
while the prior term's span is capped by the C's own `1e-12` prior clip
(`:1512`) at `log(1e12) = 27.6` nats. Three things follow, and each is a
separate arm of `inert_reason`:

- **`< 0` (the shipped `-1.0`)** — the log branch is
  `CScaleRoot·log1p(CVisitRoot + max_visit)` (`:1498`). The exponent does not
  appear in that expression at all; only its **sign** chose the branch. Every
  negative value is one search. Exact, position-independent.
- **`>= 90`** — the "use `QVisitExp` at the root too" sentinel (`:3947`). Every
  value in [90, 99] is one search.
- **`0 … 90`** — the exponent enters only as `max_visit^exp` *added to*
  `CVisitRoot`. Reported inert once `q_scale` provably exceeds the 27.6-nat
  prior span, because the cut then orders purely by completed Q and the
  transform's shape cannot re-rank. At the shipped scale that lower bound is
  `7 × 901 = 6307`.

Measured crossover on the harness position: `CScaleRoot` 2.0 (q_scale 13.6)
observable, 3.0 (20.4) not. The shipped 7.0 sits at 47.6 (log) / 6307 (power).
Lowering `CVisitRoot` to 0–1 makes the exponent observable again, which is what
`test_the_root_exponent_arm_can_still_report_live` pins.

**The bound is deliberately conservative, and under-fires.** 27.6 is derived
from the C's prior clip, not fitted to the sweep, so near the boundary the
report still says LIVE where the search is in fact unmoved (`CScaleRoot=0.01` /
`CVisitRoot=900` is such a cell). That is the safe direction: the arm never
calls a working knob dead.
`test_an_inert_root_exponent_verdict_implies_an_unobservable_search` asserts
only the implication that matters — INERT ⇒ byte-identical search — against
real searches, so the verdict and its criterion share an instrument.

Making the knob *observable* instead would mean moving the tuned
`CScaleRoot`/`CVisitRoot` play defaults. That is a play-strength change needing
an arena readout and a ledger entry, not something a UCI-surface change should
smuggle in.

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
