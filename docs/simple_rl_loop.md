# Simple self-play RL loop

`scripts/simple_rl_loop.py` (code in `chess_anti_engine/simple_rl/`) is a plain
generation loop: **collect** self-play games with the current net, **train** with
every new row presented exactly once (or twice) in shuffled order, **evaluate** with `scripts/arena_standard.py`
on held-out openings, repeat. It reuses the real finite `play_batch`, the
`Trainer` and its `compute_loss`, the shard format and the arena; it adds no
search or loss of its own.

It replaces nothing: the older `core_finite_*` / `core_generation_*` tooling is
untouched. Why it exists, from the 2026-11-19 diagnostics
([record](experiments/2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md)):
the old loop replayed 192 fixed openings per cycle with no replay window (18% of
rows were exact repeats), its policy targets were far sharper than the net, and
its value head moved a lot on about 192 game outcomes per cycle. The defaults
below answer those findings; each is a flag.

## Running it

All paths are flags; nothing reads the live checkout implicitly.

```bash
export PYTHONPATH=.
BOOKS=~/projects/chess/data/opening_books
TB=~/projects/chess/data/syzygy_3-4-5:~/projects/chess/data/syzygy_6

# 1. Once: TRAIN / EVAL opening split (deterministic; ~4 minutes for the production books).
python scripts/simple_rl_loop.py split-openings --out ~/chess-artifacts/<run>/openings \
  --book $BOOKS/8moves_v3_plus_policybeam_final145cp_plus_uho2024_060_110_plus_2move_thinbeam_dedup.pgn.zip:16 \
  --book $BOOKS/2moves_v2_no8overlap_uho060110.pgn.zip:4

# 2. The loop (from the SL base; use --random-init --model-config tiny.json for a weak start).
python scripts/simple_rl_loop.py run --run-dir ~/chess-artifacts/<run>/loop \
  --openings-dir ~/chess-artifacts/<run>/openings --init-checkpoint <sl_base.pt> \
  --generations 10 --games 2000 --device cuda --sf-path <stockfish> --syzygy "$TB" \
  --tripwire --max-repeat-rate 0.05 --min-target-entropy 0.4
```

`collect`, `train` and `evaluate` also run alone with `--generation N`. `run`
skips every stage whose directory already holds a `manifest.json` and redoes
an incomplete one from scratch, so killing it and starting it again resumes at the
last finished stage boundary (`tests/test_simple_rl_resume.py` kills it twice and
checks the result is identical to an uninterrupted run).

Control files: `<run-dir>/STOP` stops before the next stage; `<run-dir>/PAUSE`
and any `--pause-marker PATH` (for example the marker of a GPU pause window)
refuse to start a stage while present, and pause a RUNNING evaluate stage: each
arena gets them as `--pause-marker` and holds its in-flight games (no ply, models
not loaded yet if it was launched paused) until they are gone; paused time is
excluded from its timings and recorded. Exit codes: 3 tripwire, 4 blocked by a
marker. The loop never takes or checks the GPU custody lock itself; pass the
marker your GPU window uses.

The first `run` pins its settings in `<run-dir>/run_config.json`: every flag
except the operational ones (run dir, generation count, device, threads, pause
markers, tripwire thresholds). A later `run` with different settings is refused
unless `--allow-config-change` is given, which records the change. A stage takes
its settings from the flags it is started with, so a changed collect, train or
arena setting applies to every stage that runs after the change; a stage that
already completed is not redone. Settings a stage RESTORES rather than reads are
not covered by this: under `--optimizer-state continue` the optimizer's
hyperparameters come from the checkpoint (see Training details). `device` is
pinned (it switches bf16 autocast in collect and train); `eval_device` is not.
The standalone `collect` / `train` / `evaluate` subcommands are held to the same
`run_config.json` when the run dir has one, skip a stage that is already complete
unless `--redo` is given (which unseals it first), and every stage loads a model
only when its manifest is present and records that file's sha256.

**Lineage.** A sealed stage is reused (by `run`, or skipped by a standalone
subcommand) only if every input it recorded from the run's own lineage -- files in
the run dir, plus the TRAIN / EVAL openings -- still has the sha256 it was sealed
with; otherwise the command fails and names the input. External inputs (the
initial checkpoint, an SL corpus, a KL or prior-mix reference) are recorded but
not re-hashed on reuse. File hashes are cached in `<run-dir>/lineage_sha_cache.json`
by size, mtime and inode. Each stage seeds the cache with the digests it already
computed for its manifest, so the first resume does not re-read every model. A
file whose size, mtime or inode has changed since is hashed again. `--redo` of a stage while any LATER stage is sealed is
refused, because those stages were built on it. `--redo --redo-downstream` unseals
them: their directories (rows, models, logs) are moved to
`<run-dir>/superseded/<UTC stamp>[-N]/gen_NNNN/<stage>/`, where `-N` keeps two redos in
the same second apart (kept for inspection, never
read again, safe to delete). The run-level files derived from them are rewound:
- arena ledger rows of the unsealed evaluations;
- `presentation_ledger.json` (derived, rewritten by the next train);
- `probe.npz`, when generation 1's collect or train is unsealed;
- the `dup_sketch/` and `openings_seen/` files from the first unsealed collect on.

The opening cursor lives in the collect manifests, so it rewinds with them.

## What each stage writes

```
<run-dir>/
  run_config.json            settings pinned at the first run (+ recorded changes)
  ledger.jsonl               final arena records, one line per match
  probe.npz                  fixed probe rows for drift metrics
  presentation_ledger.json   per-generation histogram of presentations per row (derived)
  gen_0000/model.pt          generation-0 actor (copy of --init-checkpoint, or random init)
  gen_NNNN/collect/          shards/shard_*.zarr (+ .meta.npz row sidecar), games.jsonl,
                             starts.txt, stats.json, manifest.json
                             (while unsealed also plan.json and procs/<k>/: see Collect processes)
  gen_NNNN/train/            model.pt (Trainer checkpoint: weights, optimizer, arch),
                             presented.npz (rows this phase presented), metrics.json, manifest.json,
                             snapshots/step_<k>/ (with --snapshot-updates; see Train snapshots)
  gen_NNNN/eval/             arena_vs_previous.{jsonl,pgn,log}, arena_vs_reference.*, manifest.json
```

Every `manifest.json` holds the stage config, inputs with SHA-256, outputs with
SHA-256, seeds, counts, git SHA and dirty flag, and the runtime versions.

**split-openings.** Lines are book mainlines cut to the book's ply cap,
deduplicated, and filtered to non-terminal starts with at least two legal moves; a
game with a PGN parse error is rejected and counted (never kept as a truncated
prefix). A position's orbit is the position, its colour mirror, its left-right
reflection (the trainer's mirror augmentation) and the reflection of the colour
mirror. EVAL is the `--eval-lines` (default 4,000) lines with the smallest
`sha256(salt, line)`; two EVAL lines whose final positions share an orbit are
refused (counted as `eval_orbit_duplicates`), never silently dropped. A remaining line is excluded from TRAIN when any position on
its path is in the orbit of an EVAL start, or when its final position is in the
orbit of any position on an EVAL line's path: no TRAIN game starts at, passes
through, or is a prefix of an EVAL start. TRAIN is not de-duplicated by position
(naturally recurring positions stay); the duplicate counts are reported. On the
two production books: 158,106 lines parsed, 0 parse errors, 4,000 EVAL (0 orbit duplicates), 153,842 TRAIN, 219 excluded (139 pass through an EVAL
start, 80 end on an EVAL path), 0 TRAIN lines sharing a final position or orbit.
Rerunning with different `--eval-lines`, `--salt` or books into an existing split
is refused. Outputs `train.jsonl`, `eval.jsonl`, `eval.pgn` (for the arena's
`--openings`) and `summary.json` (whose `eval_max_plies` the arena replays).

**collect.** `--games` games from TRAIN lines, 95% / 5% by book as production
(`--book-mix`). Both seats are the net. Openings CYCLE across generations by
default (`--opening-sampling cycle`): each book has one seeded permutation per
pass, walked by a cursor that the collect manifest records
(`opening_cursor`, with `opening_cursor_before`) and the next generation
continues, so no TRAIN line is reused until its book is used up; an exhausted
book starts its next pass (a new permutation, logged). Passes are independent
permutations, so a line played near the end of one pass can recur early in the
next. A redo of a killed collect restarts from the previous generation's sealed
cursor, so it picks the same lines. The lines are drawn once per generation and
split across the collect processes (Collect processes).
`--opening-sampling independent` (the non-default alternative) redraws every
generation, without replacement only within it.

`stats.json`: games, rows, W/D/L, decisive and draw rate, mean length, mate count,
TB-adjudicated share, unresolved (ply-cap) games, the book pass per book
(`book_pass`), the share of games whose line was played in an earlier generation,
the exact-repeat rate of the encoded input (position and history planes) within
the generation, the cross-generation repeat rate, and, from the stored root
statistics, the target entropy, prior entropy, KL(target‖prior) and top-1
agreement of target and prior.
It also records `buffer_full_flushes`: how many times the generation's C searches
found their leaf buffer full and flushed. That count is observed, read from
`gumbel_c.buffer_full_flushes()` around the collect, not predicted from settings.
A count above 0 means the evaluator cap (`--eval-max-batch`) is below what the
search asks for, so the searches were re-batched. The collect writes `stats.json`
and then refuses to seal the generation, which is redone whole. See the
[buffer-full record](experiments/2026-11-20-gumbel-buffer-full-fallback.md).

The cross-generation repeat rate (`repeat_rate_vs_previous`) is a constant-memory
estimate by default. The run keeps every earlier row hash below a threshold in a
sorted array of at most `--dup-sketch-budget` hashes (default 4M, about 32 MB,
`<run-dir>/dup_sketch/`); the threshold halves whenever the array would overflow.
The `dup_sketch/` and `openings_seen/` files are diagnostics only: each collect
records their sha256 under `diagnostics_files` in its manifest. The next collect
uses one only if it matches that record; otherwise it rebuilds the file from the
sealed collects. Files older than the previous generation are pruned.
The estimate is hits among this generation's rows with a hash below the threshold,
divided by those rows. Because inclusion depends only on the hash, the estimate is
unbiased. `repeat_rate_vs_previous_estimate` holds the estimate's counts and its
binomial standard error (a lower bound when rows also repeat within the
generation), plus the same rate by plies after the book (0–3, 4–9, 10–19, 20–39,
40+). `--exact-repeat-vs-previous` computes the exact rate over every earlier row
instead; it is for small runs, because its memory grows with the run.

**train.** The default is a uniform, shuffled, exact-once schedule. Each phase is
one pass WITHOUT replacement over the rows scheduled for it, so every trainable
row is presented exactly `--presentations` times in its lifetime:

- `--presentations 1` (default): a generation's rows are presented once, in the
  phase right after that generation is collected. The phase trains on the
  current generation only.
- `--presentations 2` (the cap; larger values are refused): once in that phase and
  once in the next generation's phase. Each phase is the shuffled union of the new
  generation's rows and the previous generation's rows.

The mirror decision is drawn per presentation (per batch, by the Trainer) and is
not an extra presentation. The final partial batch is served as a short batch, so
no row is dropped or repeated; updates = ceil(rows / batch). Each phase writes the
rows it actually served (`presented.npz`, recorded from the sampler the Trainer
consumed). The run's per-row counts are derived from completed phases only (those
with a manifest), so a phase killed before its manifest contributes nothing and
its redo cannot double-present. Before training, a phase refuses rows whose count
so far differs from the schedule. Afterwards, every generation whose lifetime has
ended must show exactly `{P: trainable rows}`; the histograms are in
`metrics.json` and `presentation_ledger.json`.

`--sample-with-replacement` is the explicit non-default alternative: uniform
sampling with replacement over the last `--window` generations (default 4) for
`--updates`, or `--passes` × rows / batch. It gives no per-row guarantee and is
recorded as `sampling: with_replacement`. `--window`, `--updates` and
`--passes` are refused without it.

A `--holdout-frac` share of games (by game id) is never trained on;
the newest generation's held-out rows give policy CE to the stored target, policy
entropy and value NLL before and after the phase. Mean |ΔQ| and policy KL against
the previous generation's net are measured on a fixed probe set. `metrics.json`
also records the realised LR, value-target mix, target temperature, SL fraction
and anchor diagnostics read off the objects that consumed them.

**evaluate.** `arena_standard --mode matched_sims --search-shape training` on
`eval.pgn`, fixed seed, fixed N, no SPRT and (by default) no root noise, for
new-vs-previous and new-vs-reference (default reference: generation 0). The arena
samples moves at its default temperature 0.1, so the two games of an opening pair
are not mirror images of each other. Book plies replayed default to the split's
`eval_max_plies`; a smaller `--eval-opening-plies` is refused (games would start
mid-line). Only each arena's final record is read, and it must be complete
(`games == games_requested`, not truncated) or the stage fails. Both matches'
records are written to `ledger.jsonl` together, replacing any rows an interrupted
attempt left for that generation, so a redo never duplicates them. `--search-shape
training` takes c_scale/topk/policy temperature from this tree's production yaml,
which need not equal the loop's own collection flags; add `--arena-arg` to override.
With `--syzygy` the arena is strict: a single game still unresolved at
`--eval-max-plies` invalidates the match (the arena refuses to score it a draw).
A weak or random net at low sims can hit that; pass `--eval-no-syzygy` or a higher
`--eval-max-plies` for such runs.
The tripwire table (policy entropy, target entropy, prior entropy, KL, draw and
decisive rate, repeat rates) prints after every generation. `--tripwire` exits 3
when the newest generation crosses an armed threshold; with no threshold flag it
refuses to run rather than pass.

## Defaults and why

| Setting | Default | Why |
|---|---|---|
| Openings | 154k TRAIN lines from the two production books | 192 recurring openings gave 18% exact repeats |
| Move temperature | 0.5 on the first ply after the book, linear to 0.1 over 10 plies, then 0.1 for the rest of the game | mild early diversity without random-looking games; counted in plies from the book's end (`TemperatureConfig.schedule_plies_from_start`), so 2-move and 8-move lines get the same window. The floor never reaches 0, so every played move is SAMPLED from the stored target, not the halving survivor (`played_move_index` and `root_survivor_index` are both stored) |
| Root noise | Gumbel scale 1.0, to 0.5 over moves 12-15 | production self-play; counted in FULL MOVES from the start position (the book plies included), unlike the temperature clock |
| Sims / topk / c_scale / c_visit | 64 / 16 / 0.1 / 50 | the F12 instrument and the D1 search-gain reads |
| Tree reuse | on (`--no-tree-reuse` to turn off) | production behaviour; see below |
| Playout cap randomisation | off (`--full-prob 1.0`) | see below |
| Arena sims | 64 (`--eval-sims`) | independent of `--sims`, so changing the target budget does not move the yardstick |
| Tablebases | strict rule50 route when `--syzygy` is given | as the F12 collection: TB in search, TB adjudication, TB-correct policy rows; ply-cap games keep no rows |
| Presentations | 1 per row (2 = cap) | the owner's standing default: shuffled exact-once, at most two deliberate presentations of any row |
| Loss | policy on the stored target + WDL on the game outcome | `--w-wdl 0` for policy only, `--value-search-weight` to mix in the stored search value |
| Auxiliary heads, SF targets | off | nothing trains on Stockfish |

Stored target knobs (`--c-scale`, `--c-visit`, `--no-target-q-rescale`,
`--policy-temp`, `--topk`) each change the stored target on the compiled search
path (`tests/test_simple_rl_search_knobs.py`). With a positive move temperature the
played move is sampled from the stored target, so store-only knobs such as
`--no-target-q-rescale` also reach play. A softer target is better applied at
train time (`--train-target-temp`, a power transform of the stored target, which
equals a temperature on its logits), because every row keeps what the target was
built from.

How sigma works, for reading these knobs: sigma = c_scale × (c_visit + the
**largest** child visit count), the same in the C root halving and in the stored
target (`mcts/gumbel.py` `_completed_q_transform`), so c_scale and c_visit are near
substitutes. `target_q_rescale` exists in `GumbelConfig` but the self-play mapping
`build_selfplay_gumbel_config` did not pass it (nor `c_visit`) before this loop:
from self-play config it could not be set. Both are now `SearchConfig` fields
(`gumbel_target_q_rescale`, `gumbel_c_visit`) whose production defaults equal the
`GumbelConfig` defaults, so production behaviour is unchanged.

**Tree reuse.** By default, as in production, the chosen child's subtree is
carried into the next ply's search, so a "64-sim" target can include inherited
visits and sigma sees the inherited maximum. `--no-tree-reuse`
(`SearchConfig.gumbel_tree_reuse=False`) searches every ply from a fresh root on
exactly its own budget. The setting is in each collect manifest, and every row
records `root_start_visits`, the visits already on the root when its search began
(`stats.json` reports the mean and the share of reused roots). With few sims and a
wide topk, each candidate gets one visit and there is nothing below it to carry,
so reuse is effectively off at, for example, 8 sims / topk 16.

**Playout cap randomisation.** `--full-prob p --fast-sims n`: each ply searches
`--sims` with probability p and stores a policy row, or else searches `n`, is
played, and stores no policy target (a value-only row with `--fast-value-rows`,
otherwise no row). Every net-turn row records `search_budget`; `stats.json` counts
rows by budget. `search_budget` is the budget the ply was given, not the visits it
spent. A root proven solved mid-search stops early and keeps its statistics, which
still rebuild its target. On the SL base's real-net smoke this was 132 of 11,823
rows, all with |survivor Q| = 1.

**Stored root statistics.** With `GameConfig.record_root_stats` (on in this loop,
off by default everywhere else) each searched row also carries, in compact policy
space and float32: `root_log_prior` (the search's prior, tempered by
`--policy-temp`), `root_child_q` (completed Q per root child), `root_child_visits`
(-1 off the searched support), and three root values that differ materially:

| Field | What it is |
|---|---|
| `root_net_value` | the raw network value W - L of `root_net_wdl` (the net's softmaxed root WDL, also stored) |
| `root_survivor_q` | the completed Q of the sequential-halving survivor: the value of the move the search settles on |
| `root_node_q` | the tree's root-node Q: a visit average over every candidate halving is forced to try, so it sits below the survivor's Q. Biased; kept for diagnosis only |

plus `root_start_visits`, `root_survivor_index` and `played_move_index`.
`mcts.gumbel_c.rebuild_improved_policy` rebuilds the stored target from the
vectors and `root_net_value` (the value unvisited children are filled with) under
the run's `GumbelConfig`, so targets can be re-scored offline under different sigma
knobs without re-searching. The stored `policy_target` itself is float16, so a
rebuild matches it to half-precision rounding: within 1.5e-4 on the tiny-net smoke, but up to 9e-4 (p99 1.6e-4) on real-net rows, where probabilities near 1 carry the largest float16 absolute error. A row the search returned through an exact
shortcut rather than the completed-Q arithmetic (a solved or tablebase root, a
single legal move, or the pruned-draw fallback of #1410, which plays an unsearched
drawing move) carries NO statistics, never stale ones; `stats.json` counts them
(`rows_without_root_stats`) and their value target falls back as described under
Training details. Rows whose target was replaced by the tablebase policy are
flagged in the row sidecar (`tb_policy_override`).

## Collect evaluator: one fast path

On CUDA the collect evaluator always replays CUDA graphs of its own eager dense
forward (`DirectGPUEvaluator(cuda_graphs=True)`). There is no flag: the old
`--collect-cuda-graphs` choice is gone, because its outputs were measured
bitwise identical to the eager forward and keeping a slow twin path bought
nothing. Graphs cannot run on CPU, so a CPU collect uses the eager
`LocalModelEvaluator`. `stats.json` records the realised `evaluator_mode`
(`cuda_graphs` or `eager_cpu`) and the graph counters in `cuda_graph_stats`.
A run pinned with `collect_cuda_graphs` in its `run_config.json`, on or off,
still resumes: only current flags are compared, so the retired key is ignored,
and such a run now collects on graphs. The arena likewise replays graphs wherever its
hoisted evaluator can (see `docs/search_and_encode_paths.md`).

Measured on 2026-11-20. Setup: SL base, 128 games, 64 sims, concurrency 128,
one process, GPU otherwise idle. Every collect below has the same
`--content-digest`.

| collect evaluator | games/h | wall s |
|---|---|---|
| eager (the old default) | 2,720 | 169.4 |
| graphs (the old `--collect-cuda-graphs`) | 3,763 | 122.5 |
| graphs, the new default (run twice) | 3,760 / 3,613 | 122.5 / 127.5 |

The two runs of the new default differ by about 4%, so treat any gain under
about 5% as noise. These candidates were measured and not taken:
- **Graph row cap 1024 / 2048 instead of 512:** 3,778 / 3,787 games/h, with
  identical rows. That is within noise, and it pins more VRAM.
- **Memoising the rule-50 tablebase status per FEN:** an 11% hit rate and
  3,704 games/h, with identical rows.

## Collect processes

`--collect-procs K` runs the collect as K processes sharing the device, each
running `--concurrency` games at a time. When a flag is not given it takes the
run's pinned value (a run pinned before `--collect-procs` existed keeps 1
process), for an arm its source run's value, else the default table
`collect.CUDA_COLLECT_SHAPES` by `--sims` (both values are then pinned):

| `--sims` | processes × games per process (CUDA) | measured (2026-11-20, RTX 5090) |
|---|---|---|
| ≤ 16 | 8 × 64 | 9,689 forwarded rows/s, 0.88 of the forward ceiling, 1.85M training rows/h, 22.9 GB |
| > 16 | 3 × 128 | 64 sims: 9,487 rows/s, 0.86, 491k rows/h, 22.1 GB; 32 sims: 8,679, 0.79 |

On CPU the default is 1 × 64. K = 1 is the same code path, run in the driver's
own process. The 32-sim row has no measurement with K ≥ 6 yet.

Why: the collect search is serial within a process (C tree walk, forward, wait,
tablebase probes, C tree walk), because the collect carries persistent trees and
a supplied tree turns the 2-group pipeline off (`gumbel_c.py`, play-path audit
F5). One process therefore leaves the GPU about half idle, and raising
`--concurrency` past 256 does not fill it. Several processes overlap one
process's CPU work with another's forward.

How a generation is split:
- **Openings:** the generation's list is drawn exactly as for one process (the
  cursor advances by `--games`). Process k plays the disjoint slice
  `openings[k::K]`, so no line is played twice across processes beyond what the
  book cycle itself repeats. The collect refuses to seal unless the openings
  played equal the openings drawn.
- **Seeds:** process k uses `seed_for(seed, "collect", g, "proc", k, "of", K)`.
  K = 1 keeps the one-process seed `seed_for(seed, "collect", g)`, so a K = 1
  collect has the same `--content-digest` as one made before the flag existed.
- **Concurrency is per process:** each process runs `--concurrency` games at a
  time, with its own evaluator (its own leaf-buffer cap) and its own Stockfish
  pool.
- **Merge:** each process writes `procs/<k>/shards`, `games.jsonl` and, last,
  `done.json` (its seed, counts, evaluator record and buffer-full flushes). The
  parent hard-links the shards into `shards/` in process order, renumbered, and
  concatenates games.jsonl in process order. One manifest seals the collect;
  `seeds.collect_procs` lists every process's seed and `stats.procs` every
  process's record. The process directories are removed once the manifest is
  written.
- **Gates over the merge:** `buffer_full_flushes` is the sum over processes (any
  flush refuses the seal). `eval_max_batch_realised` and `evaluator_mode` are the
  processes' common value; if they disagree the field is null (`"mixed"` for the
  mode), so G-C2 and G-C11 fail rather than report one process for all.
  `cuda_graph_stats` counters are summed. `duplicate_game_ids` counts game ids
  that occur more than once (ids are content hashes of opening, moves and
  result, so only identical games share one; holdout membership is by game id).

K is pinned. It sets every process's RNG stream and batch composition, so rows at
K = 3 are not the rows at K = 1, and a changed K needs `--allow-config-change`.
An arm must match its source's K (`reference.settings_diff`; a source sealed
before the flag counts as 1). The distribution is argued equivalent, not
identical: each game runs the same search, targets and sampling at any K; the
collect gates (G-C1 to G-C11) are the check.

Resume: `plan.json` records K, the per-process seeds and opening ids, the
checkpoint sha256 and the settings. A killed collect whose plan is unchanged
keeps every process that wrote `done.json` (and whose shards and games.jsonl
still match it) and restarts the others from their own seed and openings. Any
other unsealed attempt is redone whole.

Process hygiene: children are spawned (never forked). Each child arms a
parent-death signal against the driver, so a SIGKILL or SIGTERM of the driver
kills it; the Stockfish engines a child starts carry the same guard against the
child (`stockfish.uci._child_reap_guard`). A failing child, an exception or
Ctrl-C in the driver stops the other children (SIGTERM, then SIGKILL after 10 s).
Pause: `<run>/PAUSE` or any `--pause-marker` pauses a RUNNING collect
cooperatively. Every process checks it before loading its model and once per
`play_batch` step (one network turn of every in-flight game). While a marker exists
the process schedules no search, forward or Stockfish work, keeps every in-flight
game in memory (its VRAM stays allocated) and writes `procs/<k>/PAUSED`; it
continues exactly where it stopped when the marker is removed, so a paused collect
makes the same rows (same `--content-digest`) as an unpaused one. Nothing on that
path has a wall-clock deadline: the engine read timeout applies only inside a
search, and searches already submitted finish on the pool's threads. Each
process records its `pause_intervals`; `stats.json` has `paused_seconds`,
`active_seconds`, games/h over the active time, and `timing_independent: false`
for a paused (or resumed) collect. `STOP` is not a pause: the driver honours it at
the next stage boundary, and the driver still refuses to START a stage while a
pause marker exists.

GPU memory: at the start the collect refuses, and never silently steps K down,
when K × 1.10 × max(48 MiB per sim + 1.11 MiB per leaf row, 2.006 MiB per leaf row
− 4348 MiB) exceeds the usable memory, min(free, 83% of the card). The first term is
a least-squares fit to the multi-process and ≤ 256-game peaks (within −8% / +9% of
each); above ~8k leaf rows in one process the peak grows faster (64 sims, 1 × 512
peaked at 29.9 GB, 25% above the line), which the second term, through 1 × 256 and
1 × 512, covers. When no process count fits, the refusal says to lower
`--concurrency`. The usable limit is measured: 8 × 96 at
16 sims peaked at 31.0 GB of 32 and ran 22% slower than 6 × 96 at 25.9 GB, which ran
at full speed.

Measured on 2026-11-20 on the RTX 5090, GPU otherwise idle (compute-process list
recorded every 2 s): SL base, 64 sims, cap auto, 0 buffer-full flushes in every
run. Each multi-process row is K independent collects with distinct seeds, run
side by side, which is what `--collect-procs` launches.

| setup | games/h | forwarded positions/s | GPU util mean | peak VRAM |
|---|---|---|---|---|
| 1 × 128 games | 2,881 | 5,282 | 46% | 8.7 GB |
| 1 × 256 | 3,372 | 6,151 | 54% | 12.2 GB |
| 1 × 512 | 3,355 | 6,047 | 53% | 29.9 GB |
| 1 × 256, graph row cap 2,048 | 2,675 | 4,880 | 43% | 13.9 GB |
| 2 × 128 | 4,546 | 8,435 | 78% | 16.2 GB |
| 2 × 256 | 4,553 | 8,412 | 78% | 22.6 GB |
| 3 × 128 | 5,021 | 9,262 | 87% | 21.9 GB |

Fewer simulations per move (same seed and openings; ceiling = the ~11k rows/s
plain-forward ceiling at batch ≥ 512):

| sims | setup | forwarded rows/s | share of ceiling | GPU util | training rows/h | peak VRAM |
|---|---|---|---|---|---|---|
| 16 | 1 × 256 | 3,146 | 0.29 | 29% | 619k | 9.1 GB |
| 16 | 3 × 128 | 7,113 | 0.65 | 64% | 1.39M | 14.8 GB |
| 16 | 4 × 128 | 8,489 | 0.77 | 78% | 1.66M | 18.7 GB |
| 16 | 8 × 64 | 9,689 | 0.88 | 90% | 1.85M | 22.9 GB |
| 16 | 6 × 96 | 9,754 | 0.89 | 90% | 1.86M | 25.9 GB |
| 16 | 8 × 96 | 7,599 | 0.69 | 81% | 1.44M | 31.0 GB (past the usable limit) |
| 32 | 1 × 256 | 4,510 | 0.41 | 39% | 452k | 12.1 GB |
| 32 | 3 × 128 | 8,679 | 0.79 | 81% | 879k | 19.7 GB |
| 64 | 1 × 256 | 6,506 | 0.59 | 58% | 337k | 12.1 GB |
| 64 | 3 × 128 | 9,487 | 0.86 | 90% | 491k | 22.1 GB |

Training rows per GPU-hour: 1.86M at 16 sims, 0.88M at 32 (3 × 128), 0.49M at 64,
so 16 sims buys 3.8× the rows of 64 once 6–8 processes fill the GPU (2.8× at
3 × 128). The Stockfish label pool (one worker, 1 node, strict TB) sustains about
1,870 labels/s per process, against about 90 rows/s per process at 6 × 96: it is not
the ceiling.

At 3 × 128, py-spy of one process puts 51% of its main thread in waiting on the
GPU and 13% in launching forwards; tablebase probing (8%), the C search (6%),
Stockfish labels (6%) and encoding (3%) are each small. The GPU is the
bottleneck there, so moving any one CPU cost into C buys single-digit percent at
most.

## Evaluator batch cap

`--eval-max-batch` controls the search, not memory. Each network turn hands one
compiled Gumbel call every active game, together with the persistent tree. The
call caps its leaf buffer at the evaluator's max batch. Before ABI 11 (the
[buffer-full fix](experiments/2026-11-20-gumbel-buffer-full-fallback.md)), a leaf
that found the buffer full was not evaluated: the C tree backed it up as a
pseudo-terminal carrying the root's Q. Since ABI 11 the batch is flushed and
resumed, and a collect whose searches flushed is refused unsealed. The rest of
this section describes the defect that rows from older builds carry.

`tree_backprop` reads a leaf value from the leaf's side to move, while the root's
Q is from the root's side to move. So a starved leaf at any ODD depth (a root
child, a great-grandchild, ...) is backed up with the wrong sign: for a losing
root those leaves look as good as the root is bad. A starved leaf at an even depth
gets the right sign, but its value is still the root's, not its own: fabricated.
One rep of a halving round requests up to concurrency × topk leaves, and the
accumulate guard keeps later reps inside the buffer, so a cap binds exactly when
it is below concurrency × topk. The old default was a fixed 512: above 32 games at
topk 16 it binds.

The 2026-11-20 diagnostic measured the cost. It used 64 sims, UHO openings and
128 identical games, and compared cap 512 against 4096:

| Setting | Cap 512 | Cap 4096 |
|---|---|---|
| Reply side: Spearman(child Q, prior) | about 0 | +0.60 (book-end mover +0.58) |
| Book-end mover wins / losses / draws | 95 / 13 / 20 | 45 / 24 / 59 |
| Mean target entropy | 0.80 | 0.69 |

At cap 4096 the mean target entropy (0.69) matches the calibration bank's
stored F12 targets. In the smoke's 256-game generation at concurrency 128, the
reply side's stored targets disagreed with a cold re-search of the same
positions: KL 2.4, top-1 agreement 0.53. The book-end mover's disagreed far less
(KL 0.4, top-1 agreement 0.84). Training one phase on those rows moved the probe
by |ΔQ| 0.30, against 0.02 on F12's rows (Training details).

The cap now defaults to the search's own leaf-buffer size,
`leaf_buffer_rows(concurrency, topk)`: 2048 at 64 games and 4096 at 128. A
smaller explicit value is refused before anything is pinned (`validate_settings`,
called by the loop's pre-pin validation). The refusal is conservative: it compares
against the 2× buffer, so an explicit 512 is refused at 17–32 games at topk 16
although the concurrency × topk request would not bind there. The realised cap is
read off the evaluator into `stats.json` (`eval_max_batch_realised`, with
`leaf_rows_required`).

Rows already collected at a binding cap are not trained on silently. Train
refuses (before pinning, and again when the phase starts) any generation in its
window whose sealed collect ran on CUDA with a cap below concurrency × topk. This
is the precise could-bind bound, not the conservative one. The cap is read from
`stats.json` when recorded, else from the pinned setting, else the legacy fixed
512 that old manifests do not record. The check covers the whole lineage, not
only the phase's window: a model descends from every earlier generation's rows,
so a capped collect in any generation 1..g taints phase g and is refused the same
way, read from the collect manifests (older metrics carry no taint field).
`--allow-capped-collect-rows` (pinned) overrides the refusal. The phase then
records `trained_on_capped_rows: true`, the window's capped generations
(`capped_collect_rows`) and the lineage's (`capped_lineage_collects`) in
`metrics.json`, the train manifest and `presentation_ledger.json`. An
`--init-checkpoint` taken from another run carries no taint record: the check
sees only this run's collects. The arena already checks the same bound and warns
(`scripts/arena_standard.py`, `arena_uncapped_leaf_rows`).

## Train snapshots

`--snapshot-updates N[,N...]` (exact-once schedule only) writes
`gen_NNNN/train/snapshots/step_<N>/` after the phase's N-th optimizer update:

- `model.pt`: weights and `arch` only (no optimizer), loadable by every checkpoint
  loader, so `scripts/arena_standard.py --candidate .../step_<N>/model.pt` plays it;
- `metrics.json`: the measurements the phase makes at its end, taken on the
  snapshot's weights: held-out policy CE / entropy / value NLL (`heldout`), probe
  `mean_abs_dq`, `policy_kl_prev_to_new` and `top1_agree_prev` against the
  phase's init model (`probe_vs_previous`), the realised lr per param group, rows
  presented so far and the rows-weighted losses so far;
- `snapshot.json`, written last: the snapshot is complete only with it.

The train manifest records every snapshot file with its SHA-256
(`outputs.snapshot_<N>_model`, `outputs.snapshot_<N>_metrics`) and
`summary.snapshots`. The counts must be strictly increasing and at most the
phase's planned updates. Both are refused before `run_config.json` is pinned; the
planned-updates bound is checked there once the phase's collects are sealed (a
standalone `train`, or a resume), and otherwise by `train` before its first
update, since a fresh `run` cannot know the row count before it collects.

Snapshots do not change training. The phase runs as one `train_steps` call per
snapshot interval on the same sampler, the measurements run under `fork_rng`, and
on CPU the final weights and `presented.npz` are bit-identical with and without
snapshots (tested).

A killed phase resumes from its newest complete snapshot: the snapshot also keeps
`resume_trainer.pt` (`Trainer.save`) and `resume_state.pt` (sampler cursor, served
rows, RNG streams, KL-anchor counters, the chunk metrics so far), restored by
`Trainer.load(exact_resume=True)`. Earlier snapshots are not rewritten, later ones
are written, and the result equals the uninterrupted phase (tested on CPU). Only
the newest snapshot keeps the resume files, and sealing the stage deletes them. A
snapshot resumes only an attempt with the same resume key, which covers the
settings, the SHA-256 of every input, and the batch plan. Snapshots from any other
attempt are deleted and redone.

## Training arms on one collection

Several training arms can train on ONE collection. Each arm is its own run dir
(its own `run_config.json`, so it pins its own train settings) started with
`--collect-from <source run dir>`. Instead of playing games, each of the arm's
collect stages seals a manifest that references the source run's sealed collect
of the same generation (`simple_rl/reference.py`). Every reader of rows and games
(training, the probe, the presentation ledger, the next collect's duplicate
statistics) resolves the reference with `common.resolve_collect_dir`. The arm
holds no copy of the rows.

```bash
python scripts/simple_rl_loop.py collect --run-dir SRC ... --generation 1     # the collection
for w in 1.0 0.0; do                                                            # the arms
  python scripts/simple_rl_loop.py run --run-dir ARM_w$w --init-checkpoint SRC/gen_0000/model.pt \
      --collect-from SRC --generations 1 --w-wdl $w <SRC's collect flags> <arm's train flags>
done
```

The reference is created only when all of these hold:

- the source collect is sealed and is not itself a reference;
- every output SHA-256 the source's manifest sealed still matches;
- the source's generation-0 model is the arm's;
- the source collect was played by the arm's own generation g-1 model (so
  in practice arms share generation 1, which the shared gen-0 model played);
- it used the arm's TRAIN openings;
- the arm's collect flags equal the source's. A different value would be pinned
  without affecting the rows. Execution-only fields are exempt: device,
  eval_max_batch, dup_sketch_budget and exact_repeat_vs_previous.
  `--seed` is NOT exempt: it also draws the train phase's batch plan, so an arm
  must use the source's seed (refused before pinning otherwise).

The cheap checks run before pinning. The reference manifest records, as `source_*`
inputs, the SHA-256 of:

- the source manifest;
- every shard file (`.zarr` and `.meta.npz`);
- `games.jsonl` and `starts.txt`;
- the source's gen-0 model.

`lineage.verify_stage` re-hashes these like in-run inputs whenever the arm reuses
the stage, and `train` re-verifies a referenced collect before loading it. A
source that is redone, unsealed or edited later therefore refuses the arm's next
stage. The arm can then re-reference with `collect --redo --redo-downstream`,
which re-checks the source against its own seal. The source's config and stats
are copied into the reference, so the capped-collect taint rules apply through it
unchanged. An arm never writes, moves or unseals the source: `--redo` and
`--redo-downstream` act only on the arm's own one-file reference directory.

**Freeze the source run dir once arms reference it.** The source cannot know its
arms exist, so nothing stops a `collect --redo` (or any edit) in the source after
arms reference it. The arms then refuse loudly at their next stage, because the
re-hashed `source_*` inputs no longer match, and must re-reference the new
collect, which means retraining.

Given the same `--seed`, arms see the same holdout split (a hash of the game id),
the same probe set (the first trained generation's held-out rows) and the same
batch plan (seeded by `--seed` and the generation). A test asserts that two arms
differing only in `--w-wdl` are served byte-identical batches in the same order
and write byte-identical `presented.npz`.

## Collect gates

`scripts/simple_rl_collect_gates.py <run>/gen_NNNN/collect` prints JSON
search-sanity metrics for a sealed collect (an arm's reference reads its source).
It replaces the scratch `side_stats.py`:

- row metrics come from a uniform seeded sample over all shards (`--sample-rows`,
  `--seed`), loaded one shard at a time; game metrics come from every game;
- sides are split by the parity of plies after the book: side 0 is the book-end
  mover, the side to move where the opening line ends;
- roots are "fresh" (`root_start_visits == 0`) or "reused".

It reports:

- the Spearman rank correlation of visited children's Q with their prior, and
  survivor == argmax Q, per side x root kind;
- the book-end mover's score (win 1, draw 0.5, loss 0). Its colour is white iff
  `start_ply` is even;
- the collect's own stats.json fields under their own names, recomputed over every
  row: `target_entropy`, `kl_target_prior`, `mean_survivor_minus_node_q`,
  `repeat_rate_within`;
- target entropy, KL(target‖prior) and survivor − node Q on the sample, overall
  and per side (`*_by_side`);
- rows per game and the draw / decisive rates;
- `buffer_full_flushes`, and `eval_max_batch_realised` against
  `leaf_rows_required` (`eval_max_batch_equals_leaf_rows`, and
  `eval_cap_covers_leaf_rows`), with the capped-collect record;
- with `--train-target-temp T` (repeatable), the target entropy after the loss's
  own `retemper_main_policy_target`.

`--thresholds FILE` exits 3 when a gate fails or its metric is missing or NaN.
Each gate is `{"dotted.metric": {"min": x, "max": y}}` (a finite number in range),
`{"dotted.metric": {"equals": v}}` (bools and numbers; False, None or a missing
metric fail `{"equals": true}`; a bool never equals a number), or `{}` (present:
a finite number, or True). For example, G-C2 is
`{"eval_max_batch_equals_leaf_rows": {"equals": true}}` and G-C1 is
`{"buffer_full_flushes": {"equals": 0}}`.

`--content-digest` prints only the identity of the collect's decoded content: a
SHA-256 over every shard's arrays and row sidecar, plus the SHA-256 of
`games.jsonl`. Use it to check that two collects made the same rows (for example
CUDA graphs on vs off). Their shard FILES are not comparable: on the 128-game
SL-base check, 263 of 2,676 zarr files differed between graphs on and off. The
differences were the shard metadata's wall-clock `generated_at_unix` and the
blosc-zstd chunk files of 11 arrays, while every decoded array was identical.

On the banked 128-game smoke collects, all rows:

| metric | clean (`fix_c128_auto`) | capped (`main_c128`) |
|---|---|---|
| book-end mover score | 0.582 | 0.820 |
| draw rate | 0.461 | 0.156 |
| rows per game | 96.7 | 56.9 |
| Spearman, mover fresh / reused | 0.596 / 0.581 | 0.577 / 0.497 |
| Spearman, reply fresh / reused | 0.596 / 0.602 | −0.053 / 0.096 |
| KL(target‖prior), mover / reply | 0.261 / 0.318 | 0.329 / 1.025 |

The Spearman values use average ranks for ties. `side_stats.py` used ordinal ranks
(argsort of argsort). With ordinal ranks this script reproduces its banked values
exactly (capped: 0.511 / 0.475 / −0.007 / 0.110). The capped collect has many
tied child Qs, from leaves backed with the root's Q, which is where the two rank
methods differ.

## Model gates

`scripts/simple_rl_model_gates.py --model <model.pt>` measures one checkpoint,
including a train snapshot, and prints JSON:

- `--audit-rows <npz>`: mean |Q| (|W − L|) and mean P(draw) of the value head on
  fixed audit rows (the `x` array). The model is loaded by the arena's loader. On
  CUDA the forward runs under bf16 autocast, as the banked references were made.
- `--collect <run dir> --gen g [--train-target-temp T]`: held-out policy CE
  against the stored target re-tempered at T by `retemper_main_policy_target`,
  plus policy entropy and value NLL. These are computed by the train stage's own
  `heldout_metrics`, on the same held-out rows (generation g's held-out games at
  the run's pinned `holdout_frac`). At T = 1 the CE equals the train stage's
  `heldout_after` (tested).
- `--thresholds FILE` works as in the collect gates (exit 3; missing or NaN fails).

On the 3,080-row D3 audit bank (`rl-audit/positions.npz`):

| net | this script, CPU fp32: mean \|Q\| / P(draw) | banked (bf16 CUDA forward) |
|---|---|---|
| SL base | 0.4086 / 0.5133 | 0.4088 / 0.5132 |
| F12 c12 | 0.3312 / 0.5278 | 0.3312 / 0.5281 |
| Q50 c12 | 0.2982 / 0.5141 | 0.2980 / 0.5144 |

The two columns differ only by fp32 vs bf16-autocast arithmetic (at most 0.0003).
The script's own `--device cuda` path, also bf16 autocast, is not bit-identical
to the banked figures either. It runs a different forward path from the study's,
and the review measured differences of up to 0.0002 from the banked values. Read
gates near a band edge with that tolerance in mind.

A held-out-CE gate at T ≠ 1 compares a model against the same measurement BEFORE
training. Arm S's G-T3 at T = 2 therefore needs the before value run on the
start model at the same temperature, for example
`--model <run>/gen_0000/model.pt --collect <run> --gen 1 --train-target-temp 2`.
The train stage's `heldout_before` is measured at T = 1 only.

## Stockfish

`play_batch` cannot run without a Stockfish handle: every self-play network turn
submits a label query, and the strict tablebase route requires an engine with the
explicit TB contract. There is no supported switch that turns the labels off, so
the collector runs the lightest valid engine (one worker, `--sf-nodes 1`,
MultiPV 1). Nothing trains on what it writes: every SF loss weight and the SF
value fraction are 0, and on the strict route the ply cap leaves a game unresolved
instead of asking Stockfish. Without `--syzygy` a ply-cap game is scored a draw.

## Training details

- **Row loading is streamed into preallocated arrays.** `load_window` loads only the
  keys the loss and the configured target source read (`TRAIN_KEYS` plus
  `target_source_keys(settings)`: survivor-Q value adds 2–4 small columns, the actor
  prior mix adds the two float32 1858-wide root arrays), and fills the train and
  held-out arrays shard by shard, so no full extra copy ever exists. Measured peak
  (banked under `rl-claude-20261119/loader-ram/`): about 28 KiB per row plus one
  decoded shard, against about 142 KiB per row before. A 758k-row generation peaks at
  about 20 GiB, 2M rows at about 54 GiB; the actor prior mix costs about 49 KiB per
  row. The output is value- and dtype-identical to the old loader
  (`tests/test_simple_rl_loader_identity.py`).
- `--optimizer-state fresh` (default) builds a new optimizer each generation. With
  `--warmup-steps 0`, its first Adam steps are full size regardless of gradient
  magnitude.
  The 2026-11-20 GPU smoke's first phase (46 policy-only updates at LR 3e-6 from
  the SL base, value head exactly frozen) moved the probe set a lot. Two readings
  exist, from different instruments:
  - the in-run `probe_vs_previous` (all 622 probe rows, GPU fp32): mean |ΔQ| 0.295,
    KL 0.148;
  - a CPU fp32 recheck on the first 256 probe rows: |ΔQ| 0.212, KL 0.237, 10.5% of
    top-1 moves changed.

  That drift is **not** an optimizer-start effect. The same fresh optimizer, warmup
  0 and LR 3e-6, run through this trainer on F12's cycle-1 rows, moves the same
  probe by |ΔQ| 0.018 and KL 0.030. The P1 harness reads 0.016 for the same update.
  Warmup does not remove the smoke's drift either: 46 warmup steps give |ΔQ| 0.357.
  Starting from the SL base's own optimizer state gives 0.249. The cause was the
  smoke's collected rows (see the collect `--eval-max-batch` note). The diagnostic
  is banked under the operations bank, `rl-claude-20261119/loop-smoke-gpu/optimizer-start/`.
- With `--warmup-start 0` (the default) and `--warmup-steps W > 0`, the first
  update of every fresh-optimizer phase runs at LR 0. Under the exact-once schedule,
  that first batch's rows are consumed without being learned from: one batch per
  phase, measured as an exactly zero first update. The default is unchanged. To
  avoid this, pass `--warmup-start <lr / W>` so the first step already moves. The
  proposed default is `lr / W`.
- `--init-optimizer-state checkpoint` (generation 1 only) starts the first phase
  from the init checkpoint's optimizer state. `--optimizer-state continue` cannot
  reach generation 1: there is no previous phase to continue from. Later
  generations follow `--optimizer-state` (`fresh` builds a fresh optimizer).
  - **Adopted:** every per-parameter state tensor (momentum buffers, Adam first
    and second moments) and the per-parameter step counts.
  - **Not adopted:** the scheduler, the per-group LR (this run's `--lr` schedule
    sets the rate; `initial_lr` is set to the schedule's peak), zclip state and
    the trainer step. They belong to the run that wrote the checkpoint.
  - **Param-group hyperparameters** (every group key except `params`, `lr`,
    `initial_lr` and SODA's step: weight decay, betas, eps, the Aurora polar
    knobs, ...) are replaced wholesale by `Optimizer.load_state_dict`. Each one is
    compared against this run's configured value. Any difference is REFUSED,
    listing every differing key with both values, unless you pick one:
    `--reapply-optimizer-hparams` uses the configured values;
    `--adopt-checkpoint-optimizer-hparams` keeps the checkpoint's (for example
    the SL base's polar knobs). The choice and the realised values of every group
    key are recorded (`optimizer_from_init_checkpoint`, `realised_optimizer`).

  Refused before anything is pinned or collected: a random init (no moments), a
  checkpoint with no optimizer state, or a state written by a different optimizer
  than `--optimizer` (read from the header). A parameter layout that differs (the
  `--matrix-optimizer-scope`) is refused when generation 1 trains.
  `--matrix-optimizer-scope` is a closed choice and is refused with an optimizer
  that does not read it (anything but Aurora, Muon or SOAP, unless
  `weight_decay_mode` is soda).

  The SL base was trained with Aurora on the `mlp_out` matrix scope: 48 matrices,
  matrix LR ×20, auxiliary AdamW betas (0.9, 0.95), eps 1e-8, polar knobs
  `polar_express` / 8 steps / fp16 / pp_beta 0.25 / pp_iterations 3. It was saved
  at step 680,276, with LR 6e-5 on the matrices and 3e-6 on the rest. Reproducing
  it needs `--optimizer aurora --matrix-optimizer-scope mlp_out`, plus one of the
  two group-hyperparameter choices, because its polar knobs are not the Trainer's
  defaults. With the default `nadamw` the phase is refused.
  `continue` restores the previous generation's optimizer, scheduler and step, so
  warmup happens once. The restore also brings the checkpoint's peak LR and LR
  floor, and every param-group key. A different `--lr` / `--lr-floor` is REFUSED
  unless `--reapply-optimizer-hparams` re-applies it (schedule phase kept). A
  param-group key difference follows the same refuse / reapply / adopt rule as
  above, so a run that adopted the SL base's knobs at generation 1 keeps them
  under `continue`. The values realised after the phase's steps, read off the
  param groups and scheduler, are recorded in the train manifest
  (`config.realised.optimizer`) and in `metrics.json`.
- Every batch of the exact-once schedule gives each row the coefficient
  1 / batch size: the short final batch is scaled by rows / batch size rather
  than averaged over its own rows. A non-finite gradient fails the phase (the
  stage is redone whole) instead of skipping an update whose rows were already
  consumed.
- `--lr-floor` defaults to `--lr` (flat after warmup).
- **Reproducing the SL base's optimizer setup on a FRESH optimizer:**
  `--optimizer aurora --optimizer-hparams-source init-checkpoint`. Every Aurora
  knob, the matrix LR multiplier, both weight decays, the clip mode
  (`--grad-clip`, zclip on the SL base) and the matrix scope are read from the
  init checkpoint's saved optimizer (the gen-0 copy once sealed). The scope is
  derived from the checkpoint's `opt_param_names` manifest: the one canonical
  scope whose filter selects exactly its matrix groups (`mlp_out` for the SL
  base). An explicit `--matrix-optimizer-scope` that builds a different split is
  refused. At train time the BUILT groups' parameter names and sizes must equal
  the checkpoint's, group by group, or the phase is refused
  (`config.realised.optimizer.built_split_matches_init_checkpoint`). An explicit
  knob (`--matrix-lr-multiplier`, `--matrix-weight-decay`, `--aux-weight-decay`,
  `--aurora-pp-iterations`, `--aurora-pp-beta`, `--aurora-polar-steps`,
  `--aurora-polar-method`, `--aurora-polar-dtype`, `--zclip-max-norm`) wins and
  is recorded under `optimizer_hparams_resolved.explicit`. `--lr` is the
  AdamW groups' rate and the matrix group runs at `--lr` × the multiplier.
  The SL base STARTED at `--lr 3e-5` (matrices 6e-4) and ENDED at 3e-6
  (matrices 6e-5); its saved groups are the end values. The train manifest
  records the realised group table, `zclip_initialized` and `zclip_buffer_len`
  at phase end (zclip does nothing before its 25 warm-up norms, so those two
  fields are the direct evidence it clipped).
- `--lr-schedule cosine-phase` decays each phase from `--lr` to `--lr-floor`
  over its planned optimizer updates, restarted each phase. The floor is per
  group: `--lr-floor` × group base / `--lr`, so the matrix group keeps its ×20
  ratio throughout (6e-4 → 6e-5 for `--lr 3e-5 --lr-floor 3e-6`). The Trainer
  steps the scheduler after each update from the update count, so updates 0 and
  1 both run at the peak and the last update sits one position short of the
  floor. It needs fresh optimizer state and the exact-once schedule. `flat`
  (the default) keeps the peak. A phase resumed from a snapshot continues the
  same cosine.
- `--kl-policy` / `--kl-wdl` add KL(reference‖student) on legal-move policy and on
  WDL; the reference defaults to generation 0 (`--kl-reference`).
- Value target: `--value-outcome-weight a --value-search-weight b` trains WDL on
  `a/(a+b)` × game outcome + `b/(a+b)` × search value. The search value is, by
  default (`--value-search-source survivor_q`), the survivor's completed Q `q`
  turned into a WDL: with draw `d` from `--value-draw-source` (`net`: the raw
  net's root draw probability; `zero`: 0), d is clipped to `1 - |q|`, then
  `W = (1 - d + q)/2`, `D = d`, `L = (1 - d - q)/2`, so W - L = q exactly. It is
  never the root-node average. Rows without a usable survivor Q (exact-shortcut rows,
  which carry no root statistics, or a survivor outside the searched support) keep the shard's stored `search_wdl`,
  whose draw also comes from the raw net's root draw (`mcts/_mcts_tree.c`); the
  counts are in `metrics.json` under `targets`. `--value-search-source
  stored_search_wdl` uses the stored field for every row.
- Policy target: `--policy-prior-mix λ` trains on `λ × stored target + (1 - λ) ×
  prior` over legal moves, where the prior is a frozen reference net
  (`--prior-mix-source reference`, `--prior-mix-reference`, default generation 0)
  or the searching net's stored prior (`actor`, which includes `--policy-temp`).
  `--train-target-temp` is applied after the mix, inside the loss
  (`policy_target_temp`). The reference prior is computed once per row on the
  un-mirrored input and mixed in before the Trainer's mirror augmentation, so a
  mirrored row trains on mirror(ref(x)) rather than ref(mirror(x)) -- the same
  convention as the stored search target, which is also searched on the
  un-mirrored position. Measured with the SL base on 512 smoke-run positions:
  mean KL(mirror(ref(x)) ‖ ref(mirror(x))) = 0.0117 nats (p95 0.055), top-1
  agreement 96.5%, mean total variation 0.029.
- `--sl-mix-dir` / `--sl-mix-frac` put exactly `round(frac × batch)` rows of an
  external shard corpus in every full batch (the same proportion, rounded, in the
  short final batch), with their own labels. Corpus rows come from one fixed
  order for the whole run, consumed from a cursor kept in the presentation
  records, so under the exact-once schedule each is used at most once per run
  (under `--sample-with-replacement` SL rows are drawn with replacement too and
  the cursor does not move); an exhausted corpus is an error. The sha256 of every
  corpus shard actually read, and of `probe.npz`, is in the train manifest; a
  corpus without a
  legal mask (or without `search_wdl` when the value target needs it) is refused.
