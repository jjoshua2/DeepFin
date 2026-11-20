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
refuse to start a stage while present. Exit codes: 3 tripwire, 4 blocked by a
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
  gen_NNNN/train/            model.pt (Trainer checkpoint: weights, optimizer, arch),
                             presented.npz (rows this phase presented), metrics.json, manifest.json
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
`sha256(salt, line)`, skipping a line whose final-position orbit repeats an EVAL
line already chosen. A remaining line is excluded from TRAIN when any position on
its path is in the orbit of an EVAL start, or when its final position is in the
orbit of any position on an EVAL line's path: no TRAIN game starts at, passes
through, or is a prefix of an EVAL start. TRAIN is not de-duplicated by position
(naturally recurring positions stay); the duplicate counts are reported. On the
two production books: 158,106 lines parsed, 0 parse errors, 4,000 EVAL (0 skipped
as duplicates or mirrors), 153,842 TRAIN, 219 excluded (139 pass through an EVAL
start, 80 end on an EVAL path), 0 TRAIN lines sharing a final position or orbit.
Rerunning with different `--eval-lines`, `--salt` or books into an existing split
is refused. Outputs `train.jsonl`, `eval.jsonl`, `eval.pgn` (for the arena's
`--openings`) and `summary.json` (whose `eval_max_plies` the arena replays).

**collect.** `--games` games from TRAIN lines drawn without replacement within the
generation, 95% / 5% by book as production (`--book-mix`). Both seats are the net.
`stats.json`: games, rows, W/D/L, decisive and draw rate, mean length, mate count,
TB-adjudicated share, unresolved (ply-cap) games, exact-repeat rate of the encoded
input (position and history planes) within the generation and against all
previous generations, and from the stored root statistics the target entropy,
prior entropy, KL(target‖prior) and top-1 agreement of target and prior.

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
rows by budget.

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
rebuild matches it to within 1.5e-4. A row the search returned through an exact
shortcut rather than the completed-Q arithmetic (a solved or tablebase root, a
single legal move, or the pruned-draw fallback of #1410, which plays an unsearched
drawing move) carries NO statistics, never stale ones; `stats.json` counts them
(`rows_without_root_stats`) and their value target falls back as described under
Training details. Rows whose target was replaced by the tablebase policy are
flagged in the row sidecar (`tb_policy_override`).

## Stockfish

`play_batch` cannot run without a Stockfish handle: every self-play network turn
submits a label query, and the strict tablebase route requires an engine with the
explicit TB contract. There is no supported switch that turns the labels off, so
the collector runs the lightest valid engine (one worker, `--sf-nodes 1`,
MultiPV 1). Nothing trains on what it writes: every SF loss weight and the SF
value fraction are 0, and on the strict route the ply cap leaves a game unresolved
instead of asking Stockfish. Without `--syzygy` a ply-cap game is scored a draw.

## Training details

- `--optimizer-state fresh` (default) builds a new optimizer each generation;
  `continue` restores the previous generation's optimizer, scheduler and step, so
  warmup happens once. The restore also brings the checkpoint's peak LR, LR floor
  and per-group weight decay, so a different `--lr` / `--lr-floor` would be
  silently ignored: the phase REFUSES such a difference unless
  `--reapply-optimizer-hparams` is given, which re-applies the requested values
  (schedule phase kept). The values realised after the phase's steps, read off
  the param groups and scheduler, are recorded in the train manifest
  (`config.realised.optimizer`) and in `metrics.json`.
- Every batch of the exact-once schedule gives each row the coefficient
  1 / batch size: the short final batch is scaled by rows / batch size rather
  than averaged over its own rows. A non-finite gradient fails the phase (the
  stage is redone whole) instead of skipping an update whose rows were already
  consumed.
- `--lr-floor` defaults to `--lr` (flat after warmup).
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
