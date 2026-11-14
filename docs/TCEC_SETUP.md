# DeepFin: tournament operator guide

DeepFin is a GPU neural-network engine (a transformer policy/value net searched with
Gumbel MCTS) that speaks UCI. It runs as a Python process. This page is for the
operator who installs it on a tournament machine and registers it in a GUI.

Every statement here was checked against the code on `main`. Commands marked
**(CPU, run)** were run on a CPU host while writing this page. Commands marked
**(GPU, not run)** need a GPU and have not been run for this page; they are also
collected in [GPU-only checks](#gpu-only-and-unloaded-machine-checks-never-run).
`tests/test_tcec_setup_doc.py` checks this page against the code. It fails when:
the options table differs from the engine's `uci` output (names, types, defaults,
spin ranges, combo values) or a string option's range differs from the code; a
named flag does not exist, or a flag default differs from the parser; an exit code
differs from what the entry points really return; a quoted UCI command does not
parse; a quoted engine message does not exist in the code; or the watchdog,
`stop` and `Threads` statements differ from the code. The "Effect" and "Rebuilds"
columns are written from reading the code and are not checked by the test.

## 1. Requirements

- Linux x86-64, an NVIDIA GPU, and a PyTorch build whose CUDA version supports the
  card ([install.md](install.md) has the details, including RTX 50-series).
- CPython 3.10.12 to 3.13, `gcc` and the Python headers (six C extensions are
  compiled at install).
- The network file (`trainer.pt`), passed on the command line.
- Disk for PyTorch and a torch.compile cache that grows to a few GB under
  `~/.cache/deepfin/worker_cache` (change it with `--compile-cache-dir` or the
  `DEEPFIN_COMPILE_CACHE` environment variable).
- The owner states that the TCEC machine has 4 GPUs. Nothing on this page was
  measured on that machine.

## 2. Install and verify

Install the engine-only set (`match` extra) as described in
[install.md](install.md) ("Match-only install"): torch from the PyTorch wheel
index first, then `pip install ".[match]"` from a source checkout. That page is
the reference; it is not repeated here.

Then run the self-check with the flags you will give the GUI:

```bash
deepfin --selfcheck --checkpoint /path/to/trainer.pt --syzygy /tb/a:/tb/b --device cuda:0   # (GPU, not run)
deepfin --selfcheck --checkpoint /path/to/trainer.pt --devices cuda:0,cuda:1,cuda:2,cuda:3  # (GPU, not run)
```

`python -m chess_anti_engine.uci --selfcheck ...` is the same command. On CPU:

```bash
python -m chess_anti_engine.uci --selfcheck --fast --device cpu \
  --checkpoint /path/to/trainer.pt --syzygy /path/to/syzygy                                  # (CPU, run)
```

prints one line per component (`python`, `torch`, `extensions`, `freshness`,
`encoder`, `movegen`, `search`, `checkpoint`, `syzygy`, `uci`) and a summary, and
exits 0. What `freshness` says depends on how the engine was installed (both run on
CPU with a small test network):

- From a source checkout (editable install, or `PYTHONPATH` at the checkout):
  `PASS  freshness   source content matches the build record for _lc0_ext, _features_ext, _mcts_tree`
  and `SELFCHECK PASS (uci: fast: --no-compile (does NOT cover the default compiled play path))`.
- From a regular install (`pip install ".[match]"`, as install.md describes):
  `SKIP  freshness   not a source checkout (regular wheel install): no sources to compare; the ABI/symbol and behavioural checks still apply`
  and `SELFCHECK PASS (uci: fast: --no-compile (does NOT cover the default compiled play path); 1 skipped: freshness)`.

A SKIP is not a failure; the exit status is 0 unless something FAILs.

Without `--fast` the UCI exchange runs the default compiled play path, so a
cold compile happens inside the check (minutes on a real net; `--uci-timeout`
raises the 900 s reply limit). The self-check's own flags are `--json`,
`--checkpoint`, `--syzygy`, `--device`, `--devices`, `--fast`, `--uci-timeout` and
`--no-uci`; any other flag is passed to the engine. [install.md](install.md)
explains what each FAIL means.

## 3. Command lines

The engine is `deepfin` (or `python -m chess_anti_engine.uci`). `--checkpoint` is
required; `DEEPFIN_CKPT` in the environment can supply it.

**One GPU** (the configuration the search was tuned on: classic Gumbel, one
walker):

```bash
deepfin --checkpoint /path/to/trainer.pt --device cuda:0                                     # (GPU, not run)
```

**Several GPUs** (the 4-GPU path): `--devices` with more than one entry turns on the
shared-tree multi-GPU PUCV pool (one worker per GPU, PUCT descent with virtual
loss) unless `--no-multi-gpu-pucv` is given. `--search-parallel gumbel` instead
runs root-parallel Gumbel over the GPUs.

```bash
deepfin --checkpoint /path/to/trainer.pt --devices cuda:0,cuda:1,cuda:2,cuda:3              # (GPU, not run)
```

The same flags on CPU, which is what the doc test and the transcripts here used:

```bash
python -m chess_anti_engine.uci --checkpoint /path/to/trainer.pt --device cpu --no-compile  # (CPU, run)
```

Every engine flag, with the default the argument parser really has:

| Flag | Default | What it does |
|---|---|---|
| `--checkpoint` | `$DEEPFIN_CKPT` | Network file (`trainer.pt`) or checkpoint directory. |
| `--device` | `auto` | `cpu`, `cuda` or `cuda:N`. |
| `--devices` | none | Comma-separated GPU list; overrides `--device`. More than one entry enables the multi-GPU pool. |
| `--chunk-sims` | 2048 | Simulations per search chunk (UCI `ChunkSims`). |
| `--topk` | 32 | Gumbel root candidates (UCI `Topk`). |
| `--c-scale` | 0.025 | Gumbel descent value scale (UCI `CScale`). |
| `--c-visit` | 50.0 | Gumbel descent `c_visit` (UCI `CVisit`). |
| `--policy-temp` | 1.0 | Prior temperature (UCI `PolicyTemperature`). |
| `--halving-div` | 2 | Sequential-halving divisor (UCI `HalvingDiv`). |
| `--gumbel-scale` | 0.0 | Root Gumbel noise; 0 = deterministic (UCI `GumbelScale`). |
| `--c-puct` | 1.75 | PUCT constant, PUCT paths only (UCI `CPuct`). |
| `--cpuct-factor` | 3.89 | UCI `CPuctFactor`. |
| `--cpuct-base` | 38739.0 | UCI `CPuctBase`. |
| `--fpu-reduction` | 0.33 | UCI `FpuReduction`. |
| `--c-visit-root` | 900.0 | Root-only `c_visit` (UCI `CVisitRoot`). |
| `--c-scale-root` | 7.0 | Root-only `c_scale` (UCI `CScaleRoot`). |
| `--q-visit-exp-root` | -1.0 | Root value-transform exponent; < 0 = log transform (UCI `QVisitExpRoot`). |
| `--max-batch` | 1024 | GPU evaluator batch (UCI `MaxBatch`). |
| `--eval-cache-entries` | 0 | Eval cache size; 0 = off (UCI `EvalCacheEntries`). |
| `--abort-factor` | 1.0 | Early-stop aggressiveness once the best move is settled. |
| `--time-budget-scale` | 1.0 | Multiplier on the per-move budget (still capped at half the remaining clock). |
| `--optimum-fraction` | 0.7 | Soft target as a fraction of the hard budget; 0 turns the early stop off. |
| `--moves-horizon` | 50 | Upper cap on the moves-left estimate. |
| `--log-level` | `WARNING` | stderr log level. |
| `--gil-profile` | off | Per-search GIL telemetry as `info string` lines. |
| `--walkers` | 1 | 1 = classic Gumbel; > 1 = PUCT walker pool (UCI `SearchWalkers`). |
| `--vloss-weight` | 3 | Virtual-loss weight (UCI `VLossWeight`). |
| `--walker-gather` | 1 | Per-walker leaf gather (UCI `LeafGather`). |
| `--vl-gather` | 512 | Leaf gather for `UseVL` / multi-GPU (UCI `VLGather`). |
| `--multi-gpu-pucv` / `--no-multi-gpu-pucv` | auto | Force the multi-GPU pool on / off (off is for debugging). |
| `--search-parallel` | `pucv` | Multi-GPU mode: `pucv` or `gumbel` (UCI `SearchParallel`). |
| `--pucv-pending-mode` | `legacy` | `legacy` or `virtual-mean` (UCI `PUCVPendingMode`). |
| `--no-coalesce` | off | Disable walker-call coalescing (debug / benchmark only). |
| `--compile` / `--no-compile` | compile on | torch.compile the model. |
| `--compile-mode` | `max-autotune` | torch.compile mode. |
| `--compile-cache-dir` | `~/.cache/deepfin/worker_cache` | Compile cache root (env `DEEPFIN_COMPILE_CACHE`). |
| `--selfcheck` | off | Run the install check instead of the engine (section 2). |

A search-option flag outside the range its UCI option accepts makes the engine
refuse to start (exit 1, the reason on stderr).

**Compile warm-up.** The first start after install compiles the network
(minutes under `max-autotune`); the cache makes later starts faster.
`scripts/uci_prewarm.py --checkpoint /path/to/trainer.pt` drives the real UCI path
once to fill it **(GPU, not run)**. Give it exactly the engine flags you will
register (`--device` or `--devices`, `--multi-gpu-pucv`, `--search-parallel`,
`--compile-mode`, `--compile-cache-dir`, `--max-batch`, `--chunk-sims`,
`--vl-gather`, `--walkers`): a flag you leave out is left out of the engine command
too, so the engine's own default applies (`--device auto`, which picks the GPU when
one is visible). The engine inherits the prewarm's environment, so
`DEEPFIN_COMPILE_CACHE` carries over; if you register a non-default
`--compile-cache-dir`, pass the same one so that directory is the one warmed.

`readyok` is sent only after the search path is warmed, so give the engine a generous start / `isready` allowance in
the GUI. `scripts/tcec_autotune.sh --checkpoint ... [--devices ...]` sweeps batching
and parallel-search settings for nps **(GPU, not run)**; nps is not playing strength,
and its output is a throughput report, not a recommended configuration.

## 4. UCI options

The table is the engine's own `uci` output (default command line, CPU). "Rebuilds"
says whether setting it rebuilds the GPU evaluator or a search pool (seconds; done
between moves, and on failure the previous settings and evaluator stay).

| Option | Type | Default | Effect | Rebuilds |
|---|---|---|---|---|
| `Hash` | spin 1024..524288 | 16384 | Tree memory cap in MB (section 6). | no |
| `Threads` | spin 1..64 | 1 | **No effect**; accepted because GUIs send it, and answered with an `info string` saying so. | no |
| `SearchWalkers` | spin 1..64 | 1 | 1 = classic Gumbel; > 1 = PUCT walker pool, where most Gumbel options are inert. | evaluator when crossing 1 <-> > 1; walker pool |
| `LeafGather` | spin 1..64 | 1 | Leaf gather per walker; walker pool only. | walker pool |
| `UseVL` | check | false | Single-GPU virtual-loss PUCV path. | search helper |
| `UseMultiGpuPUCV` | check | false | Multi-GPU PUCV pool (on automatically with several `--devices`). | pool (+ primary evaluator) |
| `PUCVPendingMode` | combo `legacy` / `virtual-mean` | legacy | Pending accounting on batched PUCV paths. | pool reinstall |
| `SearchParallel` | combo `pucv` / `gumbel` | pucv | Multi-GPU mode; `gumbel` needs two or more devices. | pool (+ primary evaluator) |
| `RPGOpenVPA` | spin -1..4096 | -1 | Root-parallel Gumbel schedule; `SearchParallel gumbel` only. | that pool |
| `RPGMinVPA` | spin -1..4096 | -1 | Same. | that pool |
| `RPGMinKeep` | spin -1..64 | -1 | Same. | that pool |
| `RPGOpenBudgetFrac` | string (0, 1] | 0.5 | Same. | that pool |
| `VLGather` | spin 32..4096 | 512 | Leaf gather for `UseVL` / multi-GPU / `SearchParallel gumbel`. | those pools |
| `MaxBatch` | spin 64..8192 | 1024 | GPU evaluator batch. | evaluator |
| `EvalCacheEntries` | spin 0..1048576 | 0 | Eval cache entries; 0 = off. | evaluator |
| `MultiPV` | spin 1..256 | 1 | Number of `info` lines (PVs) reported. | no |
| `UCI_ShowWDL` | check | false | Add `wdl` to `info` lines. | no |
| `MoveOverheadMs` | spin 0..5000 | 30 | Subtracted from every clock-derived deadline; also moves the watchdog (section 8). | no |
| `ClockBatchMarginSigmas` | string >= 0 | 2.0 | Clock safety margin in units of measured chunk-time spread. | no |
| `SyzygyPath` | string | `<empty>` | Tablebase directories (section 7). | opens the tables |
| `Syzygy50MoveRule` | check | true | Tablebase results respect the 50-move rule (section 7). | no |
| `LogFile` | string | `<empty>` | Also write the engine log to this file. | no |
| `Ponder` | check | false | Adds the ponder move to `bestmove`; not recommended (section 8). | no |
| `PolicyTemperature` | string 0.5..5.0 | 1.0 | Prior temperature. | no |
| `CScale` | string 0.0001..100.0 | 0.025 | Gumbel descent value scale. | no |
| `CVisit` | string 0.0..100000.0 | 50.0 | Gumbel descent `c_visit`. | no |
| `CScaleRoot` | string -1.0..1000.0 | 7.0 | Root-only `c_scale`. | no |
| `CVisitRoot` | string -1.0..100000.0 | 900.0 | Root-only `c_visit`. | no |
| `QVisitExpRoot` | string -10.0..99.0 | -1.0 | Root exponent; < 0 = log root transform. | no |
| `HalvingDiv` | spin 2..8 | 2 | Sequential-halving divisor. | no |
| `Topk` | spin 2..256 | 32 | Gumbel root candidates. | no |
| `GumbelScale` | string 0.0..5.0 | 0.0 | Root noise; 0 = deterministic. | no |
| `ChunkSims` | spin 32..131072 | 2048 | Simulations per chunk. | no |
| `VLossWeight` | spin 0..64 | 3 | Virtual loss on in-flight leaves. | no |
| `MinibatchSize` | spin 0..8192 | 0 | Leaves batched by the C Gumbel search; 0 = its default. Classic Gumbel only. | no |
| `CPuct` | string 0.0001..100.0 | 1.75 | PUCT constant; PUCT paths only. | PUCT helpers |
| `CPuctFactor` | string 0.0..100.0 | 3.89 | Same. | PUCT helpers |
| `CPuctBase` | string 1.0..1000000000.0 | 38739.0 | Same. | PUCT helpers |
| `FpuReduction` | string -10.0..10.0 | 0.33 | Same. | PUCT helpers |

Leave every search option at its default unless told otherwise; the defaults are
the tuned values. A search option that cannot affect the live search path says so
(`info string ... NO EFFECT on the live search ...`). The non-standard command `searchconfig` prints
every realized search parameter as `info string searchconfig ...` lines, each
marked `LIVE`, `INERT` or `BRANCH`, read off the running worker: use it to confirm
the engine runs what the GUI configured (`info string searchconfig path=...` first).

Every value the engine cannot use is answered with an `info string`; what happens
to the setting depends on the option:

| Options | A value outside the range | A value that does not parse |
|---|---|---|
| `Hash`, `Threads`, `SearchWalkers`, `LeafGather`, `VLGather`, `MaxBatch`, `EvalCacheEntries`, `MultiPV`, `MoveOverheadMs`, `RPGOpenVPA`, `RPGMinVPA`, `RPGMinKeep` | clamped to the nearest bound, then applied: `info string Hash 512 is below the advertised minimum 1024; using 1024` | ignored: `info string Hash: 'abc' is not an integer; ignored` |
| `ClockBatchMarginSigmas` | below 0 is clamped: `info string ClockBatchMarginSigmas -1.0 is below 0; using 0.0` | kept: `... is not a number; keeping ...` |
| The search options: `PolicyTemperature`, `CScale`, `CVisit`, `CScaleRoot`, `CVisitRoot`, `QVisitExpRoot`, `HalvingDiv`, `Topk`, `GumbelScale`, `ChunkSims`, `VLossWeight`, `MinibatchSize`, `CPuct`, `CPuctFactor`, `CPuctBase`, `FpuReduction` | refused, the current value kept: `info string Topk: 1 is out of range [2, 256]; keeping 32` | kept: `info string Topk: 'abc' is not an integer; keeping 32` (spin), `info string PolicyTemperature: 'abc' is not a number; keeping 1.0` (string) |
| `RPGOpenBudgetFrac` | refused, kept: `... is not a number in (0, 1]; keeping 0.5` | the same |
| `UseVL`, `UseMultiGpuPUCV`, `UCI_ShowWDL`, `Syzygy50MoveRule`, `Ponder` | not applicable | kept: `... expected true/false, got 'maybe'; keeping false` |
| `SearchParallel`, `PUCVPendingMode` | not applicable | refused, kept: `info string SearchParallel: 'bogus' is not one of pucv / gumbel; keeping pucv` |

An unknown option (`info string setoption 'Foo': unknown option, ignored`) or a
missing value (`info string setoption Hash: no value given, ignored`) changes
nothing.

## 5. Time management in plain terms

- `movetime N`: search about `N` ms, stopping one measured chunk early so the
  answer is not late.
- Clock games (`wtime` / `btime`, with `winc` / `binc` and `movestogo` if sent):
  a reserve is set aside, half of the increment is spent per move, and the rest of
  the clock is divided over an estimate of the moves left that comes from the
  number of pieces on the board (more pieces, more moves ahead). No move may take
  more than half the remaining clock (95% on the last move before a `movestogo`
  refill). The search aims at 70% of that budget (`--optimum-fraction`) and stops
  there if the best move is settled, otherwise it continues to the hard budget.
- `MoveOverheadMs` is subtracted from every deadline; `ClockBatchMarginSigmas`
  adds a margin learned from how long this machine's search chunks take.
- `go nodes N` / `go depth N`: work bounds, no clock.
- `go mate N`: stop as soon as a mate in N moves or fewer is proven. With no other
  bound the search is capped at 60 s and says so
  (`info string go: ... -- searching at most ... ms`).
  `mate` with `infinite` searches until `stop`.
- `go infinite`: until `stop`.
- A value that cannot be used (`go movetime abc`, `go mate 0`) is named in
  `info string go: ignored unusable value(s) ...`; if that leaves no bound, the 60 s
  cap applies. A non-integer value is rounded toward zero and named.

## 6. Hash

`Hash` caps the search tree's resident memory. A visit costs about 3.3 KB, so
16384 MB holds roughly 5 million visits. When the tree is full the search stops
early with `info string tree memory full ...`; a tree also cannot address more than
about 2.1 billion nodes, and then says `tree node-id space full`. Provision about
Hash + 150 MB + the eval cache of RAM for the engine process. Between moves the
tree is reused below the move played (kept, compacted, or rebuilt).

## 7. Tablebases

`SyzygyPath` takes one or more Syzygy directories joined with `:`; each is opened
with python-chess's reader. The engine answers with
`info string SyzygyPath loaded from ...` and the WDL / DTZ table counts and the
largest piece count, or
`info string SyzygyPath ... did not open any tablebase directories`. Positions up
to the largest table set present are probed: at search leaves on every wired
search path, and at the root, where a tablebase position is answered from DTZ
without a search.

**The production layout is a pair of directories, given together.** The first holds
the 3- to 6-man WDL tables and the 3- to 5-man DTZ tables, the second the 6-man DTZ
tables, so both must be in `SyzygyPath` (`/tb/first:/tb/second`). Directory names
need not describe what is inside them; check by listing. The self-check's
`--syzygy` takes the same string and probes 3-, 5- and 6-man positions when those
tables are present.

`Syzygy50MoveRule` (default `true`): with it on, a win that the 50-move rule turns
into a draw counts as a draw. With it off, cursed wins and blessed losses count as
wins and losses, at the leaves and at the root. Setting it answers
`info string Syzygy50MoveRule set to ...`.

## 8. Positions, failures and what you will see

**Positions.** A `position` the engine can answer is taken; one it cannot is
rejected as a whole:

- Castling rights without the king and rook on their squares are cleaned:
  `info string position: normalised castling rights ...`.
- An en-passant square no double push could have left is dropped:
  `info string position: normalised en-passant square ... is impossible -> none`.
  The one exception is a square that is real except that the pushed pawn's start
  square is occupied (the opponent's pawn stands just beyond it and a capture on it
  is legal): it is kept, so a move list that plays the capture is accepted, and
  `info string position: kept en-passant square ...` says so. Either way the
  position is answered, never rejected for its en-passant square. A move list
  that then plays an "en-passant capture" with nothing to capture is rejected as
  an illegal move, as any illegal move is: `moves d4e3` after
  `4k3/8/8/8/3p4/8/8/4K3 b - e3 0 1` (no pawn on e4).
- Missing or extra kings, pawns on the back rank, the side not to move in check,
  an unparseable FEN, a malformed command, and an illegal move in the move list are
  rejected: `info string position: rejected (...); keeping the previous position; go
  answers bestmove 0000 until a valid position arrives`. The next `go` answers
  `info string go refused: the last position command was rejected (...)` and
  `bestmove 0000` (a `go ponder` / `go infinite` holds it until `ponderhit` /
  `stop`). A valid `position` or `ucinewgame` clears it.

**Strict startup and exit codes.**

| Exit | When |
|---|---|
| `0` | Normal `quit`. |
| `1` | Startup failed before `readyok` (`info string engine startup failed: ...`, and `deepfin: engine startup failed: ...` on stderr); a search-option flag out of range; the health check at a game boundary failed (below); or `quit` while a search that ignored `stop` is still running. |
| `2` | Refusal before anything starts: a required C extension is missing, stale or broken (stderr names the fix; nothing on stdout), or a bad command line (for example no `--checkpoint`). |

**A failed search never loses on time.** The move then comes from the network's
cached root policy, one bounded fresh evaluation, or the first legal move, and
says so on an `info string` with `bestmove_fallback_used=N source=... move=...`.
These lines mean something went wrong and are worth a report:

- `info string search error: ...`, followed by
  `info string WARNING search failed on ... consecutive move(s); this move is UNSEARCHED (fallback)`;
- `info string search not started; bestmove_fallback_used=...`;
- `info string search did not stop in time; answering unsearched ...` (a search
  that ignored `stop` for 1 s; it is parked and later moves are answered at once);
- `info string search incomplete reason=...` (a ponder transition past its
  deadline, or a solved root whose proof names no move; the latter is a property
  of the position, not a fault).

**Deadline watchdog.** A clocked `go` that the search will not answer in time is
answered by the engine itself:
`info string WARNING search stalled: ...` then
`... deadline watchdog answers unsearched source=... move=...`. It fires when the
search is past its deadline and has completed no chunk for 4 chunk times (at least
300 ms, at most 2 s), or unconditionally 3 ms before the real budget. The move is a
fallback (the last PV's first move, the root policy, or the first legal move).
`go infinite`, `go ponder` before `ponderhit`, and pure `go nodes` / `go depth` have
no watchdog. On a heavily loaded CPU a very short `movetime` can trip it on a
healthy search (section 9). The 3 ms is the watchdog's wake-up latency (measured
p99 0.37 ms during real searches) plus its answer (under 0.5 ms).

Caveat: the watchdog is a Python thread. A hang inside a C call that holds the GIL
stalls the watchdog too, and its wake-up waits for whichever thread holds the GIL:
measured p99 0.4 ms beside a thread holding it 1 ms at a stretch, 2.2 ms at 2 ms
stretches, 8 ms beside a thread that never releases it voluntarily (5 ms switch
interval), 16 ms with two such threads. The engine runs no such thread during a
search, but anything else in the process that does makes the last-moment answer
late by that much.

**Unhealthy state and the game boundary.** Three counted fallback moves in a row,
or a failed search-path re-warm at `isready`, print
`info string WARNING engine unhealthy: ...` once; every `go` is still answered. At
the next `isready` after a game boundary (`ucinewgame`, or `position startpos` with
no moves) the engine rebuilds the evaluator and reruns the strict startup check.
The result is one of: `info string engine health check passed; unhealthy state cleared`;
`info string WARNING engine health check passed on a DEGRADED path ...`; or
`info string engine health check failed ...` (and `deepfin: engine health check failed: ...`
on stderr) and exit 1 before `readyok`. A GUI that never
sends `isready` between games gets no re-check.

**Options mid-game.** An option that rebuilds is applied at once, between moves; if
the rebuild fails, `info string ...: evaluator rebuild failed (...); keeping the
previous evaluator and settings`. While a stuck search holds the worker only
`MultiPV` and `UCI_ShowWDL` apply.

**Ponder** works at the protocol level, but it has not been validated for
tournament play and is not recommended: leave `Ponder` false.

## 9. Known limits, stated plainly

- **No degrade path when one of several GPUs fails.** If the multi-GPU pool cannot
  be (re)installed the engine restores the previous pool, else falls back to classic
  Gumbel on one GPU and says so
  (`previous configuration could not be restored ...; fell back to the single-evaluator classic Gumbel path`).
  It does not continue on the remaining healthy GPUs. A pool that fails at startup
  fails the strict startup check (exit 1).
- **Multi-GPU playing strength has never been validated.** The multi-GPU paths
  were built and measured for throughput; all search tuning was done on one GPU with
  classic Gumbel. On the multi-GPU PUCV pool most Gumbel options are inert (PUCT
  descent).
- **Ponder is not supported for tournament use** (section 8).
- The engine never resigns or claims a draw; adjudication is the GUI's.

### GPU-only and unloaded-machine checks never run

Collected from the pull requests that built the current engine (#1261, #1263,
#1264, #1265, #1270, #1273, #1275, #1276, #1294, #1300). None has been run. Run them on the
tournament machine with the real network `<C>`, a free GPU and no other load.

1. **Self-check on the real hardware** (#1273): `deepfin --selfcheck --device cuda:0
   --checkpoint <C> --syzygy <dirs>`, then with `--devices` naming every GPU. A cold
   `max-autotune` compile may take minutes; `--uci-timeout` raises the limit. The
   card's `sm_XY` check has been tested only against a fake torch.
2. **Host toolchain mismatch** (#1273): on a host whose `c++` is newer than the
   system libstdc++, the default compiled self-check failed with
   `GLIBCXX_3.4.32 not found` inside torch.compile. Run the default (compiled)
   self-check on the target host before registering; `--fast` does not cover this.
   If it FAILs this way, put a C++ compiler that matches the system libstdc++
   first on `PATH` and run the self-check again, or run the engine with
   `--no-compile`. Whether `--no-compile` is acceptable for tournament play is
   unmeasured: its speed and strength have not been compared with the compiled
   path.
3. **Stop latency** (#1261): play time-managed games and check the output never
   contains `search did not stop in time` or `parked`; the 1 s `stop` bound and the
   0.1 s minimum `go` wait were sized from CPU measurements.
4. **Multi-GPU pool and leaving it** (#1263): `uci`, `isready`, `searchconfig`,
   `go nodes 16384`, then `setoption name UseMultiGpuPUCV value false`, `isready`,
   `searchconfig`, `go nodes 16384` with `--devices` listing two or more GPUs.
   Expect `path=pucv_pool`, then `path=gumbel`, and searched bestmoves.
5. **`MaxBatch 8192` on multi-GPU** (#1263): expect `MaxBatch set to 8192` or
   `previous configuration kept`, a `searchconfig` that agrees, and a searched move.
6. **Boundary rebuild time** (#1263): with a fault-injected search failing three
   times, time `ucinewgame` to `readyok`; it must fit the GUI's `isready` timeout.
7. **GUI behaviour on the boundary exit** (#1263): whether the GUI reports a failed
   engine start for the next game, rather than hanging, when the engine exits 1 at a
   boundary. Unverified for any GUI.
8. **One bad GPU** (#1263): make one device's pool factory fail and enable
   `UseMultiGpuPUCV`; expect `previous configuration kept` or the one-GPU fallback
   line, and a searched move.
9. **Hash on the real net** (#1264): `go movetime 120000` with `Hash 16384`; RSS
   growth should stay within Hash + about 150 MB + the eval cache. On the 4-GPU
   machine with Hash >= 64 GB, a long `go infinite` then `stop`: `info` lines at a
   regular cadence (each tree growth step was measured on CPU to stall about 50 ms
   at most), and `tree memory full` if the footprint reaches Hash. Above about
   130 GB of walker-path footprint, expect `tree node-id space full` instead.
10. **Tree reuse over a game** (#1264, #1270): a 10- to 20-move game at
    `go wtime 600000 btime 600000`; reuse on every move, and no `tree memory full`
    before the hash is full. Repeat at `Hash 4096`, where the tree fills it within
    a game, so the reuse decision moves from keeping the subtree to compacting it.
11. **Forced moves and ponder release** (#1265): a position with one legal move at
    `go wtime 60000 btime 60000 winc 1000 binc 1000` answers within one small chunk;
    `go ponder` there answers only after `ponderhit`.
12. **Ponder stop on a large tree** (#1270): `go ponder` on a > 1 GiB tree, then
    `stop`: `bestmove` within one chunk.
13. **Watchdog false fires** (#1276): 200 moves of `go movetime 1000` at
    `ChunkSims 512`, also with clocks and with several devices: expect no
    `deadline watchdog` line and 200 `bestmove` lines. Then a wedged device-1
    evaluator: one `WARNING search stalled` line and exactly one `bestmove`.
    The GPU chunk-time distribution decides the false-fire rate, so record it:
    per-chunk wall times (`--log-level DEBUG`) and the gap from the search
    deadline to `bestmove`.
14. **Short `movetime` on a loaded CPU** (#1294 review): `go movetime 200` from the
    start position on a busy CPU tripped the deadline watchdog into the root-policy
    fallback. Repeat on the unloaded tournament machine and confirm it does not.
15. **The UCI compliance changes on the real net** (#1275, #1294, #1300 were CPU-only):
    `go mate N`, position normalisation and rejection, and the ponder-miss answer
    were verified with a tiny network on CPU only.
16. **Multi-GPU pool speed** (#1263): the pool's final `nps` on `go nodes 16384`
    was expected to be within noise of the code before #1263 on the same GPUs; that
    was never measured.
17. **The GPU wheel's architecture list** (#1273): the `torch` self-check reads the
    installed wheel's arch list, but the arch lists of the cu124 and cu130 wheels
    were never read, so this page makes no claim about which cards either covers.
