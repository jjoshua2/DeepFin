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
  compiled at install). On CPython 3.10 to 3.12 a finished thread's operating
  system thread can still be running torch's per-thread cleanup after Python
  reports it ended; the engine waits for it before closing anything (see §8).
  CPython 3.13 is not affected.
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
`encoder`, `movegen`, `search`, `checkpoint`, `syzygy-checksums`, `syzygy`, `uci`)
and a summary, and
exits 0. What `freshness` says depends on how the engine was installed (both run on
CPU with a small test network):

- From a source checkout (editable install, or `PYTHONPATH` at the checkout):
  `PASS  freshness   source content matches the build record for _lc0_ext, _features_ext, _mcts_tree, _tb_ext`
  and `SELFCHECK PASS (uci: fast: --no-compile (does NOT cover the default compiled play path))`.
- From a regular install (`pip install ".[match]"`, as install.md describes):
  `SKIP  freshness   not a source checkout (regular wheel install): no sources to compare; the ABI/symbol and behavioural checks still apply`
  and `SELFCHECK PASS (uci: fast: --no-compile (does NOT cover the default compiled play path); 1 skipped: freshness)`.

A SKIP is not a failure; the exit status is 0 unless something FAILs.

Without `--fast` the UCI exchange runs the default compiled play path, so a
cold compile happens inside the check (minutes on a real net; `--uci-timeout`
raises the 900 s reply limit). The self-check's own flags are `--json`,
`--checkpoint`, `--syzygy`, `--syzygy-checksums`, `--syzygy-require-checksums`,
`--no-syzygy-checksums`, `--syzygy-checksum-threads`, `--device`, `--devices`,
`--fast`, `--uci-timeout` and `--no-uci` (section 7 covers the Syzygy ones); any
other flag is passed to the engine. [install.md](install.md)
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
| `--device-startup-timeout` | 1800.0 | Seconds a device's evaluator build / warm-up may go without progress before it is dropped (section 8); 0 = no limit. |
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
the GUI. With torch.compile on, every evaluator rounds each batch up to a fixed
ladder of sizes (`_COMPILED_BATCH_BUCKETS` in `inference.py`, capped at the
evaluator's `MaxBatch`), and the warm-up runs every size of that ladder on every
device before `readyok`, one `info string warmup ...: batch ... (.../..., ...s)`
line per size (device, batch size, progress, seconds so far). Each multi-GPU pool
device runs its sizes twice (`pass 1/2` pays the compile and CUDA-graph capture,
`pass 2/2` is timed): the second pass is that device's steady-state cost, which
the stuck-device check in section 8 scales from. A size the warm-up did not cover would be compiled during a move;
if that ever happens the engine says so once per size and device:
`info string WARNING first-seen batch shape after warmup: device=... entry=... batch=...`.
That line is a bug report, not a setting to tune. `scripts/tcec_autotune.sh --checkpoint ... [--devices ...]` sweeps batching
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

`SyzygyPath` takes one or more Syzygy directories joined with `:`. One prober
reads them: the C prober built into the engine (the vendored Fathom, extension
`chess_anti_engine.tb._tb_ext`), with one table set per engine process. It
answers the root, where a tablebase position is played from DTZ without a
search, the root evaluation, and the search leaves on every wired search path.
The engine opens no python-chess tablebase. Positions up to the largest table
set present are probed. A loaded path answers with two lines:

```
info string SyzygyPath loaded from '/tb/a:/tb/b': 510 WDL + 510 DTZ tables, up to 6-piece positions; leaf prober: C Fathom
info string Syzygy root prober: C Fathom (c9c6fef0): 510 WDL + 510 DTZ tables, WDL up to 6-man, DTZ up to 6-man
```

Both lines count Fathom's tables (a material and its colour mirror are one
table). The first line says the leaf prober is loaded, the second that the
root shortcut is on. The first line alone does not prove leaves are probed:
when the active search path does not use leaf tablebases the engine also
prints

```
info string note: leaf tablebase probing is inactive on search path ...; SyzygyPath reaches only the root move shortcut and root eval on this path
```

Leaf probing is on when the load line appears without that note. When some WDL tables have no DTZ table in
any listed directory the engine adds `info string warning: ... WDL tables have no
DTZ table in SyzygyPath ...`: those materials are searched instead of answered
from the root.

Loading a path initialises every table file in it (about 0.2 s for the
production 6-man pair with the files cached). Every component must be a
readable directory, the path must load at least one table, every table must
parse, and every `.rtbw` / `.rtbz` file in the directories must be one the
prober loaded (a file whose size is wrong is dropped by the prober and
refused here, named). Otherwise the engine prints
`info string error: SyzygyPath ... REFUSED by the C tablebase prober ...; tablebase probing disabled`
(naming the path and the reason)
and plays on WITHOUT tablebases (leaves and root alike) until a good
`SyzygyPath` is set; the same line is repeated before every `readyok` while the
refused path stays configured. A missing directory in the list is refused, not
skipped.

**Load-time validation is not enough.** It checks each file's header and
layout only. In the independent review's fuzz of 300 randomly corrupted tables
(PR #1327), validation refused about 167 (roughly half); about 63 were accepted
and harmless; **62 to 67 (about one in five) were accepted and then answered
WRONG WDL/DTZ with no warning**; 2 to 3 crashed the prober at probe time; and
one probe hung once. Every Syzygy prober shares this, because none of them
checksums the data. Only a checksum of every byte finds a corruption
inside a table's compressed data, so the checksum step below is required, not
optional.

**Tablebase preflight before an event** (with the `SyzygyPath` you will give the GUI):

1. **Checksums.** Run

   ```bash
   deepfin --selfcheck --no-uci --syzygy /tb/first:/tb/second --syzygy-require-checksums
   ```

   Its `syzygy-checksums` component hashes every `.rtbw` / `.rtbz` file in every
   listed directory (duplicates included) against a checksum list, in the
   GNU `md5sum` / `sha256sum` format the standard Syzygy distribution uses
   (`<hex>  <name>`, a `*` binary-mode marker, CRLF and relative subpaths such as
   `6-wdl/KQRvKR.rtbw` are accepted; entries are matched by file name). The list
   comes from, in order: `--syzygy-checksums <list>` (repeatable; the word
   `builtin` names the shipped manifest), else every `*.md5`, `*.sha256`,
   `checksum*`, `md5` or `sha256` file inside the listed directories (one list may
   cover all of them), else the manifest shipped with the engine for the standard
   3-4-5-6-man set (`chess_anti_engine/tb/syzygy_standard_3-4-5-6.md5`: the
   lichess mirror's `https://tablebase.lichess.ovh/tables/standard/md5`, every
   table up to 6 men; it has the same 1020 entries as an independently
   downloaded copy's `3-4-5-6.md5`, and matches the files themselves).
   **A list found in a table directory takes precedence over the built-in
   manifest and is not cross-checked against it**, so a list regenerated
   locally from the files (`md5sum *.rtb? > checksum.md5`) proves nothing. Use
   the list that came with the download, or pass `--syzygy-checksums builtin`
   for the standard 3-4-5-6-man set. It FAILs
   naming every mismatched or unreadable file and every malformed list line.
   A file that no list covers (for example 7-man tables checked against the
   built-in manifest), or no list at all, is `NOT VERIFIED: ...`, naming the
   files: a SKIP without `--syzygy-require-checksums`, a FAIL with it. **Without
   the flag an uncovered set is only a SKIP and the run still ends
   `SELFCHECK PASS`**, so use the flag for an event. Listed tables absent from the path are named but are not
   a failure (a list usually covers more than one install holds).
   `--no-syzygy-checksums` skips the pass (reported as `NOT VERIFIED`).
   Measured on the production 3-4-5-6-man pair (1,385 files including the
   6-man WDL tables present in both directories, 234 GB, NVMe, about 26 GB of it
   in the page cache at the start): 130 s at 1.8 GB/s on the default 4 threads
   (`--syzygy-checksum-threads`), 2 min 19 s for the whole self-check. Hashing is
   in the self-check only; the engine never hashes at startup.
2. **Validation and probes.** The same run's `syzygy` component initialises
   every table in a child process, as the engine does at `setoption`, and FAILs
   naming a corrupt or dropped file, or the table the child was on when it
   crashed or hung; then it probes known 3-, 5- and 6-man positions through both
   probers. A crash, a hang or a FAIL names the file to replace.
3. **The engine's own load line.** After `setoption name SyzygyPath value ...`
   and `isready`, the engine must print the two `SyzygyPath loaded` /
   `Syzygy root prober` lines above and no `REFUSED` line.

`setoption name SyzygyPath value <empty>` turns tablebases off:
`info string SyzygyPath cleared; tablebase probing disabled`. Changing
`SyzygyPath` mid-session reloads both probers for the next search.

**The production layout is a pair of directories, given together.** The first holds
the 3- to 6-man WDL tables and the 3- to 5-man DTZ tables, the second the 6-man DTZ
tables, so both must be in `SyzygyPath` (`/tb/first:/tb/second`). Directory names
need not describe what is inside them; check by listing. The self-check's
`--syzygy` takes the same string and probes 3-, 5- and 6-man positions when those
tables are present, through both probers, and fails if they disagree or the C
prober refuses the path.

**Troubleshooting a refused path.** Any file ending in `.rtbw` or `.rtbz` in a
table directory that the prober cannot load refuses the WHOLE path (all
tablebases off, never a silent partial set): a stray file such as
`notes.rtbw`, a mirrored name such as `KvKR.rtbw` (the table is `KRvK.rtbw`),
a broken symlink, or a file of incomplete size. The refusal names every such
file and says what to do: `remove or rename each one` (for a mirrored name, the
message gives the real table name; for an incomplete file, re-download it).
Fix the directory and send `setoption name SyzygyPath value ...` again.
The refusal `info string` can be several thousand characters long, and some
GUIs truncate long lines; the self-check prints the same refusal in full, so
`deepfin --selfcheck --no-uci --syzygy ...` is the reliable place to read it.

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

**The last PV is the move played.** Before every `bestmove X`, the last `info`
line with a `pv` (with `MultiPV` above 1, the last `multipv 1` line) starts with X
and carries X's own score; in-search lines rank moves by visits, and the played
move need not lead them. A move the search did not produce (a fallback, the
deadline watchdog, a `stop` during ponder) gets such a line with `depth`, `pv`
and at most `nodes`: no score.

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

**One GPU failing on the multi-GPU PUCV pool.** A device whose evaluator raises
(CUDA out of memory, device lost, any error) during a search is dropped; its
unfinished leaves are given back to the other devices, the move is still searched
and answered in time, and later moves search on the devices that are left. A device
that blocks inside one GPU call for too long is dropped the same way. While the
other devices still have work, "too long" is at least 5 s, and at least 10 times
the slowest call that device has completed (starting from its timed warm-up pass),
but never more than 30 s. That is deliberately patient: while the others keep
searching, a stuck device costs only speed, whereas dropping a healthy GPU costs a
whole device. Once the move has been stopped or answered, waiting buys nothing, so
the limit drops to at least 0.25 s and 5 times the slowest call, never more than
1 s; a hung GPU cannot hold up the next `go`. When the other devices have finished
their share and sit idle while one device is still inside a call, the limit is 1 s:
an idle-survivor stall over 1 s is dropped and re-admitted at the next chunk once
it answers (shorter hiccups are waited out), so a hung GPU costs about a second of
the move it hung in. The first device (`devices[0]`) also does the root evaluations; if that fails,
they move to a remaining device. Each drop is announced once:
`info string WARNING multi-GPU DEGRADED: device ... failed mid-search ...; searching on ... of ... devices`
(or `... stopped answering ...`). While any device is dropped, every `go`, every
`isready` and `searchconfig` also say so:
`info string multi-GPU degraded: ... of ... devices (... failed: ...)`. The engine
is not marked unhealthy for this; it keeps playing on the devices it has.
A device dropped for **stopping answering** whose stuck call later comes back
without an error is re-admitted before the next search chunk (within the same
`go` if it is still searching, otherwise at the next one), announced once with
`info string multi-GPU RESTORED: device ... answered again after its stall; searching on ... of ... devices`
(the late result itself is thrown away; it never reaches the search tree or the
evaluation cache). A device that **raised** is not retried until the pool is
rebuilt (an option change that rebuilds it, or a restart). If every device has failed, each
`go` is answered from the fallback described above.
At startup, a device that fails to build or warm leaves the engine running DEGRADED
on the others, announced with `... failed at startup ...` before `readyok`; the
self-check treats that line as a failure. Only when every device fails does startup
fail (exit 1). These paths were exercised with fake CPU devices and injected faults
only (see the GPU-only checks).

A device that **stops answering several times in one move** (every call just over
the 1 s limit) would otherwise print a `DEGRADED` / `RESTORED` pair every chunk.
After 2 drops within one `go` it stays out until the next `go`, said once:
`info string WARNING multi-GPU device ... stalled ... times during this search; kept out until the next go (... of ... devices)`.
The next `go` re-admits it (with the `RESTORED` line) if its last call came back.

**A device whose evaluator build or warm-up hangs at startup.** Each device's
warm-up reports progress after every batch size it completes. A device that makes
no progress for `--device-startup-timeout` seconds (default 1800, 30 minutes:
far above one cold `max-autotune` compile step, so a healthy cold start is never
dropped) is dropped with `... failed at startup (TimeoutError: evaluator build/warm-up made no progress ...)`
and the engine reaches `readyok` on the others, DEGRADED as above. Devices waiting
for their turn to compile (they compile one at a time) are not counted as stuck,
and a hung device's turn is handed to the next. The hung device may still wake up
and compile, so while it is inside its build the devices after it build without
`torch.compile` (eager, slower), announced with `... builds without torch.compile (eager, slower): a device timed out at startup may still be compiling`:
two compiles never run at once. The same limit applies to the
`--no-multi-gpu-pucv` routing chain's warm-up. When no device comes up (every pool
device hangs, or the only device or the primary evaluator hangs, so nothing at all
progresses for the timeout plus 60 s), startup fails: `info string engine startup failed: ...` with the reason
`startup made no progress for ...`, and exit 1, never a silent hang. A thread stuck in a hung build does not block
`quit` (the process exits with the status it would have had). With
`--search-parallel gumbel` a hung device fails the start (exit 1): that mode's
start is strict, as for a device that raises. A `0` timeout waits forever (the
behaviour before this limit existed).

**After a device fault.** Once any device has raised or stopped answering this
session, the engine never lets a persistent worker thread that ran evaluator work
end, and never tears the interpreter down at exit: both intermittently crashed
the process after a fault (segfaults, PR #1333). The cause found was a join
returning before the worker's OS thread had run its C++ thread-local
destructors; joins now wait for the OS thread, and the parking rule stays as a
second line of defence. The exposed combination is CPython 3.10 to 3.12 with a
torch whose per-thread teardown outlives the join. The crash was observed with
torch 2.11, not with torch 2.14, although on CPython 3.10 both leave a thread
that ran torch work alive for about 0.2 to 8 ms after its join returns. CPython
3.13 joins the OS thread itself and is not exposed.
Every join of a thread that ran evaluator work before evaluators are closed or
the process exits waits for the OS thread (at most 1 s; if that cap is hit it
logs one warning to stderr naming the thread, at most once a minute for thread
pools, and carries on). Such a worker parks instead (blocked, no CPU) until the
process exits. What stays parked, with the evaluator and
buffers it holds: one thread per device of each pool, routing chain or
root-parallel pool closed after the fault, and one call thread per primary
evaluator replaced. Each option change that rebuilds the evaluators after a fault
adds that many again (two threads and about 30 MB RSS per rebuild with two
devices on CPU; on GPU the old evaluators also keep their device memory); a
restart frees them. Resending an option at the value already in force rebuilds
nothing (`info string ... already ...; nothing rebuilt`), so a GUI that
resends its options does not add any (`SearchParallel gumbel` / `UseMultiGpuPUCV true`
still rebuild when their pool is gone, e.g. after a root-parallel retirement). The
other side of that: re-sending the same value no longer re-admits a device that
raised. That needs an option that rebuilds set to a different value, or an engine
restart; a device that stopped answering is left out of every rebuild this session,
so only a restart brings it back. Once 16 threads are parked after a fault,
an option change that would rebuild is refused and the current value kept
(`info string WARNING ... not changed: ... parked after a device fault this session ...`).
Per-`go` search threads still end after each move. `quit` after a fault
exits with the status it would have had without closing anything:
`info string quit: a device failed this session (...); exiting with status ... without closing the evaluators`
(or, only while a device call has still not returned,
`info string quit: ...; exiting with status ... without waiting for it`, the reason
reading `a multi-GPU worker is still stuck inside a dropped device's call`; a
worker parked after the fault is not stuck and gives the first line).
It first stops the search: a search that never ended (every device stuck)
gives status 1, otherwise 0. Closing a pool after a fault waits, bounded, for
each worker to finish its cleanup and park; a worker genuinely stuck in a device
call is not waited for, but closing the CUDA-owner / batch-submitter thread of a
device that is stuck can wait up to 30 s. That happens only inside an option
change (between moves, before its `readyok`) or `quit`, never inside a `go`: a
move's `bestmove` never waits on it.
An option that rebuilds the evaluators after a device got stuck (or was dropped
for a stall) this session leaves that device out at once
(`... left out: it wedged earlier this session (...)`) instead of building on it
and waiting out `--device-startup-timeout` before `readyok`; root-parallel Gumbel
then stays retired (`root-parallel Gumbel stays retired: device ... wedged earlier this session ...`).
This holds for the rest of the session even when the device later answered
again: a stall caused by load, not a fault, still keeps that device out of every
rebuild until a restart. The operator sees it as that `left out` line at each
rebuild and as the device missing from the `multi-GPU degraded: k of n devices`
line on every `go`, `isready` and `searchconfig`, which ends
`device ... out until the engine restarts (stopped answering this session; restart between games to get it back)`.
**During an event, a `left out` / `wedged earlier this session` / `out until the
engine restarts` line means: restart the engine between games** to get the
device back (the stall may have been a false positive caused by load, see the
GPU checklist below); nothing else re-admits it.

**The primary evaluator beside the pool.** The first device also evaluates every
root and runs `go searchmoves` (classic Gumbel, which honours the restriction).
Its calls run on their own thread, warmed by one measured call at startup (so a
GPU's per-thread setup is not paid on the first move, and the stall limit starts
from a measured call, as for pool devices). If it raises, or a call blocks for
more than 1 s, root evaluations and `searchmoves` searches move to the surviving pool
devices, announced once
(`info string WARNING multi-GPU DEGRADED: the primary (root-eval) evaluator ...; root evaluations and searchmoves now run on the surviving pool devices`)
and repeated on the per-`go` degraded line. A `searchmoves` search interrupted by
the failure is rerun on the pool with the same restriction, so its `bestmove` is
still one of the `searchmoves`. On the pool devices each `searchmoves` batch is
capped at `VLGather` rows (the sizes they warmed), a narrower search than on a
healthy primary. A primary dropped for a stall whose late call came back clean
is re-admitted at the next `go`
(`multi-GPU RESTORED: the primary (root-eval) evaluator answered again after its stall; root evaluations return to it`);
one that raised stays out until it is rebuilt (an option change that rebuilds it,
or a restart). The same applies with `--search-parallel gumbel`, where root
evaluations then run on a surviving root-parallel group's own thread.

**`--no-multi-gpu-pucv` (the routing chain).** Each device's calls run on that
device's own thread. A device that raises, or whose call blocks for more than 1 s
while another device is live (at least 5 s, and at least 10 times its slowest
call, capped at 30 s, when it is the last one), is dropped, the batch it was
given is redone on another device, and the move is searched; announced and
repeated exactly as on the pool, and re-admitted after a stall the same way (with
the same 2-drops-per-`go` limit). If every device fails by raising, each `go` is
answered from the fallback (`bestmove_fallback_used`) and `quit` exits 0. If every
device stalls, the last one's call is waited on up to its cap (30 s at most), so
the deadline watchdog answers that move from the fallback (`source=first_legal`),
the next `go` finds the worker still owned (`... parked ...`, answered from the
fallback), and `quit` exits 1 (a search never ended), as with the pool.

**`--search-parallel gumbel` (root-parallel Gumbel).** Its candidate subtrees
cannot continue without the failed device, so a device that raises, or blocks one
search step for more than 1 s while another device is idle (at least 5 s while all
are busy), **retires** root-parallel Gumbel: the same move continues on classic
Gumbel, and later moves stay there. It runs on the primary evaluator (with the
configured walkers / `UseVL`) when the failed device is not `devices[0]`. When it
is `devices[0]` (which the primary shares), or the primary already failed, the
primary is set aside and classic Gumbel runs on a surviving group's own
evaluator and thread instead. Announced once, naming the device that carries on
(`info string WARNING multi-GPU DEGRADED: root-parallel Gumbel device ... failed mid-search ...; this and later moves search classic Gumbel on ... until the pool is rebuilt`,
or `... stopped answering ...`) and on every `go`, `isready` and `searchconfig`
(`multi-GPU degraded: root-parallel Gumbel retired after device ...`). When a
surviving group carries classic Gumbel and then fails too, its own
`WARNING multi-GPU DEGRADED: root-parallel Gumbel device ...` line comes before
that move's `bestmove`, and the per-`go` line ends with the group that carries on
now (`...; now searching classic Gumbel on device ...`, or `no device: every root-parallel device has failed`).

**When every device has failed.** No mode can search. With the PUCV pool or
root-parallel Gumbel, a last device that stops answering cannot be taken away
from the search thread inside it: the deadline watchdog answers that move from
the fallback (`source=first_legal`), the next `go` finds the worker still owned
(`... parked ...`, answered from the fallback), and `quit` exits 1 (a search
never ended). Devices that raise end the search normally instead: fallback
answers and `quit` exits 0. The routing chain behaves as described above. In
every mode `quit` then exits without closing the evaluators (see "After a device
fault").

**Which mode to run.** The tournament configuration is the multi-GPU PUCV pool
(the default with several `--devices`). Root-parallel Gumbel and the routing
chain are secondary: their degrade paths are tested, but they are not the
configuration the engine is tuned and measured in.

**Options mid-game.** An option that rebuilds is applied at once, between moves; if
the rebuild fails, `info string ...: evaluator rebuild failed (...); keeping the
previous evaluator and settings`. An option set to the value it already has
rebuilds nothing (`... already ...; nothing rebuilt`), so it does not re-admit
a device that raised: use a different value or restart the engine. While a stuck search holds the worker only
`MultiPV` and `UCI_ShowWDL` apply.

**Ponder** works at the protocol level, but it has not been validated for
tournament play and is not recommended: leave `Ponder` false.

## 9. Known limits, stated plainly

- **Degrade paths differ by mode** (section 8). The multi-GPU PUCV pool (the
  default with several `--devices`) and the `--no-multi-gpu-pucv` routing chain
  drop a failing device and keep searching on the rest. `--search-parallel gumbel`
  retires itself and continues on classic Gumbel on `devices[0]`: one device's
  failure costs that mode's parallelism, not just the device. A GPU that hangs
  while building or warming its evaluator is timed out after
  `--device-startup-timeout` seconds without progress (section 8); before that
  limit, a startup with a hung device waits. If an option change cannot install
  the pool at all the engine restores the previous pool, else falls back to
  classic Gumbel on one GPU and says so
  (`previous configuration could not be restored ...; fell back to the single-evaluator classic Gumbel path`).
- **A device that raised is not retried** until the pool (or chain) is rebuilt;
  a device dropped for stopping answering is re-admitted once its call returns.
- **`--no-multi-gpu-pucv` with several devices evaluates one call at a time.** It
  routes each batch to one device synchronously; it is a debugging path, not a
  way to use several GPUs at speed.
- **`go searchmoves` does not use the pool's tree search.** It runs classic Gumbel
  on the first device (`devices[0]`); if that device fails, the same restricted
  search runs on the other devices (section 8), with batches capped at `VLGather`.
- **`--selfcheck --fast` does not test the warm-up.** `--fast` forces
  `--no-compile`, so bucket padding, the warm-up of every size and its completeness
  check are not exercised, and its summary says so. **Before an event run the
  self-check without `--fast`** (with the event's `--devices` and flags): it runs
  the default compiled path, requires the engine's
  `info string warmup verified: ...` line for every evaluator (every batch size
  warmed and checked, the first-seen-size guard armed), fails on any
  `first-seen batch shape after warmup` line, and reports
  `bucket warm-up verified on N evaluator(s)` when it passes.
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
8. **One bad GPU** (#1263, updated by the degrade-path changes): make one device's
   pool factory fail and enable `UseMultiGpuPUCV`; expect one
   `multi-GPU DEGRADED: device ... failed at startup` line, `searchconfig` still on
   `path=pucv_pool`, and a searched move. Then fail a device MID-search on real
   GPUs (an injected raise in its evaluator, and separately a device made to stop
   answering, e.g. a kernel that never finishes): one `DEGRADED` line, a searched
   `bestmove` within the move's time, and searched moves afterwards. Record the
   stuck-device threshold the pool used (`stall_threshold_s`, at least 1 s) against
   the real per-batch GPU latency; it was sized on CPU only. Repeat with
   `devices[0]` failing (root evaluations and `go searchmoves` move to another
   device; the `searchmoves` bestmove must stay inside the list). Repeat the raise
   and the hang with `--no-multi-gpu-pucv` and with `--search-parallel gumbel`
   (expect the root-parallel Gumbel retirement line and searched moves on
   `devices[0]`), and `quit` while the hung kernel is still running: exit 0.
   Confirm that a GPU whose calls run just over 1 s is held out after 2 drops in a
   move (one `kept out until the next go` line), not flapping every chunk.
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
    evaluator: exactly one `bestmove`, with one `multi-GPU DEGRADED ... stopped
    answering` line (and, when the move is shorter than the stuck-device
    threshold, one `WARNING search stalled` line before it), and the following
    moves searched on the remaining devices.
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
17. **Bucket warm-up on the real net** (degrade-path change): time `uci` to
    `readyok` with the default `max-autotune` and `MaxBatch 1024` (15 batch sizes
    on the primary, up to 11 per pool device at `VLGather 512`, each pool size run
    twice), on one GPU and on all four; it must fit the GUI's start allowance.
    Measure the first real call of each size after `readyok` (CUDA-graph replay
    should make it no slower than the timed `pass 2/2` call), and over at least one
    full long-time-control game confirm no `multi-GPU DEGRADED` line appears on
    healthy GPUs (no false drop at the 5 s / 10x-slowest-call threshold). Then play 20 or more moves on
    each search path and confirm no `first-seen batch shape after warmup` line.
    Whether CUDA-graph capture of the extra sizes fits in GPU memory beside the
    model at `MaxBatch 8192` was not measured. Padding each batch up to its bucket
    gave bit-identical searches on CPU; on GPU the padded rows can change kernel
    choice, so compare node counts and moves of a fixed-node search against the
    previous build.
18. **Startup hang timeout on the real net** (degrade-gaps change): record the
    longest gap between two `info string warmup ...` lines in a cold
    `max-autotune` start on each GPU (the first size of each device pays the whole
    compile). `--device-startup-timeout` (1800 s) must stay well above it; it was
    sized from the code's own note that a cold compile "can exceed 10 min", not
    measured. Then hang one device's build (e.g. a CUDA call that never returns in
    its factory) with a short `--device-startup-timeout`: `readyok` on the others
    with the `failed at startup` line; and with every device hung, exit 1.
19. **The GPU wheel's architecture list** (#1273): the `torch` self-check reads the
    installed wheel's arch list, but the arch lists of the cu124 and cu130 wheels
    were never read, so this page makes no claim about which cards either covers.
20. **False device drops over a long game** (degrade-gaps change): a device dropped
    for a stall is left out until the engine restarts, so a false positive costs
    that GPU for the rest of the game. On all four GPUs play at least three full
    long-time-control games (and one with the machine otherwise loaded) and count
    `stopped answering` / `out until the engine restarts` lines on healthy GPUs:
    the false-drop rate per game. It was never measured; the 1 s / 5 s /
    10x-slowest-call thresholds were sized on CPU. Do not retune them without it.
