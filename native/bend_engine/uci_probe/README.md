# Experimental Bend UCI adapter

An opt-in interface for the existing bounded Bend diagnostic PUCT engine. It is
**not** DeepFin's production UCI engine or a strength/performance claim. The UCI
frontend is Python; actual board/search execution is the native Bend session.
Neural mode uses the previously exported C++ AOTI worker; diagnostic mode must be
selected explicitly and never masquerades as neural inference.

## Prepare and run

Use a separate development checkout. Bun, Clang and python-chess are sufficient
for the diagnostic interface; there is no Torch import or model build in that mode.

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python - <<'PY'
from pathlib import Path
from native.bend_engine.session_probe.run_probe import build
build(Path('build/bend_u64_toolchain/source'), Path('build/bend_uci'),
      'bun', 'clang', ['native'])
PY
python -m native.bend_engine.uci_probe --diagnostic \
  --bend-binary build/bend_uci/session-native
```

Enter, or let a UCI client send:

```text
uci
isready
position startpos moves e2e4 e7e5
go nodes 32
```

Wait for `bestmove` before sending another position/go. GUI launch settings must
run the module with the repository as its working directory (or PYTHONPATH).
No particular desktop GUI has been tested; the independent python-chess UCI client
and raw stdin/stdout exchanges are exercised by the qualification below.

For real neural inference, first run the existing checkpoint qualification with a
retained work directory. Use that exact checkpoint/package pair and its worker:

```sh
python -m native.bend_engine.uci_probe \
  --bend-binary artifacts/bend-run-01/bend/session-native \
  --checkpoint /path/to/copied/trainer.pt \
  --reuse-package artifacts/bend-run-01/checkpoint.pt2 \
  --worker-binary artifacts/bend-run-01/worker/aoti_worker \
  --device cpu --batch 4
```

Normal/SWA selection, target, device index and batch must match the package.
`--weights-key swa_model` and `--device cuda --device-index N` are explicit options.
Only trusted immutable native packages/binaries and copied checkpoints are suitable.
The adapter validates package/checkpoint identity and shape, but does not itself
redo eager numerical qualification or verify a previous PASS report. Use the
checkpoint tool on the actual target first. No automatic export/download, CUDA
fallback, weight replacement or diagnostic fallback is performed.

Model startup finishes before the first handshake. The native model stays loaded
across ordinary searches/new games. A single game has one real evaluator row;
unused rows in its static package are padded. Cross-game batching is not provided
by this interface. Each root/history change reconstructs its encoder. No shadow
CBoard PUCT tree, eager forward, or reference engine is run in the UCI hot path.
CBoard still supplies neural encoding; Python checks history/legality at requests.

## Supported contract and deliberate limits

Supports `uci`, `isready`, `ucinewgame`, `position startpos|fen ... [moves ...]`,
`setoption`, `go`, `stop`, `quit` and clean EOF. Options are `BendSimulations`
(1-256, default 32) and `BendDepth` (1-32, default 4). Go accepts nodes, depth,
movetime, wtime/btime, winc/binc, movestogo or infinite. All searches remain limited
to the existing 4096-node arena and at most 256 simulations. Larger requested node
or depth bounds are capped with the effective limits printed explicitly. `info
nodes` counts completed simulations, NOT allocated tree entries; allocated count
is labeled separately. Depth is a maximum horizon, not an alpha-beta completed
iteration. No centipawn score or full principal variation is fabricated.

`go infinite` performs the bounded search then holds its result until `stop`.
It does not continuously deepen beyond the diagnostic bounds. Clock allocation is
a simple conservative remaining/movestogo estimate plus 80% of increment, with
20ms reserve; this is not tournament time-management qualification. Movetime/clock
limits initiate cancellation; transport/cleanup overhead can exceed the deadline.

Only the input-loop thread prints UCI responses. `isready` replies while search is
active. `stop` first cancels cooperatively at an evaluation boundary, returning the
native final snapshot when available. After 250ms of cancellation grace it kills
only its own native peer/evaluator to release blocked pipe IO. With no completed
native answer it emits an explicitly labeled **unsearched legal fallback**. This
is not a last-completed-search estimate or a guarantee of sub-250ms wall latency.
A hard-aborted neural stream requires restarting this UCI process; subsequent
attempts fail explicitly, never switch to synthetic evaluations. Ordinary backend
errors are likewise diagnosed before any legal fallback. A fallback is not search
success. `quit`/EOF reap owned processes; they do not wait for the full search.

A new position/newgame/options/go during an active search is rejected; callers
must stop and consume bestmove first. This preserves response/root ordering. Valid
position commands are transactional and retain the complete move stack, including
repetition history. Normal forward extensions use acknowledged native root advances
and preserve the same Bend process/attack tables. Rewinds, alternate histories,
arbitrary FEN changes, and the finite connection budget recreate only that peer.
Every search has a fresh tree; no subtree reuse. No global live process is touched.

UCI does not carry the prototype's private claim action. Optional draw claims are
therefore left to the GUI; the adapter never sends reply status 4 and never emits
a claim key as bestmove. Automatic draw leaves still receive zero terminal replies.
For an automatically drawn root with legal moves, a clearly labeled legal fallback
keeps the move field valid while the GUI owns game adjudication. `bestmove 0000`
is emitted only when there are no legal moves (mate/stalemate). Use play_probe for
the explicit claim/PGN game-controller semantics instead.

Pondering, searchmoves, mate search, Chess960, MultiPV, tablebases, Hash/Threads,
within-tree batched selection and production Gumbel search are not implemented.
Unsupported limits/options are diagnosed, not silently treated as supported.
Input lines/history length and the command queue are bounded. No platform claim
beyond Linux, and no end-to-end speed or trained-model strength conclusion.

Protocol reference: Stockfish's official UCI documentation,
https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html

## Qualification

```sh
python -m native.bend_engine.uci_probe.run_probe \
  --report artifacts/bend-uci-interface.json
```

The opt-in command builds generic/portable/native/UBSan variants and tests the
actual executable with python-chess and raw UCI. It compares eight played moves
with the separate native diagnostic reference, checks root extensions without
PID changes, arbitrary/terminal positions, readiness during infinite search,
exactly one bestmove after stop, stale-position rejection, clocks, bounded abort
of a deliberately blocked evaluator, next-search recovery and clean shutdown.
These are diagnostic-evaluator results, not trained-model evidence. Real native
model qualification is separately recorded when executed. Ordinary pytest only
runs fake-backend/parsing contracts: no native compilation, search or inference.
No existing perft depths, workloads, production entry points or compiler sources
are changed. The new native client qualification is opt-in, not another default
CI workload. Review is self-review unless explicitly recorded otherwise.
