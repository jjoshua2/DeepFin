# Persistent Bend search/evaluator sessions

This is a **flexibility gate**, not another perft optimization or a production
engine. It builds on the U64 legal core from #773-#776 and exercises one native
process end to end:

```
Bend board -> bounded PUCT tree -> suspend with a leaf and legal actions
           -> external policy/WDL reply -> validate -> expand and back up
           -> final root visits/best move -> reset for another search epoch
```

Bend owns chess rules, tree allocation, selection, expansion, visit/value backup,
request identity checks, numerical validation, limits, and best-move selection.
C performs the existing one-time input/table transfer and a small text transport.
The generated engine does not embed Python. Python in this gate is the external
**test evaluator** and an independent tree oracle using CBoard for chess moves.
Optional python-chess also checks every distinct oracle board's moves/children.

## Run

From the repository root, with Bun, Clang and Python 3 available:

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python -m native.bend_engine.session_probe.run_probe \
  --report artifacts/bend-search-sessions.json
# Add --python-chess when python-chess is installed.
```

The compiler remains pinned to `jjoshua2/bend@57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
The installer, source fingerprint, legal engine, previous search probes and
production paths are unchanged. No perft depth or timing test is added.

## What this qualifies

- An `Array<Node>` containing real Bend boards and F32 statistics survives
  repeated IO suspension/resumption while the immutable attack table has one owner.
- PUCT follows a dynamically growing tree, not a prerecorded two-ply topology.
- Replies contain nonconstant legal-move weights and WDL probabilities. The core
  normalizes policy, uses W-L from the node's side to move, and alternates signs
  during backup. Changed policy must observably change the selected root move.
- Terminal leaves use mate/stalemate values without an evaluator request.
  Evaluated depth-cutoff leaves cache a value and can receive repeat visits.
- An epoch resets the tree but retains the same root position and attack tables.
  Epochs must increase; request and node IDs must match the outstanding ticket.
- Cancellation, explicit backend errors, stale/mismatched replies, malformed
  probabilities and zero policy mass do not partially expand or back up a node.
  Full tree snapshots must match the last successful prefix, and a later epoch
  must still work in that same process.
- Logical capacity exhaustion is reported before any partial child allocation.
  Physical storage is always **4096 nodes**, even for a lower logical cap; this
  prototype does not claim variable physical memory allocation.

## Algorithm and wire contract

This is diagnostic PUCT, not production DeepFin Gumbel parity. Selection uses
`-child.W / child.N + 1.5 * prior * sqrt(max(1, parent.N)) / (1 + child.N)`.
An unvisited child instead uses parent mean minus `reduction * sqrt(visited prior
mass)`, with reduction 0.25 at the root and 0.15 below it. Ties use the lower
packed move key. A completed simulation includes the initial root evaluation;
therefore root child visits normally sum to completed simulations minus one.
This accounting is explicit and differs from some production root setup flows.

The private startup record is the existing legal probe's numeric board record.
Then `config EPOCH SIMULATIONS CAPACITY DEPTH` uses hexadecimal U32 words:
1-256 simulations, 1-4096 logical nodes, depth 1-32. `config 0 0 0 0` or EOF
between sessions exits. The connection has a finite 1024-command fuel budget.
It is not UCI, and a reset does not accept a different FEN in the same process yet.

For each leaf the engine prints decimal `eval EPOCH REQUEST NODE COUNT`, a
`board` record, COUNT `action KEY` records and `end_eval`. The key is a private
packed move (`src | dst<<6 | promotion<<12 | flag<<15`), **not** a DeepFin policy
index. The host replies with hexadecimal words:

```
reply EPOCH REQUEST NODE STATUS WIN_BITS DRAW_BITS LOSS_BITS COUNT POLICY_BITS...
```

F32 values are raw IEEE-754 binary32 words. Status 0 is success, 1 is backend
failure, 2 is cancellation. Success requires matching legal count, finite
probabilities in [0,1], WDL mass within 1e-5 of one, and positive policy mass.
The final `result` records epoch/completed/nodes/stop/pending/next-request, followed
by every node and `best`. Stop codes are 0=budget done, 1=capacity, 2=cancelled,
3=backend failure, 4=invalid reply. No best move is available before root expansion
or at a terminal root; the sentinel is 4294967295. These are explicit diagnostic
statuses, not silent success fallbacks. Wire/configuration failures exit 2.

Cancellation is **cooperative at a reply boundary**. The transport blocks waiting
for input: it does not implement a production evaluator timeout, interrupt an
in-flight CUDA kernel, or accept asynchronous UCI `stop`. The Python verifier owns
subprocess deadlines. There is one outstanding request; no batching or virtual
loss claim is made. Bad replies end that search epoch rather than being retried.

## Scope and next decision

Earlier #768/#769 cover the AOTI/CUDA bridge; #770/#771 cover selected production
Gumbel/MCTS semantics. This gate does not supersede them or remove the real-GPU
qualification requirement. It also does not carry history planes, map to dense or
compact policy heads, adjudicate repetition/50-move draws, reuse subtrees across
played moves, implement UCI, or qualify GPU/multithread execution.

The next architecture gate should join the existing native evaluator to real
DeepFin board/history encoding and legal policy mapping, then exercise batching,
backpressure, deadlines and cancellation. Measure end-to-end throughput there
before deciding whether pure Bend boards or the existing CBoard substrate wins.
Perft already exposes a local board-speed gap; it does not identify the dominant
cost of a neural search pipeline. Keeping C/C++/CUDA for hardware/model boundaries
is compatible with this design; rewriting every component in Bend is not a goal.

Tests are path-scoped in Actions or explicit locally. Ordinary pytest only runs
small parser/evaluator contract tests. Self-review only; these checks are not a
universal theorem, an independent human review, or evidence of playing strength.


## History-aware evaluator extension

Every evaluator request now includes `path <length> <packed keys...>` after the
board record. This is the chronological root-to-leaf path, with at most 32 moves;
root requests carry `path 0`. The verifier checks it against parent links rather
than assuming a board reconstructs its history. The existing deterministic test
evaluator remains the default. An optional test callback consumes board, actions
and path for [native neural qualification](../neural_probe/README.md).
Search allocation, PUCT semantics, capacity and cancellation rules are unchanged.
