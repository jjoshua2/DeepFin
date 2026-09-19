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
- A search epoch resets the tree and retains the current root and attack tables.
  The separate idle-boundary `advance` command can change that root.
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
It is not UCI. A search reset does not accept a different FEN; an idle-boundary
`advance` command can play one exact legal move before the next search.

For each leaf the engine prints decimal `eval EPOCH REQUEST NODE COUNT`, a
`board` record, a `path LENGTH KEY...` record, COUNT `action KEY` records and
`end_eval`. The path is root-to-leaf and at most 32 plies; the root has `path 0`.
The updated verifier requires it and fails closed against an older binary. The key is a private
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
qualification requirement. The added root-to-leaf path enables the separate
[neural boundary gate](../neural_probe/README.md) to reconstruct pre-root and
search history; this session command itself does not carry history planes, map to dense or
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


## Play a move, then search the new root

At `ready`, send `advance EXPECTED_EPOCH NEW_EPOCH KEY` (hexadecimal U32 words).
The expected epoch must equal the last acknowledged search/advance epoch, and
NEW_EPOCH must be greater. KEY is the complete private move key including its
promotion and special-move flags. It may be any legal move, not just the best move
or an expanded child. C only parses this command; Bend generates the complete
legal set and applies exactly the matching move.

The reply is decimal `advance_result EXPECTED_EPOCH NEW_EPOCH KEY STATUS CURRENT_EPOCH`,
then a complete `board` record, then `ready`. Status 0 means accepted, 1 rejects a
stale/non-increasing epoch, and 2 rejects an unmatched legal move key. Rejection
changes neither root nor epoch. Success consumes NEW_EPOCH; a subsequent `config`
must use a higher epoch. The counter never wraps: after U32_MAX, quit and establish
a new connection. Malformed transport exits 2. An advance sent while waiting for
an evaluator reply is not an asynchronous stop: it is rejected as a wrong record.
Cancel/complete the search and consume its final snapshot/`ready` first.

Example from a newly loaded starting position (epoch zero):

```text
advance 0 1 70c
advance_result 0 1 1804 0 1
board ...
ready
config 2 4 1000 2
```

That plays e2e4, then starts four diagnostic simulations from Black's new root.
The previous tree has already been discarded; this is deliberately **not subtree
reuse**. The same immutable attack-table owner and process are retained. Request
paths in the following search start at the new root, rather than at game start.

`root_protocol.advance_root` validates transaction identity, status, the complete
returned board and `ready` before returning a stack-preserving host board copy.
The caller adopts that copy only on success. No input history is mutated on
rejection. A corrupt/lost acknowledgement leaves remote state uncertain: close
the peer rather than retrying blindly. An evaluator coordinator must also retire
old work and advance its epoch before using the new root; this helper does not
silently migrate existing batching actors or encode with an old HistoryEncoder.
Construct the next encoder from the returned board including its move stack.

The opt-in lifecycle gate includes the original session suite and new-root
searches, all promotions, castling, EP, played mate, stale/replayed commands,
cancellation/recovery, and atomic semantic rejection:

```sh
python -m native.bend_engine.session_probe.root_probe \
  --report artifacts/bend-root-advance.json

# Requires the CPU encoding extension; compares actual subsequent leaf/history
# inputs too, but does not compile or execute a neural model:
python -m native.bend_engine.session_probe.root_probe --modes native --check-encoding \
  --report artifacts/bend-root-encoding.json
```

This still uses the deterministic diagnostic evaluator, not trained weights or
production Gumbel. Pre-root history/clocks live in the host; Bend does not yet
adjudicate repetition, the 50/75-move rules or dead positions. Scripted repetition
therefore tests history preservation, not a complete tournament game result.
Perft and the production board/search/evaluator paths are unchanged. The existing
path-scoped session job adds these bounded shallow lifecycle checks; ordinary
pytest only gains ACK/history tests with a fake wire, no subprocess or search.
