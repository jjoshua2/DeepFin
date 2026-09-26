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

## Automatic draw leaves (host history, native terminal cache)

A history-owning coordinator can answer a selected leaf with reply status **3**,
W/D/L exactly `0 / 1 / 0`, and zero policy entries. Example hex record:

```
reply EPOCH REQUEST NODE 3 0 3f800000 0 0
```

This is a rule-adjudication assertion, **not a neural prediction**. The same epoch,
request, pending node and active-search checks apply before a reply can commit.
A noncanonical WDL or nonempty policy rejects the reply with stop 4 and leaves
all node statistics unchanged. Bend marks the leaf terminal with value zero,
allocates no children, and backs up zero. Repeated visits use that cached terminal
value without another host request. Normal status-0 WDL `(0,1,0)` still expands
or caches a depth-cutoff evaluation; it is never promoted to a rules proof.

`search_draws.reconstruct_leaf` copies the complete played-root history and applies
the exact ancestor keys, then checks the full leaf board and complete legal set.
The host recognizes fivefold repetition, 75 moves without a pawn move/capture, and
python-chess's conservative insufficient-material cases. Checkmate takes precedence;
Bend still handles mate/stalemate itself before making an evaluator request.
The batching Actor sends these replies before feature encoding or queue admission,
and consumes the sequence identity without reserving a neural row. The game report
includes `automatic_draw_leaves` with epoch, node, path and rule reason.

Threefold repetition and the 50-move rule are **claim options**, not forced endings.
They remain unmodeled as search actions here; a winning continuation must not be
removed merely because a draw could be claimed. The played-root `claim_available`
policy remains separate. General dead-position solving is also not implemented.
The history-owning host is part of the trusted chess boundary; Bend does not verify
its repetition evidence. A board-only cache cannot stand in for that history.
This protocol does not change the existing resource stop: exhaustion can stop a
search before it asks the host to adjudicate an unexpanded leaf.

The opt-in command checks native caching/backups, below-root boundaries, historyless
controls, invalid replies, and the original session suite in all four CPU modes:

```sh
python -m native.bend_engine.session_probe.draw_probe \
  --report artifacts/bend-search-draws.json
```

No perft depth or recurring benchmark is added. Direct synthetic session callers
that do not send status 3 retain their old behavior. The native-neural batching
Actor (and its game-controller subclass) enable the history-aware check by default.

## Optional claim action (explicit opt-in)

A status-4 reply is a normal, fully validated policy/WDL evaluation plus a
host-certified optional draw claim. Count is still the number of REAL legal
moves; a missing/malformed policy is invalid, not a terminal draw. Bend reserves
one additional node atomically before expansion and appends a known terminal-zero
claim child. Every ordinary child and its normalized prior remain available.
This differs from status 3, which asserts an automatic ending with no children.

The claim action's key is **131072**, outside the legal packed-move domain. It
has zero prior, a known Q of zero even before visits, and an unchanged board. It
is never applied as a chess move, encoded as a network action, or sent for neural
evaluation. Zero backup is independent of the unchanged side to move. At a depth
cutoff, max(0, the neural estimate) is a heuristic cutoff value, not a rule-based
terminal declaration. No extra node is allocated at that cutoff.

At an expanded root with a claim, final selection takes a visited real child
with positive empirical value, ranked by visits then key, or chooses the claim
when none qualifies. Roots without a claim retain the prior visit/key rule.
Positive estimates are not proofs of winning, and internal PUCT averages are not
minimax lower bounds. The added action is not production Gumbel-policy parity.

`claims.claim_option` supplies current/prospective threefold or fifty-move
proof evidence from the exact host history. A prospective witness move is NOT
played when claiming. Automatic endings retain priority. The host remains trusted
for chess adjudication; Bend validates the identity/payload and transition, not
the repetition evidence. Cached choices are safe only within this unique-history
epoch/tree; do not reuse them by board hash alone.

```sh
python -m native.bend_engine.session_probe.claim_probe \
  --report artifacts/bend-claim-options.json
```

This opt-in controlled-evaluator probe includes the original sessions and automatic
draw suite. It performs no model export, neural forward or perft. Ordinary tests
only add cheap evidence/Actor/controller contracts. No permanent native test depth
or default game/search policy changes.

## Optional connection-lifetime legal-move cache

The default session remains uncached. Set `DEEPFIN_SESSION_MOVE_CACHE_BITS=6`
explicitly for a 32-position cache shared across search epochs. Idle `cache` and
`clear_cache` commands expose and reset actual hit/fill/bypass counts without
resetting the root or epoch. See [cache configuration, lifetime and qualification](CACHE.md).
This reuses legal lists only, not evaluations, draw decisions, or search trees.
