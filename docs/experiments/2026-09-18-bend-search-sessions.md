# Bend search sessions: a functionality gate before more perft tuning

## Decision and scope

Build on PR #776 (`83edea36d9619a95befc95a1824942274e4eda49`), retaining
compiler `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9` and the existing legal core.
No production adoption, compiler change, perft-depth increase or speed claim.

The scalar-branch experiment demonstrated an avoidable allocation bottleneck
and reduced it. Its controlled perft still trails CBoard by 2.58-3.64x on the
measured positions. That does not establish whether neural search will be
board-limited, evaluator-limited, or constrained by integration requirements.
The next useful information is whether real Bend boards, mutable bounded search
state and external decisions compose in one persistent process.

## Gate and budget

The target is a diagnostic PUCT session with real legal topology, policy/WDL
responses and bounded storage. Retain the native tree and its pending ticket
across IO, validate replies before changing statistics, finish or cancel, then
reset for another epoch without restarting the process or reloading the tables.

Success requires exact request/board/action identities and complete tree-state
agreement with a separately written Python reference, whose legal transitions
come from actual CBoard. Optional python-chess checks every distinct oracle
position and child, not just the original starting FENs. Policy perturbation
must change the final best move. Failures must not partially commit expansions.
This is not a production MCTSTree/Gumbel semantic-parity claim.

Budget is local single-thread CPU checks and one path-scoped hosted confirmation;
no GPU, live checkout, training, arena or timing benchmark. The native session
has fixed 4096-node physical storage, a 1-4096 logical cap, 1-256 completed
simulations, maximum depth 1-32 and 1024 connection-command fuel. Existing perft
and benchmark sources remain byte-for-byte unchanged. Recovery is to discard
this isolated probe; earlier branches and production remain intact.

## Implementation and observations

`session_probe/Search.bend` owns selection, dynamic expansion, terminal/cutoff
values, prior normalization, sign-alternating side-to-move backup, request
validation and capacity decisions. It uses `Array<Node>` and the existing Bend
chess core. `main.bend` retains tree/table ownership in the evaluator continuation;
C transports bounded numeric messages and supplies the original input/tables.
The compiled candidate does not embed Python and does not call CBoard for moves.

All definitions check without `@unsafe` and without extending the compiler.
This supports feasibility of these particular compositions, not a universal
proof, a memory-efficiency claim, or proof that all engine features will fit.
The Python evaluator is deterministic and nonconstant, not a trained model.

Local Clang 17, exact final executable sources: generic, forced-portable U64,
native-target and UndefinedBehaviorSanitizer builds all passed. Each build ran
38 sessions, including 12 control/bad-reply cases, plus non-increasing-epoch
rejection and ten malformed-wire/configuration/EOF cases. Fifteen lightweight
parser tests passed separately. Python-chess and repository lint tools were not
installed locally; hosted confirmation must establish those checks.

Selected observations were identical across the four builds:

| Case | Completed simulations | Nodes used | Deepest node | Evaluator round trips |
| --- | ---: | ---: | ---: | ---: |
| Start position | 24 | 517 | 3 | 24 |
| Kiwipete | 24 | 1,135 | 4 | 24 |
| En passant | 24 | 101 | 4 | 23 |
| Initial checkmate / stalemate | 24 | 1 | 0 | 0 |
| Logical capacity 32 | 1 | 21 | 1 | 1 |
| Logical capacity 1 | 0 | 1 | 0 | 0 |
| Depth limit 1, budget 64 | 64 | 21 | 1 | 17 |

Changing the test policy changed the startpos best packed move from 1227 to
1162. Cancellation, backend errors and bad replies injected at the third
request left exactly the first two completed simulations and their full tree
state, with no partial third expansion or backup. A later valid epoch recovered
in the same process. Cached cutoff leaves reduced evaluator calls without
losing visit accounting. Both terminal roots correctly returned no best move.

## Limits and next decision

Root initialization counts as a completed simulation in this probe. Packed move
keys are private source/destination/promotion/flag words, not neural policy IDs.
Reset keeps the same root position; subtree reuse and played-move advance are
not implemented. The evaluator transport blocks: cancellation is cooperative
at a reply boundary, not asynchronous UCI stop or interruption of a CUDA call.
The verifier owns read/process deadlines; no production timeout is qualified.
Only one request may be outstanding. Physical allocation remains 4096 slots
regardless of the logical cap. No batching, virtual loss or backpressure claim.

The next gate should attach existing native AOTI code to real DeepFin history
planes and policy-index mapping, then exercise multiple pending evaluations,
timeouts/cancellation and end-to-end throughput. Actual CUDA execution remains
a separate required check on a suitable host. #770/#771's production semantic
probes remain relevant; this new diagnostic does not replace them.

Continue functionality work with performance measurement at integration gates,
not an indefinite demand to make every component pure Bend. CBoard and C++/CUDA
are valid fallback boundaries if the eventual workload favors them. Self-review
only; an independent reviewer and formal correctness proof are not claimed.

## Reproduce

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python -m native.bend_engine.session_probe.run_probe --python-chess \
  --report artifacts/bend-search-sessions.json
```

The path-scoped `Bend search sessions` workflow runs the same four-mode check.
Ordinary pytest adds only the cheap parser tests. Hosted results and any
qualification failures will be recorded on the PR; local validation duration
includes compilation/oracle/output work and must not be read as search speed.
