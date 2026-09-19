# Bend automatic draws inside search

## Scope / predeclared gate

Base: #785 / `322c5064c2d977cba0ab7f47b791b6871130262a`.
Compiler unchanged: `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
Continue correctness/composition rather than perft tuning. No production change,
merge, GPU allocation, trained checkpoint, or playing-strength/throughput claim.

The host already owns full game history. Add a distinct, canonical rule-draw
reply to the existing native request protocol, before neural encoding/queuing.
Bend validates it, caches terminal zero, allocates no children and backs up zero.
This is not a formal proof of the host's rule evidence or a pure-Bend history
implementation. Local mate/stalemate take precedence. Claims are optional actions:
do NOT silently force threefold/50-move draws or remove winning continuations.
Only automatic draws and conservative insufficient-material detection are in scope.

Acceptance: all four native CPU variants must match complete reference snapshots
and counts on automatic root/below-root fixtures, historyless controls, cached
revisits, no-forced-claim cases, pawn reset, mate precedence and malformed terminal
replies. A normal 100%-draw neural WDL must remain nonterminal. Reject wrong epochs,
requests/nodes and noncanonical payloads without partial updates. The shared Actor
must bypass both feature encoding and batch admission for confirmed rule draws;
local identities must not permit stale/cancelled rows to commit into newer work.
Retain the original session, root and neural lifecycle checks.

Bounded CPU-only confirmation, one compiler process and two inference threads.
Use only the previous untrained transformer fixture for neural regression. No
increased perft depth, native draw traversal or extra model export in ordinary
pytest; new native draw command is opt-in. Existing CI paths remain unchanged.
Self-review only, unless a separate reviewer is actually obtained.

## Implementation

Reply status 3 requires exact W/D/L 0/1/0 and no policy. Identity/stop/pending checks
precede payload validation; success marks status 2 and consumes one reply sequence.
Cached revisits consume a simulation but no request. Status-0 WDL is unchanged.
`search_draws` validates exact legal ancestor keys, complete leaf board and legal
set using the entire played-root history. It never substitutes FEN-only history.
The game/batching Actor checks rules before its existing CBoard feature encoder.
`Batcher.record_local` advances replay protection without reserving model capacity.
A cancelled in-flight row retains storage until completion and cannot overwrite a
new-epoch rule decision. Game reports retain per-leaf rule reasons.

Limit: arena exhaustion may stop before a host query, preserving the existing
explicit resource-limit behavior rather than claiming a chess result. There is
no subtree reuse/transposition table, so cached terminals belong to a unique
history path in one epoch. Requalify that assumption before adding either feature.

## Local readout

Clang 17 / pinned compiler: generic, portable-U64, native-target and UBSan builds
pass 21 cases each and all 38 original session cases per mode. Snapshot comparisons
cover every node's board, metadata and F32 values. A fivefold leaf below the root
is requested once, then backed up zero on 11 visits with no more evaluations.
The identical board without pre-root history is not terminal. Near the 75-move
boundary, the neural-value counterfactual gives a different backed-up value while
the adjudicated path stays exactly zero. Checkmate at halfmove 150 remains a win.

255 inexpensive contracts pass (36 new, 219 inherited). Unit cases include legal
versus pinned-illegal EP repetition identity, lost castling rights, exact history
preservation, malformed requests, optional-claim separation, encoder bypass and
late cancelled replies after a local rule decision. Focused Ruff passes.

Hosted and actual native-model regression results are recorded below only when
executed. The local model environment is Torch 2.10.0+cpu, not the locked CI build.
No result is inferred from compilation or reduced neural-call counts.
