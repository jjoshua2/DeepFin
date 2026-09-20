# Bend experimental UCI interface

## Decision and predeclared gate

Base #792 / `10dc757b3ff06ba9b37008c3d94d61105eebc74a`. Compiler pin remains
`57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`. User requested continuing viability
work. Prioritize a usable external client boundary rather than more perft/rule
features. All work is isolated, with no live checkout/process, production entry
point, compiler, model, core search or perft-depth change. No merge/deployment.

Build on the existing native session and exact-package evaluator rather than
implement another search. Test actual UCI negotiation/position/go/stop/ready with
python-chess and raw pipes. Compare played choices to the diagnostic reference,
retain history on root extension, reject illegal positions transactionally and
ensure no stale bestmove crosses to a new root. Explicitly separate protocol null
moves, private claim keys and an unsearched legal fallback on interruption/error.

Budget: one local and one hosted CPU confirmation, four Bend builds, small
four-simulation/multi-ply fixtures, no training/GPU purchase or model download.
Native client tests remain opt-in. Only cheap parsing/fake-state contracts join
pytest. Numerical qualification of an existing untrained package, if run, remains
separate from trained/GPU evidence. No performance acceptance criterion. Preserve
prior branches and recover by discarding this isolated feature, not resetting them.

## Implementation and review considerations

The Python frontend owns all stdout and the immutable root snapshot for each job.
One search worker owns the native pipes; input stays live for isready/stop. No new
job can start before the old completion is consumed. Infinite search holds a
bounded result until stop instead of claiming unbounded search. Cancellation is
cooperative, with a 250ms grace then an owned-process interrupt. No available
native result means a visibly unsearched legal fallback, never invented statistics.
Hard-aborted native neural streams require restart rather than reuse.

NativeSearch uses the unchanged Bend session, not a shadow Python/CBoard search.
It reuses the peer for compatible played histories and recreates it for rewinds,
alternate history or arbitrary FEN. The existing acknowledged-root helper checks
board/history updates. Optional claim actions are not representable in ordinary
UCI: this interface leaves those to the GUI and never sends status 4. Automatic
leaf draw adjudication is retained; root-with-legal-moves output is documented.

Strict package identity and startup validation occur before the UCI handshake.
The AOTI worker stays loaded across normal searches, while singleton rows pad its
static batch. The original checkpoint tool remains the numerical validation gate.
This adapter adds neither eager validation to the hot path nor an all-Bend or
full-UCI compliance claim. Unsupported ponder/searchmoves/mate and bounds are
explicit, as are the diagnostic PUCT limitations. No centipawn scores invented.

## Readout

Local confirmation passed on Clang 17 / Torch 2.10.0+cpu; hosted confirmation
remains pending. All four Bend modes passed the real-client/raw-protocol checks,
including eight reference-compared plies each, five special/terminal roots,
readiness during analysis, retained root PID, clocks, infinite-result holding,
stale-position rejection, forced stop and next-search recovery. The deliberately
held diagnostic evaluator stop readouts were 0.254-0.292 seconds, not general
latency guarantees or a neural inference benchmark. All 46 cheap contracts pass.
The existing saved-transformer checkpoint qualification passed in native mode
(86 forwards; maximum logit error 5.960464477539062e-7). The same untrained package
then drove a real UCI client through four continuation moves after eight plies
of repetition history: g1f3 g8f6 b1c3 b8c6, matching the direct native backend.
No trained checkpoint or CUDA run occurred.

Early checks: A real python-chess client completed native diagnostic searches and stop;
the first fixture runner found that python-chess returns Move.null(), not None,
for the correct bestmove 0000 response. Its assertion was fixed without changing
terminal engine behavior. No claim of a passing final suite follows from that.

## Remaining limits

No trained checkpoint/CUDA, strength, throughput, tournament time management,
subtree reuse, cross-game UCI batching or production Gumbel parity. The interface
is Python with native Bend search and optional native C++ inference. Full desktop
GUI compatibility and full UCI feature coverage are unqualified. Independent
review has not been obtained; self-review covers protocol/state/resource ownership.

Reproduction and exact CLI/contracts: `native/bend_engine/uci_probe/README.md`.
