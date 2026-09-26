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
An additional hosted attempt is permitted to correct a concrete validation finding.

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

## Local readout

Local confirmation passed on Clang 17 / Torch 2.10.0+cpu. All four Bend modes
passed the real-client/raw-protocol checks, including eight reference-compared
plies each, five special/terminal roots, readiness during analysis, retained root
PID, clocks, infinite-result holding, stale-position rejection, forced stop and
next-search recovery. The deliberately held diagnostic evaluator stop readouts
were 0.254-0.292 seconds, not general latency guarantees or a neural benchmark.
All 46 cheap new contracts pass.

The existing saved-transformer checkpoint qualification passed in native mode
(86 forwards; maximum logit error 5.960464477539062e-7). The same untrained package
then drove a real UCI client through four continuation moves after eight plies
of repetition history: g1f3 g8f6 b1c3 b8c6, matching the direct native backend.
No trained checkpoint or CUDA run occurred.

The first local fixture runner found that python-chess returns Move.null(), not
None, for the correct bestmove 0000 response. Its assertion was corrected without
changing terminal engine behavior. All final cases subsequently passed.

## Hosted readout: PASS

[Run 35491065057](https://github.com/jjoshua2/DeepFin/actions/runs/35491065057),
job **106025961475**, passed every stage and clean source publication. The exact
validated executable commit is `a5e6c22ae3a8accb9e9df37753c92363c4dd7399`, directly
on #792. This follow-up changes only the readout. Published UCI source blob hashes
match the local files, including the final nullable-client assertion guards.

Locked Torch **2.14.0+cpu**, Python 3.13.15, Bun 1.4.2. The retained model is the
existing **untrained 5,043,005-parameter transformer fixture**, CPU F32 batch four,
not a trained production checkpoint or a CUDA execution.

| Bend build | Reference-compared played plies | Special/terminal roots | Held diagnostic evaluator stop |
| --- | ---: | ---: | ---: |
| Generic C | 8 | 5 | 0.258181 s |
| Forced-portable U64 | 8 | 5 | 0.254757 s |
| Native CPU target | 8 | 5 | 0.254500 s |
| UBSan C | 8 | 5 | 0.253989 s |

Each actual python-chess UCI client produced the same eight moves as the separate
CBoard-backed diagnostic reference and direct native backend:
`d2d3 d7d5 d1d2 d5d4 d2d1 d8d5 d1d2 d5a2`.
These are bounded diagnostic moves, not strength evidence. The direct backend's
PID remains the same during compatible root extension. Arbitrary positions cover
both promotion colors, en passant, checkmate and stalemate. Protocol-null moves
are used only for no-legal-move roots.

Raw transcripts also demonstrate readyok while a search is pending, no bestmove
before stop in bounded infinite mode, rejection of a position during active
search, one bestmove despite repeated stop, clean shutdown, and correct next-root
responses after interrupting a deliberately blocked diagnostic evaluator. The
stop durations above include the configured 250ms grace and are not real-time
bounds or measurements of a blocked CUDA/native neural kernel. A hard-aborted
neural stream still requires restarting the UCI frontend.

Actual native-neural UCI comparison: start with eight pre-root repeated-knight
plies, then play four four-simulation searches. Both the UCI client and direct
checkpoint-backed native session choose `g1f3 g8f6 b1c3 b8c6`. No unsearched
fallback is accepted in this check. The same exact saved package is reused;
input history is preserved and the native model remains loaded across searches.
This verifies integration with the untrained CPU model, not trained chess quality.

The inherited checkpoint qualification also passes on that package: 17 search
epochs, 119 real rows and 78 native forwards across control/batching/cancellation.
Maximum absolute native-versus-eager singleton logit error is
**5.364418029785156e-7**, with the prior 2e-6 / 2e-5 tolerances unchanged. Numerical
validation lives in that existing gate; UCI's hot path does not run an eager model
or a reference tree. UBSan applies to Bend/chess, not LibTorch/model internals.

**Ruff, Basedpyright and all 325 inexpensive tests pass**: 46 new UCI contracts
plus 279 inherited claim/draw/play/checkpoint/batching/session/root contracts.
The first hosted attempt passed Ruff and all 325 tests, but stopped before native
validation on three nullable Move type diagnostics in the client test harness.
Explicit None guards fixed those without a suppression or altered engine behavior.
No numerical, source-integrity or correctness assertion was relaxed.

## Evidence

Artifact **bend-uci-confirmation**, ID **10599590094**, 30-day retention.
ZIP SHA-256: `f38cea49490dab202340af7c82d8c99f45a7200bb32a00b3dfb7a5a46b3af051`.
- UCI/raw-protocol report: `34e8e90edbac5b4ff98882038efc0724185ff9d6ef74a70998be5b83ccf653df`.
- Neural-client report: `fbc35ce9c5aad5a089d6a60e34bf58fc202ef1cd5034d292ba2db803eaabc451`.
- Checkpoint regression: `a298238c7cf7d1a404197c255398cdf9ead128da07b192d911807cd5a75bfe06`.
- Executed/reused package: `1f99ba78573c55f21359285b8cfd27a190ce5e8a7afa5ded6dcfecd118c14c19`.
- Initial applied patch: `3fad412bb2971d3177aab9fe3c8a8da4b1e3a61d136b321a1ec70e6e9b6eacee`.
- Final run_probe.py SHA-256: `021e06cec08adb95d1b5c3605b2e6a81899ef526dda9ebf24abf7b3adac4adee`.

Only compact reports and commit identity were uploaded, not checkpoint weights,
model packages or binaries. Hashes identify tested artifacts, not guaranteed
reproducible archive bytes. No temporary workflow or staging payload is included
in the feature branch. Parent #792's ordinary CI, session and neural jobs were
observed green; its older moving-release Bend chess probe remains separately red.
The new PR's broader checks are separate from this dedicated confirmation.

## Cost and remaining limits

No new permanent workflow or native UCI execution added to recurring CI. Ordinary
pytest gains the 46 inexpensive parser/fake-backend contracts only. All existing
perft depths, native workflow budgets and production entry points are unchanged.
The explicit UCI qualification command builds/runs the four modes when requested.

Still bounded to 256 simulations and 4096 allocated nodes; depth is a horizon,
not an alpha-beta iteration. Infinite holds a bounded result, not continuous
analysis. Clock allocation is deliberately basic and cancellation/cleanup can
exceed a requested movetime. Fallback is visibly unsearched. GUI owns optional
claims/adjudication; the separate play controller retains its explicit claim API.
No pondering, searchmoves, MultiPV, tablebases or full UCI compliance claim.

No trained checkpoint/CUDA, strength, throughput, tournament time management,
subtree reuse, cross-game UCI batching or production Gumbel parity. The interface
is Python with native Bend search and optional native C++ inference. Full desktop
GUI compatibility and platforms beyond Linux remain unqualified. Independent
review has not been obtained; self-review covers protocol/state/resource ownership.
The next useful measurement is the representative checkpoint/target-host workload
through this client boundary, rather than more isolated bitboard features.

Reproduction and exact CLI/contracts: `native/bend_engine/uci_probe/README.md`.
