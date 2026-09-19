# Bend: play and search the next root in one process

## Decision / contract

Base #782: `0ff37d197eda2b389129083f7b2cdcca91872d7a`. Keep compiler
`57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`, chess/perft and production paths
unchanged. The target trained-checkpoint/CUDA run still needs the user's host;
this addresses a separate missing engine lifecycle feature rather than adding
another synthetic model or more export infrastructure.

An idle connection accepts an epoch-bound exact legal move and retains its board
and attack-table owner for the next diagnostic search. Fresh tree per search;
no subtree reuse, draw adjudication, UCI or performance promise. The external
host keeps game history for existing encoders. No fresh FEN reload or process
restart is needed for a move. No CBoard generation/push in the candidate.

Acceptance: preserve all prior session contracts in generic/portable/native/
UBSan builds. Check exact roots and subsequent node/path/statistic snapshots
against the existing CBoard-backed diagnostic reference and python-chess. Test
accepted best and non-best moves, special moves, played mate, stale/replayed
commands, epoch exhaustion, invalid flags and cancellation before advancing.
Semantic rejection must preserve board/epoch; malformed commands and an advance
while a reply is pending fail closed. Old config/reply/quit syntax stays valid.

Hosted confirmation is bounded by the existing ten-minute path-scoped session
job. No timing gate, GPU allocation, training or default perft-depth increase.
Only fake-wire/parser/history unit checks join ordinary pytest. No compiler or
production model/encoder/search change. Recovery: discard this isolated branch,
not earlier work. Self-review, not an independent review or universal proof.

## Implementation

C decodes a seven-word Command with an arity assertion. Bend verifies expected
and increasing new epochs, matches the full key against its generated legal
list, then applies the matched move. Rejection and success both return an explicit
status, current epoch and full root board. The next search constructs a new arena
from that board; its ancestor paths are relative to the new root. Advance can
only execute after a completed/cancelled search is at ready, with no pending
reply. The finite connection command budget and U32 epoch limits remain.

The host helper copies the entire python-chess move stack only after validating
the reply identity and exact child board. A malformed or missing ACK does not
commit host history; remote state is then uncertain and the peer must be closed.
An already-running batching actor is not silently rebound to a different root.

## Local readout

Clang 17, the pinned Bun/compiler, CPU only. All four native build modes passed:
66 search epochs, 52 accepted root advances, 29 semantic rejections and nine
malformed/out-of-phase wire cases **per build**. All original session checks
also passed: 38 sessions and 207 CBoard/python-chess positions per build.
The root-only reference encountered 166 distinct positions; whole node snapshots
and request paths, not just final best moves, were compared.

The lifecycle covers two eight-ply engine-selected sequences, Fool's Mate,
reversible knight repetition plus a pawn move, both sides castling in a sequence,
EP, all four promotions for both colors, pinned EP rejection, attacked castling
transit rejection, wrong special flags, rejected missing promotion, unknown U32
keys, stale/future-expected/replayed epochs, and cancellation before advancement.
U32_MAX can be consumed but never wraps. The same peer object/PID stays live.

A separate native-target optional check passed **460 input encoding comparisons**:
204 newly played roots across both root-oriented history formats and 146/175
feature formats, plus 256 actual leaf requests in subsequent searches using
root-legacy-meta/v2. Existing C history/feature encoding was compared with the
Python reference, with unchanged exact-history and narrow float-feature checks.
Root-only FEN reconstructions were required to differ from the preserved played
histories. No neural model was exported or executed for this check.

Ruff and **49 cheap tests** pass locally (34 new plus 15 original session tests).
The complete native sources were unchanged between four-mode correctness and
encoding checks; a later Python-only observer hook enables the optional per-leaf
encoding assertions without affecting default search/evaluator semantics.

## Hosted readout

Pending at feature preparation. The PR-triggered native session job runs the
four modes and original contracts. The existing neural workflow also exercises
the unchanged config/reply protocol through the new command decoder. The optional
460-comparison history check is local evidence until independently run in CI.

## Reproduction / remaining limits

See `native/bend_engine/session_probe/README.md` for the exact transaction protocol.
Run `python -m native.bend_engine.session_probe.root_probe --report FILE` after
installing the pinned compiler. Add `--modes native --check-encoding` in a prepared
CPU environment for actual new-root and descendant input comparisons.

No trained-model result, CUDA pass, full-game draw adjudication, tree reuse,
production Gumbel parity or speedup is implied. A coordinator must separately
cancel/retire queued or in-flight old epochs and reconstruct its HistoryEncoder
from the acknowledged host root before requesting another search. Live production
and all prior branches are left alone; nothing is merged or deployed here.
