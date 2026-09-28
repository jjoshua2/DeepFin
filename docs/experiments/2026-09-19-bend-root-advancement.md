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

The permanent session job retains its ten-minute bound. The one-off full CPU
environment and optional encoding confirmation uses a fifteen-minute cap.
No timing gate, GPU allocation, training or default perft-depth increase.
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

## Hosted readout: PASS

[Run 35421672830](https://github.com/jjoshua2/DeepFin/actions/runs/35421672830),
job **105840416140**, passed every stage, including source-hash verification,
Ruff, Basedpyright, 49 cheap tests, all four native builds, the separate history
check and clean source publication. No tests, numerical checks or compiler
fingerprints were suppressed. The tested non-workflow source was published as
`16f532239b1c8286b68ef131135178956a8e1c5b`, directly on #782. The final follow-up
adds this readout and the already-tested path-scoped session workflow; executable
source bytes are unchanged and their published blobs match the local sources.

| Bend mode | New-root searches | Accepted advances | Semantic rejections | Invalid/out-of-phase records |
| --- | ---: | ---: | ---: | ---: |
| Generic C | 66 | 52 | 29 | 9 |
| Portable U64 | 66 | 52 | 29 | 9 |
| Native CPU | 66 | 52 | 29 | 9 |
| UBSan C | 66 | 52 | 29 | 9 |

The new four-mode lifecycle checks cover **264 search epochs and 208 accepted
moves**. The original 38-session suite also passes in each mode, including its
bad evaluator replies, cancellation, capacity and terminal behavior. Its CBoard/
python-chess reference visits 207 unique positions; the new root-advance suite
visits 166. These are separate suites, not 373 asserted unique positions.

The additional native-mode encoding run passes **460 comparisons**: 204 played
root-format/feature combinations and 256 actual leaf requests after advancing.
It repeats the same 66/52/29/9 lifecycle counts and the original session suite.
No model export or neural forward is performed by this check. The first 112
history planes are exact and the existing narrow graded-feature tolerance is
unchanged. Same-board FEN reconstruction cannot substitute for played history.

Both acknowledgements and subsequent complete request/node snapshots are checked.
The ability to change roots is established, but no subtree preservation or
learned chess quality is implied. There is no end-to-end performance measurement,
trained checkpoint, CUDA execution or all-Bend history encoder in this run.

Evidence artifact **bend-root-confirmation**, ID **10576854028**, 30-day retention.
ZIP SHA-256: `9178ae0be3b837d5f911c836a1b55e5755398d1113d75d203d88b7aeef436ede`.
- Four-mode report: `3757a80e2c63cf3dbd4231ed9dfd4d474db048c97bd8c89c0496aa727ba9d112`.
- History report: `6f456e81131683896e086bc8468861223fcef204f87a3d31b289781f6be922c6`.
- Exact applied patch: `92b8d1b1de11d1b5566ba5c66d9335657ec56884e6e6f2e571a83f57c7931c55`.
Only compact JSON and commit identity are uploaded, not generated binaries,
compiler source, model packages or weights.

The permanent session workflow replaces its old command with a root lifecycle
command that includes the old suite. Its ten-minute bound and existing shallow
search budgets remain; the optional encoding check is not an additional recurring
neural export. Ordinary pytest gains only 34 cheap fake-wire/parser/history tests.
No perft-depth change or benchmark is introduced. The final PR-triggered neural
workflow separately exercises the old config/reply ABI with native model inference;
its result is not inferred from this deterministic-evaluator confirmation.

Parent #782's ordinary CPU, capped, lint and PEXT jobs were observed green, as was
its native-neural job. Its old moving-release Bend installer still fails separately.
The new PR's wider checks are reported separately, not assumed green from this run.

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
