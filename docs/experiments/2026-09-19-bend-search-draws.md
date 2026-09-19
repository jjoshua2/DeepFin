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
pass the initial 21 cases each and all 38 original session cases per mode. Three
additional cases (capture reset, below-root material and invalid draw after an
accepted simulation) bring the final suite to 24; that expanded suite also passed
locally in native mode before the four-mode hosted confirmation. Snapshot
comparisons cover every node's board, metadata and F32 values. A fivefold leaf
below the root is requested once, then backed up zero on 11 visits with no more
evaluations. The identical board without pre-root history is not terminal. Near
the 75-move boundary, the neural-value counterfactual gives a different backed-up
value while the adjudicated leaf stays exactly zero. Checkmate at halfmove 150
remains a win.

255 inexpensive contracts pass (36 new, 219 inherited). Unit cases include legal
versus pinned-illegal EP repetition identity, lost castling rights, exact history
preservation, malformed requests, optional-claim separation, encoder bypass and
late cancelled replies after a local rule decision. Focused Ruff passes.

Torch 2.10.0+cpu local native saved-checkpoint qualification also passed. The
native neural-play regression produced 24 repeated fixture records, 108 played
moves, 417 real input rows, 316 native calls and nine automatic leaf decisions.
Maximum native/eager logit difference was 5.960464477539062e-7. The root/history
regression passed 66 searches, 52 advances, 29 semantic rejections, nine malformed
records and 460 encoding checks, plus the original session suite. These are
correctness observations, not trained-model or performance measurements.

## Hosted readout: PASS

[Run 35472609281](https://github.com/jjoshua2/DeepFin/actions/runs/35472609281),
job **105976212044**, passed all validation and clean-source publication stages.
Exact executable implementation commit:
`0940bcd7faf9faaeccd7a524b43bfc350aadca35`, directly on #785's head. The later
readout changes only this document. Published native/Python blob identities match
the locally tested source, including the final autospecced transport test.

Locked Torch **2.14.0+cpu**, Bun 1.4.2, project Python 3.13.15. The existing
source-pinned Bend compiler and all numerical tolerances are unchanged.

| Bend configuration | Draw/cache/invalid-reply cases | Original session cases |
| --- | ---: | ---: |
| Generic C | 24 | 38 |
| Forced-portable U64 | 24 | 38 |
| Native CPU target | 24 | 38 |
| UndefinedBehaviorSanitizer C | 24 | 38 |

All final node boards/metadata/statistics and result counters match the diagnostic
reference. There are 13 distinct positions in the draw probe's oracle and 207 in
the original suite, counted separately. The 24 draw cases include nine malformed
terminal replies at the root and an additional malformed reply after one completed
simulation; rejection retains the earlier complete state with no partial expansion.

Key observations, identical across builds:
- Each automatically drawn root completes 12 simulations with one adjudication
  request, zero neural evaluations, one terminal node and zero value/accumulated W.
- A fifth repetition below the root is identified only with the preserved history.
  The drawn child receives 11 visits but is requested once and stores zero. The
  initial root's own -0.75 diagnostic evaluation remains in its aggregate W;
  later zero backups do not incorrectly overwrite that previous observation.
- The same board reached without the repetition history is not terminal.
- Crossing the 75-move boundary and capturing to insufficient material terminalize
  only the affected leaf. Captures/pawn moves reset the clock. Mate at halfmove 150
  takes precedence. Threefold and 50-move claim positions still use the evaluator.
- A status-0 neural WDL of 0/1/0 does not terminalize the position.

The hosted native root/history regression also passes its original 66 searches,
52 accepted advances, 29 semantic rejections and nine invalid/out-of-phase cases,
with all **460 played-root and descendant encoding checks**. No encoder or root
advance implementation was changed to obtain that result.

The saved-model regression uses only the existing **untrained**, 5,043,005-parameter
transformer fixture, CPU F32 batch four, 175 input planes. Checkpoint qualification
in native mode passes 17 epochs / 119 real rows / 85 native forwards. The reduced
row count is due to automatic draws being answered locally, not a throughput claim.

Sustained native neural play passes three contrasts over eight bounded fixtures:

| Contrast | Real neural rows | Native forwards | Automatic leaf draw replies |
| --- | ---: | ---: | ---: |
| One-real-row padded control | 140 | 140 | 3 |
| Batched | 140 | 88 | 3 |
| Cancellation / late old-root reply | 137 | 87 | 3 |

Total **24 game records, 108 played moves, 417 real rows and 315 native forwards**.
There are **39 neural-selected moves and 69 explicitly scripted moves**. Records
are repeats, not 24 unique complete games. Nine rule replies (six 75-move, three
insufficient-material decisions) occur before input encoding/batch admission.
Every native output matches eager singleton inference, maximum absolute logit
error **5.364418029785156e-7**, under the unchanged 2e-6 / 2e-5 tolerances. Played
histories, complete later trees and outcomes agree with the padded control, and
all PGNs replay. The fault contrast still discards one held old-root reply after
the new root has queued work. No stale completion is admitted by the local-rule
sequence update. UBSan covers the draw/chess executable in the four-mode probe,
not LibTorch or the compiled model internals; actual model regression here is
native mode only.

**Ruff, Basedpyright and all 255 cheap tests pass.** Two preceding hosted attempts
passed Ruff and all tests but stopped before native execution on test-double typing:
first an invalid cast, then a subclass constructor without a super call. The final
signature-checked `create_autospec` mock preserves the no-process test and passes
strict typing without a suppression or unsafe cast. Engine/coordinator semantics,
source hashes, tolerances and assertions were not relaxed.

## Evidence and reproduction

Artifact **bend-draw-confirmation**, ID **10593496351**, 30-day retention.
ZIP SHA-256: `dd1ee763ee5cd6b9d87dd3438a1f7f67632d96ee5fd0b06c9f95240171de2264`.
- Draw/session report: `a501e7446b7ef1d472733debbea820bffad30d79a9134b2aa8f203eec62db1cc`.
- Root/history report: `6f456e81131683896e086bc8468861223fcef204f87a3d31b289781f6be922c6`.
- Checkpoint report: `8a4422dbb9be319b167d20935441a5b5cd2437c0ffc5b047596c2a4121cbf87e`.
- Neural play report: `3edeedbfbd91f28a41b08ca702476c6a7527d7893e9fb018504496ec5bee47f9`.
- Executed package: `33bce058d4342a009ea3fed93786ecc169c05ffd81b4d24b7b03cf7675d0da6c`.
- Fixture checkpoint: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.

Only JSON reports and commit identity are uploaded, not weights/packages/binaries.
Hashes identify executed artifacts, not guaranteed reproducible archive bytes.
The clean branch contains no development workflow or patch transport files.

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python -m native.bend_engine.session_probe.draw_probe \
  --report artifacts/bend-search-draws.json
```

The draw command needs the pinned compiler, Clang and python-chess; it does not
export or execute a neural model. The shared batched Actor and GameActor apply
automatic draw adjudication by default; synthetic callers that never send status
3 retain their old diagnostic behavior. Existing path-scoped session/neural jobs
exercise changed source normally. No permanent workflow was added or widened,
no perft depths increased, and no native draw traversal/model export was added to
ordinary pytest. Only 36 new inexpensive tests enter its discovery.

## Remaining limits / decision

The host's history/rules implementation is trusted, not proven by Bend. Optional
claim actions (threefold/50 moves) are still not represented in search; the existing
played-root claim policy is separate. General dead-position solving, production
Gumbel parity, subtree reuse and transposition-history safety remain unqualified.
A resource-limit stop can precede adjudication and must never be reported as a draw.
The extra Python history/legal work has not been performance-measured; avoiding
model calls alone is not a speedup claim. No trained-model/CUDA execution, playing
strength, universal proof, independent review, merge or deployment occurred.

The useful next correctness step is to represent optional draw claims as choices,
not forced results, and qualify them against the intended production draw policy.
Actual trained-checkpoint/CUDA execution still requires the target host. Broader
PR checks are separate from this successful focused confirmation.
