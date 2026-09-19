# Bend sustained neural play and host draw adjudication

Base: #784 / 389cb8561599f54b98a6aa86be73218ade2ff426.
Compiler pin remains 57bc84edc0df32780e2c4dde44e3a2ee1a500cc9.

## Plan before confirmation

Connect existing Actor, bounded native evaluator batching and root-advance ACKs.
Bend owns each board/tree; the host owns game history, root draws and coordination.
Keep a fresh tree per played move. No production, perft, compiler or Bend source
change; no GPU, live process, training, timing gate or learned-strength claim.

Use the prior saved untrained transformer fixture and exact package reuse; do
not add a new model architecture or increase a tolerance. Compare per-row native
logits with eager singleton outputs and every tree snapshot with the existing
diagnostic PUCT reference. Compare one-real-row control, normal batching and two
cancelled old-root searches followed by scripted external moves. Hold one old
reply until the new root's first request is queued; it must be discarded. All
later trees and played moves must match the control. Reparse PGNs, including
pre-root history. No root change while a live pending row exists.

Fixture suite: neural-selected continuations from start/Kiwipete/history and a
promotion position; scripted mate, fivefold repetition, a 75-move endpoint and an
initial draw claim. Script choices must be labeled separately from neural moves.
At played roots distinguish automatic draws, an explicit optional policy to claim
(including an intended-move witness), and unfinished experiment limits/errors.
Draws inside search and exhaustive dead-position detection remain unimplemented.
Use python-chess's material-based test, not a claim of full dead-position proof.

Budget: isolated CPU, two inference threads, single-threaded Bend, at most eight
games in a group, at most 128 plies and 64 simulations per ply (fixtures use four),
180-second group deadline. Four native build modes and source-hash-checked hosted
confirmation under 15 minutes. Preserve inherited batching/session checks. No
added model export/game execution in default pytest. Add only cheap unit contracts;
the new native command stays opt-in. Self-review unless independently reviewed.

## Implementation boundary

GameActor extends the existing Actor through a small after_search hook. The
original Actor's default reset/done behavior is unchanged and regression tested.
A completed search has no active pending row before advance. The controller uses
the existing exact-key ACK helper, registers the advance epoch, adopts the copied
played history, reconstructs HistoryEncoder, then starts and registers the next
search epoch. The same native process and attack tables remain; every tree is new.

Cancelled in-flight rows can remain reserved until their batch returns. They are
not live pending work and cannot affect the newer root/request. Cancellation does
not play the partial best move: only an explicitly scripted external move can
advance after a cancelled fixture search. An unexpected stop ends unfinished and
fails the qualification command. A bad ACK leaves remote state uncertain; all
owned peers are closed rather than retried blindly.

The host uses python-chess outcomes at played roots. Automatic draws are separate
from the optional claim_available policy; prospective claims include a legal
intended-move witness without pushing that move. Mate/actual outcomes take
precedence over the experiment limit. The limit is '*', not a draw. PGN plus JSON
retain actual moves, full pre-root history, result and reason. General dead-position
solving and search-node draw values are not implemented by this controller.

## Local readout

CPU Torch 2.10.0 / Clang 17, same prior untrained 5,043,005-parameter transformer
fixture. All four modes passed three contrasts (one-real-row, batched and faults).
Each contrast played 36 moves and included one zero-search initial draw claim.
Totals: 96 game records, 432 played moves/search epochs, 1716 real inference rows,
1265 native batch calls. Eight epochs were deliberately cancelled after two
completed simulations and followed by the same scripted external move as control.
The four submitted old-root outputs were held until a new-root request was queued,
then discarded without completing that request. All subsequent complete trees
and played histories matched the control. Max logit error was 7.152557373046875e-7
against singleton eager inference, inside unchanged 2e-6 / 2e-5 CPU tolerances.
Calls/fills depend on arrival timing and do not establish throughput.

Ruff and 219 inexpensive contracts pass (37 new, 182 inherited). Actual PGNs
parse/replay with full prior history. Scripted mate, fivefold, 75 moves and a
claimed threefold produce the expected endings; ordinary ply caps remain '*'.
Initial mate/stalemate/insufficient/75-move tests send no search command.
The original session suite (38 per mode, 207 oracle positions) and original
batching/control/cancellation groups were rerun in all four modes and passed
with the existing saved transformer evaluator after the Actor refactor.

The ordinary single-game CLI also passed from pre-root moves e2e4/e7e5, producing
four neural-selected continuation plies and 16 native calls. Its PGN replayed the
complete six-ply history and reported unfinished at the explicit four-ply limit.
This is a separate local usability check, not another trained-game or speed claim.

## Hosted readout: PASS

[Confirmation run 35423665470](https://github.com/jjoshua2/DeepFin/actions/runs/35423665470),
job **105845773809**, passed every stage, including source verification and clean
publication. Exact tested non-workflow implementation commit:
`9cfc217858a34f3304cdaa5d516ba59768b6ce38`, directly on #784. The final follow-up adds
this readout and the already-checked permanent cheap-test workflow update; native
and Python executable source bytes are unchanged and match local blob hashes.

Locked Torch **2.14.0+cpu**, Bun 1.4.2, Python 3.13.15. The saved checkpoint and
package are the existing untrained transformer fixture, CPU F32 batch four with
175 root-legacy-meta/v2_threats input planes and repetition fix enabled. This is
not a trained production network or a CUDA run. The package was exported once in
the inherited checkpoint qualification, then reused for all neural-play contrasts.

All four Bend builds passed: generic C, forced-portable U64, native CPU, UBSan.
UBSan covers Bend/chess, not LibTorch or the compiled model. Per mode there are
three contrasts and eight fixtures, including a zero-move claimed draw:

| Mode | One-real-row calls / rows | Batched calls / rows | Fault calls / rows | Late old-root rows discarded |
| --- | ---: | ---: | ---: | ---: |
| Generic | 144 / 144 | 88 / 144 | 87 / 141 | 1 |
| Portable | 144 / 144 | 88 / 144 | 88 / 141 | 1 |
| Native | 144 / 144 | 88 / 144 | 88 / 141 | 1 |
| UBSan | 144 / 144 | 80 / 144 | 87 / 141 | 1 |

Total **96 game records, 432 played moves/search epochs, 1716 real neural rows and
1270 native forwards**. These are repeats of eight bounded fixtures across modes
and contrasts, NOT 96 distinct full games. Of the moves, **156 are selected by
neural search and 276 are explicitly scripted**. The scripted moves exercise
adjudication and external/opponent transitions; they are never presented as
learned choices. Eight first searches were cancelled after two simulations;
their explicit scripted move, not partial best move, advanced the root.

Every real output, including cancelled rows, matched singleton eager inference.
Maximum absolute logit error: **5.364418029785156e-7**. All later complete tree
snapshots, histories and outcomes matched the one-real-row control. Full and
partial fills 1/2/3/4 occurred. Peak reserved rows were seven normally and eight
in faults under the declared capacity. The four held old replies were discarded
only after the next root/encoder had queued an evaluation. No pending/flight work
remained at completion. Seventy-one distinct oracle positions were encountered.

| Fixture | Played plies per contrast | Result / reason |
| --- | ---: | --- |
| Start (first move scripted) | 6 | * / ply_limit |
| Kiwipete (first move scripted) | 4 | * / ply_limit |
| Pre-root played history | 4 | * / ply_limit |
| Black promotion | 1 | * / ply_limit |
| Scripted Fool's Mate | 4 | 0-1 / checkmate |
| Scripted repetition | 16 | 1/2-1/2 / fivefold_repetition |
| Scripted 75-move endpoint | 1 | 1/2-1/2 / seventyfive_moves |
| Initial optional claim | 0 | 1/2-1/2 / threefold_repetition (claimed) |

PGNs for every record parsed and replayed to the identical final full move stack
and FEN. No extra search was sent after a played endpoint. The intended-move
claim cases and initial terminal/no-command cases are additionally unit tested.

Hosted regressions also passed:
- inherited saved-checkpoint four-mode qualification: 68 epochs, 500 real rows,
  332 native calls, complete control/batched/cancellation comparisons, maximum
  logit error 5.364418029785156e-7;
- native root lifecycle: 66 searches, 52 accepted moves, 29 semantic rejections,
  nine invalid/out-of-phase records, plus 460 played-root/leaf encoding checks;
- original native session suite: 38 cases, 207 python-chess oracle positions;
- Ruff, Basedpyright with zero errors/warnings, and all **219 cheap tests**.

The first hosted attempt passed Ruff and all tests but stopped on a fake Wire's
expect parameter name not matching the Protocol keyword contract. Renaming the
test parameter fixed the four type diagnostics; no executable engine/coordinator,
numerical tolerance, test assertion or integrity check was relaxed. The final
test bytes were hash-checked in the successful run.

Call-count differences are not throughput evidence: the control also runs the
same physical batch-four package with padding, and arrival timing changes fill.
No CPU/GPU performance claim is made from this correctness verification.

## Evidence

Artifact **bend-neural-play-confirmation**, ID **10578671161**, 30-day retention.
ZIP SHA-256: `7a0863af25180436b0d4f5c14d7f47bb84d53a2b71934b389717374c9e1d28af`.
- Games report: `c64f87c3e282578cabfdb32fc94d2f590d8f676351315afc9a807fec10d82796`.
- Checkpoint regression: `ae5a6c2aaa648a933116899df8a1f6cc2c34cc6362ad062426f91e9c66ad6eac`.
- Root/history regression: `6f456e81131683896e086bc8468861223fcef204f87a3d31b289781f6be922c6`.
- Checkpoint: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Executed package: `19812a02a34de588ef63587fbbd56a31626ecd916e146a4590d80381a30f20f6`.
- Applied patch: `a4291ce441b9ab12ca93c7d92092e899c5a52cb4e5f9f46d6f020320af3c746b`.
- Corrected test SHA-256: `a5da6564fb9d527ad05f92b5ee2832629fdd356e9b648950bbfa160daac684d1`.

Only JSON reports and commit identity were uploaded, not packages, weights or
binaries. Hashes identify executed artifacts, not reproducible archive bytes.
The commit/readout preserve compact outcomes after transient artifacts expire.

## Run, cost and limits

The new command requires the EXACT checkpoint/package pair already qualified by
checkpoint_probe with a retained work directory; it does not export another model:

```sh
python -m native.bend_engine.neural_probe.play_probe \
  --checkpoint /path/to/copied/trainer.pt \
  --reuse-package artifacts/bend-run-01/checkpoint.pt2 \
  --moves e2e4 e7e5 --max-plies 8 --simulations 4 \
  --report artifacts/bend-neural-play.json
```

Use --qualification-suite for the bounded fixture/control/fault comparisons,
optionally --modes generic portable native ubsan. Device, dtype, batch and encoding
must match the reused package. CUDA retains explicit target/tolerance checks but
was NOT executed here. Use only trusted packages/checkpoints and a safe compute
window. PGNs are stored in the JSON game records. The host owns coordination,
clocks, history and root adjudication; Bend owns legal chess and diagnostic search,
and C++ owns inference. No all-Bend game-controller claim is made.

The permanent neural workflow adds only this cheap test file/path to its static
and pytest steps. No native-play run or extra export is added there. All existing
native checks and the fifteen-minute limit remain. No perft-depth changes, no
benchmark, no compiler or Bend source change, no production change, merge or
deployment. Temporary development workflow and patch transport are absent from
the feature diff. Broader PR checks are separate from the passing confirmation.

Self-review only, not independent review or a universal proof. Still unqualified:
trained-checkpoint/CUDA execution, throughput/strength, production Gumbel semantics,
search-node draw adjudication, subtree reuse, general dead positions and UCI.
This test establishes short played-root neural sequences and lifecycle composition,
not tournament readiness. The search can still misvalue a future draw because
only played roots are adjudicated by this controller.
