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
Initial mate/stalemate/insufficient/75-move tests send no search command. The
normalizer/checker, compiler, generated Bend source and perft are unchanged.

Original session and batching regressions are running separately. Hosted
confirmation and its artifact/source identities are pending.
