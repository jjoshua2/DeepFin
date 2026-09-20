# Optional draw choices in Bend search

Base #788 / 9e4a8a6db0693d2bcccfd1dfb73ffc63cd969619. Compiler remains
57bc84edc0df32780e2c4dde44e3a2ee1a500cc9. No production or perft change.

## Predeclared scope and acceptance

Add an opt-in `search_choice` policy: current/prospective threefold and fifty-move
claims add a distinct terminal zero action, with ALL normal chess continuations
preserved. Default `automatic` and `claim_available` behavior remain. The host
retains exact history and intended-move evidence; Bend implements the choice,
selection and zero backup, not a proof of that evidence. No phantom board move
may enter policy mapping, feature encoding, root advancement, or played PGN.

Qualify in generic/portable/native/UBSan CPU modes with deterministic evaluation:
losing/tied roots may claim; a discovered winning/mating continuation beats it;
prospective witnesses are not played; current-board-only history is insufficient;
below-root options and cutoff estimates are exercised. Reject malformed identity,
count, WDL and policy with no partial commit. Include extra claim-node capacity
in the atomic allocation check. Preserve original session and automatic draw suites.
Run focused cheap contracts, plus native game/controller integration where feasible.
No GPU, trained checkpoint or strength/throughput result; no repeated full suite,
no additional perft depth, no permanent benchmark. Bound hosted confirmation at
15 minutes and keep default CI costs limited to inexpensive contracts.

## Semantics

Status 4 carries a normal evaluation plus a host-certified optional claim.
The synthetic child uses reserved key 131072, unchanged board, terminal zero and
zero policy prior. It is not a chess move, cannot be sent to an encoder and has
no model slot. Zero backup is independent of the side-to-move convention. PUCT
knows the claim Q is zero even before a visit; normal moves keep their priors.
At a depth cutoff use max(0, neural value); the node remains a heuristic cutoff,
NOT an automatic game ending. At an expanded root with a claim, choose a visited
normal child with positive empirical value, ranked by visits then key; otherwise
claim. Other roots retain their original visit/key selection. Estimates and
sample averages are not minimax proofs or guaranteed-value bounds.

Tests and hosted evidence will be recorded after execution. Self-review only.

## Local readout

Clang 17 / source-fingerprinted compiler: all four modes (generic, portable U64,
native CPU, UBSan) pass 25 targeted claim cases, 38 original session cases and
24 automatic-draw cases each. Complete node board/statistic snapshots are checked.
A mating continuation wins over a known claim; pessimistic/tied continuations can
choose the claim. Both current and prospective claims preserve ALL real children.
A root with 20 legal moves requires 22 total nodes when the claim is offered:
cap 21 fails without an accepted simulation or partial allocation; cap 22 passes.
Malformed replies are rejected at the root and after a completed simulation.
No automatic-draw behavior or no-claim root-selection behavior is changed.

All 279 cheap tests pass locally (24 new, 255 inherited). New tests cover exact
history/witness handling, action-domain separation, the actual Actor/batcher
metadata and reply status, cancellation/reset, default policy preservation and
the game controller ending without pushing a claim witness. Local NumPy and Torch
are environment versions rather than the locked hosted environment; only the
focused C encoding extensions were built in this disposable checkout for tests.
Ruff passes. Independent review and hosted confirmation are reported separately.
No model is exported or run by the new cheap tests or native claim probe.

The node array is unchanged in shape. The host is still trusted for claim evidence;
this is diagnostic PUCT, not a formal rule proof or production Gumbel equivalence.
No trained network, CUDA, end-to-end speed or playing strength was evaluated.
Defaults remain unchanged; use `play_probe --claims search_choice` explicitly.

Rules source: FIDE Laws of Chess, Articles 9.2/9.3 (current or intended-move claim)
and 9.6 (automatic endings): https://handbook.fide.com/chapter/E012023.
