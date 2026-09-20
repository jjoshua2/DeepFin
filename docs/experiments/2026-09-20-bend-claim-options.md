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

## Hosted readout: PASS

[Run 35489187825](https://github.com/jjoshua2/DeepFin/actions/runs/35489187825),
job **106021020785**, passed all stages on the first confirmation attempt. The
exact executable implementation was published as
`883d6161312327756c43d8a532cba6e5d69a86dd`, directly on #788's head. This later
commit only records observations; no executable changes were made after the pass.
The applied patch SHA-256 was checked before testing. The temporary workflow and
transport blobs are not part of the published feature tree.

| Bend mode | Optional-claim cases | Original session cases | Automatic-draw cases |
| --- | ---: | ---: | ---: |
| Generic C | 25 | 38 | 24 |
| Portable U64 | 25 | 38 | 24 |
| Native CPU | 25 | 38 | 24 |
| UBSan C | 25 | 38 | 24 |

The claim suite's oracle encounters 17 distinct board positions; history-aware
claim availability is checked separately and is not cached by that board identity.
Complete node/board/statistic snapshots match the diagnostic reference. Important
observations, identical across native variants:

| Case | Completed simulations | Nodes | Selected action |
| --- | ---: | ---: | --- |
| Current threefold | 1 | 22 | claim, all 20 real root moves retained |
| Prospective threefold | 1 | 24 | claim, witness f6g8 not played |
| Current fifty-move | 1 | 17 | claim, all real moves retained |
| Prospective fifty-move | 1 | 17 | claim, witness a1a2 not played |
| Tied continuations | 16 | 22 | claim |
| Losing sampled continuations | 32 | 22 | claim |
| Immediate mate versus available claim | 8 | 25 | c7b7, not claim |
| Capacity 21, 20 real moves plus claim | 0 | 1 | explicit capacity stop, no partial write |
| Capacity 22, same root | 1 | 22 | claim offered successfully |

Other cases cover optional children below root, cutoff zero floors without forced
endings, preservation of positive cutoff estimates, historyless/disabled controls,
automatic-ending precedence, nine malformed root replies, and malformed count
after a previously accepted simulation. PUCT estimates are not claims of optimal
play; the one-simulation cases only expose the optional action and safe fallback.

**Ruff, Basedpyright and all 279 inexpensive contracts pass** in the locked CPU
environment. The native played-root regression remains green: 66 searches, 52
accepted advances, 29 semantic rejections, nine malformed/out-of-phase cases and
460 history/feature comparisons, plus the original session suite.

### Actual native neural regression

The existing saved **untrained 5,043,005-parameter transformer fixture** was exported
once, with Torch **2.14.0+cpu**, CPU F32 batch four, and reused for the game checks.
No architecture, training, encoder, compiler or numerical tolerance was changed.

The inherited checkpoint/control/batching/cancellation test in native mode passes
17 epochs, 119 real rows and 78 forwards. Maximum native/eager singleton logit
difference remains **5.364418029785156e-7**. Forward counts vary with queue arrival
and are not a performance result.

Three ordinary `play_probe --claims search_choice` invocations also pass with the
exact same package, four simulations per ply and a two-ply limit:

| Initial claim opportunity | Played continuation | Optional leaf offers | Real neural forwards | Ending |
| --- | --- | ---: | ---: | --- |
| Fifty moves, clock 100 | a1a4 e8d7 | 8 | 8 | * / ply_limit |
| Prospective fifty, clock 99 | a1a4 e8d7 | 8 | 8 | * / ply_limit |
| Current threefold after eight pre-root plies | g1f3 b8c6 | 5 | 8 | * / ply_limit |

These six moves are actually selected by neural search, not scripted. The untrained
model elected to CONTINUE in these three examples; do not relabel them as completed
claimed draws. The cases confirm native status-4 integration, positive-continuation
selection, per-node reference agreement, history/encoder reset and PGN replay.
They do not establish good chess or show an actual neural-selected claim ending.
The prospective a1a2 witness is recorded, but the chosen a1a4 move is what is played.
Across the three games there are 24 actual native forwards and 21 optional offers;
maximum logit error is 5.364418029785156e-7. UBSan covers the four-mode Bend claim
suite, not LibTorch/model internals; actual model execution here is native mode.

### Additional local claim-ending integration

A separate local check uses the actual native Bend executable, real GameActor,
HistoryEncoder and Batcher, but controlled zero-valued evaluator completions (no
neural forward). Current and prospective threefold roots each complete four
simulations and select key 131072. The controller ends a claimed draw without
any advance command, new board move or encoder rebound; PGNs replay exactly.
The prospective case preserves seven pre-root plies and reports f6g8 only as its
intended-move witness. The current case preserves eight pre-root plies. This is
local controlled-evaluator evidence, not a hosted neural-selected-claim result.

## Evidence

Artifact **bend-claim-options-confirmation**, ID **10598825341**, 30-day retention.
ZIP SHA-256: `abde6f30e6834faafb1e31d04a1ebb9c83e37e8759f27ac4d05da1fe82067645`.
- Claims/sessions/automatic draws: `4eeabbb1dd8aea0ed52e2d90cfc9024145ffadc86ccb4af171c1c6ff2e10a9c5`.
- Checkpoint regression: `0b463d8604ce9181a2cf64c5b1aefdcc286bb556f5e347f18fea0d2f9f7cc481`.
- Root/history: `6f456e81131683896e086bc8468861223fcef204f87a3d31b289781f6be922c6`.
- Fifty-move neural play: `d427f21f9309bc0e06aca84c2831657e05ad63ff6427b58076c9503aea0f9855`.
- Prospective-fifty neural play: `8f939297dad1d55d998bc196a88ea894c7417f43c382744c536eaf2ff6769584`.
- Repetition neural play: `6bdbd92c88c02fdab83211450146783a0754d763a467a1f424f3f67396a95ded`.
- Saved fixture: `bb9934134c0d9a39c515c10fa870122c3dfd28259bece392be026e32a9ca60d7`.
- Executed package: `3f22b8aa8f1b4a057bb022b7a6b6a2af2d4c1f61a338ccba57afdb3cafbbd4aa`.
- Applied patch: `a359d7abaa764f968b696f71e051c2356e1f0c97061c063dfddf5d91f062a7c2`.
- Additional local controller JSON: `443c888744f1603307bd09032e2c8de510f099ab4acce7004d81ac92f427167c`.

Only reports/commit identity were uploaded, not weights or executable packages.
The compact observations above are retained when transient reports expire.

## Reproduce / limits

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python -m native.bend_engine.session_probe.claim_probe \
  --report artifacts/bend-claim-options.json

# Use a trusted, matching checkpoint/package already qualified by checkpoint_probe.
python -m native.bend_engine.neural_probe.play_probe \
  --checkpoint /path/to/copied/trainer.pt \
  --reuse-package artifacts/bend-run-01/checkpoint.pt2 \
  --claims search_choice --max-plies 8 --simulations 4 \
  --report artifacts/bend-claim-play.json
```

The native controlled probe includes existing sessions and automatic draws.
Existing game defaults remain automatic; optional choices are explicitly enabled.
No new permanent workflow, recurring native claim traversal, model export or
benchmark was added. Ordinary pytest gains only 24 inexpensive contracts. Perft
depths, source-pinned compiler, production search/encoder/model and live state are
unchanged. The feature is self-reviewed, not independently reviewed or proven.

Remaining: trained-checkpoint/CUDA execution, production Gumbel equivalence,
optimal claim/search policy, subtree reuse and transposition-history safety,
general dead-position solving, UCI and end-to-end throughput/strength. Extra host
history/claim work is unbenchmarked. The action's guaranteed zero does not turn
Monte Carlo estimates into minimax bounds. No merge or deployment occurred.
Broader PR checks are separate from this passing focused confirmation.
