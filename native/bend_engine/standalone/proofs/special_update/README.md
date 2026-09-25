# Typed promotion and en-passant representation updates

This bounded opt-in extension reuses the ordinary clearing/reinsertion proof.
Four universal contracts connect actual make_move to complete Board updates and
representation preservation. Promotion has four choices (tags1..4) and flag0.
En-passant uses promotion0/flag1. Endpoints and metadata are raw U32 in the source;
native tests use square0..63. No source-occupancy, moving-side, pawn-rank,
legal-capture, king-safety or legal-metadata conclusion follows. Castling is not
covered. Production functions and all older accepted laws are unchanged.

The exact-update specifications retain current rights/EP helpers and raw turn
arithmetic; those are not independently proved FIDE metadata semantics. Typed
labels are fixed by independent closed consumer witnesses. Invalid raw promotion
tag6 can leave color without any kind; its counterexample prevents silently
widening the promotion domain.

The en-passant mask removes source, destination and the off-target victim
(destination XOR8). Target freshness is derived from removal, not assumed.
Arbitrary-mask clearing and valid typed insertion then preserve the partition.

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/special_update/focused.js /path/to/pinned/bend --report /tmp/special-proofs.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/special_update/verify_native.js /path/to/pinned/bend --report /tmp/special-native.json
# Expensive whole inherited chain, never implied by focused qualification:
bun native/bend_engine/standalone/proofs/special_update/verify.js /path/to/pinned/bend
```

Success requires status0 and exactly `All terms check.`. Eight semantic/refinement
or witness controls, eight manifest/import checks and one synthetic warning-output
unit are retained. Crashes, missing imports, malformed terms and timeouts do not
count as semantic rejections.

The independent native oracle uses64-square sets, not the proof model. It compares
all eight bitboards and metadata for promotion-shaped/EP-shaped fixtures, broad
raw inputs and excluded inconsistent-board diagnostics. Candidate inputs contain
only Boards and moves. Probe rejection is not validation by make_move. Modes
repeat fixtures, not exhaustive Boards or games. Actual ignored-promotion and
wrong-victim corruptions must compile/run before value rejection.
