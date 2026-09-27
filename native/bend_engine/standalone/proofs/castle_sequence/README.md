# Initialized castling producer and checked-filter sequence

Two public laws connect the actual sequential table-threaded control flow to the
independent target-centred geometry in initialized_attacks/Spec.bend:

- `initialized_castle_producer_matches_geometry`: actual castle_side returns the
  original initialized array and the exact original caller tail, prepending the
  exact castle move precisely when both initial and transit checks are false.
- `initialized_filtered_castle_matches_three_stage_geometry`: actual castle_side
  from an empty candidate list followed by actual filter_legal returns the original
  array and exactly one castling move iff initial, transit and completed checks
  are all false; otherwise it returns the empty list.

Both laws require actual producer guard truth, initial representation consistency,
exactly one moving-side king at home, valid Boolean turn, and depth17/128 slider
blocks/64 extras initialization. Seed is arbitrary. The public body derives all
stage/query certificates from the checked castle_stage_geometry law; no correct
mask, returned table, desired output or stage singleton is assumed by callers.
Both child Boards are made directly from the original Board. The moving side is
kept fixed even when a child flips the turn.

`Runtime` is only a proof/test adapter over imported actual Chess.castle_side and
Chess.filter_legal. It does not replace production move generation. It omits the
ordinary scan, second castling side and optimized filter_prepare; this is not yet
an initialized full-legal_moves theorem. Tail entries are preserved by the first
law, not certified safe or duplicate-free. The second law starts with an empty
list; it does not silently validate arbitrary inherited moves.

The geometrical Boolean is the existing independent target-centred witness.
Universal attacker-origin/target-centred correspondence, legal history and
castling-right validity remain separate. No full chess-legality or completeness
claim follows. Complete source array equality is not native allocation/lifetime
or full-buffer qualification.

## Reproduction

Use the unchanged standalone pin aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae.

```sh
BUN=/path/to/bun python3 native/bend_engine/standalone/proofs/castle_sequence/focused.py \
  /path/to/pinned/bend --report /tmp/castle-sequence-focused.json
BUN=/path/to/bun CC=clang python3 native/bend_engine/standalone/proofs/castle_sequence/verify_native.py \
  /path/to/pinned/bend --report /tmp/castle-sequence-native.json
```

The source gate checks the complete consumer, then eight narrowly targeted
semantic/refinement corruptions of actual source/adapter wiring, eight manifest
and import policies, and one synthetic warning-output unit. Mutants run the
lightweight Wire proof; the original full initialization-linked consumer is checked
once, not re-normalized for every mutant. Parser/affine errors, missing files,
crashes and timeouts never count as semantic rejection.

The native probe builds real Tables.build, then threads its array through actual
producer and filtered calls. Complete ordered lists and child Boards are compared
with an external forward-coordinate/set oracle. One nonzero unused-slot marker is
read after each result; marker preservation is sampled state evidence, not a
full-buffer or lifetime proof. The marker-only mutant must preserve every move-list
and Board value while losing that slot. Modes repeat fixtures; mutation builds
are generic only. No routine perft, search or GPU workload is added.
