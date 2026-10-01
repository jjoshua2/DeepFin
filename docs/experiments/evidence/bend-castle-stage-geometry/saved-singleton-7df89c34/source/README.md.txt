# Moving-side singleton kings through castling

This opt-in suite connects the actual ordinary king step, raw castling update and
king-selection code. It adds five related public laws, not a full legal-castling
soundness proof.

## Contracts and domains

- `transit_preserves_moving_king_singleton`: input representation consistency and one
  moving-side king at a typed route source imply one king at the actual ordinary
  transit-step destination. Destination freshness is not assumed: that update clears
  its target.
- `castle_preserves_moving_king_singleton`: the same assumptions plus initial rook
  landing-square freshness imply one moving-side king at the actual castling target.
- `guard_supplies_castling_singleton`: the existing implementation-linked producer
  guard derives that freshness; applying its exact castling move preserves the
  singleton at the destination.
- `guarded_checks_target_original_side_king`: at Start, Transit and Castled stages,
  actual `in_check` for the **input side** equals actual `attacked` at its certified
  king square with the implementation's opposite-side argument. The complete
  array/Boolean result pair is equal, for arbitrary actual arrays.
- `move_flips_valid_side`: any actual move maps a Boolean-encoded turn to its opposite.

The four routes are 4→6, 4→2, 60→62 and 60→58. General route lemmas use the input
Board's actual color selector; the guard corollaries select the route through the
actual producer's side convention. Raw invalid turn metadata is not silently
assumed Boolean. Only the last law adds a valid-side premise.

Only the moving side's king is certified. Opposing king count/preservation, attack
truth, historical rights, complete metadata legality, reachability and full legal
move soundness/completeness remain separate. `Spec.at` calls actual moves at the
certified route coordinates; it is a proof interface, not a replacement generator.
The laws do not assert that every stage is reached or every guarded move is emitted.

## Proof structure

`Rows` checks complete finite Boolean-row cases for existing representation validity.
`Prerequisites` derives the actual king decoder tag and absence of king bits on a
fresh mask. Structural Word identities in `Algebra`, then both actual U64 limbs,
provide the selected-color king-plane update in `Raw`. `Finish` removes the original
singleton and eliminates any rook-landing contribution. `Checks` derives the actual
trailing-zero king index and keeps the **original** side explicit through the turn
flip. It reuses the unchanged implementation-linked producer guard, ordinary update
and castling update proofs.

No output-singleton, correct-decoder, empty-rook-target or desired check answer is
silently supplied to the guard-based public results. Input singleton and consistency
remain explicit assumptions. The source examples show why both matter, including an
opposing king on a blocked rook landing square becoming an additional moving-side
owned king under the raw update.

## Reproduction

Use the unchanged compiler revision
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` and Bun 1.4.2:

```sh
bun native/bend_engine/standalone/verify_compiler.js /path/to/bend
bun native/bend_engine/standalone/proofs/castle_singleton/focused.js \
  /path/to/bend --report /tmp/castle-singleton-focused.json
BUN=bun CC=clang python3 \
  native/bend_engine/standalone/proofs/castle_singleton/verify_native.py \
  /path/to/bend --report /tmp/castle-singleton-native.json
```

The focused command checks the complete importing consumer, then 18 controls:
9 semantic/refinement rejections, 8 manifest/import-policy checks and 1 synthetic
warning-output unit. Two actual production mutations deliberately fail at the
inherited `move_update/Ordinary.unfold` link; seven mutate the new public contracts.
The synthetic check is not another compiler execution. Missing imports, syntax or
ownership errors, signals and timeouts never count as semantic proof rejection.

The native candidate imports actual Position/Chess/Text, not the proof model. The
separate square-set reference compares all 19 raw Board fields, the original-side
king plane (two limbs), and its selected index. Native testing does not execute
`in_check` or prove attack geometry. Generic, portable, native-target and UBSan repeat
the same fixtures; the three deliberate mutation builds use generic flags only.

512 premise-satisfying starting boards are each observed at three stages (1,536
requests), plus 20 off-premise diagnostics, for 1,556 distinct request tuples and
34,232 numeric fields per mode. This is 532 distinct starting Boards, not 1,556.
Nine malformed bounded requests per mode test the probe rather than a production
validator. Mutation detection stops at the first mismatch, here a transit-stage
record, rather than asserting independent detection at every stage.

This suite does not increase ordinary perft budgets or change production code.
