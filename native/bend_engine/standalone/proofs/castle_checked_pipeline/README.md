# Sequential initialized castling checks

Three related public contracts compose the qualified initialized per-stage checks
with actual `Chess.castle_side` and actual full `Chess.filter_legal`. This suite
is not a replacement production generator.

## Public scope

- `initialized_castle_producer_matches_checks`: the actual producer preserves the
  complete initialized table and returns the arbitrary original tail or prepends
  the exact castling move according to independent initial/transit check values.
- `initialized_castle_pipeline_matches_three_checks`: one producer starting with
  an empty tail, followed by the full filter on its returned table/list, preserves
  the initialized table and returns exactly the singleton move when all three
  independent stage values are false, otherwise the empty list.
- `accepted_castle_pipeline_has_safe_stages`: membership in that isolated pipeline
  implies all three independent stage check values are false.

Input assumptions: representation consistency; one moving-side king at home;
valid Boolean-encoded side to move; actual producer guard true; initialization
with depth 17, all 128 slider blocks and all 64 extras. Initial seed is arbitrary.
King-stage certificates, square bounds, check answers and returned-table equality
are derived by the public composition from the qualified #899 producers, not
supplied as caller assumptions. Internal Wire certificates are all instantiated
by the checked public Composition definitions.

The first query's returned table supplies the real transit query. The table/list
returned by the real producer supplies the real full destination filter. An early
rejection short-circuits the final filter over an empty list. Each child Board is
made from the original Board; checks use the original moving side, not flipped
child turn. Full equality is stronger than Boolean equality or pointer identity.

The first contract permits arbitrary existing tails and duplicates; it does not
certify inherited entries. The final two deliberately start with an empty tail.
The pipeline is an audit composition of production calls, not full `legal_moves`:
it does not include the ordinary scan, both wings, `filter_prepare`, or the
optimized sensitive-mask path. Bridging that full producer/filter pipeline remains
separate. These results use the independently specified target-centred geometry;
universal attacker-origin reversal, historical rights and FIDE reachability remain
separate. Source table equality does not prove native allocation or lifetime.

## Reproduction

Use the unchanged pinned compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae
with Bun 1.4.2. Its manifest and original proof gate remain unchanged.

```sh
BUN=bun python3 native/bend_engine/standalone/proofs/castle_checked_pipeline/focused.py \
  /path/to/pinned/bend --report /tmp/castle-pipeline-focused.json
CC=clang bun native/bend_engine/standalone/proofs/castle_checked_pipeline/verify_native.js \
  /path/to/pinned/bend --report /tmp/castle-pipeline-native.json
```

The full focused gate checks the public consumer and separate small satisfiable
premise examples. `--controls-only` is a developer-only partial check and never
reports a complete proof pass. Rejection checks target explicit new control-flow
and certificate functions. A timeout, crash, invalid import or affine-use error
is not a semantic rejection. The warning-output unit is synthetic, not another
Bend execution.

The native verifier uses a separately written coordinate/set/attack reference
retained from the earlier producer verifier. It observes both arbitrary-tail
producer results and empty-tail producer-plus-full-filter results. Requests run
with actual Tables.build and a patterned-seed complete initialization. It compares
full ordered lists, all 19 child-Board fields, and an unused-slot marker after each
batch. Modes reuse fixtures; marker reads are not full-buffer verification.

No production function, old proof, routine perft budget or application language
boundary is changed. Current qualification status belongs in the dated readout
and exact execution receipts, not inferred from source files merely existing.
