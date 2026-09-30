# Initialized non-slider attack geometry

Three public contracts connect independent knight, king and two pawn-direction
geometry to actual computation, complete extras initialization and actual queries.

- `computed_leaper_matches_coordinates`: all four typed classes at every square
  below64. Independent file/rank coordinates, not production wrapping deltas.
- `extras_store_coordinate_mask`: all64 actual extras iterations on an arbitrary
  complete depth17 array establish every selected mask. The complete final array
  is included in the read result; no expected original mask is assumed.
- `initialized_leaper_query_matches_coordinates`: actual allocation and any
  preceding actual `Tables.tables` call supply the array shape. After full extras,
  typed non-slider `Chess.attack` returns the independent mask and final array,
  for arbitrary seed, preceding-loop parameters and occupancy.

## Argument

`Computed` checks64 finite scalar coordinate tuples inside the compiler and lifts
those facts with structural accumulation, rather than expanding complete bitboards
at every leaf or importing a host-generated answer table. `Facts` derives slot
bounds and separation. `One` connects the real four writes to their selected values
and unaffected other squares. `Stored` proves preservation through all subsequent
iterations. `Loop` relates this proof schedule to actual `Tables.extras`, and
`Full` binds every query into one uniform complete loop. `Initialized` derives shape
from actual allocation/table operations instead of assuming correct stored values.

Actual affine arrays are consumed through the existing representation bridge;
logical results retain the entire final array, not merely a read value or the
original input. Native pointer identity, allocation and lifetime are not proved.

The consumer also derives the actual reversed pawn query for both Boolean attacker
colors. This is the query-direction specialization, not a new general forward/reverse
attack-membership theorem.

## Reproduce (opt-in)

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/attack_geometry/consumer.bend
bun native/bend_engine/standalone/proofs/attack_geometry/focused.js /path/to/pinned/bend \
  --report /tmp/attack-geometry-focused.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/attack_geometry/verify_native.py \
  /path/to/pinned/bend --report /tmp/attack-geometry-native.json
```

The focused gate checks the consumer and19 controls:10 semantic/refinement,
8 manifest/import policy and1 synthetic warning-output unit. It rejects unsafe
success output and does not accept crashes, parser/ownership errors or timeouts as
semantic rejection. Each invocation is bounded to300 seconds.

The native probe runs actual `Chess.attack` on four initialization contexts: full
`Tables.build`, all-ones seed plus full extras, a partial actual table loop plus
full extras, and a planted last-slot sentinel plus full extras. Each context checks
all256 masks under three occupancies:3,072 distinct requests/6,144 U32 mask fields
per build mode. Modes repeat fixtures. All nine malformed-batch tests belong to the
bounded probe, not raw Chess validation or transactional input rejection.

Three actual implementation corruptions exercise king query routing, black-pawn
storage address separation and white-pawn direction. They must compile and execute
before the independent signed-coordinate oracle finds wrong values. The candidate
never imports these proof functions or receives reference answers.

## Boundaries

Depth17 is the supported engine allocation shape, not claimed minimal. The preceding
loop need not initialize valid slider data; those slots are not queried by these
non-slider contracts. A direct equality for the enormous closed `Tables.build()`
term is not claimed. Its actual native execution is one test context.

No public law here yet composes all five mask classes into initialized `attacked`,
proves forward/reverse slider geometry, establishes singleton kings after castling,
or proves legal chess moves. Existing initialized-slider, attack-witness and mandatory
check-path results remain unchanged for that subsequent composition. No production
function, prior accepted law or compiler pin is changed. Self-review only.
