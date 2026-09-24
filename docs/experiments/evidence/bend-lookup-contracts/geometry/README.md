# Independent geometry foundations: checked supplementary source

These two derived statements are separate from the four promoted public lookup
laws and their sixteen controls. They are newly constructed and checked here,
not a relabeling of earlier native comparisons as proof. Their exact source is
archived as `.bend.txt` for deliberate future promotion. No temporary executable
proof folder is left in the primary source tree.

`Step.actual` proves, for every U32 square whose Nat value is below 64 and every
direction below 8, that actual `Tables.jump(sq, Tables.delta(dir))` agrees with
`Grid.next` in natural-number coordinates, including the off-board sentinel 64.
`Grid` imports only Base. It uses unbounded coordinates, explicit positive and
negative moves, and rejects edge crossings rather than using U32 wraparound.
The 512 finite cases reduce inside the source checker; the scaffold supplies only
case indices, not host-computed expected answers.

`Masks.actual` proves equality of the actual relevant-mask computation
`Tables.slider(key % 64, key >= 64, U64.zero(), True)` and `Rays.mask(nat(key))`
for every key below 128. `Rays` imports only Base and Grid; it builds coordinate
paths, removes the terminal square from each ray, and converts the resulting
interior positions to a bit set. The bitboards are computed in the specification,
not imported from a host-generated table. This is an equality of full masks, not
only their population counts. The board-bounded paths use seven steps. No general
arbitrary-fuel ray theorem or arbitrary-occupancy blocker theorem is claimed.

Both actual-value statements include their domain proof and use existing
U32/Nat recovery lemmas, so the finite enumeration is connected to arbitrary valid
machine-word inputs. The source checks complete with exact safe success.

## Reproduction

In a disposable checkout of the parent plus this patch, copy `*.bend.txt` into
`native/bend_engine/standalone/proofs/geometry_review/`, removing only `.txt`.
Check the source manifest, then run:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/geometry_review/Step.bend
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/geometry_review/Masks.bend
python /path/to/verify_geometry.py.txt /path/to/checkout /path/to/pinned/bend --bun /path/to/bun --report /tmp/geometry.json
```

The native driver compares all 512 actual single steps and all 128 actual masks
against an independent signed-coordinate Python reference, in generic and UBSan
builds. This is exhaustive only over those finite primitive inputs, not over
occupancies, complete table buffers or the whole engine. It receives no oracle
answers as candidate inputs. Two isolated mutations compile/run and fail the
reference, and separately fail at intended source refinements: admitting file 8
as on-board and retaining terminal edge squares in relevant masks. These two
regressions do not increase the sixteen primary source-control count.

Two initial construction failures are retained: a match after computed coordinate
bindings violated the checker's direct-parameter restriction, and an implicitly
linear List was explicitly duplicated. A direct-parameter helper and an explicitly
Data-kind Nat list fix these language-use errors without changing the intended
geometry or checker. They are not counted as proof-rejection controls.

The next independent geometry obligation is blocker traversal/inclusion and the
irrelevance of excluded terminal occupancy bits for every occupancy, then its
composition with the promoted actual lookup. The current statements are not that
remaining theorem. Source checker/Base, native lowering and hardware remain trust
boundaries; no independent reviewer or hosted qualification is claimed.
