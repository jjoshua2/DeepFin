# Independent blocker-ray and actual lookup refinement

Opt-in continuation of PR #873. Five source contracts connect production ray
traversal, masking and actual initialized lookup to independent coordinate paths.
No production or previous proof source is changed. No routine perft cost is added.

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/ray/consumer.bend
bun native/bend_engine/standalone/proofs/ray/focused.js /path/to/pinned/bend --controls-only --report /tmp/ray-controls.json
bun native/bend_engine/standalone/proofs/ray/verify_native.js /path/to/pinned/bend --report /tmp/ray-native.json
# Complete inherited chain, explicitly opt-in; not implied by modular receipts:
bun native/bend_engine/standalone/proofs/ray/verify.js /path/to/pinned/bend --report /tmp/ray-aggregate.json
```

## Specification and public contracts

`Grid`, `Rays`, `Masks`, `Path` and `Frame` preserve the exact earlier archived
coordinate, relevant-mask and terminal-path sources. Grid uses Nat file/rank
coordinates and explicit boundaries, not production wrapping deltas. Rays.path
lists successive squares. Path.attack includes the first occupied square and
stops before later squares. Spec.rays unions the appropriate four independent
paths. That function's result does not call production ray/slider/lookup. Although
Spec also contains actual-input route certificates and production observations,
the independent attack definition depends only on Path, Rays, Grid and Base.

| Law | Domain and guarantee |
| --- | --- |
| seven_steps_reach_edge | All 64 squares and eight directions; the square after the final independent seven-step path is off board. The path budget is complete, not an unproved truncation. |
| ray_matches_independent_path | For every valid square/direction and arbitrary U64 occupancy and accumulator, actual Tables.ray equals the accumulator OR first-blocker-inclusive independent path attacks. |
| slider_matches_independent_rays | For every key <128 and arbitrary occupancy, actual unmasked slider equals independent rook/bishop ray attacks. |
| relevant_mask_preserves_attacks | Actual slider on occupancy masked by its actual relevant mask equals independent attacks on the original arbitrary occupancy. |
| initialized_lookup_matches_independent_rays | Actual Chess.slide on the real allocation/table/extras pipeline returns the complete final array and independent attacks. Only symbolic depth=17 and bounded table/extras budgets are caller premises. |

The initialized domain is `before + 1 + after + start_key <=128` and extras
count+start<=64; the selected key is `before + start_key`. Seed and occupancy are
arbitrary. Existing producers derive shape, masks, prefixes, index bounds and
computed data. Callers supply no expected attacks or correct stored values.
The universal domain includes the full 128-block/64-square build configuration.
A separate closed normalization of the literal huge Tables.build term is not
claimed; the source theorem uses symbolic initialization and budget certificates.

## Proof structure

Traversal proves a general accumulator-loop refinement by induction on fuel and
an arbitrary independent path. Its internal trace predicate observes only actual
coordinate steps and path entries, never occupancy or expected attacks. Trace
constructs this certificate for all 512 finite square/direction domains inside
the checker. Termination independently proves seven steps reach the board edge.
The finite scaffolds enumerate domain cases, not host-computed bitboards or
occupancy cases. The occupancy and accumulator arguments remain universal.

Fold transports accumulator traversal to the independent right-fold attack
specification. Slider composes all four rays, with finite scalar-only key-mapping
certificates. Algebra and Masking structurally prove that each interior bit is
retained by the union of relevant ray interiors. The earlier Frame theorem then
removes any dependence on the terminal square's occupancy. Masks links that
independent bitboard to the actual production mask; Geometry composes masking
and traversal. Lookup finally composes the existing actual initialized-lookup
result with this independent geometry, preserving both returned components.

No result is supplied by an external test oracle. There are no new axioms,
unsafe dependencies, foreign equality witnesses, holes, modified checker inputs,
or weakened earlier accepted statements. Source results trust the pinned checker
and Base semantics; they do not verify native lowering or physical ownership.

## Qualification and mutation discipline

The consumer must exit zero with exactly `All terms check.`. Controls-only output
explicitly marks the consumer and full focused gate NOT_RUN. Missing files,
linearity/termination errors, crashes and timeouts are not semantic rejections.
Ten semantic/refinement controls alter actual blocker behavior, accumulator,
coordinates/masks, independent path budgets/observations, and the complete-array
result. Eight enforce manifests/imports/regular files/no unsafe or holes. One
synthetic warning test checks the output wrapper, not another compiler execution.
The complete wrapper retains the unchanged 110-law/276-control parent and adds
five laws/19 controls; modular reuse must not be called a fresh 115-law run.

The native candidate executes actual Tables.ray and masked/unmasked Tables.slider.
The independent flat reference walks signed coordinates. It supplies only key,
direction, occupancy and accumulator inputs to the candidate, never answers or
proof certificates. Generic/portable/native/UBSan repeat 7,008 distinct cases:
4,960 rays, 1,024 unmasked sliders and 1,024 masked sliders. All 512 ray domains,
128 keys and 1,456 possible first-blocker positions are represented, including
terminal blockers, off-ray noise and arbitrary sampled nonzero accumulators.
Nine malformed/budget requests are rejected per mode. Actual ignore-blocker and
omit-blocker mutations must compile and run before a wrong-value rejection.

These are not exhaustive U64 inputs, full-array tests or a new engine benchmark.
The unchanged actual lookup native gate can be run separately to exercise the
real builder and repeated Chess.slide calls. Native tests do not establish an
end-to-end compiler or allocation/lifetime theorem. Existing compiler limitations
remain. This increment moves no additional Python application responsibility.
