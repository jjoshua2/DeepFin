# Initialized attack composition

This opt-in suite composes existing initialized non-slider and blocker-ray lookup
proofs with the actual attack reducer and singleton-king selector. All three public
contracts return the entire initialized array as well as the query result.

## Public contracts

- `initialized_attack_query_matches_geometry`: all six typed piece classes,
  including the two sequential slider lookups for a queen.
- `initialized_attacked_matches_coordinate_witness`: the actual attack decision
  equals a per-square selected-color/piece witness over independent reverse-query
  masks. No board-partition, non-overlap or king-count premise is needed.
- `initialized_single_king_check_matches_coordinate_witness`: the same result for
  actual `in_check`, with one selected-side king exactly at the bounded query square.

Every contract uses the same symbolic construction: depth `d == 17`, `n == 128`
slider-table blocks from key zero/offset 512, and `extra == 64` extras iterations.
Seed is arbitrary, square is below 64, and side is explicitly Boolean (not an
unrestricted raw metadata value). The mask-query contract permits arbitrary U64
occupancy; attack decisions use the actual board color occupancy.

`Evidence` and the small `Queen`, `Flow`, and `Check` lemmas take exact query-pair
certificates internally. `Query` and `Initialized` discharge ALL of these from the
existing checked allocation, storage, geometry and query producers. The public
caller never supplies a correct mask, correct returned table, or attack answer.
The split lets regression controls check the new composition without repeatedly
normalizing the entire inherited table-build proof chain.

## Reproduce

```sh
export BEND_NO_TELEMETRY=1
bun native/bend_engine/standalone/proofs/attack_composition/focused.js /path/to/pinned/bend --report /tmp/composition-source.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/attack_composition/verify_native.py /path/to/pinned/bend --report /tmp/composition-native.json
```

`--controls-only` deliberately reports `consumer: NOT_RUN` and `focused_gate:
NOT_RUN`; it is not a replacement for the complete focused gate. Safe source
acceptance is status zero and exactly `All terms check.`. Expected/observed
refinement failures at named functions count as semantic controls; missing imports,
syntax/ownership errors, crashes and timeouts never do.

## Native scope

The new probe executes real `Tables.build()` or full tables-plus-extras builds from
zero, all-ones and patterned seeds. Each batch threads the returned array across
six mask queries (including queen), actual `attacked` and bounded `in_check`.
An external forward-coordinate reference provides expected values. The candidate
does not execute the proof geometry or receive reference-produced decisions.
A read of unused slot 131071 checks seed preservation after each batch, not the
entire native array. The source theorem's complete-array equality is stronger than
that sampled runtime observation; native ownership/lifetime remains a trust boundary.

Missing kings produce diagnostic sentinel 2 in the PROBE instead of an out-of-domain
`in_check` call. Multiple kings test actual lowest-bit selection, not whole-position
king safety. Raw inconsistent boards are legitimate attack-function test inputs.

## Still separate

This proves independent **reverse-query** geometry and its actual reduction, not a
new universal forward/reverse ray-membership equivalence theorem. The native oracle
uses forward attacks, but finite test agreement is not that theorem. Deriving
singleton kings through castling, historic rights/metadata legality, semantic king
safety of every generated move, and generator soundness/completeness remain separate.

No literal normalization of the huge closed `Tables.build()` term is claimed. The
formal initializer is symbolic with explicit equality premises; the native probe
also executes the actual public zero-argument builder. Existing public-builder
source guards and their limitations remain unchanged.

No production functions, older contracts, compiler/checker inputs, permanent
workflows, model/GPU, search/training, benchmark or perft budget are changed.
