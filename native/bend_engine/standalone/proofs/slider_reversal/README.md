# Blocker-aware slider attack reversal

Two public source laws establish reciprocity of independent geometric masks and
actual unmasked Tables.slider computations for rook/bishop, plus their union for
queen. Source and target are below64; occupancy is one shared arbitrary U64.
Neither endpoint must be empty. The first occupied square is included; only the
strict intervening path may block an edge. Disconnected/equal endpoints are covered.

Bits.path derives ray-bit membership from a generic first-blocker-inclusive list
scan. Found turns positive visibility into path membership. Nine generated files
supply512 coordinate-only source/direction certificates containing1,456 on-ray
prefix equalities. They compare reversed strict interiors without inspecting
occupancy. Booleans.reverse_clear works for arbitrary list/occupancy by induction.
Directed composes those certificates, then Reversal combines opposite directions.
Computed links the same geometry to the existing actual Tables.slider proof.
The generator emits equality constructors, not trusted answers or occupancy cases.

Commands (pinned compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae):

    python3 generate_certificates.py
    BUN=/path/to/bun python3 focused.py /path/to/compiler --report focused.json
    BUN=/path/to/bun CC=clang python3 -O verify_native.py /path/to/compiler --report native.json
    python3 test_harness.py

Native tests reuse the unchanged production-only ray/probe.bend. They execute
actual unmasked slider computation, not initialized Chess.attack lookup. Queen
observations combine actual rook/bishop results, not a newly qualified production
queen dispatch. The independent oracle walks signed coordinates and includes the
first blocker. Every aligned pair is tested against every subset of its strict
interior, both endpoint bits and bounded off-line noise, plus global occupancy
contexts covering all endpoint pairs. Modes repeat fixtures. Whole masks are
compared separately from reciprocity; ignoring blockers can remain reciprocal.

The source laws quantify arbitrary occupancy, while native cases are finite.
This is not a new table-initialization, native-lifetime, full attacked witness,
accepted-castling starting/transit safety, historical-rights or move-completeness
result. Existing proofs and production functions remain unchanged. Reports start
NOT_COMPLETED and pass only after complete validation; independent runs need
separate report paths. Host tests are not added to formal-law/control totals.
