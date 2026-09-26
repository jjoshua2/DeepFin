# Castling cannot bypass the full check through blocker filtering

Four public source contracts establish an operational property of the actual engine.
For a move certified by either castling producer, its source is an owned king bit.
The actual sensitive mask includes owned king bits regardless of the supplied rays.
Consequently the actual fast-step takes `filter_step`, not unchecked insertion.

`Connect` obtains that producer certificate from the saved full-generator provenance
proof. Membership in actual `Chess.legal_moves` plus flag 2 suffices; no independent
input-consistency, caller-tail, ray correctness, or desired safety premise is supplied.
The generated-step law preserves the complete ownership-threaded table/list pair.
For a list consisting entirely of certified castles, actual `filter_blockers` equals
`filter_legal` on the entire returned pair, even with repeated moves and arbitrary rays.
This last law is not asserted for unrestricted mixed move lists with arbitrary rays.

The optimized filter does not special-case flag 2. The closed empty-source witness
shows that a forged raw flag-2 move can bypass checking. The initialized generator's
provenance is essential. The importing consumer also instantiates a generated castle
with zero rays and a duplicate certified list. Its symbolic zero-table example is
not a claim of correct attack tables. Native tests use actual current `Tables.build()`.

## Reproduce (opt-in)

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/castle_safety/consumer.bend
bun native/bend_engine/standalone/proofs/castle_safety/focused.js /path/to/pinned/bend
bun native/bend_engine/standalone/proofs/castle_safety/verify_native.js /path/to/pinned/bend
```

The focused gate checks all four laws and 15 controls: four ordinary implementation/
refinement failures, two false closed boundary witnesses, eight manifest/import checks,
and one synthetic warning-output check (not a compiler execution). It isolates the
new implementation bridges from inherited consumers when testing corruptions, so an
unrelated inherited failure cannot receive credit. Errors, crashes and timeouts do
not substitute for intended source diagnostics. The compiler is unchanged and the
checker must return exactly `All terms check.`.

Native tests execute actual both-side/full-generator castling subsets, the real
`filter_requires`, `filter_blockers` and `filter_legal`, including their complete raw
child Boards. The external coordinate/set reference checks those ordered castling
subsets and the filtered lists. It does not validate the entire noncastling move set.
Each generic/portable/native-target/UBSan mode repeats the same 1,080 distinct requests,
including 270 zero-ray requests, 360 duplicate-list requests and four destination-only
attack cases with zero rays. Both full-filter and blocker-filter results must match.
Candidate probes receive no expected decisions and import no proof predicates.

## Limits

These laws prove which actual check path is taken and complete result equivalence,
not that `in_check` computes independently specified chess attacks. No standalone
semantic king-safety, rights-history, metadata legality, reachability, generation
completeness, native storage/lifetime or performance theorem follows. In particular,
no equality is asserted between a check using unrelated table states: actual arrays
are kept explicit. No production code, prior proof, compiler input, historical pin,
permanent workflow, routine perft or Python application responsibility is changed.
