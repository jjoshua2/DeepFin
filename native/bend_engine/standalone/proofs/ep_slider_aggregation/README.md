# Actual EP four-ray slider aggregation

This increment composes the prior checked EP single-ray result into the actual
four-ray `Tables.slider(sq,diagonal,occupied(board),False)` for either selected
class. `Rays.four` proves structural inclusion of the exact nested OR;
`Rays.subset` applies unchanged `EP.actual_ray` to each production ray.
Fuel is7, both flags are False and each accumulator is zero. The production
selector starts at0 for rook or4 for bishop and includes the next three
directions.

Each of the four input-only coordinate traces has its own OLD first-blocker-
inclusive `Path.attack` observation. BOTH source and captured victim
(`dst xor8`) must be outside each observation. Neither legal membership nor
parent validity replaces those conditions. The occupancy layer permits
arbitrary board rows, raw turn, query/source/destination U32s and needs no pawn
decoder, query range, canonical turn or target geometry certificate.

`Slider.subset` composes the actual four-ray masks with PR1050
`Hits.colored_subset` under only decoder0, through the actual nested
`AND(AND(slider,OR(selected R/B,queens)),color(board,by))` gate.
The class matches the actual production direction selector and the query
side is fixed. `Slider.clear` transfers an explicit zero parent gate.

The public `legal_member` and `clear_member` each call `Tag.legal_member`
once with the exact arbitrary affine table and full-Ply occurrence. They
derive promotion0/flag1 and actual pawn priority decoder0. Results retain the
PR1050 first-ray occupancy/king/gate result at the production base direction,
under explicit parent `Board.valid` and actual `Position.valid_ep`
certificates. The first trace/disjoint equalities are explicit duplicable
Data; the whole affine facts tuple and table are consumed once.

Both actual EP directions have nonzero opponent R/B/Q rows and public
four-ray instances. White rook and both bishop gates shrink tozero. The
black rook gate loses rook23 but retains queen8 in another ray, demonstrating
why a single empty ray does not establish a zero whole class. Witnesses
include the fourth bishop direction, malformed raw-turn7 erasure, an
out-of-range U32 query72 without added bounds, and a validated-parent actual
legal EP occurrence whose observed victim opens the king32-to-rook39 gate.
Exact table identity, duplicate count3 and wrong promotion/EP flag absence
remain checked through the unchanged occurrence producer.

This is conditional geometric aggregation of one four-ray class. It does
not establish the eight-ray R/B union, arbitrary-table `Chess.attack` or
`in_check` lookup geometry, whole attack safety, king safety, accepted-root
preservation, suffix-array equality or full legal-move correctness.

Run from the repository root using a pinned checker and its qualification
manifest, with fresh report/evidence paths outside the checkout:

```text
python -B -m native.bend_engine.standalone.proofs.ep_slider_aggregation.qualify_ep_slider_aggregation CHECKER --checker-manifest MANIFEST --report REPORT --evidence-dir FRESH_EVIDENCE
```

The fail-closed qualifier freezes the union of Fixtures, Witnesses, RawWitnesses
Counterexample and OccurrenceWitnesses import closures, checks those five entries separately and runs29 isolated
one-declaration controls:18 contract couplings and11 concrete false witnesses.
Only typed rejection at the expected obligation counts; parser/import/affine/
inference, crash, timeout, resource and output failures do not qualify.

Exact PR1050 base: `fe211b3a0af14ec9c95b3e83ec96a0db7d83683e`, tree
`b5e41a7290b7dee2c96b0c21f7852cb9edd48568`. Bend2.0.21+U64:
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`; Bun1.4.2; verified84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
The original combined Witnesses entry failed with MemoryExhaustion under the
unchanged address-space limit and is excluded as a runtime/resource failure.
The first split concrete entry failed with a machine stack overflow while
expanding deep recursion or a large literal; that failure is also excluded.
Only the optional concrete out-of-range query changed from4294967295 to72;
the arbitrary-U32 structural theorem and all premises remain unchanged.
The query72 combined concrete recheck also failed with MemoryExhaustion.
All three failures are excluded. Final concrete statements are split into gate
effects, raw-row boundaries and occurrence counterexamples under the same limits. The final source keeps public Fixtures
and the four concrete entries in separate bounded entries; all declarations and controls
remain covered. Each check has86400 seconds, CPUs1,3,6GiB AS/RSS and16MiB per output stream.
Commands, raw logs, snapshots and hashes remain in fresh external evidence;
no prior seal is overwritten.

The first Q003 false source-only-sufficiency control exhausted memory before a typed rejection. Its crash is excluded; the 23 prior typed controls remain partial evidence. Counterexample now copies only the exact immutable parent board and OLD path, retains the same six geometry propositions, and separates them from the legal-occurrence/count imports. No premise, proposition, resource cap or expected rejection rule is weakened.
