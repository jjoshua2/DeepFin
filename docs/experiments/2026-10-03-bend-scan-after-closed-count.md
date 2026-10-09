# Closed full-Ply occurrence count for actual multi-bit scan_after

This increment stacks on verified PR1006 head
1c7fbc68bc81f7b59ae1b6eb903ffb5450381dd5 in branch
proof/bend-scan-after-predicate-20261003.

## Checked theorem

QueryFactor.scan_after_closed_count proves, for arbitrary U64 targets, U32
source and EP square, pawn flag, exact input table, full-Ply query, and arbitrary
full-Ply tail:

    pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)), query)
      = S.count(tail,query)
        + Bool.pick(Nat,
            Nat.is_lt(U32.to_nat(query.dst),64)
              && U64.test_bit(targets,U32.to_nat(query.dst)),
            query_emission_count(src,pawn,ep,query), 0)

The theorem compares all four query fields. query_emission_count is the count
of the exact non-destination fields for moves generated at query.dst: for a
pawn on a back rank it sums the four actual promotion tags 1, 2, 3, and 4, each
with flag 0; otherwise it checks promotion 0 and
Bool.to_u32(pawn && query.dst == ep). Thus a set target contributes only when
the actual generated source, promotion, and flag fields match. The arbitrary
tail remains S.count(tail,query), preserving duplicate multiplicity.

destination_equal_projection proves the Spec.same destination split uses
exactly the query's U32 destination field. key_factor factors each actual
specification emission through U32 equality. weighted_occurrences preserves
frequency for arbitrary duplicate-containing key lists. actual_emission_count
then composes PR1000's actual_bit_squares_frequency_all_u32 with its emitted
range guarantee. Finally actual_tally_count composes the closed count with
the retained full-Ply tail, and scan_after_closed_count composes that result
through actual Chess.scan_after and #1006's exact input-table proof.

consumer.bend exposes the full theorem and the below-64 destination range
result. The implementation match is grounded in legal_probe/Chess.bend and
the existing checked Emission.destinations, Count.destinations, and
AfterCount.scan_after_count modules. No board-validity or target-geometry
premise is assumed; no full legal-move correctness claim is made.

The pinned-checker qualification record and raw stdout, stderr, resource logs,
hashes, and mutation controls are in
evidence/2026-10-03/scan-after-closed-count.
