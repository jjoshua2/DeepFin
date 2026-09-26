import Base
import ./Spec.bend as S
import ../castling/Route.bend as R
import ../castling/Actual.bend as C

law producer_source_is_home_king:
  for +r: R.Route
  {R.src(r) == U32.add(S.home(r),4) : U32}

law producer_destination_matches_side:
  for +r: R.Route
  {R.dst(r) == U32.add(S.home(r),Bool.pick(U32,S.king_side(r),6,2)) : U32}

law producer_rook_source_matches_update:
  for +r: R.Route
  {S.rook_source(r) == C.rook_from(R.dst(r)) : U32}

law producer_transit_is_rook_landing:
  for +r: R.Route
  {U64.bit(U32.to_nat(S.transit(r))) == R.rook(r) : U64}

law producer_between_contains_transit:
  for +r: R.Route
  {U64.and(S.between(r),U64.bit(U32.to_nat(S.transit(r)))) == U64.bit(U32.to_nat(S.transit(r))) : U64}

law producer_between_contains_destination:
  for +r: R.Route
  {U64.and(S.between(r),R.target(r)) == R.target(r) : U64}
