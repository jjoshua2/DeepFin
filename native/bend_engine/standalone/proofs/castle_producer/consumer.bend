import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../castling/Route.bend as R
import ../castling/Actual.bend as C

def all_source(+r: R.Route) -> {R.src(r) == U32.add(S.home(r),4) : U32}:
  Laws.producer_source_is_home_king(r)

def all_destination(+r: R.Route) -> {R.dst(r) == U32.add(S.home(r),Bool.pick(U32,S.king_side(r),6,2)) : U32}:
  Laws.producer_destination_matches_side(r)

def all_rook_source(+r: R.Route) -> {S.rook_source(r) == C.rook_from(R.dst(r)) : U32}:
  Laws.producer_rook_source_matches_update(r)

def all_transit(+r: R.Route) -> {U64.bit(U32.to_nat(S.transit(r))) == R.rook(r) : U64}:
  Laws.producer_transit_is_rook_landing(r)

def all_between_transit(+r: R.Route) -> {U64.and(S.between(r),U64.bit(U32.to_nat(S.transit(r)))) == U64.bit(U32.to_nat(S.transit(r))) : U64}:
  Laws.producer_between_contains_transit(r)

def all_between_destination(+r: R.Route) -> {U64.and(S.between(r),R.target(r)) == R.target(r) : U64}:
  Laws.producer_between_contains_destination(r)
