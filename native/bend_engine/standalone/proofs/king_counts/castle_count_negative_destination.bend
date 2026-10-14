# MUST reject at reject_occupied_destination: row validity and landing clearance do not imply path clearance.
import Base
import ./CastleClear.bend as C
import ./castle_count_fixtures.bend as F
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def reject_occupied_destination() ->
  {U64.and(U64.bit(U32.to_nat(E.dst(F.destination_occupied(),True{}))),Chess.get_kings(F.destination_occupied())) == U64.zero() : U64}:
  C.destination(F.destination_occupied(),True{},F.destination_rows(),{==})
