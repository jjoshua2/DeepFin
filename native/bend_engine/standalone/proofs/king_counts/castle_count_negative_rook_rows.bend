# MUST reject at reject_overlapping_rook: guard true does not imply parent row partition.
import Base
import ./CastleClear.bend as C
import ./castle_count_fixtures.bend as F
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def reject_overlapping_rook() ->
  {U64.and(U64.bit(U32.to_nat(E.rook(F.overlapping_rook(),True{}))),Chess.get_kings(F.overlapping_rook())) == U64.zero() : U64}:
  C.rook_source(F.overlapping_rook(),True{},{==},F.overlapping_guard())
