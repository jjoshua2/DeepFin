# MUST reject at reject_missing_source_color: source bit false cannot equal side True.
import Base
import ./CastleCount.bend as C
import ./castle_count_fixtures.bend as F
import ./Frame.bend as Frame
import ../color_counts/U64Count.bend as U
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess
def reject_missing_source_color() ->
  {U.count(Frame.model(F.zero_color(),E.move(F.zero_color(),True{}),U64.zero(),True{})) ==
   U.count(U64.and(Chess.get_kings(F.zero_color()),U64.zero())) : Nat}:
  C.plane(F.zero_color(),True{},U64.zero(),True{},F.zero_color_rows(),F.zero_color_guard(),{==})
