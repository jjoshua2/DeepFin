# MUST reject at reject_false_guard: the valid-row blocked fixture has false guard.
import Base
import ./CastleFrame.bend as C
import ./castle_frame_fixtures.bend as F
import ../castle_emission/Spec.bend as E
def reject_false_guard() -> C.both(F.blocked(),E.move(F.blocked(),True{})):
  C.guarded(F.blocked(),True{},F.blocked_rows(),{==})
