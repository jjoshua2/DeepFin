# Core geometry, fixtures and exact production-routing baseline.
import Base
import ./Bridge.bend as Bridge
import ./Spec.bend as Forward
import ./Fixtures.bend as Fixtures
import ./Runtime.bend as Runtime
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def use_geometry(+b: Chess.Board,+target: Nat,+by: Bool,
  bound: {Nat.is_lt(target,64n) == True{} : Bool}) ->
  {Geometry.attacked(b,target,by) == Forward.attacked(b,target,by) : Bool}:
  Bridge.geometry(b,target,by,bound)
