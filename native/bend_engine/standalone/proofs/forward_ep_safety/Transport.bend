# Reflect target-centred facts through the existing complete forward-source bridge.
import Base
import ../forward_attacks/Bridge.bend as Bridge
import ../forward_attacks/Spec.bend as Forward
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def equivalent(+b: Chess.Board,+q: Nat,+by: Bool,bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Forward.attacked(b,q,by) == Geometry.attacked(b,q,by) : Bool}:
  Equal.sym(Bool,Geometry.attacked(b,q,by),Forward.attacked(b,q,by),Bridge.geometry(b,q,by,bound))

def clear(+b: Chess.Board,+q: Nat,+by: Bool,bound: {Nat.is_lt(q,64n) == True{} : Bool},
  no: {Geometry.attacked(b,q,by) == False{} : Bool}) ->
  {Forward.attacked(b,q,by) == False{} : Bool}:
  Equal.trans(Bool,Forward.attacked(b,q,by),Geometry.attacked(b,q,by),False{},equivalent(b,q,by,bound),no)

def hit(+b: Chess.Board,+q: Nat,+by: Bool,bound: {Nat.is_lt(q,64n) == True{} : Bool},
  yes: {Geometry.attacked(b,q,by) == True{} : Bool}) ->
  {Forward.attacked(b,q,by) == True{} : Bool}:
  Equal.trans(Bool,Forward.attacked(b,q,by),Geometry.attacked(b,q,by),True{},equivalent(b,q,by,bound),yes)
