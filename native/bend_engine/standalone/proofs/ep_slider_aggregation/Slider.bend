# Actual four-ray slider gates with fixed query side and the production R/B selector.
import Base
import ./Rays.bend as Rays
import ../ep_geometric_hits/Hits.bend as Hit
import ../ep_geometric_hits/Plane.bend as Plane
import ../ep_occupancy_ray/EP.bend as EP
import ../ordinary_slider_preservation/Algebra.bend as A
import ../ordinary_slider_preservation/Check.bend as Check
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def raw(+sq: U32,+b: C.Board,+diagonal: Bool,+by: U32) -> U64:
  U64.and(U64.and(Tables.slider(sq,diagonal,C.occupied(b),False{}),
    U64.or(Plane.bits(b,Hit.piece(diagonal)),C.get_queens(b))),C.color(b,by))
def gate(+sq: U32,+b: C.Board,+diagonal: Bool,+by: U32) ->
  {raw(sq,b,diagonal,by) == U64.and(Tables.slider(sq,diagonal,C.occupied(b),False{}),Hit.colored(b,diagonal,by)) : U64}:
  Check.gate(Tables.slider(sq,diagonal,C.occupied(b),False{}),Plane.bits(b,Hit.piece(diagonal)),C.get_queens(b),C.color(b,by))

def subset(+sq: U32,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  kind: {C.piece(b,s) == 0 : U32},
  r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(s,d),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(s,d),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(s,d),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(s,d),Path.attack(x3,C.occupied(b)))) ->
  L.subset(raw(sq,Plane.moved(b,s,d),diagonal,by),raw(sq,b,diagonal,by)):
  L.rewrite(U64.and(Tables.slider(sq,diagonal,C.occupied(Plane.moved(b,s,d)),False{}),Hit.colored(Plane.moved(b,s,d),diagonal,by)),
    U64.and(Tables.slider(sq,diagonal,C.occupied(b),False{}),Hit.colored(b,diagonal,by)),
    raw(sq,Plane.moved(b,s,d),diagonal,by),raw(sq,b,diagonal,by),
    gate(sq,Plane.moved(b,s,d),diagonal,by),gate(sq,b,diagonal,by),
    A.and_mono(Tables.slider(sq,diagonal,C.occupied(Plane.moved(b,s,d)),False{}),Tables.slider(sq,diagonal,C.occupied(b),False{}),
      Hit.colored(Plane.moved(b,s,d),diagonal,by),Hit.colored(b,diagonal,by),
      Rays.subset(sq,b,s,d,diagonal,x0,x1,x2,x3,r0,r1,r2,r3,e0,e1,e2,e3),
      Hit.colored_subset(b,s,d,diagonal,by,kind)))

def clear(+sq: U32,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  kind: {C.piece(b,s) == 0 : U32},
  r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(s,d),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(s,d),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(s,d),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(s,d),Path.attack(x3,C.occupied(b))),
  old_clear: {raw(sq,b,diagonal,by) == U64.zero() : U64}) ->
  {raw(sq,Plane.moved(b,s,d),diagonal,by) == U64.zero() : U64}:
  A.zero_subset(raw(sq,Plane.moved(b,s,d),diagonal,by),raw(sq,b,diagonal,by),
    subset(sq,b,s,d,diagonal,by,x0,x1,x2,x3,kind,r0,r1,r2,r3,e0,e1,e2,e3),old_clear)
