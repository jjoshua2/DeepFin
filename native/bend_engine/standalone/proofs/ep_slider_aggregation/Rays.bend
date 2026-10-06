# Four production rays: fuel7, False/False flags, zero accumulator.
import Base
import ../ep_occupancy_ray/EP.bend as EP
import ../ep_geometric_hits/Plane.bend as Plane
import ../ordinary_slider_preservation/Algebra.bend as A
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def start(diagonal: Bool) -> U32: Bool.pick(U32,diagonal,4,0)
def mask(+sq: U32,+dir: U32,+b: C.Board) -> U64:
  Tables.ray(7n,False{},sq,dir,C.occupied(b),False{},U64.zero())

def four(+a0: U64,+b0: U64,+a1: U64,+b1: U64,+a2: U64,+b2: U64,+a3: U64,+b3: U64,
  e0: L.subset(a0,b0),e1: L.subset(a1,b1),e2: L.subset(a2,b2),e3: L.subset(a3,b3)) ->
  L.subset(U64.or(U64.or(a0,a1),U64.or(a2,a3)),U64.or(U64.or(b0,b1),U64.or(b2,b3))):
  A.or_mono(U64.or(a0,a1),U64.or(b0,b1),U64.or(a2,a3),U64.or(b2,b3),
    A.or_mono(a0,b0,a1,b1,e0,e1),A.or_mono(a2,b2,a3,b3,e2,e3))

# Each certificate observes BOTH removals on that OLD first-blocker-inclusive ray.
# Routes contain coordinates only; no board-validity or query-bound premise.
def subset(+sq: U32,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  r0: {Route.trace(7n,sq,start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(s,d),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(s,d),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(s,d),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(s,d),Path.attack(x3,C.occupied(b)))) ->
  L.subset(Tables.slider(sq,diagonal,C.occupied(Plane.moved(b,s,d)),False{}),
    Tables.slider(sq,diagonal,C.occupied(b),False{})):
  four(mask(sq,start(diagonal),Plane.moved(b,s,d)),mask(sq,start(diagonal),b),
    mask(sq,U32.inc(start(diagonal)),Plane.moved(b,s,d)),mask(sq,U32.inc(start(diagonal)),b),
    mask(sq,U32.add(start(diagonal),2),Plane.moved(b,s,d)),mask(sq,U32.add(start(diagonal),2),b),
    mask(sq,U32.add(start(diagonal),3),Plane.moved(b,s,d)),mask(sq,U32.add(start(diagonal),3),b),
    EP.actual_ray(7n,sq,start(diagonal),x0,b,s,d,U64.zero(),r0,e0),
    EP.actual_ray(7n,sq,U32.inc(start(diagonal)),x1,b,s,d,U64.zero(),r1,e1),
    EP.actual_ray(7n,sq,U32.add(start(diagonal),2),x2,b,s,d,U64.zero(),r2,e2),
    EP.actual_ray(7n,sq,U32.add(start(diagonal),3),x3,b,s,d,U64.zero(),r3,e3))
