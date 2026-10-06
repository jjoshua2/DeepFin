# One actual geometric ray gated by a fixed-side rook/bishop plus queen plane.
import Base
import ./Plane.bend as Plane
import ../ep_occupancy_ray/EP.bend as EP
import ../ordinary_slider_preservation/Algebra.bend as A
import ../ordinary_slider_preservation/Check.bend as Check
import ../board/Spec.bend as B
import ../ray/Path.bend as Path
import ../ray/Spec.bend as Route
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def piece(diagonal: Bool) -> B.Piece:
  match diagonal:
    case False{}: B.Rook{}
    case True{}: B.Bishop{}
def nonpawn(diagonal: Bool) -> {U32.is_eq(0,B.tag(piece(diagonal))) == False{} : Bool}:
  match diagonal:
    case False{}: {==}
    case True{}: {==}
def colored(+b: C.Board,+diagonal: Bool,+by: U32) -> U64:
  U64.or(Plane.plane(b,piece(diagonal),by),Plane.plane(b,B.Queen{},by))
def ray(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,+acc: U64) -> U64:
  Tables.ray(n,False{},sq,dir,C.occupied(b),False{},acc)
def raw(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,+diagonal: Bool,+by: U32,+acc: U64) -> U64:
  U64.and(U64.and(ray(n,sq,dir,b,acc),U64.or(Plane.bits(b,piece(diagonal)),C.get_queens(b))),C.color(b,by))
def gate(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,+diagonal: Bool,+by: U32,+acc: U64) ->
  {raw(n,sq,dir,b,diagonal,by,acc) == U64.and(ray(n,sq,dir,b,acc),colored(b,diagonal,by)) : U64}:
  Check.gate(ray(n,sq,dir,b,acc),Plane.bits(b,piece(diagonal)),C.get_queens(b),C.color(b,by))

def colored_subset(+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,
  +kind: {C.piece(b,s) == 0 : U32}) ->
  L.subset(colored(Plane.moved(b,s,d),diagonal,by),colored(b,diagonal,by)):
  A.or_mono(Plane.plane(Plane.moved(b,s,d),piece(diagonal),by),Plane.plane(b,piece(diagonal),by),
    Plane.plane(Plane.moved(b,s,d),B.Queen{},by),Plane.plane(b,B.Queen{},by),
    Plane.subset(b,s,d,piece(diagonal),by,kind,nonpawn(diagonal)),Plane.subset(b,s,d,B.Queen{},by,kind,{==}))

# Same fuel, square, direction, False/False flags, accumulator and fixed query side.
# BOTH removed squares are outside the OLD first-blocker-inclusive observation.
def subset(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,+acc: U64,
  kind: {C.piece(b,s) == 0 : U32},route: {Route.trace(n,sq,dir,xs) == True{} : Bool},
  empty: D.disjoint(EP.removed(s,d),Path.attack(xs,C.occupied(b)))) ->
  L.subset(raw(n,sq,dir,Plane.moved(b,s,d),diagonal,by,acc),raw(n,sq,dir,b,diagonal,by,acc)):
  L.rewrite(U64.and(ray(n,sq,dir,Plane.moved(b,s,d),acc),colored(Plane.moved(b,s,d),diagonal,by)),
    U64.and(ray(n,sq,dir,b,acc),colored(b,diagonal,by)),
    raw(n,sq,dir,Plane.moved(b,s,d),diagonal,by,acc),raw(n,sq,dir,b,diagonal,by,acc),
    gate(n,sq,dir,Plane.moved(b,s,d),diagonal,by,acc),gate(n,sq,dir,b,diagonal,by,acc),
    A.and_mono(ray(n,sq,dir,Plane.moved(b,s,d),acc),ray(n,sq,dir,b,acc),
      colored(Plane.moved(b,s,d),diagonal,by),colored(b,diagonal,by),
      EP.actual_ray(n,sq,dir,xs,b,s,d,acc,route,empty),colored_subset(b,s,d,diagonal,by,kind)))

def clear(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,+acc: U64,
  kind: {C.piece(b,s) == 0 : U32},route: {Route.trace(n,sq,dir,xs) == True{} : Bool},
  empty: D.disjoint(EP.removed(s,d),Path.attack(xs,C.occupied(b))),
  old_clear: {raw(n,sq,dir,b,diagonal,by,acc) == U64.zero() : U64}) ->
  {raw(n,sq,dir,Plane.moved(b,s,d),diagonal,by,acc) == U64.zero() : U64}:
  A.zero_subset(raw(n,sq,dir,Plane.moved(b,s,d),diagonal,by,acc),raw(n,sq,dir,b,diagonal,by,acc),
    subset(n,sq,dir,xs,b,s,d,diagonal,by,acc,kind,route,empty),old_clear)
