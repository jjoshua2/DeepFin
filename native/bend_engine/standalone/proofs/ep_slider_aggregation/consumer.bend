# One exact affine input table/full-Ply occurrence; retain the first-ray parent frame.
import Base
import ./Rays.bend as Rays
import ./Slider.bend as Slider
import ../ep_geometric_hits/consumer.bend as Prior
import ../ep_occurrence_tag/consumer.bend as Tag
import ../ep_occurrence_tag/Spec.bend as Move
import ../ep_occupancy_ray/EP.bend as EP
import ../ordinary_slider_preservation/Algebra.bend as A
import ../occupancy_insertion/Lattice.bend as L
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as C

def result(+sq: U32,+b: C.Board,-m: C.Ply,+diagonal: Bool,+by: U32) -> Type:
  Prior.result(7n,sq,Rays.start(diagonal),b,m,diagonal,by,U64.zero()) &
  L.subset(Slider.raw(sq,C.make_move(b,m),diagonal,by),Slider.raw(sq,b,diagonal,by))
def clear_result(+sq: U32,+b: C.Board,-m: C.Ply,+diagonal: Bool,+by: U32) -> Type:
  Prior.result(7n,sq,Rays.start(diagonal),b,m,diagonal,by,U64.zero()) &
  {Slider.raw(sq,C.make_move(b,m),diagonal,by) == U64.zero() : U64}

def known(+sq: U32,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  +r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  +e0: {U64.and(EP.removed(s,d),Path.attack(x0,C.occupied(b))) == U64.zero() : U64},
  e1: D.disjoint(EP.removed(s,d),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(s,d),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(s,d),Path.attack(x3,C.occupied(b))),
  cert: Move.facts(b,C.Ply{s,d,0,1})) -> result(sq,b,C.Ply{s,d,0,1},diagonal,by):
  (promotion,(+pawn,rest))=cert
  (Prior.known(7n,sq,Rays.start(diagonal),x0,b,s,d,diagonal,by,U64.zero(),good,coherent,r0,e0,(promotion,(pawn,rest))),
    Slider.subset(sq,b,s,d,diagonal,by,x0,x1,x2,x3,Tag.pawn_kind(b,s,pawn),r0,r1,r2,r3,e0,e1,e2,e3))

def shape(+sq: U32,+b: C.Board,m: C.Ply,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x3,C.occupied(b))),
  tag: {Move.flag(m) == 1 : U32},cert: Move.facts(b,m)) -> result(sq,b,m,diagonal,by):
  match m:
    case C.Ply{+s,+d,+p,+f}:
      (+promo,rest)=cert
      %Equal.sym(U32,f,1,tag) : result(sq,b,C.Ply{s,d,p,_},diagonal,by)
      %Equal.sym(U32,p,0,promo) : result(sq,b,C.Ply{s,d,_,1},diagonal,by)
      known(sq,b,s,d,diagonal,by,x0,x1,x2,x3,good,coherent,r0,r1,r2,r3,e0,e1,e2,e3,({==},rest))

def legal_member(table: Array<U64>,+sq: U32,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x3,C.occupied(b))),
  here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {Move.flag(m) == 1 : U32}) -> result(sq,b,m,diagonal,by):
  shape(sq,b,m,diagonal,by,x0,x1,x2,x3,good,coherent,r0,r1,r2,r3,e0,e1,e2,e3,tag,
    Tag.legal_member(table,b,m,here,tag))

def clear_from(+sq: U32,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,
  cert: result(sq,b,m,diagonal,by),old_clear: {Slider.raw(sq,b,diagonal,by) == U64.zero() : U64}) ->
  clear_result(sq,b,m,diagonal,by):
  (frame,included)=cert
  (frame,A.zero_subset(Slider.raw(sq,C.make_move(b,m),diagonal,by),Slider.raw(sq,b,diagonal,by),included,old_clear))

def clear_member(table: Array<U64>,+sq: U32,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,
  +x0: List<&2,Nat>,+x1: List<&2,Nat>,+x2: List<&2,Nat>,+x3: List<&2,Nat>,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  r0: {Route.trace(7n,sq,Rays.start(diagonal),x0) == True{} : Bool},
  r1: {Route.trace(7n,sq,U32.inc(Rays.start(diagonal)),x1) == True{} : Bool},
  r2: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),2),x2) == True{} : Bool},
  r3: {Route.trace(7n,sq,U32.add(Rays.start(diagonal),3),x3) == True{} : Bool},
  e0: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x0,C.occupied(b))),
  e1: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x1,C.occupied(b))),
  e2: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x2,C.occupied(b))),
  e3: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(x3,C.occupied(b))),
  here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {Move.flag(m) == 1 : U32},
  old_clear: {Slider.raw(sq,b,diagonal,by) == U64.zero() : U64}) -> clear_result(sq,b,m,diagonal,by):
  clear_from(sq,b,m,diagonal,by,
    shape(sq,b,m,diagonal,by,x0,x1,x2,x3,good,coherent,r0,r1,r2,r3,e0,e1,e2,e3,tag,
      Tag.legal_member(table,b,m,here,tag)),old_clear)
