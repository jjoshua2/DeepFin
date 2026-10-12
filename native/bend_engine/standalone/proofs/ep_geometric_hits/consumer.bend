# One exact affine table and full-Ply occurrence, retaining explicit parent certificates.
import Base
import ./Hits.bend as Hits
import ../ep_occupancy_ray/EP.bend as EP
import ../ep_occupancy_ray/consumer.bend as OldPublic
import ../ep_occurrence_tag/consumer.bend as Tag
import ../ep_occurrence_tag/Spec.bend as Move
import ../ordinary_slider_preservation/Algebra.bend as A
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../ray/Path.bend as Path
import ../ray/Spec.bend as Route
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as C

def result(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,-m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64) -> Type:
  OldPublic.result(n,sq,dir,b,m,acc) &
  L.subset(Hits.raw(n,sq,dir,C.make_move(b,m),diagonal,by,acc),Hits.raw(n,sq,dir,b,diagonal,by,acc))
def clear_result(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,-m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64) -> Type:
  OldPublic.result(n,sq,dir,b,m,acc) &
  {Hits.raw(n,sq,dir,C.make_move(b,m),diagonal,by,acc) == U64.zero() : U64}

def known(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  +route: {Route.trace(n,sq,dir,xs) == True{} : Bool},+empty: {U64.and(EP.removed(s,d),Path.attack(xs,C.occupied(b))) == U64.zero() : U64},
  cert: Move.facts(b,C.Ply{s,d,0,1})) -> result(n,sq,dir,b,C.Ply{s,d,0,1},diagonal,by,acc):
  (promotion,(+pawn,rest))=cert
  (OldPublic.known(n,sq,dir,xs,b,s,d,acc,good,coherent,route,empty,(promotion,(pawn,rest))),
    Hits.subset(n,sq,dir,xs,b,s,d,diagonal,by,acc,Tag.pawn_kind(b,s,pawn),route,empty))

def shape(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  tag: {Move.flag(m) == 1 : U32},cert: Move.facts(b,m)) -> result(n,sq,dir,b,m,diagonal,by,acc):
  match m:
    case C.Ply{+s,+d,+p,+f}:
      (+promo,rest)=cert
      %Equal.sym(U32,f,1,tag) : result(n,sq,dir,b,C.Ply{s,d,p,_},diagonal,by,acc)
      %Equal.sym(U32,p,0,promo) : result(n,sq,dir,b,C.Ply{s,d,_,1},diagonal,by,acc)
      known(n,sq,dir,xs,b,s,d,diagonal,by,acc,good,coherent,route,empty,({==},rest))

def legal_member(table: Array<U64>,+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {Move.flag(m) == 1 : U32}) -> result(n,sq,dir,b,m,diagonal,by,acc):
  shape(n,sq,dir,xs,b,m,diagonal,by,acc,good,coherent,route,empty,tag,Tag.legal_member(table,b,m,here,tag))

# Destructure a certificate parameter; Bend matches parameters rather than calls.
def clear_from(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64,
  cert: result(n,sq,dir,b,m,diagonal,by,acc),old_clear: {Hits.raw(n,sq,dir,b,diagonal,by,acc) == U64.zero() : U64}) ->
  clear_result(n,sq,dir,b,m,diagonal,by,acc):
  (frame,included)=cert
  (frame,A.zero_subset(Hits.raw(n,sq,dir,C.make_move(b,m),diagonal,by,acc),Hits.raw(n,sq,dir,b,diagonal,by,acc),included,old_clear))

def clear_known(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(s,d),Path.attack(xs,C.occupied(b))),
  cert: Move.facts(b,C.Ply{s,d,0,1}),old_clear: {Hits.raw(n,sq,dir,b,diagonal,by,acc) == U64.zero() : U64}) ->
  clear_result(n,sq,dir,b,C.Ply{s,d,0,1},diagonal,by,acc):
  clear_from(n,sq,dir,b,C.Ply{s,d,0,1},diagonal,by,acc,
    known(n,sq,dir,xs,b,s,d,diagonal,by,acc,good,coherent,route,empty,cert),old_clear)

def clear_shape(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  tag: {Move.flag(m) == 1 : U32},cert: Move.facts(b,m),old_clear: {Hits.raw(n,sq,dir,b,diagonal,by,acc) == U64.zero() : U64}) ->
  clear_result(n,sq,dir,b,m,diagonal,by,acc):
  match m:
    case C.Ply{+s,+d,+p,+f}:
      (+promo,rest)=cert
      %Equal.sym(U32,f,1,tag) : clear_result(n,sq,dir,b,C.Ply{s,d,p,_},diagonal,by,acc)
      %Equal.sym(U32,p,0,promo) : clear_result(n,sq,dir,b,C.Ply{s,d,_,1},diagonal,by,acc)
      clear_known(n,sq,dir,xs,b,s,d,diagonal,by,acc,good,coherent,route,empty,({==},rest),old_clear)

def clear_member(table: Array<U64>,+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+m: C.Ply,+diagonal: Bool,+by: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {Move.flag(m) == 1 : U32},
  old_clear: {Hits.raw(n,sq,dir,b,diagonal,by,acc) == U64.zero() : U64}) -> clear_result(n,sq,dir,b,m,diagonal,by,acc):
  clear_shape(n,sq,dir,xs,b,m,diagonal,by,acc,good,coherent,route,empty,tag,Tag.legal_member(table,b,m,here,tag),old_clear)
