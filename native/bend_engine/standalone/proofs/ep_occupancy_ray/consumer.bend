# One exact arbitrary table and full-Ply occurrence certificate; explicit parents.
import Base
import ./EP.bend as EP
import ../ep_king_frame/consumer.bend as King
import ../ep_occurrence_tag/consumer.bend as Tag
import ../ep_occurrence_tag/Spec.bend as Move
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../ray/Path.bend as Path
import ../ray/Spec.bend as Route
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../../Position.bend as Position
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def result(+n: Nat,+sq: U32,+dir: U32,+b: C.Board,-m: C.Ply,+acc: U64) -> Type:
  {C.occupied(C.make_move(b,m)) == EP.normalized(b,Move.src(m),Move.dst(m)) : U64} &
  (King.preservation(b,m) & L.subset(Tables.ray(n,False{},sq,dir,C.occupied(C.make_move(b,m)),False{},acc),
    Tables.ray(n,False{},sq,dir,C.occupied(b),False{},acc)))

def known(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(s,d),Path.attack(xs,C.occupied(b))),
  cert: Move.facts(b,C.Ply{s,d,0,1})) -> result(n,sq,dir,b,C.Ply{s,d,0,1},acc):
  (EP.normalization(b,s,d),(King.known(b,s,d,good,coherent,cert),EP.actual_ray(n,sq,dir,xs,b,s,d,acc,route,empty)))

def shape(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,m: C.Ply,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  tag: {Move.flag(m) == 1 : U32},cert: Move.facts(b,m)) -> result(n,sq,dir,b,m,acc):
  match m:
    case C.Ply{+s,+d,+p,+f}:
      (+promo,rest)=cert
      %Equal.sym(U32,f,1,tag) : result(n,sq,dir,b,C.Ply{s,d,p,_},acc)
      %Equal.sym(U32,p,0,promo) : result(n,sq,dir,b,C.Ply{s,d,_,1},acc)
      known(n,sq,dir,xs,b,s,d,acc,good,coherent,route,empty,({==},rest))

def legal_member(table: Array<U64>,+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+m: C.Ply,+acc: U64,
  good: {Board.valid(b) == True{} : Bool},coherent: {Position.valid_ep(C.get_ep(b),b) == True{} : Bool},
  route: {Route.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(EP.removed(Move.src(m),Move.dst(m)),Path.attack(xs,C.occupied(b))),
  here: E.member(m,E.moves(C.legal_moves(table,b))),+tag: {Move.flag(m) == 1 : U32}) -> result(n,sq,dir,b,m,acc):
  shape(n,sq,dir,xs,b,m,acc,good,coherent,route,empty,tag,Tag.legal_member(table,b,m,here,tag))
