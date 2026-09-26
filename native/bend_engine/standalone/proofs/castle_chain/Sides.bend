# Dependent continuation form of the actual side producer, retaining the real array.
# Used to thread a pointwise certificate through the second side and final filter.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S

def finish(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +m: Chess.Ply,start_check: Bool,+tail: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  good: S.every(P,tail),pm: P(m),next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.castle_finish(m,start_check,tail,r)):
  match start_check:
    case True{}:
      (table,transit) = r
      next((table,tail),good)
    case False{}:
      (table,transit) = r
      match transit:
        case True{}: next((table,tail),good)
        case False{}: next((table,Con{m,tail}),(pm,good))

def start(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & Bool,
  good: S.every(P,tail),pm: P(Chess.Ply{src,dst,0,2}),next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.castle_start(b,src,dst,transit,tail,r)):
  (table,check) = r
  finish(P,Goal,Chess.Ply{src,dst,0,2},check,tail,
    Chess.in_check(table,Chess.make_move(b,Chess.Ply{src,transit,0,0}),Chess.get_turn(b)),good,pm,next)

def checked(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  ok: Bool,table: Array<U64>,+b: Chess.Board,+src: U32,+dst: U32,+transit: U32,+tail: List<&2,Chess.Ply>,
  good: S.every(P,tail),pm: {ok == True{} : Bool} -> P(Chess.Ply{src,dst,0,2}),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.castle_checked(ok,table,b,src,dst,transit,tail)):
  match ok:
    case False{}: next((table,tail),good)
    case True{}: start(P,Goal,b,src,dst,transit,tail,Chess.in_check(table,b,Chess.get_turn(b)),good,pm({==}),next)

def side(-P: Chess.Ply -> Type,-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+ks: Bool,r: Array<U64> & List<&2,Chess.Ply>,good: S.every(P,E.moves(r)),
  pm: {E.guard(b,ks) == True{} : Bool} -> P(E.move(b,ks)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(P,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.castle_side(b,ks,r)):
  (table,tail) = r
  checked(P,Goal,E.guard(b,ks),table,b,E.src(b),E.dst(b,ks),E.transit(b,ks),tail,good,pm,next)
