# Project two complete pairs; the public proof constructs both certificates.
import Base
import ../accepted_castle/Spec.bend as S
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess

def old_turn(-a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+side: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  no: {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}) ->
  {Chess.in_check(a,Chess.make_move(b,m),Bool.to_u32(side)) == (a,False{}) : Array<U64> & Bool}:
  %turn :
    {Chess.in_check(a,Chess.make_move(b,m),_) == (a,False{}) : Array<U64> & Bool}
  no

def geometry(-a: Array<U64>,+child: Chess.Board,+q: Nat,+side: Bool,
  query: {Chess.in_check(a,child,Bool.to_u32(side)) ==
    (a,G.attacked(child,q,Bool.not(side))) : Array<U64> & Bool},
  no: {Chess.in_check(a,child,Bool.to_u32(side)) == (a,False{}) : Array<U64> & Bool}) ->
  {G.attacked(child,q,Bool.not(side)) == False{} : Bool}:
  Equal.trans(Bool,G.attacked(child,q,Bool.not(side)),
    S.answer(Chess.in_check(a,child,Bool.to_u32(side))),False{},
    Equal.sym(Bool,S.answer(Chess.in_check(a,child,Bool.to_u32(side))),
      G.attacked(child,q,Bool.not(side)),
      Equal.cong(Array<U64> & Bool,Bool,S.answer,
        Chess.in_check(a,child,Bool.to_u32(side)),(a,G.attacked(child,q,Bool.not(side))),query)),
    Equal.cong(Array<U64> & Bool,Bool,S.answer,
      Chess.in_check(a,child,Bool.to_u32(side)),(a,False{}),no))
