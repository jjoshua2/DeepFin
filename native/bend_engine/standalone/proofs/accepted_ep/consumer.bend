# Public actual-full-Ply consumer: one affine table, old moving side.
import Base
import ./Public.bend as P
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../../../legal_probe/Chess.bend as Chess

def use_prepared(a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  tag: {C.flag(m) == 1 : U32},here: E.member(m,E.moves(Chess.filter_prepare(b,(a,xs))))) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  P.prepared(a,b,xs,m,tag,here)

def use_generated(a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 1 : U32}) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  P.legal(a,b,m,here,tag)
