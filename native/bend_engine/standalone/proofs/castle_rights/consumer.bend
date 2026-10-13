import Base
import ./Spec.bend as S
import ./Words.bend as U
import ./Replay.bend as Replay
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def lift(-a: Array<U64>,+limit: U32,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.within(limit,maybe),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.replay(a,limit,maybe,xs):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  %same : S.replay(_,limit,maybe,xs)
  Replay.tail(c,limit,xs,(O.pack(c),maybe),({==},good))

# Arbitrary arrays, raw boards and ordered duplicate-containing command tails.
def use_replay(a: Array<U64>,+limit: U32,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.within(limit,maybe)) ->
  S.replay(a,limit,maybe,xs):
  lift(a,limit,maybe,xs,good,R.reify(a))
def use_game(a: Array<U64>,+g: Pos.Game,+xs: List<&2,String>) ->
  S.replay(a,Chess.get_rights(Pos.board(g)),Some{g},xs):
  use_replay(a,Chess.get_rights(Pos.board(g)),Some{g},xs,U.self(Chess.get_rights(Pos.board(g))))
def zero_result(-a: Array<U64>,r: Array<U64> & Maybe<&2,Pos.Game>,checked: S.result(a,0,r)) -> S.zero_result(a,r):
  (table,+maybe)=r
  (kept,good)=checked
  (kept,S.certified(maybe,good))
def zero_lift(-a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.empty_rights(maybe),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.zero_replay(a,maybe,xs):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  %same : S.zero_replay(_,maybe,xs)
  zero_result(O.pack(c),Protocol.moves(xs,(O.pack(c),maybe)),
    Replay.tail(c,0,xs,(O.pack(c),maybe),({==},S.initial(maybe,good))))
def use_zero(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.empty_rights(maybe)) ->
  S.zero_replay(a,maybe,xs):
  zero_lift(a,maybe,xs,good,R.reify(a))
