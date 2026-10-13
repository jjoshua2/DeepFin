import Base
import ./Spec.bend as S
import ./Replay.bend as Replay
import ./Actual.bend as A
import ./Emission.bend as Emission
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def lift(-a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.good(maybe),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.replay(a,maybe,xs):
  (+c,same)=cert
  %same : S.replay(_,maybe,xs)
  Replay.tail(c,S.white_limit(maybe),S.black_limit(maybe),xs,(O.pack(c),maybe),({==},S.initial(maybe,good)))
def use_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.good(maybe)) -> S.replay(a,maybe,xs):
  lift(a,maybe,xs,good,R.reify(a))
def use_child(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,zero: {Chess.get_rights(b) == 0 : U32},here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> A.decreases(b,m):
  Emission.legal_member(table,b,m,zero,here)
