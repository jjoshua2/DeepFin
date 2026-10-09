import Base
import ./Spec.bend as S
import ./Replay.bend as Replay
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../Position.bend as Pos

def lift(-a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.good(maybe),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.replay(a,maybe,xs):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  %same : S.replay(_,maybe,xs)
  Replay.tail(c,xs,(O.pack(c),maybe),({==},good))
def use_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,good: S.good(maybe)) -> S.replay(a,maybe,xs):
  lift(a,maybe,xs,good,R.reify(a))
