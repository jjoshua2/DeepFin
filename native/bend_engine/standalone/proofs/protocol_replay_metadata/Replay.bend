import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def tail(+c: O.Cells,texts: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,
  checked: S.root(c,r)) -> S.root(c,Protocol.moves(texts,r)):
  match texts:
    case Nil{}: checked
    case Con{+text,+rest}:
      (a,+maybe)=r
      (same,good)=checked
      %Equal.sym(Array<U64>,a,O.pack(c),same) :
        S.root(c,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,S.weaken(maybe,good)))
def first(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+text: String,+rest: List<&2,String>,
  good: S.parent(maybe)) -> S.root(c,Protocol.moves(Con{text,rest},(O.pack(c),maybe))):
  tail(c,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,good))
def validated(+c: O.Cells,+text: String,+rest: List<&2,String>,
  r: Array<U64> & Maybe<&2,Pos.Game>,checked: Root.root(c,r)) ->
  S.root(c,Protocol.moves(Con{text,rest},r)):
  (a,+maybe)=r
  (same,good)=checked
  %Equal.sym(Array<U64>,a,O.pack(c),same) : S.root(c,Protocol.moves(Con{text,rest},(_,maybe)))
  first(c,maybe,text,rest,S.accepted(maybe,good))
