import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../storage/Observe.bend as O
import ../../Protocol.bend as Protocol
import ../../Position.bend as Pos

# Induction follows each physical command cons and the exact actual output pair.
def tail(+c: O.Cells,xs: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,checked: S.root(c,r)) ->
  S.root(c,Protocol.moves(xs,r)):
  match xs:
    case Nil{}: checked
    case Con{+text,+rest}:
      (a,+maybe)=r
      (same,good)=checked
      %Equal.sym(Array<U64>,a,O.pack(c),same) : S.root(c,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,good))
