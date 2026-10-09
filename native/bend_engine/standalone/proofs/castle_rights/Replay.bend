import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../storage/Observe.bend as O
import ../../Protocol.bend as Protocol
import ../../Position.bend as Pos

def tail(+c: O.Cells,+limit: U32,xs: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,checked: S.root(c,limit,r)) ->
  S.root(c,limit,Protocol.moves(xs,r)):
  match xs:
    case Nil{}: checked
    case Con{+text,+rest}:
      (a,+maybe)=r
      (same,good)=checked
      %Equal.sym(Array<U64>,a,O.pack(c),same) : S.root(c,limit,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,limit,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,limit,maybe,text,good))
