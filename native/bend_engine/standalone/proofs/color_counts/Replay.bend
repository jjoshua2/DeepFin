import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def tail(+c: O.Cells,+w: Nat,+bl: Nat,xs: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,parent: S.root(c,w,bl,r)) ->
  S.root(c,w,bl,Protocol.moves(xs,r)):
  match xs:
    case Nil{}: parent
    case Con{+text,+rest}:
      (table,+maybe)=r
      (same,good)=parent
      %Equal.sym(Array<U64>,table,O.pack(c),same) : S.root(c,w,bl,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      tail(c,w,bl,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,w,bl,good))
