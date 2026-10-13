# Induction follows every command and the exact returned pair, including None.
import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ./Validate.bend as Validate
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def moves(+c: O.Cells,texts: List<&2,String>,r: Array<U64> & Maybe<&2,Pos.Game>,
  checked: S.root(c,r)) -> S.root(c,Protocol.moves(texts,r)):
  match texts:
    case Nil{}: checked
    case Con{+text,+rest}:
      (a,+maybe)=r
      (same,good)=checked
      %Equal.sym(Array<U64>,a,O.pack(c),same) :
        S.root(c,Protocol.moves(rest,Protocol.apply_one(text,maybe,_)))
      moves(c,rest,Protocol.apply_one(text,maybe,O.pack(c)),Apply.one(c,maybe,text,good))

def initial(+c: O.Cells,+maybe: Maybe<&2,Pos.Game>,+texts: List<&2,String>,
  good: Root.rows(maybe)) ->
  S.root(c,Protocol.moves(texts,Protocol.validate(maybe,O.pack(c)))):
  moves(c,texts,Protocol.validate(maybe,O.pack(c)),Validate.known(c,maybe,good))
