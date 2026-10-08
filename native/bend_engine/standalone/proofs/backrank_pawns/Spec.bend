import Base
import ./Mask.bend as M
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def good(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: M.safe(Pos.board(g))
def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & good(Root.value(r))
def result(-a: Array<U64>,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} & good(Root.value(r))
def replay(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-xs: List<&2,String>) -> Type:
  result(a,Protocol.moves(xs,(a,maybe)))
