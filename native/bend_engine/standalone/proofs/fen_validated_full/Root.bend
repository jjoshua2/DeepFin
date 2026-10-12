import Base
import ./Turn.bend as Turn
import ../board/Spec.bend as B
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

# Parsed row partition remains separate from the actual Position.valid flag.
def rows(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: {B.valid(Pos.board(g)) == True{} : Bool}
def accepted(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}:
      {B.valid(Pos.board(g)) == True{} : Bool} &
      (Turn.canonical(Chess.get_turn(Pos.board(g))) &
        {Pos.valid(Pos.board(g)) == True{} : Bool})
def value(r: Array<U64> & Maybe<&2,Pos.Game>) -> Maybe<&2,Pos.Game>:
  (a,g)=r
  g
def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & accepted(value(r))
