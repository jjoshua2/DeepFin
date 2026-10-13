import Base
import ./Words.bend as U
import ./Rights.bend as Rights
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def within(-limit: U32,out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: U.subset(limit,Chess.get_rights(Pos.board(g)))
def empty_rights(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: {Chess.get_rights(Pos.board(g)) == 0 : U32}
def valid(-b: Chess.Board) -> Type:
  {Pos.valid_right(b,1,7,4,1) == True{} : Bool} &
    ({Pos.valid_right(b,2,0,4,1) == True{} : Bool} &
      ({Pos.valid_right(b,4,63,60,0) == True{} : Bool} &
        {Pos.valid_right(b,8,56,60,0) == True{} : Bool}))
def zero_output(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: {Chess.get_rights(Pos.board(g)) == 0 : U32} & valid(Pos.board(g))
def root(-c: O.Cells,-limit: U32,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & within(limit,Root.value(r))
def result(-a: Array<U64>,-limit: U32,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} & within(limit,Root.value(r))
def zero_result(-a: Array<U64>,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} & zero_output(Root.value(r))
def replay(-a: Array<U64>,-limit: U32,-maybe: Maybe<&2,Pos.Game>,-xs: List<&2,String>) -> Type:
  result(a,limit,Protocol.moves(xs,(a,maybe)))
def zero_replay(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-xs: List<&2,String>) -> Type:
  zero_result(a,Protocol.moves(xs,(a,maybe)))
def initial(+maybe: Maybe<&2,Pos.Game>,good: empty_rights(maybe)) -> within(0,maybe):
  match maybe:
    case None{}: Unit{}
    case Some{g}:
      %Equal.sym(U32,Chess.get_rights(Pos.board(g)),0,good) : U.subset(0,_)
      U.self(0)
def certified(+maybe: Maybe<&2,Pos.Game>,good: within(0,maybe)) -> zero_output(maybe):
  match maybe:
    case None{}: Unit{}
    case Some{g}:
      +same: {Chess.get_rights(Pos.board(g)) == 0 : U32}=U.subset_zero(Chess.get_rights(Pos.board(g)),good)
      (same,(Rights.valid_right(Pos.board(g),1,7,4,1,same),
        (Rights.valid_right(Pos.board(g),2,0,4,1,same),
          (Rights.valid_right(Pos.board(g),4,63,60,0,same),Rights.valid_right(Pos.board(g),8,56,60,0,same)))))
