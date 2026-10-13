import Base
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def canonical(b: Chess.Board) -> Type:
  {Bool.or(U32.is_eq(Chess.get_turn(b),0),U32.is_eq(Chess.get_turn(b),1)) == True{} : Bool}
def parent(-b: Chess.Board) -> Type:
  {Pos.valid_right(b,1,7,4,1) == True{} : Bool} &
  ({Pos.valid_right(b,2,0,4,1) == True{} : Bool} &
  ({Pos.valid_right(b,4,63,60,0) == True{} : Bool} &
  {Pos.valid_right(b,8,56,60,0) == True{} : Bool}))
def white_kingside(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {Pos.valid_right(Chess.make_move(b,m),1,7,4,1) == True{} : Bool}
