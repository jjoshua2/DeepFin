# Raw flag-one updates. This does not certify legal en passant.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ./Actual.bend as EP

law en_passant_make_move_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  {Chess.make_move(b,Chess.Ply{src,dst,0,1}) == EP.expected(b,src,dst) : Chess.Board}

law en_passant_make_move_preserves_representation:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}
