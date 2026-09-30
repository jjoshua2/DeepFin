# Raw flag-two updates. This does not certify legal castling.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ./Actual.bend as C
import ./Route.bend as R

law castling_make_move_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  {Chess.make_move(b,Chess.Ply{src,dst,0,2}) == C.expected(b,src,dst) : Chess.Board}

law castling_make_move_preserves_representation:
  for +b: Chess.Board
  for +route: R.Route
  for good: {Board.valid(b) == True{} : Bool}
  for fresh: {Board.fresh_mask(b,R.rook(route)) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Chess.Ply{R.src(route),R.dst(route),0,2})) == True{} : Bool}
