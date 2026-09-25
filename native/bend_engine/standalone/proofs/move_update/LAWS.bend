# Actual scalar deletion lift and ordinary make_move, not move legality.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ./Spec.bend as Spec
import ./ClearActual.bend as Clear
import ./Ordinary.bend as Ordinary

law clear_mask_exact_update:
  for +b: Chess.Board
  for +mask: U64
  {Clear.run(b,mask) == Spec.erase(b,mask) : Chess.Board}

law clear_mask_preserves_representation:
  for +b: Chess.Board
  for +mask: U64
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Clear.run(b,mask)) == True{} : Bool}

law cleared_target_is_fresh:
  for +b: Chess.Board
  for +source: U64
  for +target: U64
  {Board.fresh_mask(Clear.run(b,Spec.removal(source,target)),target) == True{} : Bool}

law ordinary_make_move_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  {Chess.make_move(b,Chess.Ply{src,dst,0,0}) == Ordinary.expected(b,src,dst) : Chess.Board}

law ordinary_make_move_preserves_representation:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Chess.Ply{src,dst,0,0})) == True{} : Bool}
