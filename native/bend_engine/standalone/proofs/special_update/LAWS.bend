# Four universal representation contracts; no king-safety or move-legality claim.
import Base
import ./Spec.bend as Spec
import ../board/Spec.bend as Board
import ../../../legal_probe/Chess.bend as Chess

law typed_promotion_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for +p: Spec.Promotion
  {Chess.make_move(b,Chess.Ply{src,dst,Spec.tag(p),0}) == Spec.promotion_expected(b,src,dst,p) : Chess.Board}

law typed_promotion_preserves_representation:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for +p: Spec.Promotion
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Chess.Ply{src,dst,Spec.tag(p),0})) == True{} : Bool}

law en_passant_exact_update:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  {Chess.make_move(b,Chess.Ply{src,dst,0,1}) == Spec.ep_expected(b,src,dst) : Chess.Board}

law en_passant_preserves_representation:
  for +b: Chess.Board
  for +src: U32
  for +dst: U32
  for good: {Board.valid(b) == True{} : Bool}
  {Board.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}
