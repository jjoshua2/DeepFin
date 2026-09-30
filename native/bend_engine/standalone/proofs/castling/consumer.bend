import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Examples.bend as Examples
import ./Actual.bend as C
import ./Route.bend as R
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess

def use_exact(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(b,Chess.Ply{src,dst,0,2}) == C.expected(b,src,dst) : Chess.Board}:
  Laws.castling_make_move_exact_update(b,src,dst)

def use_preserve(+b: Chess.Board,+r: R.Route,
  good: {B.valid(b) == True{} : Bool},
  fresh: {B.fresh_mask(b,R.rook(r)) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{R.src(r),R.dst(r),0,2})) == True{} : Bool}:
  Laws.castling_make_move_preserves_representation(b,r,good,fresh)
