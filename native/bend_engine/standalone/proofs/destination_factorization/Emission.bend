# Reuse the accepted promotion expansion; independently describe full Ply blocks.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../promotion/LAWS.bend as Promotion
import ../promotion/PROOF.bend as PromotionProof
import ./Spec.bend as S
import ./Factor.bend as F

def put(promote: Bool,+src: U32,+dst: U32,+tag: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.put_move(promote,src,dst,tag,tail) == List.append(&2,Chess.Ply,S.block(promote,src,dst,tag),tail) : List<&2,Chess.Ply>}:
  match promote:
    case True{}: Promotion.promotion_choices_exact(src,dst,tag,tail)
    case False{}: {==}

def keys(xs: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {F.emit(xs,src,pawn,ep,tail) == S.expand(xs,src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+dst,+rest}:
      +promo = S.promotes(pawn,dst)
      +tag = S.flag(pawn,dst,ep)
      Equal.trans(List<&2,Chess.Ply>,F.emit(Con{dst,rest},src,pawn,ep,tail),
        List.append(&2,Chess.Ply,S.block(promo,src,dst,tag),F.emit(rest,src,pawn,ep,tail)),
        S.expand(Con{dst,rest},src,pawn,ep,tail),
        put(promo,src,dst,tag,F.emit(rest,src,pawn,ep,tail)),
        Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,ms => List.append(&2,Chess.Ply,S.block(promo,src,dst,tag),ms),
          F.emit(rest,src,pawn,ep,tail),S.expand(rest,src,pawn,ep,tail),keys(rest,src,pawn,ep,tail)))

def destinations(+n: Nat,+empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,tail) ==
    S.expand(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  Equal.trans(List<&2,Chess.Ply>,Chess.destinations(n,empty,bb,src,pawn,ep,tail),
    F.emit(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail),
    S.expand(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail),
    F.plain(n,empty,bb,src,pawn,ep,tail),keys(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail))
