# Shared-prefix structural identity through the actual producers.
import Base
import ../../../legal_probe/Chess.bend as Chess

def emit(keys: List<&2,U32>,+src: U32,+pawn: Bool,+ep: U32,tail: List<&2,Chess.Ply>) -> List<&2,Chess.Ply>:
  match keys:
    case Nil{}: tail
    case Con{+dst,rest}:
      +rank = U32.div(dst,8)
      Chess.put_move(Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7))),src,dst,
        Bool.to_u32(Bool.and(pawn,U32.is_eq(dst,ep))),emit(rest,src,pawn,ep,tail))

def structural(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,
  +keys: List<&2,U32>,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,emit(keys,src,pawn,ep,tail)) ==
    emit(Chess.bit_squares(n,empty,bb,keys),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  match n:
    case 0n: {==}
    case 1n+p:
      match empty:
        case True{}: {==}
        case False{}:
          +dst = U64.ctz(bb)
          +rest = U64.clear_lsb(bb)
          structural(p,U64.is_zero(rest),rest,src,pawn,ep,Con{dst,keys},tail)

def plain(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,tail) ==
    emit(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  structural(n,empty,bb,src,pawn,ep,Nil{},tail)
