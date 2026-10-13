import Base
import ./Fields.bend as Fields
import ../promotion/Spec.bend as Promotion
import ../castle_chain/Spec.bend as Chain
import ../../../legal_probe/Chess.bend as Chess

# The nonpromotion flag is specifically Bool.to_u32, not an arbitrary raw flag.
# Promotion ignores that flag and emits all four typed choices with flag zero.
def put(promote: Bool,+src: U32,+dst: U32,ep: Bool,+tail: List<&2,Chess.Ply>,
  good: Chain.every(m => Fields.scanned(m),tail)) ->
  Chain.every(m => Fields.scanned(m),Chess.put_move(promote,src,dst,Bool.to_u32(ep),tail)):
  match promote:
    case False{}:
      match ep:
        case False{}: (Inl{(src,(dst,{==}))},good)
        case True{}: (Inr{Inl{(src,(dst,{==}))}},good)
    case True{}:
      (Inr{Inr{(src,(dst,(Promotion.Knight{},{==})))}},
       (Inr{Inr{(src,(dst,(Promotion.Bishop{},{==})))}},
        (Inr{Inr{(src,(dst,(Promotion.Rook{},{==})))}},
         (Inr{Inr{(src,(dst,(Promotion.Queen{},{==})))}},good))))

# Fuel and empty are the actual implementation parameters; neither is normalized.
def destinations(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  good: Chain.every(m => Fields.scanned(m),tail)) ->
  Chain.every(m => Fields.scanned(m),Chess.destinations(n,empty,bb,src,pawn,ep,tail)):
  match n:
    case 0n: good
    case 1n+p:
      match empty:
        case True{}: good
        case False{}:
          +dst = U64.ctz(bb)
          +rank = U32.div(dst,8)
          +rest = U64.clear_lsb(bb)
          +promote = Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))
          +ep_capture = Bool.and(pawn,U32.is_eq(dst,ep))
          destinations(p,U64.is_zero(rest),rest,src,pawn,ep,
            Chess.put_move(promote,src,dst,Bool.to_u32(ep_capture),tail),
            put(promote,src,dst,ep_capture,tail,good))
