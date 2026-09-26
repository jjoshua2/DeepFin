# Prove actual ordinary/pawn scan cannot introduce flag two.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E

def put(+promote: Bool,+src: U32,+dst: U32,+ep: Bool,+tail: List<&2,Chess.Ply>,p: S.all(S.not_castle,tail)) -> S.all(S.not_castle,Chess.put_move(promote,src,dst,Bool.to_u32(ep),tail)):
  match promote:
    case True{}: ({==},({==},({==},({==},p))))
    case False{}:
      match ep:
        case False{}: ({==},p)
        case True{}: ({==},p)

def destinations(+n: Nat,+empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep_sq: U32,+tail: List<&2,Chess.Ply>,p: S.all(S.not_castle,tail)) -> S.all(S.not_castle,Chess.destinations(n,empty,bb,src,pawn,ep_sq,tail)):
  match n:
    case 0n: p
    case 1n+k:
      match empty:
        case True{}: p
        case False{}:
          +dst = U64.ctz(bb)
          +rank = U32.div(dst,8)
          +rest = U64.clear_lsb(bb)
          +promote = Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))
          +ep = Bool.and(pawn,U32.is_eq(dst,ep_sq))
          destinations(k,U64.is_zero(rest),rest,src,pawn,ep_sq,
            Chess.put_move(promote,src,dst,Bool.to_u32(ep),tail),put(promote,src,dst,ep,tail,p))

def after_then(+src: U32,+pawn: Bool,+ep: U32,+acc: List<&2,Chess.Ply>,r: Array<U64> & U64,
  -Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,ps: S.all(S.not_castle,acc),
  next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(S.not_castle,E.moves(r)) -> Q(r)) -> Q(Chess.scan_after(src,pawn,ep,acc,r)):
  (table,+targets) = r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,acc)),
    destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,acc,ps))

def step_then(+b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  -Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,ps: S.all(S.not_castle,E.moves(r)),
  next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(S.not_castle,E.moves(r)) -> Q(r)) -> Q(Chess.scan_step(b,src,r)):
  (table,acc) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  after_then(src,pawn,Chess.get_ep(b),acc,Chess.piece_targets(pawn,table,b,src),Q,ps,next)

def loop(+xs: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  -Q: @-r: (Array<U64> & List<&2,Chess.Ply>) -> Type,ps: S.all(S.not_castle,E.moves(r)),
  next: @r: (Array<U64> & List<&2,Chess.Ply>) -> S.all(S.not_castle,E.moves(r)) -> Q(r)) -> Q(Chess.scan(xs,b,r)):
  match xs:
    case Nil{}: next(r,ps)
    case Con{+src,+rest}:
      step_then(b,src,r,out => Q(Chess.scan(rest,b,out)),ps,
        out => p => loop(rest,b,out,Q,p,next))

def no_castle(+xs: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,ps: S.all(S.not_castle,E.moves(r))) -> S.all(S.not_castle,E.moves(Chess.scan(xs,b,r))):
  loop(xs,b,r,out => S.all(S.not_castle,E.moves(out)),ps,out => p => p)
