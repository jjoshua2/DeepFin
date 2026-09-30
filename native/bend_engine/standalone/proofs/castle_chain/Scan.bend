# Actual ordinary piece scan cannot introduce the castling tag.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ./Spec.bend as S

def bool_flag(b: Bool) -> {U32.is_eq(Bool.to_u32(b),2) == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def put(promote: Bool,+src: U32,+dst: U32,+flag: U32,+tail: List<&2,Chess.Ply>,
  tag: {U32.is_eq(flag,2) == False{} : Bool},good: S.every(S.ordinary,tail)) ->
  S.every(S.ordinary,Chess.put_move(promote,src,dst,flag,tail)):
  match promote:
    case False{}: (tag,good)
    case True{}: ({==},({==},({==},({==},good))))

def destinations(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  good: S.every(S.ordinary,tail)) ->
  S.every(S.ordinary,Chess.destinations(n,empty,bb,src,pawn,ep,tail)):
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
            put(promote,src,dst,Bool.to_u32(ep_capture),tail,bool_flag(ep_capture),good))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  good: S.every(S.ordinary,tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(S.ordinary,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets) = r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)),
    destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail,good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(S.ordinary,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(S.ordinary,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,src,r)):
  (table,tail) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  after(Goal,src,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,src),good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  good: S.every(S.ordinary,E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(S.ordinary,E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{src,+rest}:
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,good,
        z => pz => scan(Goal,rest,b,z,pz,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> S.every(S.ordinary,E.moves(r)) -> Goal(r)) ->
  Goal(S.candidates(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),Unit{},next)
