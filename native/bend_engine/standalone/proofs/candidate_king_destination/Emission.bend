import Base
import ./Algebra.bend as A
import ./Bits.bend as Bits
import ./Targets.bend as T
import ../board/Spec.bend as B
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as Chess

def clear(b: Chess.Board,m: Chess.Ply) -> Type:
  match m:
    case Chess.Ply{src,dst,promotion,flag}: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b))

def put(+b: Chess.Board,promote: Bool,+src: U32,+dst: U32,flag: U32,+tail: List<&2,Chess.Ply>,
  +dst_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64},
  good: C.every(m => clear(b,m),tail)) -> C.every(m => clear(b,m),Chess.put_move(promote,src,dst,flag,tail)):
  match promote:
    case False{}: (dst_clear,good)
    case True{}: (dst_clear,(dst_clear,(dst_clear,(dst_clear,good))))

# Empty flags agree with the actual mask; arbitrary fuel and caller tails remain explicit.
def destinations(+b: Chess.Board,n: Nat,+empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  flag: {empty == U64.is_zero(bb) : Bool},
  +target_clear: {U64.and(bb,Chess.get_kings(b)) == U64.zero() : U64},
  good: C.every(m => clear(b,m),tail)) -> C.every(m => clear(b,m),Chess.destinations(n,empty,bb,src,pawn,ep,tail)):
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
          +tag = Bool.to_u32(Bool.and(pawn,U32.is_eq(dst,ep)))
          +head_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64} = Bits.ctz_clear(bb,Chess.get_kings(b),target_clear,Equal.sym(Bool,False{},U64.is_zero(bb),flag))
          destinations(b,p,U64.is_zero(rest),rest,src,pawn,ep,Chess.put_move(promote,src,dst,tag,tail),{==},
            A.gate_clear(bb,U64.sub(bb,U64.one()),Chess.get_kings(b),target_clear),put(b,promote,src,dst,tag,tail,head_clear,good))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  target_clear: D.disjoint(T.mask(r),Chess.get_kings(b)),good: C.every(m => clear(b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => clear(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets) = r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)),
    destinations(b,64n,U64.is_zero(targets),targets,src,pawn,ep,tail,{==},target_clear,good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  valid: {B.valid(b) == True{} : Bool},good: C.every(m => clear(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => clear(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan_step(b,src,r)):
  (table,+tail) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  T.piece(z => Goal(Chess.scan_after(src,pawn,Chess.get_ep(b),tail,z)),pawn,table,b,src,valid,
    z => target_clear => after(Goal,b,src,pawn,Chess.get_ep(b),tail,z,target_clear,good,next))

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +valid: {B.valid(b) == True{} : Bool},good: C.every(m => clear(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => clear(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+src,+rest}:
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,valid,good,z => pz => scan(Goal,rest,b,z,valid,pz,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,valid: {B.valid(b) == True{} : Bool},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => clear(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.candidates(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),valid,Unit{},next)

def candidates(table: Array<U64>,+b: Chess.Board,valid: {B.valid(b) == True{} : Bool}) ->
  C.every(m => clear(b,m),E.moves(C.candidates(table,b))):
  initial(r => C.every(m => clear(b,m),E.moves(r)),table,b,valid,r => good => good)

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,valid: {B.valid(b) == True{} : Bool},
  here: E.member(m,E.moves(C.candidates(table,b)))) -> clear(b,m):
  initial(r => E.member(m,E.moves(r)) -> clear(b,m),table,b,valid,r => good => Lists.extract(m => clear(b,m),r,m,good))(here)

