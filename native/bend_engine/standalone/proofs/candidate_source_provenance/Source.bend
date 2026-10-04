import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../castle_safety/Spec.bend as S
import ../bitboard_membership/AllEmittedSound.bend as All

def source(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {Nat.is_lt(U32.to_nat(S.source(m)),64n) == True{} : Bool} &
  {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(S.source(m))) == True{} : Bool}

# Both promotion blocks and ordinary moves preserve the caller-tail predicate.
def put(+b: Chess.Board,promote: Bool,+src: U32,+dst: U32,flag: U32,+tail: List<&2,Chess.Ply>,
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  good: C.every(m => source(b,m),tail)) ->
  C.every(m => source(b,m),Chess.put_move(promote,src,dst,flag,tail)):
  match promote:
    case False{}: ((bound,owned),good)
    case True{}: ((bound,owned),((bound,owned),((bound,owned),((bound,owned),good))))

def destinations(+b: Chess.Board,n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  good: C.every(m => source(b,m),tail)) ->
  C.every(m => source(b,m),Chess.destinations(n,empty,bb,src,pawn,ep,tail)):
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
          destinations(b,p,U64.is_zero(rest),rest,src,pawn,ep,Chess.put_move(promote,src,dst,tag,tail),
            bound,owned,put(b,promote,src,dst,tag,tail,bound,owned,good))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  good: C.every(m => source(b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => source(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets) = r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)),
    destinations(b,64n,U64.is_zero(targets),targets,src,pawn,ep,tail,bound,owned,good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  good: C.every(m => source(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => source(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,src,r)):
  (table,tail) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  after(Goal,b,src,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,src),bound,owned,good,next)

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +below: {All.all_below64(keys) == True{} : Bool},
  +set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool},
  good: C.every(m => source(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => source(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+src,+rest}:
      +own = Chess.color(b,Chess.get_turn(b))
      +bound = All.bool_and_true_left(Nat.is_lt(U32.to_nat(src),64n),All.all_below64(rest),below)
      +rest_below = All.bool_and_true_right(Nat.is_lt(U32.to_nat(src),64n),All.all_below64(rest),below)
      +owned = All.bool_and_true_left(U64.test_bit(own,U32.to_nat(src)),All.all_bits_set(rest,own),set)
      +rest_set = All.bool_and_true_right(U64.test_bit(own,U32.to_nat(src)),All.all_bits_set(rest,own),set)
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,bound,owned,good,
        z => pz => scan(Goal,rest,b,z,rest_below,rest_set,pz,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => source(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(C.candidates(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  +keys = Chess.bit_squares(64n,U64.is_zero(own),own,Nil{})
  scan(Goal,keys,b,(table,Nil{}),All.actual_bit_squares_below64(own),
    All.actual_bit_squares_all_set(own),Unit{},next)

def candidates(table: Array<U64>,+b: Chess.Board) ->
  C.every(m => source(b,m),E.moves(C.candidates(table,b))):
  initial(r => C.every(m => source(b,m),E.moves(r)),table,b,r => good => good)

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(C.candidates(table,b)))) -> source(b,m):
  initial(r => E.member(m,E.moves(r)) -> source(b,m),table,b,
    r => good => Lists.extract(m => source(b,m),r,m,good))(here)
