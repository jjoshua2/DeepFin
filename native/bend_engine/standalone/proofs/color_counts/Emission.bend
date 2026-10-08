import Base
import ./Actual.bend as A
import ../u64/Words.bend as Old
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../bitboard_membership/AllEmittedSound.bend as All
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../../../legal_probe/Chess.bend as Chess

def ordinary(+b: Chess.Board,+s: U32,+d: U32,ep: Bool,
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  db: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool}) -> A.emitted(b,Chess.Ply{s,d,0,Bool.to_u32(ep)}):
  match ep:
    case False{}: (sb,(owned,(db,{==})))
    case True{}: (sb,(owned,(db,{==})))
def put(+b: Chess.Board,promote: Bool,+s: U32,+d: U32,ep: Bool,+tail: List<&2,Chess.Ply>,
  +sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},+owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +db: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},good: C.every(m => A.emitted(b,m),tail)) ->
  C.every(m => A.emitted(b,m),Chess.put_move(promote,s,d,Bool.to_u32(ep),tail)):
  match promote:
    case False{}: (ordinary(b,s,d,ep,sb,owned,db),good)
    case True{}:
      ((sb,(owned,(db,{==}))),((sb,(owned,(db,{==}))),((sb,(owned,(db,{==}))),((sb,(owned,(db,{==}))),good))))
def destinations(+b: Chess.Board,n: Nat,+empty: Bool,+bb: U64,+s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  coherent: {empty == U64.is_zero(bb) : Bool},+sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},good: C.every(m => A.emitted(b,m),tail)) ->
  C.every(m => A.emitted(b,m),Chess.destinations(n,empty,bb,s,pawn,ep,tail)):
  match n:
    case 0n: good
    case 1n+p:
      match empty:
        case True{}: good
        case False{}:
          +d=U64.ctz(bb)
          +rank=U32.div(d,8)
          +rest=U64.clear_lsb(bb)
          +promote=Bool.and(pawn,Bool.or(U32.is_zero(rank),U32.is_eq(rank,7)))
          +tag=Bool.and(pawn,U32.is_eq(d,ep))
          destinations(b,p,U64.is_zero(rest),rest,s,pawn,ep,Chess.put_move(promote,s,d,Bool.to_u32(tag),tail),{==},sb,owned,
            put(b,promote,s,d,tag,tail,sb,owned,Ctz.actual_ctz_range(bb,Equal.sym(Bool,False{},U64.is_zero(bb),coherent)),good))
def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+s: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  r: Array<U64> & U64,sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},good: C.every(m => A.emitted(b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan_after(s,pawn,ep,tail,r)):
  (table,+targets)=r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,s,pawn,ep,tail)),
    destinations(b,64n,U64.is_zero(targets),targets,s,pawn,ep,tail,{==},sb,owned,good))
def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+s: U32,r: Array<U64> & List<&2,Chess.Ply>,
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  good: C.every(m => A.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan_step(b,s,r)):
  (table,tail)=r
  +pawn=U64.test_bit(Chess.get_pawns(b),U32.to_nat(s))
  after(Goal,b,s,pawn,Chess.get_ep(b),tail,Chess.piece_targets(pawn,table,b,s),sb,owned,good,next)
def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +below: {All.all_below64(keys) == True{} : Bool},+set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool},
  good: C.every(m => A.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+s,+rest}:
      +own=Chess.color(b,Chess.get_turn(b))
      +sb=All.bool_and_true_left(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +rb=All.bool_and_true_right(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +owned=All.bool_and_true_left(U64.test_bit(own,U32.to_nat(s)),All.all_bits_set(rest,own),set)
      +rs=All.bool_and_true_right(U64.test_bit(own,U32.to_nat(s)),All.all_bits_set(rest,own),set)
      step(z => Goal(Chess.scan(rest,b,z)),b,s,r,sb,owned,good,z => pz => scan(Goal,rest,b,z,rb,rs,pz,next))
def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.candidates(table,b)):
  +own=Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),All.actual_bit_squares_below64(own),All.actual_bit_squares_all_set(own),Unit{},next)

def right_cases(white: Bool,ks: Bool) ->
  {Bool.not(U32.is_zero(U32.and(0,U32.mul(Bool.pick(U32,white,1,4),Bool.pick(U32,ks,1,2))))) == False{} : Bool}:
  match white ks:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}
def guard_false(+b: Chess.Board,+ks: Bool,zero: {Chess.get_rights(b) == 0 : U32}) -> {E.guard(b,ks) == False{} : Bool}:
  %Equal.sym(U32,Chess.get_rights(b),0,zero) :
    {Bool.and(Bool.not(U32.is_zero(U32.and(_,E.right(b,ks)))),Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)))) == False{} : Bool}
  %Equal.sym(Bool,Bool.not(U32.is_zero(U32.and(0,E.right(b,ks)))),False{},right_cases(U32.is_eq(Chess.get_turn(b),1),ks)) :
    {Bool.and(_,Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)))) == False{} : Bool}
  {==}
def castle(+b: Chess.Board,+ks: Bool,zero: {Chess.get_rights(b) == 0 : U32},guard: {E.guard(b,ks) == True{} : Bool}) -> A.emitted(b,E.move(b,ks)):
  Old.false_true(A.emitted(b,E.move(b,ks)),Equal.trans(Bool,False{},E.guard(b,ks),True{},Equal.sym(Bool,E.guard(b,ks),False{},guard_false(b,ks,zero)),guard))
def suffix(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +zero: {Chess.get_rights(b) == 0 : U32},good: C.every(m => A.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.suffix(b,r)):
  Chain.suffix(m => A.emitted(b,m),Goal,b,r,good,g => castle(b,True{},zero,g),g => castle(b,False{},zero,g),next)
def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,+zero: {Chess.get_rights(b) == 0 : U32},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => A.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.legal_moves(table,b)):
  initial(z => Goal(C.suffix(b,z)),table,b,z => good => suffix(Goal,b,z,zero,good,next))
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,zero: {Chess.get_rights(b) == 0 : U32},here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> A.decreases(b,m):
  A.post(b,m,legal(r => E.member(m,E.moves(r)) -> A.emitted(b,m),table,b,zero,r => good => Lists.extract(x => A.emitted(b,x),r,m,good))(here))
