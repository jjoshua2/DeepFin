import Base
import ./Actual.bend as K
import ./Selector.bend as Selector
import ../board/Spec.bend as Board
import ../backrank_pawns/Mask.bend as M
import ../color_counts/Actual.bend as A
import ../color_counts/Emission.bend as Prior
import ../candidate_king_destination/Targets.bend as Targets
import ../candidate_king_destination/Bits.bend as Bits
import ../candidate_king_destination/Algebra.bend as Algebra
import ../ep_king_frame/Parent.bend as EP
import ../bitboard_membership/IndexEquality.bend as Index
import ../bitboard_membership/AllEmittedSound.bend as All
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../u64/Words.bend as Old
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def ep_here(+b: Chess.Board,+d: U32,pawn: Bool,tag: {Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b))) == True{} : Bool},
  parent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool}) -> {Pos.valid_ep(d,b) == True{} : Bool}:
  %Equal.sym(U32,d,Chess.get_ep(b),Index.u32_eq_true_same(d,Chess.get_ep(b),All.bool_and_true_right(pawn,U32.is_eq(d,Chess.get_ep(b)),tag))) :
    {Pos.valid_ep(_,b) == True{} : Bool}
  parent
def victim_clear(+b: Chess.Board,+d: U32,pair: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64} &
  {U64.and(U64.bit(U32.to_nat(U32.xor(d,8))),Chess.get_kings(b)) == U64.zero() : U64}) ->
  {U64.and(U64.bit(U32.to_nat(U32.xor(d,8))),Chess.get_kings(b)) == U64.zero() : U64}:
  (dc,vc)=pair
  vc
def cap_flag(+b: Chess.Board,+s: U32,+d: U32,+pawn: Bool,flag: Bool,
  selected: {Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b))) == flag : Bool},
  rows: {Board.valid(b) == True{} : Bool},db: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},dst: K.target_clear(b,Chess.Ply{s,d,0,Bool.to_u32(flag)})) ->
  K.cap_clear(b,Chess.Ply{s,d,0,Bool.to_u32(flag)}):
  match flag:
    case False{}: dst
    case True{}:
      victim_clear(b,d,EP.clear(b,d,rows,db,ep_here(b,d,pawn,selected,ep)))
def ordinary(+b: Chess.Board,+s: U32,+d: U32,+pawn: Bool,+rows: {Board.valid(b) == True{} : Bool},
  +sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},+owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +db: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  +dst: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64}) ->
  K.emitted(b,Chess.Ply{s,d,0,Bool.to_u32(Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b))))}):
  +flag=Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b)))
  (Prior.ordinary(b,s,d,flag,sb,owned,db),(Selector.ordinary(b,s,d,Bool.to_u32(flag),rows,sb,owned),
    (dst,cap_flag(b,s,d,pawn,flag,{==},rows,db,ep,dst))))

# A promoted pawn inserts only selectors 1--4 and the implementation resets its flag to zero.
def put(+b: Chess.Board,+s: U32,+d: U32,pawn: Bool,rank: Bool,+tail: List<&2,Chess.Ply>,
  source: {M.pawn(b,s) == pawn : Bool},+rows: {Board.valid(b) == True{} : Bool},
  +sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},+owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +db: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},+ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  +dst: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64},good: C.every(m => K.emitted(b,m),tail)) ->
  C.every(m => K.emitted(b,m),Chess.put_move(Bool.and(pawn,rank),s,d,Bool.to_u32(Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b)))),tail)):
  match pawn rank:
    case False{} _: (ordinary(b,s,d,False{},rows,sb,owned,db,ep,dst),good)
    case True{} False{}: (ordinary(b,s,d,True{},rows,sb,owned,db,ep,dst),good)
    case True{} True{}:
      +pawn_set=source
      (((sb,(owned,(db,{==}))),(Selector.promoted(b,s,d,1,0,rows,sb,pawn_set,{==},{==}),(dst,dst))),
       (((sb,(owned,(db,{==}))),(Selector.promoted(b,s,d,2,0,rows,sb,pawn_set,{==},{==}),(dst,dst))),
        (((sb,(owned,(db,{==}))),(Selector.promoted(b,s,d,3,0,rows,sb,pawn_set,{==},{==}),(dst,dst))),
         (((sb,(owned,(db,{==}))),(Selector.promoted(b,s,d,4,0,rows,sb,pawn_set,{==},{==}),(dst,dst))),good))))

# Each physical tail cons has its own certificate; fuel and empty flags are actual inputs.
def destinations(+b: Chess.Board,n: Nat,+empty: Bool,+bb: U64,+s: U32,+pawn: Bool,+tail: List<&2,Chess.Ply>,
  +coherent: {empty == U64.is_zero(bb) : Bool},+source: {M.pawn(b,s) == pawn : Bool},+rows: {Board.valid(b) == True{} : Bool},
  +sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},+owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  +ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},+clear: {U64.and(bb,Chess.get_kings(b)) == U64.zero() : U64},
  good: C.every(m => K.emitted(b,m),tail)) ->
  C.every(m => K.emitted(b,m),Chess.destinations(n,empty,bb,s,pawn,Chess.get_ep(b),tail)):
  match n:
    case 0n: good
    case 1n+p:
      match empty:
        case True{}: good
        case False{}:
          +d=U64.ctz(bb)
          +rest=U64.clear_lsb(bb)
          +rank=M.backrank(d)
          +flag=Bool.to_u32(Bool.and(pawn,U32.is_eq(d,Chess.get_ep(b))))
          +dst: {U64.and(U64.bit(U32.to_nat(d)),Chess.get_kings(b)) == U64.zero() : U64}=Bits.ctz_clear(bb,Chess.get_kings(b),clear,Equal.sym(Bool,False{},U64.is_zero(bb),coherent))
          destinations(b,p,U64.is_zero(rest),rest,s,pawn,Chess.put_move(Bool.and(pawn,rank),s,d,flag,tail),{==},source,rows,sb,owned,ep,
            Algebra.gate_clear(bb,U64.sub(bb,U64.one()),Chess.get_kings(b),clear),
            put(b,s,d,pawn,rank,tail,source,rows,sb,owned,Ctz.actual_ctz_range(bb,Equal.sym(Bool,False{},U64.is_zero(bb),coherent)),ep,dst,good))
def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+s: U32,+pawn: Bool,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  source: {M.pawn(b,s) == pawn : Bool},rows: {Board.valid(b) == True{} : Bool},
  sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},
  ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},clear: {U64.and(Targets.mask(r),Chess.get_kings(b)) == U64.zero() : U64},
  good: C.every(m => K.emitted(b,m),tail),next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(s,pawn,Chess.get_ep(b),tail,r)):
  (table,+targets)=r
  next((table,Chess.destinations(64n,U64.is_zero(targets),targets,s,pawn,Chess.get_ep(b),tail)),
    destinations(b,64n,U64.is_zero(targets),targets,s,pawn,tail,{==},source,rows,sb,owned,ep,clear,good))
def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+s: U32,r: Array<U64> & List<&2,Chess.Ply>,
  +rows: {Board.valid(b) == True{} : Bool},sb: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(s)) == True{} : Bool},ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  good: C.every(m => K.emitted(b,m),E.moves(r)),next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,s,r)):
  (table,+tail)=r
  +pawn=M.pawn(b,s)
  Targets.piece(z => Goal(Chess.scan_after(s,pawn,Chess.get_ep(b),tail,z)),pawn,table,b,s,rows,
    z => clear => after(Goal,b,s,pawn,tail,z,{==},rows,sb,owned,ep,clear,good,next))
def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +rows: {Board.valid(b) == True{} : Bool},+below: {All.all_below64(keys) == True{} : Bool},+set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool},
  +ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},good: C.every(m => K.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,good)
    case Con{+s,+rest}:
      +own=Chess.color(b,Chess.get_turn(b))
      +sb=All.bool_and_true_left(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +rb=All.bool_and_true_right(Nat.is_lt(U32.to_nat(s),64n),All.all_below64(rest),below)
      +owned=All.bool_and_true_left(U64.test_bit(own,U32.to_nat(s)),All.all_bits_set(rest,own),set)
      +rs=All.bool_and_true_right(U64.test_bit(own,U32.to_nat(s)),All.all_bits_set(rest,own),set)
      step(z => Goal(Chess.scan(rest,b,z)),b,s,r,rows,sb,owned,ep,good,z => pz => scan(Goal,rest,b,z,rows,rb,rs,ep,pz,next))
def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  rows: {Board.valid(b) == True{} : Bool},ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.candidates(table,b)):
  +own=Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),rows,All.actual_bit_squares_below64(own),All.actual_bit_squares_all_set(own),ep,Unit{},next)
def castle(+b: Chess.Board,+ks: Bool,zero: {Chess.get_rights(b) == 0 : U32},guard: {E.guard(b,ks) == True{} : Bool}) -> K.emitted(b,E.move(b,ks)):
  Old.false_true(K.emitted(b,E.move(b,ks)),Equal.trans(Bool,False{},E.guard(b,ks),True{},Equal.sym(Bool,E.guard(b,ks),False{},Prior.guard_false(b,ks,zero)),guard))
def suffix(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +zero: {Chess.get_rights(b) == 0 : U32},good: C.every(m => K.emitted(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.suffix(b,r)):
  Chain.suffix(m => K.emitted(b,m),Goal,b,r,good,g => castle(b,True{},zero,g),g => castle(b,False{},zero,g),next)
def legal(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,
  rows: {Board.valid(b) == True{} : Bool},ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},+zero: {Chess.get_rights(b) == 0 : U32},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => K.emitted(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.legal_moves(table,b)):
  initial(z => Goal(C.suffix(b,z)),table,b,rows,ep,z => good => suffix(Goal,b,z,zero,good,next))
def legal_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rows: {Board.valid(b) == True{} : Bool},
  ep: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},zero: {Chess.get_rights(b) == 0 : U32},here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> K.preserved(b,m):
  K.post(b,m,rows,legal(r => E.member(m,E.moves(r)) -> K.emitted(b,m),table,b,rows,ep,zero,r => good => Lists.extract(x => K.emitted(b,x),r,m,good))(here))
