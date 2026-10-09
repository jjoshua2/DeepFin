import Base
import ./Emission.bend as Destination
import ./Targets.bend as Targets
import ../candidate_source_provenance/Source.bend as Source
import ../board/Spec.bend as B
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as Lists
import ../castle_emission/Spec.bend as E
import ../bitboard_membership/AllEmittedSound.bend as All
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as Chess

# Pair two predicates over the same concrete list; neither table nor membership is copied.
def pair_lists(-P: Chess.Ply -> Type,-Q: Chess.Ply -> Type,+xs: List<&2,Chess.Ply>,
  left: C.every(P,xs),right: C.every(Q,xs)) -> C.every(m => P(m) & Q(m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{h,t}:
      (ph,pt) = left
      (qh,qt) = right
      ((ph,qh),pair_lists(P,Q,t,pt,qt))

def after(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,
  +b: Chess.Board,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,r: Array<U64> & U64,
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  target_clear: D.disjoint(Targets.mask(r),Chess.get_kings(b)),
  source_good: C.every(m => Source.source(b,m),tail),destination_good: C.every(m => Destination.clear(b,m),tail),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => Source.source(b,m),E.moves(r)) -> C.every(m => Destination.clear(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_after(src,pawn,ep,tail,r)):
  (table,+targets) = r
  +moves = Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)
  next((table,moves),Source.destinations(b,64n,U64.is_zero(targets),targets,src,pawn,ep,tail,bound,owned,source_good),
    Destination.destinations(b,64n,U64.is_zero(targets),targets,src,pawn,ep,tail,{==},target_clear,destination_good))

def step(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+b: Chess.Board,+src: U32,r: Array<U64> & List<&2,Chess.Ply>,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  valid: {B.valid(b) == True{} : Bool},source_good: C.every(m => Source.source(b,m),E.moves(r)),destination_good: C.every(m => Destination.clear(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => Source.source(b,m),E.moves(r)) -> C.every(m => Destination.clear(b,m),E.moves(r)) -> Goal(r)) ->
  Goal(Chess.scan_step(b,src,r)):
  (table,+tail) = r
  +pawn = U64.test_bit(Chess.get_pawns(b),U32.to_nat(src))
  Targets.piece(z => Goal(Chess.scan_after(src,pawn,Chess.get_ep(b),tail,z)),pawn,table,b,src,valid,
    z => clear => after(Goal,b,src,pawn,Chess.get_ep(b),tail,z,bound,owned,clear,source_good,destination_good,next))

def scan(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  +below: {All.all_below64(keys) == True{} : Bool},+set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool},
  +valid: {B.valid(b) == True{} : Bool},source_good: C.every(m => Source.source(b,m),E.moves(r)),destination_good: C.every(m => Destination.clear(b,m),E.moves(r)),
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => Source.source(b,m),E.moves(r)) -> C.every(m => Destination.clear(b,m),E.moves(r)) -> Goal(r)) -> Goal(Chess.scan(keys,b,r)):
  match keys:
    case Nil{}: next(r,source_good,destination_good)
    case Con{+src,+rest}:
      +own = Chess.color(b,Chess.get_turn(b))
      +bound = All.bool_and_true_left(Nat.is_lt(U32.to_nat(src),64n),All.all_below64(rest),below)
      +rest_below = All.bool_and_true_right(Nat.is_lt(U32.to_nat(src),64n),All.all_below64(rest),below)
      +owned = All.bool_and_true_left(U64.test_bit(own,U32.to_nat(src)),All.all_bits_set(rest,own),set)
      +rest_set = All.bool_and_true_right(U64.test_bit(own,U32.to_nat(src)),All.all_bits_set(rest,own),set)
      step(z => Goal(Chess.scan(rest,b,z)),b,src,r,bound,owned,valid,source_good,destination_good,
        z => ps => pd => scan(Goal,rest,b,z,rest_below,rest_set,valid,ps,pd,next))

def initial(-Goal: @-r: Array<U64> & List<&2,Chess.Ply> -> Type,table: Array<U64>,+b: Chess.Board,valid: {B.valid(b) == True{} : Bool},
  next: @r: Array<U64> & List<&2,Chess.Ply> -> C.every(m => Source.source(b,m),E.moves(r)) -> C.every(m => Destination.clear(b,m),E.moves(r)) -> Goal(r)) -> Goal(C.candidates(table,b)):
  +own = Chess.color(b,Chess.get_turn(b))
  scan(Goal,Chess.bit_squares(64n,U64.is_zero(own),own,Nil{}),b,(table,Nil{}),All.actual_bit_squares_below64(own),All.actual_bit_squares_all_set(own),valid,Unit{},Unit{},next)

def extract(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,+m: Chess.Ply,
  source_good: C.every(x => Source.source(b,x),E.moves(r)),destination_good: C.every(x => Destination.clear(b,x),E.moves(r))) ->
  E.member(m,E.moves(r)) -> (Source.source(b,m) & Destination.clear(b,m)):
  (table,+xs) = r
  here => Lists.at(x => Source.source(b,x) & Destination.clear(b,x),xs,m,
    pair_lists(x => Source.source(b,x),x => Destination.clear(b,x),xs,source_good,destination_good),here)

def member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,valid: {B.valid(b) == True{} : Bool},
  here: E.member(m,E.moves(C.candidates(table,b)))) -> Source.source(b,m) & Destination.clear(b,m):
  initial(r => E.member(m,E.moves(r)) -> (Source.source(b,m) & Destination.clear(b,m)),table,b,valid,r => ps => pd => extract(b,r,m,ps,pd))(here)
