import Base
import ./Source.bend as N
import ../../../legal_probe/Chess.bend as Chess
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../castle_safety/Spec.bend as S
import ../bitboard_membership/AllEmittedSound.bend as All
import ../filter_ray_exclusion/Filter.bend as F
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Path.bend as P
import ../ray/Spec.bend as R
import ../../Tables.bend as Tables
import ../destination_factorization/Spec.bend as Count

def use_candidates(table: Array<U64>,+b: Chess.Board) ->
  C.every(m => N.source(b,m),E.moves(C.candidates(table,b))):
  N.candidates(table,b)

def use_put(+b: Chess.Board,promote: Bool,+src: U32,+dst: U32,flag: U32,+tail: List<&2,Chess.Ply>,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  good: C.every(m => N.source(b,m),tail)) ->
  C.every(m => N.source(b,m),Chess.put_move(promote,src,dst,flag,tail)):
  N.put(b,promote,src,dst,flag,tail,bound,owned,good)

def use_scan(+keys: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,
  below: {All.all_below64(keys) == True{} : Bool},
  set: {All.all_bits_set(keys,Chess.color(b,Chess.get_turn(b))) == True{} : Bool},
  good: C.every(m => N.source(b,m),E.moves(r))) ->
  C.every(m => N.source(b,m),E.moves(Chess.scan(keys,b,r))):
  N.scan(z => C.every(m => N.source(b,m),E.moves(z)),keys,b,r,below,set,good,z => pz => pz)

def use_member(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(C.candidates(table,b)))) -> N.source(b,m):
  N.member(table,b,m,here)

# Ownership and index bounds are now derived from actual candidate membership.
def exclusion_known(+b: Chess.Board,+rays: U64,+m: Chess.Ply,cert: N.source(b,m),
  bypass: {Chess.filter_requires(S.sensitive(b,rays),m) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(S.source(m))),rays):
  match m:
    case Chess.Ply{src,dst,promotion,flag}:
      (bound,owned) = cert
      F.source_exclusion(b,rays,src,dst,promotion,flag,bound,owned,bypass)

def use_exclusion(table: Array<U64>,+b: Chess.Board,+rays: U64,+m: Chess.Ply,
  here: E.member(m,E.moves(C.candidates(table,b))),
  bypass: {Chess.filter_requires(S.sensitive(b,rays),m) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(S.source(m))),rays):
  exclusion_known(b,rays,m,N.member(table,b,m,here),bypass)

def ray_known(+b: Chess.Board,+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+src: U32,+dst: U32,+rays: U64,+acc: U64,
  cert: N.source(b,Chess.Ply{src,dst,0,0}),
  route: {R.trace(n,sq,dir,xs) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  coverage: L.subset(P.attack(xs,Chess.occupied(b)),rays)) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(b),False{},acc)):
  (bound,owned) = cert
  F.actual_ray(n,sq,dir,xs,b,src,dst,rays,acc,route,bound,owned,bypass,coverage)

def use_ray(table: Array<U64>,+b: Chess.Board,+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+src: U32,+dst: U32,+rays: U64,+acc: U64,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(table,b))),
  route: {R.trace(n,sq,dir,xs) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  coverage: L.subset(P.attack(xs,Chess.occupied(b)),rays)) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(b),False{},acc)):
  ray_known(b,n,sq,dir,xs,src,dst,rays,acc,N.member(table,b,Chess.Ply{src,dst,0,0},here),route,bypass,coverage)

def board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    U64.from_u32(12),U64.zero(),1,0,64}
def mixed_tail() -> List<&2,Chess.Ply>:
  Chess.Ply{3,1,0,0} <> Chess.Ply{3,1,0,0} <> Nil{}
def tail_good() -> C.every(m => N.source(board(),m),mixed_tail()):
  (({==},{==}),(({==},{==}),Unit{}))
def mixed_source_output() ->
  C.every(m => N.source(board(),m),Chess.put_move(False{},2,1,0,mixed_tail())):
  N.put(board(),False{},2,1,0,mixed_tail(),{==},{==},tail_good())
def promotions_with_mixed_tail() ->
  C.every(m => N.source(board(),m),Chess.put_move(True{},2,56,1,mixed_tail())):
  N.put(board(),True{},2,56,1,mixed_tail(),{==},{==},tail_good())
def duplicate_tail_count() ->
  {Count.count(Chess.put_move(False{},2,1,0,mixed_tail()),Chess.Ply{3,1,0,0}) == 2n : Nat}: {==}
def bad_tail_bound() ->
  {Nat.is_lt(U32.to_nat(S.source(Chess.Ply{64,1,0,0})),64n) == False{} : Bool}: {==}
def foreign_source() ->
  {U64.test_bit(Chess.color(board(),Chess.get_turn(board())),2n) == True{} : Bool}: {==}
def opposite_source() ->
  {U64.test_bit(Chess.color(board(),U32.xor(Chess.get_turn(board()),1)),2n) == False{} : Bool}: {==}

# Total-function witnesses: ownership must come from the actual input keys.
def pawn_board(own: U64) -> Chess.Board:
  Chess.Board{U64.from_u32(4),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    own,U64.zero(),1,0,64}
def unowned_scan() ->
  {E.moves(Chess.scan(2 <> Nil{},pawn_board(U64.zero()),(ALeaf{U64.zero()},Nil{}))) ==
    Chess.Ply{2,10,0,0} <> Nil{} : List<&2,Chess.Ply>}: {==}
def unowned_scan_source() ->
  {U64.test_bit(Chess.color(pawn_board(U64.zero()),Chess.get_turn(pawn_board(U64.zero()))),2n) == False{} : Bool}: {==}
def actual_candidate() ->
  {E.moves(C.candidates(ALeaf{U64.zero()},pawn_board(U64.from_u32(4)))) ==
    Chess.Ply{2,10,0,0} <> Nil{} : List<&2,Chess.Ply>}: {==}
def actual_member() ->
  E.member(Chess.Ply{2,10,0,0},E.moves(C.candidates(ALeaf{U64.zero()},pawn_board(U64.from_u32(4))))):
  Inl{{==}}
def actual_source() -> N.source(pawn_board(U64.from_u32(4)),Chess.Ply{2,10,0,0}):
  N.member(ALeaf{U64.zero()},pawn_board(U64.from_u32(4)),Chess.Ply{2,10,0,0},actual_member())
def actual_exclusion() -> D.disjoint(U64.bit(2n),U64.from_u32(2)):
  use_exclusion(ALeaf{U64.zero()},pawn_board(U64.from_u32(4)),U64.from_u32(2),Chess.Ply{2,10,0,0},actual_member(),{==})
