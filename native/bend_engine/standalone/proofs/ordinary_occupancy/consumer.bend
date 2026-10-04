import Base
import ./Ordinary.bend as O
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def ordinary(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})) ==
    U64.or(U64.and_not(Chess.occupied(b),U64.bit(U32.to_nat(src))),U64.bit(U32.to_nat(dst))) : U64}:
  O.normalization(b,src,dst)

def ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: Chess.Board,+src: U32,+dst: U32,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},
  empty: D.disjoint(U64.bit(U32.to_nat(src)),P.attack(xs,Chess.occupied(b)))) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(b),False{},acc)):
  O.actual_ray(n,sq,dir,xs,b,src,dst,acc,route,empty)

# Deliberately inconsistent piece rows and arbitrary raw turn; occupancy is color-only.
def board(+white: U64,+black: U64,turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),white,black,turn,0,64}

def closed() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(6),U64.zero(),7),Chess.Ply{1,3,0,0})) == U64.from_u32(12) : U64}: {==}
def closed_by_theorem() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(6),U64.zero(),7),Chess.Ply{1,3,0,0})) ==
    O.normalized(board(U64.from_u32(6),U64.zero(),7),1,3) : U64}:
  O.normalization(board(U64.from_u32(6),U64.zero(),7),1,3)
def same_square() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(6),U64.from_u32(6),7),Chess.Ply{1,1,0,0})) == U64.from_u32(6) : U64}: {==}
def absent_source() ->
  {Chess.occupied(Chess.make_move(board(U64.zero(),U64.zero(),9),Chess.Ply{4,3,0,0})) == U64.from_u32(8) : U64}: {==}
def out_of_range() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(6),U64.zero(),7),Chess.Ply{64,65,0,0})) == U64.from_u32(6) : U64}: {==}

def path() -> List<&2,Nat>: 1n <> 2n <> 3n <> 4n <> Nil{}
def board_ray(+acc: U64) ->
  L.subset(Tables.ray(4n,False{},0,0,Chess.occupied(Chess.make_move(board(U64.from_u32(12),U64.zero(),7),Chess.Ply{3,1,0,0})),False{},acc),
    Tables.ray(4n,False{},0,0,Chess.occupied(board(U64.from_u32(12),U64.zero(),7)),False{},acc)):
  O.actual_ray(4n,0,0,path(),board(U64.from_u32(12),U64.zero(),7),3,1,acc,{==},{==})
def new_ray() ->
  {Tables.ray(4n,False{},0,0,Chess.occupied(Chess.make_move(board(U64.from_u32(12),U64.zero(),7),Chess.Ply{3,1,0,0})),False{},U64.zero()) == U64.from_u32(2) : U64}: {==}
def old_ray() ->
  {Tables.ray(4n,False{},0,0,Chess.occupied(board(U64.from_u32(12),U64.zero(),7)),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}

# Special flags change extra occupancy; these are total-function witnesses, not legal moves.
def ep_actual() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(2),U64.from_u32(4),1),Chess.Ply{1,10,0,1})) == U64.from_u32(1024) : U64}: {==}
def ep_not_ordinary() ->
  {U64.is_eq(Chess.occupied(Chess.make_move(board(U64.from_u32(2),U64.from_u32(4),1),Chess.Ply{1,10,0,1})),
    O.normalized(board(U64.from_u32(2),U64.from_u32(4),1),1,10)) == False{} : Bool}: {==}
def castle_actual() ->
  {Chess.occupied(Chess.make_move(board(U64.from_u32(144),U64.zero(),1),Chess.Ply{4,6,0,2})) == U64.from_u32(96) : U64}: {==}
def castle_not_ordinary() ->
  {U64.is_eq(Chess.occupied(Chess.make_move(board(U64.from_u32(144),U64.zero(),1),Chess.Ply{4,6,0,2})),
    O.normalized(board(U64.from_u32(144),U64.zero(),1),4,6)) == False{} : Bool}: {==}
