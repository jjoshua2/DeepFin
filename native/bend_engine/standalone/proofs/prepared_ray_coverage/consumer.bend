import Base
import ./Coverage.bend as G
import ./Query.bend as Q
import ../storage/Observe.bend as O
import ../candidate_source_provenance/consumer.bend as Candidate
import ../accepted_castle/Spec.bend as Prepared
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../castle_safety/Spec.bend as S
import ../occupancy_insertion/Lattice.bend as L
import ../ray/Rays.bend as Rays
import ../ray/Trace.bend as Trace
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables

def use_query(+c: O.Cells,+q: Nat,+side: U32,+occ: U64,
  r: Q.rook(c,q,side,occ),d: Q.bishop(c,q,side,occ)) ->
  {Chess.attack(4,O.pack(c),U32.from_nat(q),side,occ) ==
    (O.pack(c),G.actual(q,occ)) : Array<U64> & U64}:
  Q.queen(c,q,side,occ,r,d)

def use_coverage(+c: O.Cells,+b: Chess.Board,+q: Nat,+dir: Nat,
  +qb: {Nat.is_lt(q,64n) == True{} : Bool},+db: {Nat.is_lt(dir,8n) == True{} : Bool},
  chosen: {Q.king(b) == U32.from_nat(q) : U32},
  r: Q.rook(c,q,Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,q,Chess.get_turn(b),Chess.occupied(b))) ->
  L.subset(G.path(q,dir,Chess.occupied(b)),Prepared.rays(c,b)):
  Q.coverage(c,b,q,dir,qb,db,chosen,r,d)

# Actual candidate membership supplies source bounds and ownership. This bridge
# supplies both the old path coverage and actual 7-step route, with no such caller
# premises. It retains the actual old/new False flags and the same accumulator.
def use_ray(generation_table: Array<U64>,+c: O.Cells,+b: Chess.Board,+q: Nat,+dir: Nat,+src: U32,+dst: U32,+acc: U64,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  +qb: {Nat.is_lt(q,64n) == True{} : Bool},+db: {Nat.is_lt(dir,8n) == True{} : Bool},
  chosen: {Q.king(b) == U32.from_nat(q) : U32},
  r: Q.rook(c,q,Chess.get_turn(b),Chess.occupied(b)),
  d: Q.bishop(c,q,Chess.get_turn(b),Chess.occupied(b)),
  bypass: {Chess.filter_requires(S.sensitive(b,Prepared.rays(c,b)),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  L.subset(Tables.ray(7n,False{},U32.from_nat(q),U32.from_nat(dir),Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(7n,False{},U32.from_nat(q),U32.from_nat(dir),Chess.occupied(b),False{},acc)):
  Candidate.use_ray(generation_table,b,7n,U32.from_nat(q),U32.from_nat(dir),Rays.path(7n,q,dir),src,dst,Prepared.rays(c,b),acc,
    here,Trace.finite(q,dir,qb,db),bypass,Q.coverage(c,b,q,dir,qb,db,chosen,r,d))

def corner() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(0n),
    U64.bit(0n),U64.zero(),1,0,64}
def selected_corner() -> {Q.king(corner()) == 0 : U32}: {==}

# An arbitrary supplied table can violate lookup geometry. The nonzero old path
# exists even though this actual queen lookup returns zero and preserves the table.
def empty_table_query() ->
  {Chess.attack(4,ALeaf{U64.zero()},0,1,Chess.occupied(corner())) ==
    (ALeaf{U64.zero()},U64.zero()) : Array<U64> & U64}: {==}
def uncovered_supplied_table() ->
  {U64.is_eq(U64.and(Prepared.rays(O.Cell{U64.zero()},corner()),G.path(0n,0n,Chess.occupied(corner()))),
    G.path(0n,0n,Chess.occupied(corner()))) == False{} : Bool}: {==}

def orthogonal_high() ->
  L.subset(G.path(0n,2n,U64.zero()),G.actual(0n,U64.zero())):
  G.actual_coverage(0n,2n,U64.zero(),{==},{==})
def diagonal_high() ->
  L.subset(G.path(0n,4n,U64.zero()),G.actual(0n,U64.zero())):
  G.actual_coverage(0n,4n,U64.zero(),{==},{==})
def first_blocker() ->
  L.subset(G.path(27n,0n,U64.bit(29n)),G.actual(27n,U64.bit(29n))):
  G.actual_coverage(27n,0n,U64.bit(29n),{==},{==})
