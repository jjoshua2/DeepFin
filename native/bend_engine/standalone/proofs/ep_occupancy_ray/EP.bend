# Actual flag-one occupancy clears source and captured pawn, then inserts target.
import Base
import ../en_passant/Actual.bend as EP
import ../en_passant/Mask.bend as Mask
import ../move_update/Spec.bend as M
import ../move_update/Ordinary.bend as Move
import ../ordinary_occupancy/Ordinary.bend as O
import ../ordinary_occupancy/Algebra.bend as A
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../ray_clear_disjoint/Algebra.bend as D
import ../ray_clear_disjoint/Clear.bend as Clear
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as C

def removed(+s: U32,+d: U32) -> U64: EP.source(s,d)
def normalized(+b: C.Board,+s: U32,+d: U32) -> U64:
  U64.or(U64.and_not(C.occupied(b),removed(s,d)),U64.bit(U32.to_nat(d)))

def simplify(+occ: U64,+s: U32,+d: U32) ->
  {U64.or(U64.and_not(occ,EP.mask(s,d)),U64.bit(U32.to_nat(d))) == U64.or(U64.and_not(occ,removed(s,d)),U64.bit(U32.to_nat(d))) : U64}:
  Equal.trans(U64,U64.or(U64.and_not(occ,EP.mask(s,d)),U64.bit(U32.to_nat(d))),
    U64.or(U64.and_not(occ,M.removal(removed(s,d),U64.bit(U32.to_nat(d)))),U64.bit(U32.to_nat(d))),
    U64.or(U64.and_not(occ,removed(s,d)),U64.bit(U32.to_nat(d))),
    Equal.cong(U64,U64,m => U64.or(U64.and_not(occ,m),U64.bit(U32.to_nat(d))),EP.mask(s,d),
      M.removal(removed(s,d),U64.bit(U32.to_nat(d))),Mask.same(removed(s,d),U64.bit(U32.to_nat(d)))),
    A.reinsert(occ,removed(s,d),U64.bit(U32.to_nat(d))))

# This identity is total: arbitrary board rows, turn and source/destination U32s.
def normalization(+b: C.Board,+s: U32,+d: U32) ->
  {C.occupied(C.make_move(b,C.Ply{s,d,0,1})) == normalized(b,s,d) : U64}:
  Equal.trans(U64,C.occupied(C.make_move(b,C.Ply{s,d,0,1})),C.occupied(EP.dispatch(b,s,d,C.piece(b,s))),normalized(b,s,d),
    Equal.cong(C.Board,U64,board => C.occupied(board),C.make_move(b,C.Ply{s,d,0,1}),EP.dispatch(b,s,d,C.piece(b,s)),EP.unfold(b,s,d)),
    Equal.trans(U64,C.occupied(EP.dispatch(b,s,d,C.piece(b,s))),
      U64.or(U64.and_not(C.occupied(b),EP.mask(s,d)),U64.bit(U32.to_nat(d))),normalized(b,s,d),
      O.raw_occupied(b,EP.mask(s,d),U64.bit(U32.to_nat(d)),C.piece(b,s),U32.is_eq(C.get_turn(b),1),U32.xor(C.get_turn(b),1),
        Move.rights(b,s,d,C.piece(b,s)),Move.ep(s,d,C.piece(b,s))),simplify(C.occupied(b),s,d)))

# Both removals must be outside the OLD first-blocker-inclusive observation.
def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: C.Board,+s: U32,+d: U32,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},
  empty: D.disjoint(removed(s,d),P.attack(xs,C.occupied(b)))) ->
  L.subset(Tables.ray(n,False{},sq,dir,C.occupied(C.make_move(b,C.Ply{s,d,0,1})),False{},acc),
    Tables.ray(n,False{},sq,dir,C.occupied(b),False{},acc)):
  L.rewrite(Tables.ray(n,False{},sq,dir,normalized(b,s,d),False{},acc),Tables.ray(n,False{},sq,dir,C.occupied(b),False{},acc),
    Tables.ray(n,False{},sq,dir,C.occupied(C.make_move(b,C.Ply{s,d,0,1})),False{},acc),Tables.ray(n,False{},sq,dir,C.occupied(b),False{},acc),
    Equal.cong(U64,U64,occ => Tables.ray(n,False{},sq,dir,occ,False{},acc),C.occupied(C.make_move(b,C.Ply{s,d,0,1})),normalized(b,s,d),normalization(b,s,d)),{==},
    Clear.clear_then_insert(n,sq,dir,xs,C.occupied(b),removed(s,d),U64.bit(U32.to_nat(d)),acc,route,empty))
