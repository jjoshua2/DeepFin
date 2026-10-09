import Base
import ./Bits.bend as Bits
import ../castle_safety/Spec.bend as S
import ../ray/Path.bend as P
import ../ray/Spec.bend as R
import ../ray_clear_disjoint/Algebra.bend as D
import ../occupancy_insertion/Lattice.bend as L
import ../ordinary_occupancy/Ordinary.bend as O
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def mask_exclusion(+own: U64,+rays: U64,+kings: U64,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(own,U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(U64.and(own,U64.or(rays,kings)),Chess.Ply{src,dst,promotion,flag}) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(src)),rays):
  Bits.bit_zero(rays,U32.to_nat(src),bound,
    Bits.test_exclusion(own,rays,kings,U32.to_nat(src),bound,owned,
      Bits.or_false_right(U32.is_eq(flag,1),U64.test_bit(U64.and(own,U64.or(rays,kings)),U32.to_nat(src)),bypass)))

def source_exclusion(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,promotion,flag}) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(src)),rays):
  mask_exclusion(Chess.color(b,Chess.get_turn(b)),rays,Chess.get_kings(b),src,dst,promotion,flag,bound,owned,bypass)

def path_disjoint(+b: Chess.Board,+rays: U64,+xs: List<&2,Nat>,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,promotion,flag}) == False{} : Bool},
  coverage: L.subset(P.attack(xs,Chess.occupied(b)),rays)) ->
  D.disjoint(U64.bit(U32.to_nat(src)),P.attack(xs,Chess.occupied(b))):
  D.disjoint_subset(U64.bit(U32.to_nat(src)),P.attack(xs,Chess.occupied(b)),rays,coverage,
    source_exclusion(b,rays,src,dst,promotion,flag,bound,owned,bypass))

def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+b: Chess.Board,+src: U32,+dst: U32,+rays: U64,+acc: U64,
  route: {R.trace(n,sq,dir,xs) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owned: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool},
  coverage: L.subset(P.attack(xs,Chess.occupied(b)),rays)) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(b,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(b),False{},acc)):
  O.actual_ray(n,sq,dir,xs,b,src,dst,acc,route,
    path_disjoint(b,rays,xs,src,dst,0,0,bound,owned,bypass,coverage))
