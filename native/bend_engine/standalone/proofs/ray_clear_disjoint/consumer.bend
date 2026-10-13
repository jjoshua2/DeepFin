import Base
import ./Algebra.bend as D
import ./Clear.bend as C
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../blocker_frame/Blocker.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables

def actual_clear(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+removed: U64,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(removed,P.attack(xs,occ))) ->
  {Tables.ray(n,False{},sq,dir,C.clear(occ,removed),False{},acc) ==
    Tables.ray(n,False{},sq,dir,occ,False{},acc) : U64}:
  C.actual_ray(n,sq,dir,xs,occ,removed,acc,route,empty)

def composed(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+removed: U64,+added: U64,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},empty: D.disjoint(removed,P.attack(xs,occ))) ->
  L.subset(Tables.ray(n,False{},sq,dir,U64.or(C.clear(occ,removed),added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc)):
  C.clear_then_insert(n,sq,dir,xs,occ,removed,added,acc,route,empty)

def path() -> List<&2,Nat>: 1n <> 2n <> 3n <> 4n <> Nil{}

def beyond_disjoint() -> D.disjoint(U64.from_u32(8),P.attack(path(),U64.from_u32(12))): {==}
def beyond_agreement() ->
  {B.agree(path(),U64.from_u32(12),C.clear(U64.from_u32(12),U64.from_u32(8))) == True{} : Bool}:
  C.agreement(path(),U64.from_u32(12),U64.from_u32(8),beyond_disjoint())
def old_interior_fails() ->
  {P.agree(path(),U64.from_u32(12),C.clear(U64.from_u32(12),U64.from_u32(8))) == False{} : Bool}: {==}

def actual_old() ->
  {Tables.ray(4n,False{},0,0,U64.from_u32(12),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}
def beyond_preserved() ->
  {Tables.ray(4n,False{},0,0,C.clear(U64.from_u32(12),U64.from_u32(8)),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}
def first_blocker_deleted() ->
  {Tables.ray(4n,False{},0,0,C.clear(U64.from_u32(12),U64.from_u32(4)),False{},U64.zero()) == U64.from_u32(14) : U64}: {==}
def first_blocker_not_disjoint() ->
  {U64.is_eq(U64.and(U64.from_u32(4),P.attack(path(),U64.from_u32(12))),U64.zero()) == False{} : Bool}: {==}

def bounded_source(+acc: U64) ->
  {Tables.ray(4n,False{},0,0,C.clear(U64.from_u32(12),U64.bit(3n)),False{},acc) ==
    Tables.ray(4n,False{},0,0,U64.from_u32(12),False{},acc) : U64}:
  C.actual_source(4n,0,0,path(),U64.from_u32(12),3,acc,{==},{==},beyond_disjoint())

def compose_closed(+acc: U64) ->
  L.subset(Tables.ray(4n,False{},0,0,U64.or(C.clear(U64.from_u32(12),U64.from_u32(8)),U64.from_u32(2)),False{},acc),
    Tables.ray(4n,False{},0,0,U64.from_u32(12),False{},acc)):
  C.clear_then_insert(4n,0,0,path(),U64.from_u32(12),U64.from_u32(8),U64.from_u32(2),acc,{==},beyond_disjoint())

# Equal attacks cannot manufacture occupancy agreement, even before the last square.
def duplicate_attack_equal() ->
  {P.attack(1n <> 1n <> Nil{},U64.from_u32(2)) == P.attack(1n <> 1n <> Nil{},U64.zero()) : U64}: {==}
def duplicate_agreement_false() ->
  {B.agree(1n <> 1n <> Nil{},U64.from_u32(2),U64.zero()) == False{} : Bool}: {==}
def duplicate_old_agreement_false() ->
  {P.agree(1n <> 1n <> Nil{},U64.from_u32(2),U64.zero()) == False{} : Bool}: {==}
def duplicate_valid_clear() ->
  {P.attack(1n <> 1n <> Nil{},C.clear(U64.from_u32(2),U64.from_u32(4))) ==
    P.attack(1n <> 1n <> Nil{},U64.from_u32(2)) : U64}:
  C.attack(1n <> 1n <> Nil{},U64.from_u32(2),U64.from_u32(4),{==})
