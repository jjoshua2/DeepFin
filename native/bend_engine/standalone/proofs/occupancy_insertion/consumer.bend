import Base
import ./Lattice.bend as L
import ./Insertion.bend as I
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../../Tables.bend as Tables

def path_insertion(+xs: List<&2,Nat>,+occ: U64,+added: U64) ->
  L.subset(P.attack(xs,U64.or(occ,added)),P.attack(xs,occ)):
  I.attack_subset(xs,occ,added)

def scan_insertion(+xs: List<&2,Nat>,+occ: U64,+added: U64,+acc: U64) ->
  L.subset(S.scan(xs,U64.or(occ,added),acc),S.scan(xs,occ,acc)):
  I.scan_subset(xs,occ,added,acc)

def actual_insertion(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+added: U64,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool}) ->
  L.subset(Tables.ray(n,False{},sq,dir,U64.or(occ,added),False{},acc),
    Tables.ray(n,False{},sq,dir,occ,False{},acc)):
  I.actual_ray(n,sq,dir,xs,occ,added,acc,route)

def path() -> List<&2,Nat>: 1n <> 2n <> 3n <> Nil{}

def old_attack() -> {P.attack(path(),U64.zero()) == U64.from_u32(14) : U64}: {==}
def inserted_attack() -> {P.attack(path(),U64.from_u32(4)) == U64.from_u32(6) : U64}: {==}

def actual_old() ->
  {Tables.ray(3n,False{},0,0,U64.zero(),False{},U64.zero()) == U64.from_u32(14) : U64}: {==}
def actual_inserted() ->
  {Tables.ray(3n,False{},0,0,U64.from_u32(4),False{},U64.zero()) == U64.from_u32(6) : U64}: {==}

def concrete_subset() ->
  L.subset(Tables.ray(3n,False{},0,0,U64.from_u32(4),False{},U64.zero()),
    Tables.ray(3n,False{},0,0,U64.zero(),False{},U64.zero())): {==}

# Mutating this insertion to deletion yields new attack 14 versus old 6.
def deletion_trap() ->
  L.subset(Tables.ray(3n,False{},0,0,U64.or(U64.from_u32(4),U64.from_u32(4)),False{},U64.zero()),
    Tables.ray(3n,False{},0,0,U64.from_u32(4),False{},U64.zero())): {==}

def concrete_shared_acc(+acc: U64) ->
  L.subset(Tables.ray(3n,False{},0,0,U64.or(U64.zero(),U64.from_u32(4)),False{},acc),
    Tables.ray(3n,False{},0,0,U64.zero(),False{},acc)):
  I.actual_ray(3n,0,0,path(),U64.zero(),U64.from_u32(4),acc,{==})

# Closed edge witnesses accompany the arbitrary-list theorem.
def duplicate_path() ->
  L.subset(P.attack(1n <> 2n <> 1n <> 3n <> Nil{},U64.from_u32(4)),
    P.attack(1n <> 2n <> 1n <> 3n <> Nil{},U64.zero())): {==}
def outside_path() ->
  L.subset(P.attack(65n <> 1n <> Nil{},U64.from_u32(2)),
    P.attack(65n <> 1n <> Nil{},U64.zero())): {==}

# Direct production witnesses for the excluded reverse/deletion claims.
def actual_deleted() ->
  {Tables.ray(3n,False{},0,0,U64.and(U64.from_u32(4),U64.not(U64.from_u32(4))),False{},U64.zero()) == U64.from_u32(14) : U64}: {==}

def reverse_subset_fails() ->
  {U64.is_eq(U64.and(
    Tables.ray(3n,False{},0,0,U64.from_u32(4),False{},U64.zero()),
    Tables.ray(3n,False{},0,0,U64.zero(),False{},U64.zero())),
    Tables.ray(3n,False{},0,0,U64.zero(),False{},U64.zero())) == False{} : Bool}: {==}

def deletion_subset_fails() ->
  {U64.is_eq(U64.and(
    Tables.ray(3n,False{},0,0,U64.from_u32(4),False{},U64.zero()),
    Tables.ray(3n,False{},0,0,U64.and(U64.from_u32(4),U64.not(U64.from_u32(4))),False{},U64.zero())),
    Tables.ray(3n,False{},0,0,U64.and(U64.from_u32(4),U64.not(U64.from_u32(4))),False{},U64.zero())) == False{} : Bool}: {==}

def closed_bit_preserved() ->
  {U64.test_bit(U64.or(U64.from_u32(4),U64.from_u32(12)),2n) == True{} : Bool}: {==}
