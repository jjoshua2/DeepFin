import Base
import ./Spec.bend as S
import ./Path.bend as Path
import ./Rays.bend as Rays
import ./Grid.bend as Grid
import ./Termination.bend as End
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess
import ../layout/Spec.bend as Layout
import ../data/Spec.bend as Data

law seven_steps_reach_edge:
  for +sq: Nat
  for +dir: Nat
  for sb: {Nat.is_lt(sq,64n) == True{} : Bool}
  for db: {Nat.is_lt(dir,8n) == True{} : Bool}
  {Grid.next(End.endpoint(Rays.path(7n,sq,dir),sq),dir) == 64n : Nat}

law ray_matches_independent_path:
  for +sq: Nat
  for +dir: Nat
  for +occ: U64
  for +acc: U64
  for sb: {Nat.is_lt(sq,64n) == True{} : Bool}
  for db: {Nat.is_lt(dir,8n) == True{} : Bool}
  {Tables.ray(7n,False{},U32.from_nat(sq),U32.from_nat(dir),occ,False{},acc) ==
    U64.or(acc,Path.attack(Rays.path(7n,sq,dir),occ)) : U64}

law slider_matches_independent_rays:
  for +key: Nat
  for +occ: U64
  for valid: {Nat.is_lt(key,128n) == True{} : Bool}
  {S.actual_slider(key,occ) == S.rays(key,occ) : U64}

law relevant_mask_preserves_attacks:
  for +key: Nat
  for +occ: U64
  for valid: {Nat.is_lt(key,128n) == True{} : Bool}
  {S.actual_slider(key,U64.and(occ,Layout.mask(U32.from_nat(key)))) == S.rays(key,occ) : U64}

law initialized_lookup_matches_independent_rays:
  for +d: Nat
  for +seed: U64
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +extra: Nat
  for +start: Nat
  for +occ: U64
  for depth: {d == 17n : Nat}
  for budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool}
  for ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),S.rays(Nat.add(before,k),occ)) : Array<U64> & U64}
