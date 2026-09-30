# Specialize selected-block correctness to one complete table, uniformly in its key.
import Base
import ../layout/Domain.bend as Domain
import ./Spec.bend as B
import ./Domain.bend as D
import ../data/Spec.bend as Data
import ../prefix/Nats.bend as N
import ../ray/Spec.bend as S
import ../ray/LAWS.bend as Ray
import ../ray/PROOF.bend as RayProof
import ../../../legal_probe/Chess.bend as Chess

import ./Binding.bend as Bind

def actual(+d: Nat,+seed: U64,+n: Nat,+key: Nat,+extra: Nat,+start: Nat,+occ: U64,
  +valid: {Nat.is_lt(key,128n) == True{} : Bool},depth: {d == 17n : Nat},
  +full: {n == 128n : Nat},eb: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(B.run(d,seed,n,extra,start),U32.from_nat(key),occ) ==
    (B.run(d,seed,n,extra,start),S.rays(key,occ)) : Array<U64> & U64}:
  %Bind.image(d,seed,n,key,extra,start,valid,full) :
    {Chess.slide(_,U32.from_nat(key),occ) == (B.run(d,seed,n,extra,start),S.rays(key,occ)) : Array<U64> & U64}
  %Bind.image(d,seed,n,key,extra,start,valid,full) :
    {Chess.slide(Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start),U32.from_nat(key),occ) ==
      (_,S.rays(key,occ)) : Array<U64> & U64}
  %N.right_zero(key) :
    {Chess.slide(Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start),U32.from_nat(_),occ) ==
      (Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start),S.rays(key,occ)) : Array<U64> & U64}
  %N.right_zero(key) :
    {Chess.slide(Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start),U32.from_nat(Nat.add(key,0n)),occ) ==
      (Data.run(Array.new(U64,d,seed),key,D.suffix(key),0n,extra,start),S.rays(_,occ)) : Array<U64> & U64}
  Ray.initialized_lookup_matches_independent_rays(d,seed,key,D.suffix(key),0n,extra,start,occ,
    depth,Bind.budget(key,D.full_configuration(key,valid)),eb)

# Direct engine key, not an additional caller-supplied conversion certificate.

def u32(+d: Nat,+seed: U64,+n: Nat,+key: U32,+extra: Nat,+start: Nat,+occ: U64,
  +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},depth: {d == 17n : Nat},
  full: {n == 128n : Nat},eb: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(B.run(d,seed,n,extra,start),key,occ) ==
    (B.run(d,seed,n,extra,start),S.rays(U32.to_nat(key),occ)) : Array<U64> & U64}:
  %Domain.injective(U32.from_nat(U32.to_nat(key)),key,Domain.exact(U32.to_nat(key),valid)) :
    {Chess.slide(B.run(d,seed,n,extra,start),_,occ) ==
      (B.run(d,seed,n,extra,start),S.rays(U32.to_nat(key),occ)) : Array<U64> & U64}
  actual(d,seed,n,U32.to_nat(key),extra,start,occ,valid,depth,full,eb)
