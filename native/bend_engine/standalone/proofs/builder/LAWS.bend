# Uniform full-table API. Public build() expression linkage is a separate source guard.
import Base
import ./Spec.bend as B
import ../ray/Spec.bend as S
import ../../../legal_probe/Chess.bend as Chess

law full_table_lookup:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +key: Nat
  for +extra: Nat
  for +start: Nat
  for +occ: U64
  for valid: {Nat.is_lt(key,128n) == True{} : Bool}
  for depth: {d == 17n : Nat}
  for full: {n == 128n : Nat}
  for eb: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Chess.slide(B.run(d,seed,n,extra,start),U32.from_nat(key),occ) ==
    (B.run(d,seed,n,extra,start),S.rays(key,occ)) : Array<U64> & U64}

law u32_full_table_lookup:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +key: U32
  for +extra: Nat
  for +start: Nat
  for +occ: U64
  for valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for depth: {d == 17n : Nat}
  for full: {n == 128n : Nat}
  for eb: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Chess.slide(B.run(d,seed,n,extra,start),key,occ) ==
    (B.run(d,seed,n,extra,start),S.rays(U32.to_nat(key),occ)) : Array<U64> & U64}
