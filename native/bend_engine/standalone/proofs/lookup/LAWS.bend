# Public actual-lookup contracts promoted from the checked archive.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../../bitboard_probe/Sliders.bend as Sliders
import ../data/Spec.bend as Data
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence
import ./MaskedLookup.bend as Result

# Conditional routing helper; initialized_lookup derives these certificates.
law lookup_from_headers:
  for a: Array<U64>
  for +key: U32
  for +occ: U64
  for +mask: U64
  for +at: U32
  for mh: {Array.get(U64,a,key) == (a,mask) : Array<U64> & U64}
  for oh: {Array.get(U64,a,U32.add(128,key)) == (a,U64.from_u32(at)) : Array<U64> & U64}
  {Chess.slide(a,key,occ) == Array.get(U64,a,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}

law selected_state_is_masked:
  for +key: U32
  for +occ: U64
  for valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}

law initialized_lookup:
  for +d: Nat
  for +seed: U64
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +extra: Nat
  for +start: Nat
  for +occ: U64
  for +depth: {d == 17n : Nat}
  for +budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool}
  for ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Result.expected(Nat.add(before,k),occ)) : Array<U64> & U64}
