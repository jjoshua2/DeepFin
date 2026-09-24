# Supplementary source-to-source result. The mask and slider remain production definitions.
import Base
import ./Pipeline.bend as Lookup
import ./Canonical.bend as Canonical
import ../../../legal_probe/Chess.bend as Chess
import ../data/Spec.bend as Data
import ../data/Index.bend as Index
import ../data/Pipeline.bend as DP
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence
import ../../Tables.bend as Tables

def expected(+k: Nat,+occ: U64) -> U64:
  +key = U32.from_nat(k)
  Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.and(occ,Layout.mask(key)),False{})

def actual(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,+occ: U64,
  +depth: {d == 17n : Nat},
  +budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),expected(Nat.add(before,k),occ)) : Array<U64> & U64}:
  +selected = Nat.add(before,k)
  +key = U32.from_nat(selected)
  Equal.trans(Array<U64> & U64,
    Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),key,occ),
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Data.expected(selected,U32.to_nat(Lookup.index(selected,occ)))),
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),expected(selected,occ)),
    Lookup.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget),
    Equal.cong(U64,Array<U64> & U64,v =>
      (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Tables.slider(U32.mod(key,64),U32.is_ge(key,64),v,False{})),
      Sequence.at(U32.to_nat(Lookup.index(selected,occ)),Layout.mask(key)),U64.and(occ,Layout.mask(key)),
      Canonical.state(key,occ,Index.key_valid(selected,DP.selected_valid(before,after,k,budget)))))
