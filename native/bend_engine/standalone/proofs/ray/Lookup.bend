# Compose independently specified attacks with the actual initialized lookup.
import Base
import ./Spec.bend as S
import ./Geometry.bend as Geometry
import ../lookup/MaskedLookup.bend as L
import ../data/Spec.bend as Data
import ../data/Pipeline.bend as DP
import ../../../legal_probe/Chess.bend as Chess

def actual(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,+occ: U64,
  +depth: {d == 17n : Nat},
  +budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),S.rays(Nat.add(before,k),occ)) : Array<U64> & U64}:
  +key = Nat.add(before,k)
  Equal.trans(Array<U64> & U64,
    Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(key),occ),
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),L.expected(key,occ)),
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),S.rays(key,occ)),
    L.actual(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget),
    Equal.cong(U64,Array<U64> & U64,v => (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),v),
      L.expected(key,occ),S.rays(key,occ),Geometry.masked_slider(key,occ,DP.selected_valid(before,after,k,budget))))
