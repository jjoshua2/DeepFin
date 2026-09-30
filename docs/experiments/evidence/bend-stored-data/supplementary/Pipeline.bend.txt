# Supplementary actual Chess.slide composition; independent ray geometry remains open.
import Base
import ./Route.bend as Route
import ../../../legal_probe/Chess.bend as Chess
import ../../../bitboard_probe/Sliders.bend as Sliders
import ../data/Spec.bend as Data
import ../data/Index.bend as Index
import ../data/Pipeline.bend as DP
import ../data/LAWS.bend as DL
import ../data/PROOF.bend as DProof
import ../headers/LAWS.bend as HL
import ../headers/PROOF.bend as HProof
import ../layout/Spec.bend as Layout
import ../layout/LAWS.bend as LL
import ../layout/PROOF.bend as LProof
import ../layout/Domain.bend as Domain
import ../prefix/Spec.bend as P
import ../prefix/Nats.bend as N

def index(+k: Nat,occ: U64) -> U32:
  Sliders.pext_index(occ,Layout.mask(U32.from_nat(k)))

def index_bound(+k: Nat,+occ: U64,valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index(k,occ)),U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}:
  LL.lookup_index_bound(U32.from_nat(k),occ,Index.key_valid(k,valid))

def index_roundtrip(+k: Nat,+occ: U64,+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.from_nat(U32.to_nat(index(k,occ))) == index(k,occ) : U32}:
  Domain.injective(U32.from_nat(U32.to_nat(index(k,occ))),index(k,occ),
    Index.exact(U32.to_nat(index(k,occ)),Layout.size(U32.from_nat(k)),
      N.lt_to_le(U32.to_nat(index(k,occ)),U32.to_nat(Layout.size(U32.from_nat(k))),index_bound(k,occ,valid)),
      Index.size_bound(k,valid)))

def actual(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,+occ: U64,
  +depth: {d == 17n : Nat},
  +budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  +ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),
      Data.expected(Nat.add(before,k),U32.to_nat(index(Nat.add(before,k),occ)))) : Array<U64> & U64}:
  +selected = Nat.add(before,k)
  +valid = DP.selected_valid(before,after,k,budget)
  Equal.trans(Array<U64> & U64,
    Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(selected),occ),
    Array.get(U64,Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.add(P.prefix(selected),index(selected,occ))),
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Data.expected(selected,U32.to_nat(index(selected,occ)))),
    Route.actual(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(selected),occ,Layout.mask(U32.from_nat(selected)),P.prefix(selected),
      HL.stored_header_after_extras(d,seed,before,after,k,False{},extra,start,depth,budget,ebudget),
      HL.stored_header_after_extras(d,seed,before,after,k,True{},extra,start,depth,budget,ebudget)),
    %index_roundtrip(selected,occ,valid) :
      {Array.get(U64,Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.add(P.prefix(selected),_)) ==
        (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Data.expected(selected,U32.to_nat(index(selected,occ)))) : Array<U64> & U64}
    DL.stored_data_after_extras(d,seed,before,after,k,U32.to_nat(index(selected,occ)),extra,start,depth,budget,index_bound(selected,occ,valid),ebudget))
