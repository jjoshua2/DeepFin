# Source-to-source lookup guarantees; independent geometry is separate.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../../../bitboard_probe/Sliders.bend as Sliders
import ../data/Spec.bend as Data
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence
import ./Pipeline.bend as Lookup
import ./MaskedLookup.bend as Masked

import ./LAWS.bend as Laws
import ./PROOF.bend as Proof

def use_certified_header_route(a: Array<U64>,
  +key: U32,
  +occ: U64,
  +mask: U64,
  +at: U32,
  mh: {Array.get(U64,a,key) == (a,mask) : Array<U64> & U64},
  oh: {Array.get(U64,a,U32.add(128,key)) == (a,U64.from_u32(at)) : Array<U64> & U64}) ->
  {Chess.slide(a,key,occ) == Array.get(U64,a,U32.add(at,Sliders.pext_index(occ,mask))) : Array<U64> & U64}:
  Laws.certified_header_route(a,key,occ,mask,at,mh,oh)

def use_selected_state_is_masked(+key: U32,
  +occ: U64,
  valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}:
  Laws.selected_state_is_masked(key,occ,valid)

def use_initialized_indexed_lookup(+d: Nat,
  +seed: U64,
  +before: Nat,
  +after: Nat,
  +k: Nat,
  +extra: Nat,
  +start: Nat,
  +occ: U64,
  depth: {d == 17n : Nat},
  budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Data.expected(Nat.add(before,k),U32.to_nat(Lookup.index(Nat.add(before,k),occ)))) : Array<U64> & U64}:
  Laws.initialized_indexed_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)

def use_initialized_masked_lookup(+d: Nat,
  +seed: U64,
  +before: Nat,
  +after: Nat,
  +k: Nat,
  +extra: Nat,
  +start: Nat,
  +occ: U64,
  depth: {d == 17n : Nat},
  budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Masked.expected(Nat.add(before,k),occ)) : Array<U64> & U64}:
  Laws.initialized_masked_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)

# Inhabited full-range and partial suffix budgets; these do not expand Array.new.
def full_first() -> {Nat.is_lt(Nat.add(Data.count(0n,127n),0n),129n) == True{} : Bool}: {==}
def full_last() -> {Nat.is_lt(Nat.add(Data.count(127n,0n),0n),129n) == True{} : Bool}: {==}
def suffix() -> {Nat.is_lt(Nat.add(Data.count(1n,62n),64n),129n) == True{} : Bool}: {==}
def extras() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
def oversized() -> {Nat.is_lt(Nat.add(Data.count(128n,0n),0n),129n) == False{} : Bool}: {==}
