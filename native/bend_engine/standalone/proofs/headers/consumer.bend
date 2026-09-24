import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Facts.bend as F
import ./Addresses.bend as A
import ./OneBlock.bend as One
import ./Stored.bend as Stored
import ./Pipeline.bend as Pipeline
import ./Split.bend as Split
import ../layout/Spec.bend as Layout
import ../prefix/Spec.bend as P
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables

# Both alternatives are consumed; no assumed header value enters either proof.
def own(a: Array<U64>,+k: Nat,+which: Bool,valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape}) ->
  {Array.get(U64,Tables.tables(1n,F.key(k),P.prefix(k),a),A.query(k,which)) ==
    (Tables.tables(1n,F.key(k),P.prefix(k),a),One.expected(k,which)) : Array<U64> & U64}:
  Laws.own_block_header(a,k,which,valid,shape)

def through_blocks(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),a),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),a),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  Laws.stored_header_after_tables(a,before,after,k,which,shape,budget)

def final_mask(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,
  depth: {d == 17n : Nat},budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,Pipeline.run(Array.new(U64,d,seed),before,after,k,extra,start),F.key(Nat.add(before,k))) ==
    (Pipeline.run(Array.new(U64,d,seed),before,after,k,extra,start),Layout.mask(F.key(Nat.add(before,k)))) : Array<U64> & U64}:
  Laws.stored_header_after_extras(d,seed,before,after,k,False{},extra,start,depth,budget,ebudget)

def final_prefix(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,
  depth: {d == 17n : Nat},budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,Pipeline.run(Array.new(U64,d,seed),before,after,k,extra,start),F.offset(Nat.add(before,k))) ==
    (Pipeline.run(Array.new(U64,d,seed),before,after,k,extra,start),U64.from_u32(P.prefix(Nat.add(before,k)))) : Array<U64> & U64}:
  Laws.stored_header_after_extras(d,seed,before,after,k,True{},extra,start,depth,budget,ebudget)

# Concrete positive-count budgets include a full 128-block run and both ends.
def first_full_budget() -> {Nat.is_lt(Nat.add(Stored.count(0n,127n),0n),129n) == True{} : Bool}: {==}
def rook_boundary_budget() -> {Nat.is_lt(Nat.add(Stored.count(63n,64n),0n),129n) == True{} : Bool}: {==}
def bishop_boundary_budget() -> {Nat.is_lt(Nat.add(Stored.count(64n,63n),0n),129n) == True{} : Bool}: {==}
def last_full_budget() -> {Nat.is_lt(Nat.add(Stored.count(127n,0n),0n),129n) == True{} : Bool}: {==}
def full_extras_budget() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
def overrun_rejected() -> {Nat.is_lt(Nat.add(Stored.count(128n,0n),0n),129n) == False{} : Bool}: {==}

def schedule_partition(+p: Nat,+n: Nat,+k: Nat,a: Array<U64>) ->
  {P.scheduled(Nat.add(p,n),k,a) == P.scheduled(n,Nat.add(p,k),P.scheduled(p,k,a)) : Array<U64>}:
  Split.split(p,n,k,a)
