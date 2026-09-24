import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./MaskedLookup.bend as Result
import ../../../legal_probe/Chess.bend as Chess
import ../data/Spec.bend as Data

# No assumed headers, data, shape or index values for the initialized pipeline.
def use(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+extra: Nat,+start: Nat,+occ: U64,
  depth: {d == 17n : Nat},
  budget: {Nat.is_lt(Nat.add(Data.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Chess.slide(Data.run(Array.new(U64,d,seed),before,after,k,extra,start),U32.from_nat(Nat.add(before,k)),occ) ==
    (Data.run(Array.new(U64,d,seed),before,after,k,extra,start),Result.expected(Nat.add(before,k),occ)) : Array<U64> & U64}:
  Laws.initialized_lookup(d,seed,before,after,k,extra,start,occ,depth,budget,ebudget)

# Full-pipeline witnesses without expanding a literal huge allocation.
def first_budget() -> {Nat.is_lt(Nat.add(Data.count(0n,127n),0n),129n) == True{} : Bool}: {==}
def last_budget() -> {Nat.is_lt(Nat.add(Data.count(127n,0n),0n),129n) == True{} : Bool}: {==}
def full_extras() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
def excluded_key() -> {Nat.is_lt(U32.to_nat(128),128n) == False{} : Bool}: {==}
