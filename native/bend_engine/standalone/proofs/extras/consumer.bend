import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ./Certified.bend as Certified
import ../../Tables.bend as Tables
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

# The public contract is consumed with symbolic arrays, counts and addresses.
# Closed scalar witnesses below demonstrate both nonempty protected regions.
def bounded(a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},outside: {Spec.outside(q) == True{} : Bool}) ->
  {Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q) ==
    (Tables.extras(n,U32.from_nat(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  Laws.bounded_extras_read(a,n,k,q,shape,budget,qb,outside)

# The actual producer supplies shape; the theorem caller does not assume shape.
def produced(+d: Nat,+n: Nat,+key: U32,+at: U32,+seed: U64,+extra: Nat,+k: Nat,+q: U32,
  depth: {d == 17n : Nat},budget: {Nat.is_le(Nat.add(extra,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},outside: {Spec.outside(q) == True{} : Bool}) ->
  {Array.get(U64,Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),q) ==
    (Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),
      S.value(Array.get(U64,Tables.tables(n,key,at,Array.new(U64,d,seed)),q))) : Array<U64> & U64}:
  Laws.table_pipeline_extras_read(d,n,key,at,seed,extra,k,q,depth,budget,qb,outside)

def excluded_first_write() -> {Spec.outside(256) == False{} : Bool}: {==}
def excluded_last_write() -> {Spec.outside(511) == False{} : Bool}: {==}
def excluded_overrun() -> {Nat.is_le(Nat.add(65n,0n),64n) == False{} : Bool}: {==}
def excluded_wrapping_query() -> {U32.is_lt(131072,131072) == False{} : Bool}: {==}

def full_count() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
def positive_last_square() -> {Nat.is_le(Nat.add(1n,63n),64n) == True{} : Bool}: {==}
def depth_17() -> {17n == 17n : Nat}: {==}
def last_header() -> {Spec.outside(255) == True{} : Bool}: {==}
def first_slider() -> {Spec.outside(512) == True{} : Bool}: {==}
def allocation_last() -> {U32.is_lt(131071,131072) == True{} : Bool} & {Spec.outside(131071) == True{} : Bool}: ({==},{==})
