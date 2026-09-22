import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

law allocation_complete:
  for +d: Nat
  for +seed: U64
  {O.shape(Array.new(U64,d,seed)) == Spec.full(d) : O.Shape}

law table_pipeline_complete:
  for +d: Nat
  for +n: Nat
  for +extra: Nat
  for +key: U32
  for +at: U32
  for +sq: U32
  for +seed: U64
  {O.shape(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))) == Spec.full(d) : O.Shape}

# Complete shape, not reported capacity alone, discharges the old path premise.
law bounded_complete_routes:
  for a: Array<U64>
  for +i: U32
  for +j: U32
  for shape: {O.shape(a) == Spec.full(17n) : O.Shape}
  for ib: {U32.is_lt(i,131072) == True{} : Bool}
  for jb: {U32.is_lt(j,131072) == True{} : Bool}
  for ne: {U32.is_eq(i,j) == False{} : Bool}
  {S.separate(a,i,j) == True{} : Bool}

law bounded_other_index_write:
  for a: Array<U64>
  for +i: U32
  for +j: U32
  for +v: U64
  for shape: {O.shape(a) == Spec.full(17n) : O.Shape}
  for ib: {U32.is_lt(i,131072) == True{} : Bool}
  for jb: {U32.is_lt(j,131072) == True{} : Bool}
  for ne: {U32.is_eq(i,j) == False{} : Bool}
  {Array.get(U64,Array.set(U64,a,i,v),j) ==
    (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}
