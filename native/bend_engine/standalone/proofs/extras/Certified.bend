import Base
import ./Actual.bend as Actual
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../storage/Build.bend as Build
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../separation/Spec.bend as S

# Symbolic depth avoids expanding a closed 131072-cell constructor tree.
# The explicit equality fixes the intended depth, not an assumed shape/read result.
def pipeline(+d: Nat,+n: Nat,+key: U32,+at: U32,+seed: U64,+extra: Nat,+k: Nat,+q: U32,
  depth: {d == 17n : Nat},
  budget: {Nat.is_le(Nat.add(extra,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},outside: {Spec.outside(q) == True{} : Bool}) ->
  {Array.get(U64,Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),q) ==
    (Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),
      S.value(Array.get(U64,Tables.tables(n,key,at,Array.new(U64,d,seed)),q))) : Array<U64> & U64}:
  Actual.pair(Tables.tables(n,key,at,Array.new(U64,d,seed)),extra,k,q,
    Equal.trans(O.Shape,O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),C.full(d),C.full(17n),
      Equal.trans(O.Shape,O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),O.shape(Array.new(U64,d,seed)),C.full(d),
        Build.tables(n,key,at,Array.new(U64,d,seed)),Complete.allocation(d,seed)),
      Equal.cong(Nat,O.Shape,x => C.full(x),d,17n,depth)),budget,qb,outside)
