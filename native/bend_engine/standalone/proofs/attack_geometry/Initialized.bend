# Allocation and actual prior table loop supply shape, not assumed mask values.
import Base
import ./Geometry.bend as G
import ./Query.bend as Q
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as O
import ../storage/Build.bend as Build
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete

def run(+d: Nat,seed: U64,n: Nat,key: U32,at: U32,extra: Nat) -> Array<U64>:
  Tables.extras(extra,0,Tables.tables(n,key,at,Array.new(U64,d,seed)))

def query(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+kind: G.Leaper,+q: Nat,+occ: U64,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},qb: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Chess.attack(Q.tag(kind),run(d,seed,n,key,at,extra),U32.from_nat(q),Q.side(kind),occ) ==
    (run(d,seed,n,key,at,extra),G.mask(kind,q)) : Array<U64> & U64}:
  Q.actual(extra,Tables.tables(n,key,at,Array.new(U64,d,seed)),kind,q,occ,
    Equal.trans(O.Shape,O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),C.full(d),C.full(17n),
      Equal.trans(O.Shape,O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),O.shape(Array.new(U64,d,seed)),C.full(d),
        Build.tables(n,key,at,Array.new(U64,d,seed)),Complete.allocation(d,seed)),
      Equal.cong(Nat,O.Shape,x => C.full(x),d,17n,depth)),qb,full)
