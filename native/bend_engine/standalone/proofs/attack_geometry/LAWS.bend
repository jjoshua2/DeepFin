# Independent non-slider geometry through the actual initialization and query path.
import Base
import ./Geometry.bend as G
import ./Slots.bend as S
import ./Query.bend as Q
import ./Initialized.bend as I
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as O
import ../complete/Spec.bend as C

law computed_leaper_matches_coordinates:
  for +kind: G.Leaper
  for +sq: Nat
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  {S.value(kind,sq) == G.mask(kind,sq) : U64}

law extras_store_coordinate_mask:
  for +extra: Nat
  for a: Array<U64>
  for +kind: G.Leaper
  for +sq: Nat
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  for full: {extra == 64n : Nat}
  {Array.get(U64,Tables.extras(extra,0,a),S.index(kind,sq)) ==
    (Tables.extras(extra,0,a),G.mask(kind,sq)) : Array<U64> & U64}

law initialized_leaper_query_matches_coordinates:
  for +d: Nat
  for +seed: U64
  for +n: Nat
  for +key: U32
  for +at: U32
  for +extra: Nat
  for +kind: G.Leaper
  for +sq: Nat
  for +occ: U64
  for depth: {d == 17n : Nat}
  for full: {extra == 64n : Nat}
  for bound: {Nat.is_lt(sq,64n) == True{} : Bool}
  {Chess.attack(Q.tag(kind),I.run(d,seed,n,key,at,extra),U32.from_nat(sq),Q.side(kind),occ) ==
    (I.run(d,seed,n,key,at,extra),G.mask(kind,sq)) : Array<U64> & U64}
