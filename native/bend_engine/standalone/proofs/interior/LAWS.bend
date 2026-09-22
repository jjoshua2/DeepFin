import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O

law allocation_mask_exact:
  for +x: U32
  for bounded: {U32.is_lt(x,131072) == True{} : Bool}
  {S.normalize(x) == x : U32}

law allocation_mask_injective:
  for +x: U32
  for +y: U32
  for bx: {U32.is_lt(x,131072) == True{} : Bool}
  for by: {U32.is_lt(y,131072) == True{} : Bool}
  for same: {S.normalize(x) == S.normalize(y) : U32}
  {x == y : U32}

law prefix_interior_certificate:
  for +k: Nat
  for +x: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for within: S.member(k,x)
  S.interior(x)

law ordered_normalized_addresses_distinct:
  for +i: Nat
  for +j: Nat
  for +x: U32
  for +y: U32
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  for ix: S.member(i,x)
  for jy: S.member(j,y)
  {U32.is_ne(S.normalize(x),S.normalize(y)) == True{} : Bool}

law interior_read_uses_unmasked_address:
  for a: Array<U64>
  for +k: Nat
  for +x: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for within: S.member(k,x)
  for size: {O.count(Array.size(U64,a)) == 131072 : U32}
  {Array.get(U64,a,x) == Array.get.go(U64,a,131072,x) : Array<U64> & U64}
