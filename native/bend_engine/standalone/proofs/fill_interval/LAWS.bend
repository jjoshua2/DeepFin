import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as Prefix
import ../storage/Observe.bend as O
import ../complete/Spec.bend as Complete
import ../separation/Spec.bend as S

law bounded_clear_before:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +end: U32
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for qb: {U32.is_lt(q,at) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}
  {S.clear(O.shape(a),n,at,q) == True{} : Bool}

law bounded_clear_after:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(q)) == True{} : Bool}
  {S.clear(O.shape(a),n,at,q) == True{} : Bool}

law bounded_fill_preserves_before:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +end: U32
  for +q: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for qb: {U32.is_lt(q,at) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}

law bounded_fill_preserves_after:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +q: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(q)) == True{} : Bool}
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}

law block_fill_preserves_before:
  for a: Array<U64>
  for +k: Nat
  for +q: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for qb: {U32.is_lt(q,Prefix.prefix(k)) == True{} : Bool}
  {S.value(Array.get(U64,Spec.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}

law block_fill_preserves_after:
  for a: Array<U64>
  for +k: Nat
  for +q: U32
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for start: {U32.is_le(Prefix.prefix(1n+k),q) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  {S.value(Array.get(U64,Spec.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}
