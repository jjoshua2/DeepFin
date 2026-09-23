import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../relative/Spec.bend as Relative
import ../complete/Spec.bend as Complete
import ../separation/Spec.bend as S
import ../storage/Observe.bend as O

law interval_clear:
  for +t: O.Shape
  for +n: Nat
  for +at: U32
  for +end: U32
  for +q: U32
  for shape: {t == Complete.full(17n) : O.Shape}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for room: {Spec.room(n,at,end) == True{} : Bool}
  for outside: {Spec.outside(q,at,end) == True{} : Bool}
  {S.clear(t,n,at,q) == True{} : Bool}

law interval_fill_preserves_query:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +end: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for eb: {U32.is_lt(end,131072) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for room: {Spec.room(n,at,end) == True{} : Bool}
  for outside: {Spec.outside(q,at,end) == True{} : Bool}
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}

law chess_fill_clear:
  for +t: O.Shape
  for +k: Nat
  for +q: U32
  for shape: {t == Complete.full(17n) : O.Shape}
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for outside: {Spec.outside(q,Prefix.prefix(k),Prefix.prefix(1n+k)) == True{} : Bool}
  {S.clear(t,U32.to_nat(Layout.size(U32.from_nat(k))),Prefix.prefix(k),q) == True{} : Bool}

law chess_fill_preserves_query:
  for a: Array<U64>
  for +k: Nat
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for outside: {Spec.outside(q,Prefix.prefix(k),Prefix.prefix(1n+k)) == True{} : Bool}
  {S.value(Array.get(U64,Spec.block(a,k),q)) == S.value(Array.get(U64,a,q)) : U64}

law fill_preserves_later_lookup:
  for a: Array<U64>
  for +i: Nat
  for +j: Nat
  for +occ: U64
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for +before: {Nat.is_lt(i,j) == True{} : Bool}
  for +valid: {Nat.is_lt(j,128n) == True{} : Bool}
  {S.value(Array.get(U64,Spec.block(a,i),Relative.address(j,Relative.index(j,occ)))) ==
    S.value(Array.get(U64,a,Relative.address(j,Relative.index(j,occ)))) : U64}

law fill_preserves_reserved_query:
  for a: Array<U64>
  for +k: Nat
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for +valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for +reserved: {U32.is_lt(q,512) == True{} : Bool}
  {S.value(Array.get(U64,Spec.block(a,k),q)) == S.value(Array.get(U64,a,q)) : U64}
