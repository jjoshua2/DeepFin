import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables

law write_preserves_other_location:
  for a: Array<U64>
  for +i: U32
  for +j: U32
  for +v: U64
  for e: {S.separate(a,i,j) == True{} : Bool}
  {Array.get(U64,Array.set(U64,a,i,v),j) ==
    (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}

law write_preserves_separation:
  for a: Array<U64>
  for +i: U32
  for +j: U32
  for at: U32
  for v: U64
  {S.separate(Array.set(U64,a,at,v),i,j) == S.separate(a,i,j) : Bool}

law fill_preserves_unwritten_location:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for +j: U32
  for e: {S.clear(O.shape(a),n,at,j) == True{} : Bool}
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),j)) == S.value(Array.get(U64,a,j)) : U64}

law fill_preserves_separation:
  for a: Array<U64>
  for +n: Nat
  for +at: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for +i: U32
  for +j: U32
  {S.separate(Tables.fill(n,at,sq,bishop,mask,subset,a),i,j) == S.separate(a,i,j) : Bool}
