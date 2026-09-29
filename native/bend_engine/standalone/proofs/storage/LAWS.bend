# Actual affine arrays occur in every contract. No bounds/geometry claim is
# implied by shape preservation or by same-location read-after-write.
import Base
import ../../Tables.bend as Tables
import ./Observe.bend as O

law read_preserves_storage:
  for a: Array<U64>
  for i: U32
  {O.storage(Array.get(U64,a,i)) == a : Array<U64>}

law write_read_same_location:
  for a: Array<U64>
  for +i: U32
  for +v: U64
  {Array.get(U64,Array.set(U64,a,i,v),i) == (Array.set(U64,a,i,v),v) : Array<U64> & U64}

law initialized_read:
  for +d: Nat
  for +v: U64
  for +i: U32
  {Array.get(U64,Array.new(U64,d,v),i) == (Array.new(U64,d,v),v) : Array<U64> & U64}

law write_preserves_shape:
  for a: Array<U64>
  for i: U32
  for v: U64
  {O.shape(Array.set(U64,a,i,v)) == O.shape(a) : O.Shape}

law fill_preserves_shape:
  for n: Nat
  for +at: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for a: Array<U64>
  {O.shape(Tables.fill(n,at,sq,bishop,mask,subset,a)) == O.shape(a) : O.Shape}

law tables_preserve_shape:
  for n: Nat
  for +key: U32
  for +at: U32
  for a: Array<U64>
  {O.shape(Tables.tables(n,key,at,a)) == O.shape(a) : O.Shape}

law extras_preserve_shape:
  for n: Nat
  for +sq: U32
  for a: Array<U64>
  {O.shape(Tables.extras(n,sq,a)) == O.shape(a) : O.Shape}

law fill_one_read:
  for a: Array<U64>
  for +at: U32
  for +sq: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  {Array.get(U64,Tables.fill(1n,at,sq,bishop,mask,subset,a),at) ==
    (Tables.fill(1n,at,sq,bishop,mask,subset,a),Tables.slider(sq,bishop,subset,False{})) : Array<U64> & U64}
