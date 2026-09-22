# Storage-shape induction over the actual table builders. Index safety and
# geometric correctness are separate obligations; shape does not imply either.
import Base
import ../../Tables.bend as Tables
import ../../Subsets.bend as Subsets
import ../layout/Spec.bend as Layout
import ./Observe.bend as O
import ./Representation.bend as R
import ./Write.bend as Write

def fill(n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, a: Array<U64>) ->
  {O.shape(Tables.fill(n,at,sq,bishop,mask,subset,a)) == O.shape(a) : O.Shape}:
  match n:
    case 0n: {==}
    case 1n+p:
      R.shape_transfer(a,x => Tables.fill(1n+p,at,sq,bishop,mask,subset,x),
        +c => Equal.trans(O.Shape,
          O.shape(Tables.fill(p,U32.inc(at),sq,bishop,mask,Subsets.next(subset,mask),Array.set(U64,O.pack(c),at,Tables.slider(sq,bishop,subset,False{})))),
          O.shape(Array.set(U64,O.pack(c),at,Tables.slider(sq,bishop,subset,False{}))),O.shape(O.pack(c)),
          fill(p,U32.inc(at),sq,bishop,mask,Subsets.next(subset,mask),Array.set(U64,O.pack(c),at,Tables.slider(sq,bishop,subset,False{}))),
          Write.model(c,at,Tables.slider(sq,bishop,subset,False{}))),R.reify(a))

# Abbreviations of the actual calls, not a replacement memory model.

def headers(+c: O.Cells, +key: U32, at: U32) -> Array<U64>:
  Array.set(U64,Array.set(U64,O.pack(c),key,Layout.mask(key)),U32.add(128,key),U64.from_u32(at))

def header_shape(+c: O.Cells, +key: U32, +at: U32) ->
  {O.shape(headers(c,key,at)) == O.shape(O.pack(c)) : O.Shape}:
  Equal.trans(O.Shape,O.shape(headers(c,key,at)),O.shape(Array.set(U64,O.pack(c),key,Layout.mask(key))),O.shape(O.pack(c)),
    Write.write(Array.set(U64,O.pack(c),key,Layout.mask(key)),U32.add(128,key),U64.from_u32(at)),
    Write.model(c,key,Layout.mask(key)))

def block(+c: O.Cells, +key: U32, +at: U32) -> Array<U64>:
  Tables.fill(U32.to_nat(Layout.size(key)),at,U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),U64.zero(),headers(c,key,at))

def block_shape(+c: O.Cells, +key: U32, +at: U32) ->
  {O.shape(block(c,key,at)) == O.shape(O.pack(c)) : O.Shape}:
  Equal.trans(O.Shape,O.shape(block(c,key,at)),O.shape(headers(c,key,at)),O.shape(O.pack(c)),
    fill(U32.to_nat(Layout.size(key)),at,U32.mod(key,64),U32.is_ge(key,64),Layout.mask(key),U64.zero(),headers(c,key,at)),
    header_shape(c,key,at))

def tables(n: Nat, +key: U32, +at: U32, a: Array<U64>) ->
  {O.shape(Tables.tables(n,key,at,a)) == O.shape(a) : O.Shape}:
  match n:
    case 0n: {==}
    case 1n+p:
      R.shape_transfer(a,x => Tables.tables(1n+p,key,at,x),
        +c => Equal.trans(O.Shape,
          O.shape(Tables.tables(p,U32.inc(key),U32.add(at,Layout.size(key)),block(c,key,at))),
          O.shape(block(c,key,at)),O.shape(O.pack(c)),
          tables(p,U32.inc(key),U32.add(at,Layout.size(key)),block(c,key,at)),block_shape(c,key,at)),R.reify(a))

def knight(+c: O.Cells, +sq: U32) -> Array<U64>:
  Array.set(U64,O.pack(c),U32.add(256,sq),Tables.leaps(8n,sq,0,False{},U64.zero()))

def king(+c: O.Cells, +sq: U32) -> Array<U64>:
  Array.set(U64,knight(c,sq),U32.add(320,sq),Tables.leaps(8n,sq,0,True{},U64.zero()))

def white(+c: O.Cells, +sq: U32) -> Array<U64>:
  Array.set(U64,king(c,sq),U32.add(384,sq),Tables.pawns(sq,1))

def black(+c: O.Cells, +sq: U32) -> Array<U64>:
  Array.set(U64,white(c,sq),U32.add(448,sq),Tables.pawns(sq,4294967295))

def extra_shape(+c: O.Cells, +sq: U32) ->
  {O.shape(black(c,sq)) == O.shape(O.pack(c)) : O.Shape}:
  Equal.trans(O.Shape,O.shape(black(c,sq)),O.shape(white(c,sq)),O.shape(O.pack(c)),
    Write.write(white(c,sq),U32.add(448,sq),Tables.pawns(sq,4294967295)),
    Equal.trans(O.Shape,O.shape(white(c,sq)),O.shape(king(c,sq)),O.shape(O.pack(c)),
      Write.write(king(c,sq),U32.add(384,sq),Tables.pawns(sq,1)),
      Equal.trans(O.Shape,O.shape(king(c,sq)),O.shape(knight(c,sq)),O.shape(O.pack(c)),
        Write.write(knight(c,sq),U32.add(320,sq),Tables.leaps(8n,sq,0,True{},U64.zero())),
        Write.model(c,U32.add(256,sq),Tables.leaps(8n,sq,0,False{},U64.zero())))))

def extras(n: Nat, +sq: U32, a: Array<U64>) ->
  {O.shape(Tables.extras(n,sq,a)) == O.shape(a) : O.Shape}:
  match n:
    case 0n: {==}
    case 1n+p:
      R.shape_transfer(a,x => Tables.extras(1n+p,sq,x),
        +c => Equal.trans(O.Shape,O.shape(Tables.extras(p,U32.inc(sq),black(c,sq))),O.shape(black(c,sq)),O.shape(O.pack(c)),
          extras(p,U32.inc(sq),black(c,sq)),extra_shape(c,sq)),R.reify(a))
