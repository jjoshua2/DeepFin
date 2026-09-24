# Same two public-array writes used in actual Tables.tables. No lookup oracle.
import Base
import ./Facts.bend as F
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../storage/Roundtrip.bend as Round
import ../storage/Write.bend as Write
import ../separation/Spec.bend as S
import ../relative/Order.bend as Order

def first(a: Array<U64>,+k: Nat) -> Array<U64>:
  Array.set(U64,a,F.key(k),Layout.mask(F.key(k)))

def headers(a: Array<U64>,+k: Nat,+at: U32) -> Array<U64>:
  Array.set(U64,first(a,k),F.offset(k),U64.from_u32(at))

def first_shape(+c: O.Cells,+k: Nat,shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(first(O.pack(c),k)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(first(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),
    Write.model(c,F.key(k),Layout.mask(F.key(k))),shape)

def shape(+c: O.Cells,+k: Nat,+at: U32,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(headers(O.pack(c),k,at)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(headers(O.pack(c),k,at)),O.shape(first(O.pack(c),k)),C.full(17n),
    Write.write(first(O.pack(c),k),F.offset(k),U64.from_u32(at)),first_shape(c,k,sh))

def mask_value(+c: O.Cells,+k: Nat,+at: U32,
  sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},facts: F.bounds(k)) ->
  {S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))) == Layout.mask(F.key(k)) : U64}:
  (kb,(ob,ne)) = facts
  Equal.trans(U64,S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))),
    S.value(Array.get(U64,first(O.pack(c),k),F.key(k))),Layout.mask(F.key(k)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,headers(O.pack(c),k,at),F.key(k)),
      (headers(O.pack(c),k,at),S.value(Array.get(U64,first(O.pack(c),k),F.key(k)))),
      Complete.frame(first(O.pack(c),k),F.offset(k),F.key(k),U64.from_u32(at),first_shape(c,k,sh),
        Order.lt_lt(F.offset(k),256,131072,ob,{==}),Order.lt_lt(F.key(k),256,131072,kb,{==}),ne)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,first(O.pack(c),k),F.key(k)),
      (first(O.pack(c),k),Layout.mask(F.key(k))),Round.model(c,F.key(k),Layout.mask(F.key(k)))))

def prefix_value(+c: O.Cells,+k: Nat,+at: U32) ->
  {S.value(Array.get(U64,headers(O.pack(c),k,at),F.offset(k))) == U64.from_u32(at) : U64}:
  Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,headers(O.pack(c),k,at),F.offset(k)),
    (headers(O.pack(c),k,at),U64.from_u32(at)),Round.write_read(first(O.pack(c),k),F.offset(k),U64.from_u32(at)))
