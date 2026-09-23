# Apply the derived numeric certificate to the actual affine fill implementation.
import Base
import ./Spec.bend as Spec
import ./Clear.bend as Clear
import ../../Tables.bend as Tables
import ../complete/Spec.bend as Complete
import ../separation/Spec.bend as S
import ../separation/Fill.bend as Fill
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R

def lift(-a: Array<U64>,+n: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,+q: U32,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},
  room: {Spec.room(n,at,end) == True{} : Bool},
  outside: {Spec.outside(q,at,end) == True{} : Bool},
  cert: Exists(O.Cells,c=>{O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,_),q)) == S.value(Array.get(U64,a,q)) : U64}
  %same : {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),q)) == S.value(Array.get(U64,_,q)) : U64}
  Fill.frame(O.pack(c),n,at,sq,bishop,mask,subset,q,
    Clear.certificate(O.shape(O.pack(c)),n,at,end,q,
      Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),Complete.full(17n),
        Equal.cong(Array<U64>,O.Shape,b=>O.shape(b),O.pack(c),a,same),shape),eb,qb,room,outside))

def preserve(a: Array<U64>,+n: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,+q: U32,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},
  room: {Spec.room(n,at,end) == True{} : Bool},
  outside: {Spec.outside(q,at,end) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  lift(a,n,at,end,sq,bishop,mask,subset,q,shape,eb,qb,room,outside,R.reify(a))
