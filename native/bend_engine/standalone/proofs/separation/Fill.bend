# Other-slot preservation over the actual fill loop, including U32 address wrap.
import Base
import ./Spec.bend as S
import ./Actual.bend as A
import ../../Tables.bend as Tables
import ../../Subsets.bend as Subsets
import ../storage/Observe.bend as O
import ../storage/Model.bend as M
import ../storage/Roundtrip.bend as Round
import ../storage/Representation.bend as R
import ../storage/Write.bend as Write
import ../storage/Build.bend as Build

def and_left(+a: Bool, +b: Bool, e: {Bool.and(a,b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case False{}: e
    case True{}: {==}

def and_right(+a: Bool, +b: Bool, e: {Bool.and(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a b:
    case False{} False{}: e
    case False{} True{}: {==}
    case True{} False{}: e
    case True{} True{}: {==}

def image_shape(+c: O.Cells, +i: U32, +v: U64) ->
  {O.shape(O.pack(M.updated(c,i,v))) == O.shape(O.pack(c)) : O.Shape}:
  Equal.trans(O.Shape,O.shape(O.pack(M.updated(c,i,v))),O.shape(Array.set(U64,O.pack(c),i,v)),O.shape(O.pack(c)),
    Equal.cong(Array<U64>,O.Shape,a => O.shape(a),O.pack(M.updated(c,i,v)),Array.set(U64,O.pack(c),i,v),
      Equal.sym(Array<U64>,Array.set(U64,O.pack(c),i,v),O.pack(M.updated(c,i,v)),Round.set(c,i,v))),Write.write(O.pack(c),i,v))

def remaining(+c: O.Cells, +n: Nat, +at: U32, +j: U32, +v: U64,
  e: {S.clear(O.shape(O.pack(c)),1n+n,at,j) == True{} : Bool}) ->
  {S.clear(O.shape(O.pack(M.updated(c,at,v))),n,U32.inc(at),j) == True{} : Bool}:
  Equal.trans(Bool,S.clear(O.shape(O.pack(M.updated(c,at,v))),n,U32.inc(at),j),
    S.clear(O.shape(O.pack(c)),n,U32.inc(at),j),True{},
    Equal.cong(O.Shape,Bool,t => S.clear(t,n,U32.inc(at),j),
      O.shape(O.pack(M.updated(c,at,v))),O.shape(O.pack(c)),image_shape(c,at,v)),
    and_right(S.separate(O.pack(c),at,j),S.clear(O.shape(O.pack(c)),n,U32.inc(at),j),e))

def model(+n: Nat, +c: O.Cells, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +j: U32,
  +e: {S.clear(O.shape(O.pack(c)),n,at,j) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),j)) ==
    S.value(Array.get(U64,O.pack(c),j)) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      +v = Tables.slider(sq,bishop,subset,False{})
      +next = Subsets.next(subset,mask)
      +updated = M.updated(c,at,v)
      +stored = Round.set(c,at,v)
      %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),at,v),O.pack(updated),stored) :
        {S.value(Array.get(U64,Tables.fill(p,U32.inc(at),sq,bishop,mask,next,_),j)) == S.value(Array.get(U64,O.pack(c),j)) : U64}
      Equal.trans(U64,S.value(Array.get(U64,Tables.fill(p,U32.inc(at),sq,bishop,mask,next,O.pack(updated)),j)),
        S.value(Array.get(U64,O.pack(updated),j)),S.value(Array.get(U64,O.pack(c),j)),
        model(p,updated,U32.inc(at),sq,bishop,mask,next,j,remaining(c,p,at,j,v,e)),
        Equal.trans(U64,S.value(Array.get(U64,O.pack(updated),j)),
          S.value(Array.get(U64,Array.set(U64,O.pack(c),at,v),j)),S.value(Array.get(U64,O.pack(c),j)),
          Equal.cong(Array<U64>,U64,a => S.value(Array.get(U64,a,j)),O.pack(updated),Array.set(U64,O.pack(c),at,v),
            Equal.sym(Array<U64>,Array.set(U64,O.pack(c),at,v),O.pack(updated),stored)),
          Equal.cong(Array<U64> & U64,U64,r => S.value(r),
            Array.get(U64,Array.set(U64,O.pack(c),at,v),j),(Array.set(U64,O.pack(c),at,v),S.value(Array.get(U64,O.pack(c),j))),
            A.model(c,at,j,v,and_left(S.separate(O.pack(c),at,j),S.clear(O.shape(O.pack(c)),p,U32.inc(at),j),e)))))

def lift(-a: Array<U64>, +n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +j: U32,
  e: {S.clear(O.shape(a),n,at,j) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),j)) == S.value(Array.get(U64,a,j)) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,_),j)) == S.value(Array.get(U64,a,j)) : U64}
  %same : {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),j)) == S.value(Array.get(U64,_,j)) : U64}
  model(n,c,at,sq,bishop,mask,subset,j,Equal.trans(Bool,
    S.clear(O.shape(O.pack(c)),n,at,j),S.clear(O.shape(a),n,at,j),True{},
    Equal.cong(Array<U64>,Bool,b => S.clear(O.shape(b),n,at,j),O.pack(c),a,same),e))

def frame(a: Array<U64>, +n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +j: U32,
  e: {S.clear(O.shape(a),n,at,j) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),j)) == S.value(Array.get(U64,a,j)) : U64}:
  lift(a,n,at,sq,bishop,mask,subset,j,e,R.reify(a))

def stable(a: Array<U64>, +n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +i: U32, +j: U32) ->
  {S.separate(Tables.fill(n,at,sq,bishop,mask,subset,a),i,j) == S.separate(a,i,j) : Bool}:
  Equal.cong(O.Shape,Bool,t => S.disjoint(t,i,j),O.shape(Tables.fill(n,at,sq,bishop,mask,subset,a)),O.shape(a),
    Build.fill(n,at,sq,bishop,mask,subset,a))
