# Transfer frame lemmas to real affine buffers through existing certified reification.
import Base
import ./Spec.bend as S
import ./Model.bend as Proof
import ../storage/Observe.bend as O
import ../storage/Model.bend as M
import ../storage/Refinement.bend as Ref
import ../storage/Roundtrip.bend as Round
import ../storage/Representation.bend as R
import ../storage/Write.bend as Write

def normalized(+c: O.Cells, +i: U32, +j: U32,
  e: {S.separate(O.pack(c),i,j) == True{} : Bool}) ->
  {S.apart(O.shape(O.pack(c)),O.size(c),M.normalized(c,i),M.normalized(c,j)) == True{} : Bool}:
  Equal.trans(Bool,
    S.apart(O.shape(O.pack(c)),O.size(c),M.normalized(c,i),M.normalized(c,j)),
    S.separate(O.pack(c),i,j),True{},
    Equal.cong(U32,Bool,n => S.apart(O.shape(O.pack(c)),n,U32.and(i,U32.sub(n,1)),U32.and(j,U32.sub(n,1))),
      O.size(c),S.size(O.shape(O.pack(c))),Equal.sym(U32,S.size(O.shape(O.pack(c))),O.size(c),Proof.size_image(c))),e)

def before(+c: O.Cells, +j: U32) ->
  {S.value(Array.get(U64,O.pack(c),j)) == M.read_sized(c,O.size(c),j) : U64}:
  Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,O.pack(c),j),
    (O.pack(c),M.read_sized(c,O.size(c),j)),Round.get(c,j))

def updated_value(+c: O.Cells, +i: U32, +j: U32, +v: U64,
  e: {S.separate(O.pack(c),i,j) == True{} : Bool}) ->
  {M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),j) == M.read_sized(c,O.size(c),j) : U64}:
  %Equal.sym(U32,O.size(M.updated(c,i,v)),O.size(c),Ref.size_update(c,O.size(c),M.normalized(c,i),v)) :
    {M.read_sized(M.updated(c,i,v),_,j) == M.read_sized(c,O.size(c),j) : U64}
  Proof.other(c,O.size(c),M.normalized(c,i),M.normalized(c,j),v,normalized(c,i,j,e))

def model(+c: O.Cells, +i: U32, +j: U32, +v: U64,
  e: {S.separate(O.pack(c),i,j) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,O.pack(c),i,v),j) ==
    (Array.set(U64,O.pack(c),i,v),S.value(Array.get(U64,O.pack(c),j))) : Array<U64> & U64}:
  +stored = Round.set(c,i,v)
  %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),i,v),O.pack(M.updated(c,i,v)),stored) :
    {Array.get(U64,_,j) == (Array.set(U64,O.pack(c),i,v),S.value(Array.get(U64,O.pack(c),j))) : Array<U64> & U64}
  %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),i,v),O.pack(M.updated(c,i,v)),stored) :
    {Array.get(U64,O.pack(M.updated(c,i,v)),j) == (_,S.value(Array.get(U64,O.pack(c),j))) : Array<U64> & U64}
  Equal.trans(Array<U64> & U64,Array.get(U64,O.pack(M.updated(c,i,v)),j),
    (O.pack(M.updated(c,i,v)),M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),j)),
    (O.pack(M.updated(c,i,v)),S.value(Array.get(U64,O.pack(c),j))),
    Round.get(M.updated(c,i,v),j),
    Equal.cong(U64,Array<U64> & U64,x => (O.pack(M.updated(c,i,v)),x),
      M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),j),S.value(Array.get(U64,O.pack(c),j)),
      Equal.trans(U64,M.read_sized(M.updated(c,i,v),O.size(M.updated(c,i,v)),j),
        M.read_sized(c,O.size(c),j),S.value(Array.get(U64,O.pack(c),j)),updated_value(c,i,j,v,e),
        Equal.sym(U64,S.value(Array.get(U64,O.pack(c),j)),M.read_sized(c,O.size(c),j),before(c,j)))))

def lift(-a: Array<U64>, +i: U32, +j: U32, +v: U64,
  e: {S.separate(a,i,j) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Array.set(U64,a,i,v),j) == (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Array.set(U64,_,i,v),j) == (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}
  %same : {Array.get(U64,Array.set(U64,O.pack(c),i,v),j) == (Array.set(U64,_,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}
  %same : {Array.get(U64,Array.set(U64,O.pack(c),i,v),j) == (Array.set(U64,O.pack(c),i,v),S.value(Array.get(U64,_,j))) : Array<U64> & U64}
  model(c,i,j,v,Equal.trans(Bool,S.separate(O.pack(c),i,j),S.separate(a,i,j),True{},
    Equal.cong(Array<U64>,Bool,t => S.separate(t,i,j),O.pack(c),a,same),e))

def frame(a: Array<U64>, +i: U32, +j: U32, +v: U64,
  e: {S.separate(a,i,j) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,i,v),j) == (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}:
  lift(a,i,j,v,e,R.reify(a))

def stable(a: Array<U64>, +i: U32, +j: U32, at: U32, v: U64) ->
  {S.separate(Array.set(U64,a,at,v),i,j) == S.separate(a,i,j) : Bool}:
  Equal.cong(O.Shape,Bool,t => S.disjoint(t,i,j),O.shape(Array.set(U64,a,at,v)),O.shape(a),Write.write(a,at,v))
