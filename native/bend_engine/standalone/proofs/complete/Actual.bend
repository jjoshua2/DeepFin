import Base
import ./Spec.bend as Spec
import ./Core.bend as Core
import ./Arithmetic.bend as A
import ../../Tables.bend as Tables
import ../normalization/Actual.bend as Normalize
import ../separation/Spec.bend as S
import ../separation/Actual.bend as Frame
import ../storage/Observe.bend as O
import ../storage/Build.bend as Build
import ../storage/Representation.bend as R

def allocation(+d: Nat,+seed: U64) ->
  {O.shape(Array.new(U64,d,seed)) == Spec.full(d) : O.Shape}:
  match d:
    case 0n: {==}
    case 1n+p:
      Equal.cong(O.Shape,O.Shape,t => O.Node{t,t},O.shape(Array.new(U64,p,seed)),Spec.full(p),allocation(p,seed))

def pipeline(+d: Nat,+n: Nat,+extra: Nat,+key: U32,+at: U32,+sq: U32,+seed: U64) ->
  {O.shape(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))) == Spec.full(d) : O.Shape}:
  Equal.trans(O.Shape,O.shape(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))),
    O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),Spec.full(d),
    Build.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed))),
    Equal.trans(O.Shape,O.shape(Tables.tables(n,key,at,Array.new(U64,d,seed))),O.shape(Array.new(U64,d,seed)),Spec.full(d),
      Build.tables(n,key,at,Array.new(U64,d,seed)),allocation(d,seed)))

# Keep depth symbolic during the route proof; only closed scalar certificates
# are specialized at depth 17, avoiding expansion of a 131072-leaf proof term.
def raw(+d: Nat,+i: U32,+j: U32,
  +cap: {A.cap(d) == 131072 : U32},
  depth: {Nat.is_lt(d,18n) == True{} : Bool},
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) ->
  {S.apart(Spec.full(d),131072,i,j) == True{} : Bool}:
  %cap : {S.apart(Spec.full(d),_,i,j) == True{} : Bool}
  Core.routes(d,i,j,depth,
    Equal.trans(Bool,U32.is_lt(i,A.cap(d)),U32.is_lt(i,131072),True{},
      Equal.cong(U32,Bool,h => U32.is_lt(i,h),A.cap(d),131072,cap),ib),
    Equal.trans(Bool,U32.is_lt(j,A.cap(d)),U32.is_lt(j,131072),True{},
      Equal.cong(U32,Bool,h => U32.is_lt(j,h),A.cap(d),131072,cap),jb),
    eq => A.different(i,j,ne,eq))

def general(+d: Nat,a: Array<U64>,+i: U32,+j: U32,
  +shape: {O.shape(a) == Spec.full(d) : O.Shape},
  size: {S.size(Spec.full(d)) == 131072 : U32},
  cap: {A.cap(d) == 131072 : U32},
  depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +ib: {U32.is_lt(i,131072) == True{} : Bool},
  +jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) -> {S.separate(a,i,j) == True{} : Bool}:
  Equal.trans(Bool,S.separate(a,i,j),S.apart(O.shape(a),131072,i,j),True{},
    Normalize.paths(a,i,j,Equal.trans(U32,S.size(O.shape(a)),S.size(Spec.full(d)),131072,
      Equal.cong(O.Shape,U32,t => S.size(t),O.shape(a),Spec.full(d),shape),size),ib,jb),
    Equal.trans(Bool,S.apart(O.shape(a),131072,i,j),S.apart(Spec.full(d),131072,i,j),True{},
      Equal.cong(O.Shape,Bool,t => S.apart(t,131072,i,j),O.shape(a),Spec.full(d),shape),
      raw(d,i,j,cap,depth,ib,jb,ne)))

def paths(a: Array<U64>,+i: U32,+j: U32,
  shape: {O.shape(a) == Spec.full(17n) : O.Shape},
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) -> {S.separate(a,i,j) == True{} : Bool}:
  general(17n,a,i,j,shape,{==},{==},{==},ib,jb,ne)

def frame_lift(-a: Array<U64>,+i: U32,+j: U32,+v: U64,
  shape: {O.shape(a) == Spec.full(17n) : O.Shape},
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Array.set(U64,a,i,v),j) ==
    (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  Frame.lift(a,i,j,v,
    Equal.trans(Bool,S.separate(a,i,j),S.separate(O.pack(c),i,j),True{},
      Equal.cong(Array<U64>,Bool,t => S.separate(t,i,j),a,O.pack(c),Equal.sym(Array<U64>,O.pack(c),a,same)),
      paths(O.pack(c),i,j,
        Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),Spec.full(17n),
          Equal.cong(Array<U64>,O.Shape,t => O.shape(t),O.pack(c),a,same),shape),ib,jb,ne)),(c,same))

def frame(a: Array<U64>,+i: U32,+j: U32,+v: U64,
  shape: {O.shape(a) == Spec.full(17n) : O.Shape},
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,i,v),j) ==
    (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}:
  frame_lift(a,i,j,v,shape,ib,jb,ne,R.reify(a))
