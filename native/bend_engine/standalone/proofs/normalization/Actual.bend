# Real affine API observations. Capacity is observed from the actual constructor shape.
import Base
import ./Bound.bend as Bound
import ../separation/Spec.bend as S
import ../storage/Observe.bend as O
import ../storage/Read.bend as Read
import ../storage/Representation.bend as R

def image_size(+c: O.Cells) -> {S.size(O.shape(O.pack(c))) == O.size(c) : U32}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      Equal.cong(U32,U32,n => U32.shl(n),S.size(O.shape(O.pack(xs))),O.size(xs),image_size(xs))

def image_pair(+c: O.Cells) ->
  {Array.size(U64,O.pack(c)) == (O.pack(c),S.size(O.shape(O.pack(c)))) : Array<U64> & U32}:
  Equal.trans(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),
    (O.pack(c),S.size(O.shape(O.pack(c)))),Read.size_pair(c),
    Equal.cong(U32,Array<U64> & U32,n => (O.pack(c),n),O.size(c),S.size(O.shape(O.pack(c))),
      Equal.sym(U32,S.size(O.shape(O.pack(c))),O.size(c),image_size(c))))

def lift_size(-a: Array<U64>, cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.size(U64,a) == (a,S.size(O.shape(a))) : Array<U64> & U32}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.size(U64,_) == (a,S.size(O.shape(a))) : Array<U64> & U32}
  %same : {Array.size(U64,O.pack(c)) == (_,S.size(O.shape(a))) : Array<U64> & U32}
  %same : {Array.size(U64,O.pack(c)) == (O.pack(c),S.size(O.shape(_))) : Array<U64> & U32}
  image_pair(c)

def size(a: Array<U64>) -> {Array.size(U64,a) == (a,S.size(O.shape(a))) : Array<U64> & U32}:
  lift_size(a,R.reify(a))

def read(a: Array<U64>, +i: U32,
  capacity: {S.size(O.shape(a)) == 131072 : U32},
  bound: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {Array.get(U64,a,i) == Array.get.go(U64,a,131072,i) : Array<U64> & U64}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,a),(a,S.size(O.shape(a))),size(a)) :
    {Array.get.at(U64,i,_) == Array.get.go(U64,a,131072,i) : Array<U64> & U64}
  %Equal.sym(U32,S.size(O.shape(a)),131072,capacity) :
    {Array.get.go(U64,a,_,U32.and(i,U32.sub(_,1))) == Array.get.go(U64,a,131072,i) : Array<U64> & U64}
  Equal.cong(U32,Array<U64> & U64,j => Array.get.go(U64,a,131072,j),U32.and(i,U32.sub(131072,1)),i,Bound.table_address(i,bound))

def write(a: Array<U64>, +i: U32, +v: U64,
  capacity: {S.size(O.shape(a)) == 131072 : U32},
  bound: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {Array.set(U64,a,i,v) == Array.set.fin(U64,Array.swap.go(U64,a,131072,i,v)) : Array<U64>}:
  %Equal.sym(Array<U64> & U32,Array.size(U64,a),(a,S.size(O.shape(a))),size(a)) :
    {Array.set.fin(U64,Array.swap.at(U64,i,v,_)) == Array.set.fin(U64,Array.swap.go(U64,a,131072,i,v)) : Array<U64>}
  %Equal.sym(U32,S.size(O.shape(a)),131072,capacity) :
    {Array.set.fin(U64,Array.swap.go(U64,a,_,U32.and(i,U32.sub(_,1)),v)) == Array.set.fin(U64,Array.swap.go(U64,a,131072,i,v)) : Array<U64>}
  Equal.cong(U32,Array<U64>,j => Array.set.fin(U64,Array.swap.go(U64,a,131072,j,v)),U32.and(i,U32.sub(131072,1)),i,Bound.table_address(i,bound))

def paths(a: Array<U64>, +i: U32, +j: U32,
  capacity: {S.size(O.shape(a)) == 131072 : U32},
  ib: {U32.is_lt(i,131072) == True{} : Bool},
  jb: {U32.is_lt(j,131072) == True{} : Bool}) ->
  {S.separate(a,i,j) == S.apart(O.shape(a),131072,i,j) : Bool}:
  %Equal.sym(U32,S.size(O.shape(a)),131072,capacity) :
    {S.apart(O.shape(a),_,U32.and(i,U32.sub(_,1)),U32.and(j,U32.sub(_,1))) == S.apart(O.shape(a),131072,i,j) : Bool}
  %Equal.sym(U32,U32.and(i,U32.sub(131072,1)),i,Bound.table_address(i,ib)) :
    {S.apart(O.shape(a),131072,_,U32.and(j,U32.sub(131072,1))) == S.apart(O.shape(a),131072,i,j) : Bool}
  Equal.cong(U32,Bool,k => S.apart(O.shape(a),131072,i,k),U32.and(j,U32.sub(131072,1)),j,Bound.table_address(j,jb))
