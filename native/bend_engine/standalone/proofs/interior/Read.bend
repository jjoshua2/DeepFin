# The actual affine get API does not change a bounded address at normalization.
import Base
import ./Mask.bend as Mask
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../storage/Read.bend as Earlier
import ../storage/Representation.bend as R

def model(+c: O.Cells, +x: U32,
  size: {O.count(Array.size(U64,O.pack(c))) == 131072 : U32},
  bounded: {U32.is_lt(x,131072) == True{} : Bool}) ->
  {Array.get(U64,O.pack(c),x) == Array.get.go(U64,O.pack(c),131072,x) : Array<U64> & U64}:
  +same = Equal.trans(U32,O.size(c),O.count(Array.size(U64,O.pack(c))),131072,
    Equal.sym(U32,O.count(Array.size(U64,O.pack(c))),O.size(c),
      Equal.cong(Array<U64> & U32,U32,r => O.count(r),Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),Earlier.size_pair(c))),size)
  %Equal.sym(Array<U64> & U32,Array.size(U64,O.pack(c)),(O.pack(c),O.size(c)),Earlier.size_pair(c)) :
    {Array.get.at(U64,x,_) == Array.get.go(U64,O.pack(c),131072,x) : Array<U64> & U64}
  %Equal.sym(U32,O.size(c),131072,same) :
    {Array.get.go(U64,O.pack(c),_,U32.and(x,U32.sub(_,1))) == Array.get.go(U64,O.pack(c),131072,x) : Array<U64> & U64}
  Equal.cong(U32,Array<U64> & U64,i => Array.get.go(U64,O.pack(c),131072,i),S.normalize(x),x,Mask.allocation(x,bounded))

def lift(-a: Array<U64>, +x: U32,
  size: {O.count(Array.size(U64,a)) == 131072 : U32}, bounded: {U32.is_lt(x,131072) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,a,x) == Array.get.go(U64,a,131072,x) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,_,x) == Array.get.go(U64,a,131072,x) : Array<U64> & U64}
  %same : {Array.get(U64,O.pack(c),x) == Array.get.go(U64,_,131072,x) : Array<U64> & U64}
  model(c,x,Equal.trans(U32,O.count(Array.size(U64,O.pack(c))),O.count(Array.size(U64,a)),131072,
    Equal.cong(Array<U64>,U32,b => O.count(Array.size(U64,b)),O.pack(c),a,same),size),bounded)

def actual(a: Array<U64>, +x: U32,
  size: {O.count(Array.size(U64,a)) == 131072 : U32}, bounded: {U32.is_lt(x,131072) == True{} : Bool}) ->
  {Array.get(U64,a,x) == Array.get.go(U64,a,131072,x) : Array<U64> & U64}:
  lift(a,x,size,bounded,R.reify(a))
