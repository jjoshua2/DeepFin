import Base
import ./Spec.bend as S
import ./Facts.bend as Facts
import ../storage/Observe.bend as O
import ../storage/Write.bend as Write
import ../separation/Spec.bend as Sep
import ../separation/Actual.bend as Actual
import ../fill_interval/Clear.bend as Clear
import ../complete/Spec.bend as C
import ../complete/Arithmetic.bend as CA
import ../relative/Order.bend as Order
import ../u64/Words.bend as Old

def choose(-T: Type,+a: Bool,+b: Bool,
  low: {a == True{} : Bool} -> T,
  high: {b == True{} : Bool} -> T,
  e: {Bool.or(a,b) == True{} : Bool}) -> T:
  match a b:
    case True{} True{}: low({==})
    case True{} False{}: low({==})
    case False{} True{}: high({==})
    case False{} False{}: Old.false_true(T,e)

def routes(+d: Nat,+i: U32,+q: U32,ib: Facts.bounds(i),+qb: {U32.is_lt(q,131072) == True{} : Bool},
  e: {S.outside(q) == True{} : Bool},
  +size: {Sep.size(C.full(d)) == 131072 : U32},+cap: {CA.cap(d) == 131072 : U32},
  +depth: {Nat.is_lt(d,18n) == True{} : Bool}) -> {Sep.disjoint(C.full(d),i,q) == True{} : Bool}:
  (+lo,+hi) = ib
  choose({Sep.disjoint(C.full(d),i,q) == True{} : Bool},U32.is_lt(q,256),U32.is_le(512,q),
    low => Clear.paths(d,i,q,size,cap,depth,Order.lt_lt(i,512,131072,hi,{==}),qb,
      eq => CA.different(q,i,Order.lt_ne(q,i,Order.lt_le(q,256,i,low,lo)),Equal.sym(U32,i,q,eq))),
    high => Clear.paths(d,i,q,size,cap,depth,Order.lt_lt(i,512,131072,hi,{==}),qb,
      eq => CA.different(i,q,Order.lt_ne(i,q,Order.lt_le(i,512,q,hi,high)),eq)),e)

def general(+d: Nat,+c: O.Cells,+i: U32,+q: U32,+v: U64,
  shape: {O.shape(O.pack(c)) == C.full(d) : O.Shape},
  ib: Facts.bounds(i),qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool},
  size: {Sep.size(C.full(d)) == 131072 : U32},cap: {CA.cap(d) == 131072 : U32},
  depth: {Nat.is_lt(d,18n) == True{} : Bool}) ->
  {Sep.value(Array.get(U64,Array.set(U64,O.pack(c),i,v),q)) == Sep.value(Array.get(U64,O.pack(c),q)) : U64}:
  Equal.cong(Array<U64> & U64,U64,r => Sep.value(r),Array.get(U64,Array.set(U64,O.pack(c),i,v),q),
    (Array.set(U64,O.pack(c),i,v),Sep.value(Array.get(U64,O.pack(c),q))),
    Actual.model(c,i,q,v,Equal.trans(Bool,Sep.separate(O.pack(c),i,q),Sep.disjoint(C.full(d),i,q),True{},
      Equal.cong(O.Shape,Bool,t => Sep.disjoint(t,i,q),O.shape(O.pack(c)),C.full(d),shape),routes(d,i,q,ib,qb,e,size,cap,depth))))


def write(+c: O.Cells,+i: U32,+q: U32,+v: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  ib: Facts.bounds(i),qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool}) ->
  {Sep.value(Array.get(U64,Array.set(U64,O.pack(c),i,v),q)) == Sep.value(Array.get(U64,O.pack(c),q)) : U64}:
  general(17n,c,i,q,v,shape,ib,qb,e,{==},{==},{==})
