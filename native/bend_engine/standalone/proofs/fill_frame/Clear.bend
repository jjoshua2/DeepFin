# Derive the existing clear-write certificate from a numeric half-open interval.
import Base
import ./Spec.bend as Spec
import ./Nats.bend as N
import ../complete/Spec.bend as Complete
import ../complete/Arithmetic.bend as A
import ../complete/Actual.bend as C
import ../normalization/Bound.bend as Bound
import ../relative/Order.bend as Order
import ../separation/Spec.bend as S
import ../storage/Observe.bend as O
import ../u64/Words.bend as Old

def path(+d: Nat,+t: O.Shape,+i: U32,+j: U32,
  shape: {t == Complete.full(d) : O.Shape},
  size: {S.size(Complete.full(d)) == 131072 : U32},
  +cap: {A.cap(d) == 131072 : U32},
  depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +ib: {U32.is_lt(i,131072) == True{} : Bool},
  +jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) -> {S.disjoint(t,i,j) == True{} : Bool}:
  %Equal.sym(O.Shape,t,Complete.full(d),shape) : {S.disjoint(_,i,j) == True{} : Bool}
  %Equal.sym(U32,S.size(Complete.full(d)),131072,size) :
    {S.apart(Complete.full(d),_,U32.and(i,U32.sub(_,1)),U32.and(j,U32.sub(_,1))) == True{} : Bool}
  %Equal.sym(U32,U32.and(i,U32.sub(131072,1)),i,Bound.table_address(i,ib)) :
    {S.apart(Complete.full(d),131072,_,U32.and(j,U32.sub(131072,1))) == True{} : Bool}
  %Equal.sym(U32,U32.and(j,U32.sub(131072,1)),j,Bound.table_address(j,jb)) :
    {S.apart(Complete.full(d),131072,i,_) == True{} : Bool}
  C.raw(d,i,j,cap,depth,ib,jb,ne)

def either(-T: Type,+a: Bool,+b: Bool,
  left: {a == True{} : Bool} -> T,
  right: {b == True{} : Bool} -> T,
  e: {Bool.or(a,b) == True{} : Bool}) -> T:
  match a:
    case True{}: left({==})
    case False{}: right(e)

def or_left(+a: Bool,+b: Bool,e: {a == True{} : Bool}) -> {Bool.or(a,b) == True{} : Bool}:
  Equal.cong(Bool,Bool,x=>Bool.or(x,b),a,True{},e)

def or_right(+a: Bool,+b: Bool,e: {b == True{} : Bool}) -> {Bool.or(a,b) == True{} : Bool}:
  match a:
    case True{}: {==}
    case False{}: e

def outside_ne(+q: U32,+at: U32,+end: U32,
  head: {U32.is_lt(at,end) == True{} : Bool},
  e: {Spec.outside(q,at,end) == True{} : Bool}) -> {U32.is_eq(at,q) == False{} : Bool}:
  either({U32.is_eq(at,q) == False{} : Bool},U32.is_lt(q,at),U32.is_le(end,q),
    p=>N.reverse_ne(at,q,p),p=>Order.lt_ne(at,q,Order.lt_le(at,end,q,head,p)),e)

def outside_next(+q: U32,+at: U32,+end: U32,
  safe: {U32.is_lt(at,131072) == True{} : Bool},
  e: {Spec.outside(q,at,end) == True{} : Bool}) -> {Spec.outside(q,U32.inc(at),end) == True{} : Bool}:
  either({Spec.outside(q,U32.inc(at),end) == True{} : Bool},U32.is_lt(q,at),U32.is_le(end,q),
    p=>or_left(U32.is_lt(q,U32.inc(at)),U32.is_le(end,q),N.before_next(q,at,safe,p)),
    p=>or_right(U32.is_lt(q,U32.inc(at)),U32.is_le(end,q),p),e)

def induction(+d: Nat,+t: O.Shape,+n: Nat,+at: U32,+end: U32,+q: U32,
  +shape: {t == Complete.full(d) : O.Shape},
  +size: {S.size(Complete.full(d)) == 131072 : U32},
  +cap: {A.cap(d) == 131072 : U32},
  +depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +eb: {U32.is_lt(end,131072) == True{} : Bool},
  +qb: {U32.is_lt(q,131072) == True{} : Bool},
  +room: {Spec.room(n,at,end) == True{} : Bool},
  +outside: {Spec.outside(q,at,end) == True{} : Bool}) -> {S.clear(t,n,at,q) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      +head = N.room_head(p,at,end,room)
      +safe = Order.lt_lt(at,end,131072,head,eb)
      +first = path(d,t,at,q,shape,size,cap,depth,safe,qb,outside_ne(q,at,end,head,outside))
      +rest = induction(d,t,p,U32.inc(at),end,q,shape,size,cap,depth,eb,qb,N.next_room(p,at,end,safe,room),outside_next(q,at,end,safe,outside))
      %Equal.sym(Bool,S.disjoint(t,at,q),True{},first) : {Bool.and(_,S.clear(t,p,U32.inc(at),q)) == True{} : Bool}
      rest

def certificate(+t: O.Shape,+n: Nat,+at: U32,+end: U32,+q: U32,
  shape: {t == Complete.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},
  room: {Spec.room(n,at,end) == True{} : Bool},
  outside: {Spec.outside(q,at,end) == True{} : Bool}) -> {S.clear(t,n,at,q) == True{} : Bool}:
  induction(17n,t,n,at,end,q,shape,{==},{==},{==},eb,qb,room,outside)
