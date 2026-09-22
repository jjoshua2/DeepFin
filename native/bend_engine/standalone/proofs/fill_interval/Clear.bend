# Numeric budgets imply the existing path-only clear-write predicate.
import Base
import ./Naturals.bend as N
import ../relative/Order.bend as Order
import ../complete/Core.bend as Core
import ../complete/Arithmetic.bend as CA
import ../complete/Spec.bend as Complete
import ../normalization/Bound.bend as Mask
import ../separation/Spec.bend as S

# Depth stays symbolic until the public certificate selects the actual allocation.
def paths(+d: Nat,+i: U32,+j: U32,
  size: {S.size(Complete.full(d)) == 131072 : U32},
  +cap: {CA.cap(d) == 131072 : U32},
  depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +ib: {U32.is_lt(i,131072) == True{} : Bool},
  +jb: {U32.is_lt(j,131072) == True{} : Bool},
  different: {i == j : U32} -> {False{} == True{} : Bool}) ->
  {S.disjoint(Complete.full(d),i,j) == True{} : Bool}:
  Equal.trans(Bool,S.disjoint(Complete.full(d),i,j),
    S.apart(Complete.full(d),131072,U32.and(i,131071),U32.and(j,131071)),True{},
    Equal.cong(U32,Bool,n => S.apart(Complete.full(d),n,U32.and(i,U32.sub(n,1)),U32.and(j,U32.sub(n,1))),
      S.size(Complete.full(d)),131072,size),
    Equal.trans(Bool,S.apart(Complete.full(d),131072,U32.and(i,131071),U32.and(j,131071)),
      S.apart(Complete.full(d),131072,i,j),True{},
      Equal.trans(Bool,S.apart(Complete.full(d),131072,U32.and(i,131071),U32.and(j,131071)),
        S.apart(Complete.full(d),131072,i,U32.and(j,131071)),S.apart(Complete.full(d),131072,i,j),
        Equal.cong(U32,Bool,x => S.apart(Complete.full(d),131072,x,U32.and(j,131071)),U32.and(i,131071),i,Mask.table_address(i,ib)),
        Equal.cong(U32,Bool,x => S.apart(Complete.full(d),131072,i,x),U32.and(j,131071),j,Mask.table_address(j,jb))),
      Equal.trans(Bool,S.apart(Complete.full(d),131072,i,j),S.apart(Complete.full(d),CA.cap(d),i,j),True{},
        Equal.cong(U32,Bool,n => S.apart(Complete.full(d),n,i,j),131072,CA.cap(d),Equal.sym(U32,CA.cap(d),131072,cap)),
        Core.routes(d,i,j,depth,
          Equal.trans(Bool,U32.is_lt(i,CA.cap(d)),U32.is_lt(i,131072),True{},Equal.cong(U32,Bool,n => U32.is_lt(i,n),CA.cap(d),131072,cap),ib),
          Equal.trans(Bool,U32.is_lt(j,CA.cap(d)),U32.is_lt(j,131072),True{},Equal.cong(U32,Bool,n => U32.is_lt(j,n),CA.cap(d),131072,cap),jb),different))))

def both(+a: Bool,+b: Bool,ae: {a == True{} : Bool},be: {b == True{} : Bool}) -> {Bool.and(a,b) == True{} : Bool}:
  Equal.trans(Bool,Bool.and(a,b),Bool.and(True{},b),True{},Equal.cong(Bool,Bool,x => Bool.and(x,b),a,True{},ae),be)

def before(+d: Nat,+n: Nat,+at: U32,+end: U32,+q: U32,
  +size: {S.size(Complete.full(d)) == 131072 : U32},
  +cap: {CA.cap(d) == 131072 : U32},
  +depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +eb: {U32.is_lt(end,131072) == True{} : Bool},
  +qb: {U32.is_lt(q,at) == True{} : Bool},
  +budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.clear(Complete.full(d),n,at,q) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      +ae = N.current(p,at,end,budget)
      +ab = Order.lt_lt(at,end,131072,ae,eb)
      +qq = Order.lt_lt(q,at,131072,qb,ab)
      both(S.disjoint(Complete.full(d),at,q),S.clear(Complete.full(d),p,U32.inc(at),q),
        paths(d,at,q,size,cap,depth,ab,qq,eq => CA.different(q,at,Order.lt_ne(q,at,qb),Equal.sym(U32,at,q,eq))),
        before(d,p,U32.inc(at),end,q,size,cap,depth,eb,N.below_next(q,at,ab,qb),N.tail_budget(p,at,end,ab,budget)))

def after(+d: Nat,+n: Nat,+at: U32,+q: U32,
  +size: {S.size(Complete.full(d)) == 131072 : U32},
  +cap: {CA.cap(d) == 131072 : U32},
  +depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +qb: {U32.is_lt(q,131072) == True{} : Bool},
  +budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(q)) == True{} : Bool}) ->
  {S.clear(Complete.full(d),n,at,q) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      +aq = N.current(p,at,q,budget)
      +ab = Order.lt_lt(at,q,131072,aq,qb)
      both(S.disjoint(Complete.full(d),at,q),S.clear(Complete.full(d),p,U32.inc(at),q),
        paths(d,at,q,size,cap,depth,ab,qb,eq => CA.different(at,q,Order.lt_ne(at,q,aq),eq)),
        after(d,p,U32.inc(at),q,size,cap,depth,qb,N.tail_budget(p,at,q,ab,budget)))
