import Base
import ./Arithmetic.bend as A
import ./Spec.bend as Spec
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../u64/Words.bend as Old

def below_zero(-T: Type,n: Nat,e: {Nat.is_lt(n,0n) == True{} : Bool}) -> T:
  match n:
    case 0n: Old.false_true(T,e)
    case 1n+p: Old.false_true(T,e)

def loosen(+n: Nat,m: Nat,e: {Nat.is_lt(n,m) == True{} : Bool}) ->
  {Nat.is_lt(n,1n+m) == True{} : Bool}:
  match n m:
    case 0n _: {==}
    case 1n+p 0n: Old.false_true({Nat.is_lt(1n+p,1n) == True{} : Bool},e)
    case 1n+p 1n+q: loosen(p,q,e)

# Only the 17 possible branch depths are enumerated, never the addresses.
def half(d: Nat, e: {Nat.is_lt(d,17n) == True{} : Bool}) ->
  {U32.shr(A.cap(1n+d)) == A.cap(d) : U32}:
  match d:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n+p:
      below_zero({U32.shr(A.cap(18n+p)) == A.cap(17n+p) : U32},p,e)

def choose(-T: Type,+a: Bool,+b: Bool,+l: Bool,+r: Bool,arg: T,
  left: T -> {a == True{} : Bool} -> {b == True{} : Bool} -> {l == True{} : Bool},
  right: T -> {a == False{} : Bool} -> {b == False{} : Bool} -> {r == True{} : Bool}) ->
  {S.choice(l,r,a,b) == True{} : Bool}:
  match a b:
    case True{} True{}: left(arg,{==},{==})
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: right(arg,{==},{==})

# Numeric distinction is transported recursively; no path-separation premise.
def routes(+d: Nat,+i: U32,+j: U32,
  +depth: {Nat.is_lt(d,18n) == True{} : Bool},
  +ib: {U32.is_lt(i,A.cap(d)) == True{} : Bool},
  +jb: {U32.is_lt(j,A.cap(d)) == True{} : Bool},
  different: {i == j : U32} -> {False{} == True{} : Bool}) ->
  {S.apart(Spec.full(d),A.cap(d),i,j) == True{} : Bool}:
  match d:
    case 0n:
      Old.false_true({S.apart(Spec.full(0n),A.cap(0n),i,j) == True{} : Bool},
        different(Equal.trans(U32,i,0,j,A.one32(i,ib),Equal.sym(U32,j,0,A.one32(j,jb)))))
    case 1n+p:
      Equal.trans(Bool,S.apart(Spec.full(1n+p),A.cap(1n+p),i,j),
        S.choice(S.apart(Spec.full(p),A.cap(p),i,j),
          S.apart(Spec.full(p),A.cap(p),U32.sub(i,A.cap(p)),U32.sub(j,A.cap(p))),
          U32.is_lt(i,A.cap(p)),U32.is_lt(j,A.cap(p))),True{},
        Equal.cong(U32,Bool,h => S.choice(S.apart(Spec.full(p),h,i,j),
          S.apart(Spec.full(p),h,U32.sub(i,h),U32.sub(j,h)),U32.is_lt(i,h),U32.is_lt(j,h)),
          U32.shr(A.cap(1n+p)),A.cap(p),half(p,depth)),
        choose({i == j : U32} -> {False{} == True{} : Bool},U32.is_lt(i,A.cap(p)),U32.is_lt(j,A.cap(p)),
          S.apart(Spec.full(p),A.cap(p),i,j),
          S.apart(Spec.full(p),A.cap(p),U32.sub(i,A.cap(p)),U32.sub(j,A.cap(p))),
          different,df => ie => je => routes(p,i,j,loosen(p,17n,depth),ie,je,df),
          df => ie => je => routes(p,U32.sub(i,A.cap(p)),U32.sub(j,A.cap(p)),loosen(p,17n,depth),
            A.right32(p,i,ib,ie),A.right32(p,j,jb,je),
            eq => df(A.subtract_injective(i,j,A.cap(p),eq)))))
