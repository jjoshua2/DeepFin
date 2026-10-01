import Base
import ./Spec.bend as S
import ../u64/Words.bend as Old

def at(-P: Nat -> Type,xs: List<&2,Nat>,+x: Nat,good: S.every(P,xs),here: S.member(x,xs)) -> P(x):
  match xs:
    case Nil{}: Empty.absurd(P(x),here)
    case h <> t:
      (ph,pt) = good
      match here:
        case Inl{same}:
          %Equal.sym(Nat,x,h,same) : P(_)
          ph
        case Inr{later}: at(P,t,x,pt,later)

def nat_eq(+a: Nat,+b: Nat,e: {Nat.is_eq(a,b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+p: Old.false_true({0n == 1n+p : Nat},e)
    case 1n+p 0n: Old.false_true({1n+p == 0n : Nat},e)
    case 1n+p 1n+q: Equal.cong(Nat,Nat,n => 1n+n,p,q,nat_eq(p,q,e))

def same(+a: Nat) -> {Nat.is_eq(a,a) == True{} : Bool}:
  match a:
    case 0n: {==}
    case 1n+p: same(p)

def branch(-A: Type,+a: Bool,+b: Bool,e: {Bool.or(a,b) == True{} : Bool},left: {a == True{} : Bool} -> A,right: {b == True{} : Bool} -> A) -> A:
  match a:
    case True{}: left({==})
    case False{}: right(e)

def from_true(+x: Nat,xs: List<&2,Nat>,e: {S.contains(x,xs) == True{} : Bool}) -> S.member(x,xs):
  match xs:
    case Nil{}: Old.false_true(Empty,e)
    case +h <> +t:
      branch(S.member(x,h <> t),Nat.is_eq(x,h),S.contains(x,t),e,
        eq => Inl{nat_eq(x,h,eq)},later => Inr{from_true(x,t,later)})

def or_left(+a: Bool,+b: Bool,e: {a == True{} : Bool}) -> {Bool.or(a,b) == True{} : Bool}:
  Equal.cong(Bool,Bool,x => Bool.or(x,b),a,True{},e)

def or_right(+a: Bool,+b: Bool,e: {b == True{} : Bool}) -> {Bool.or(a,b) == True{} : Bool}:
  match a:
    case False{}: e
    case True{}: {==}

def to_true(+x: Nat,xs: List<&2,Nat>,here: S.member(x,xs)) -> {S.contains(x,xs) == True{} : Bool}:
  match xs:
    case Nil{}: Empty.absurd({False{} == True{} : Bool},here)
    case +h <> +t:
      match here:
        case Inl{eq}:
          or_left(Nat.is_eq(x,h),S.contains(x,t),
            Equal.trans(Bool,Nat.is_eq(x,h),Nat.is_eq(h,h),True{},
              Equal.cong(Nat,Bool,n => Nat.is_eq(n,h),x,h,eq),same(h)))
        case Inr{later}: or_right(Nat.is_eq(x,h),S.contains(x,t),to_true(x,t,later))

def equivalent(+a: Bool,+b: Bool,ab: {a == True{} : Bool} -> {b == True{} : Bool},ba: {b == True{} : Bool} -> {a == True{} : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}: {==}
    case True{} True{}: {==}
    case True{} False{}: Old.false_true({True{} == False{} : Bool},ab({==}))
    case False{} True{}: Old.false_true({False{} == True{} : Bool},ba({==}))
