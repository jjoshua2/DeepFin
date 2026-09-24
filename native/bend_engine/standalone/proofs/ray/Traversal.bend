import Base
import ./Spec.bend as S
import ./Frame.bend as B
import ./Algebra.bend as A
import ../../Tables.bend as T
import ../u64/Words.bend as Old
import ../layout/Domain.bend as Domain

def nat_eq(+a: Nat,+b: Nat,e: {Nat.is_eq(a,b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+p: Old.false_true({0n == 1n+p : Nat},e)
    case 1n+p 0n: Old.false_true({1n+p == 0n : Nat},e)
    case 1n+p 1n+q: Equal.cong(Nat,Nat,n => 1n+n,p,q,nat_eq(p,q,e))

def not_false(+b: Bool,e: {Bool.not(b) == True{} : Bool}) -> {b == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: Old.false_true({True{} == False{} : Bool},e)

def stopped(n: Nat,sq: U32,dir: U32,occ: U64,+acc: U64) ->
  {T.ray(n,True{},sq,dir,occ,False{},acc) == acc : U64}:
  match n:
    case 0n: {==}
    case 1n+p: {==}

def choose(+n: Nat,+stop: Bool,+sq: U32,+dir: U32,+occ: U64,+acc: U64,+value: U64,
  e: {T.ray(n,False{},sq,dir,occ,False{},acc) == value : U64}) ->
  {T.ray(n,stop,sq,dir,occ,False{},acc) == Bool.pick(U64,stop,acc,value) : U64}:
  match stop:
    case False{}: e
    case True{}: stopped(n,sq,dir,occ,acc)

def advance(+n: Nat,+q: U32,+dir: U32,+occ: U64,+acc: U64,valid: Bool,off: Bool) -> U64:
  T.ray(n,Bool.or(off,U64.test_bit(occ,U32.to_nat(q))),q,dir,occ,False{},
    U64.or(acc,Bool.pick(U64,Bool.and(valid,True{}),U64.bit(U32.to_nat(q)),U64.zero())))

def pulse(+n: Nat,+q: U32,+dir: U32,+occ: U64,+acc: U64,+x: Nat) -> U64:
  T.ray(n,U64.test_bit(occ,x),q,dir,occ,False{},U64.or(acc,U64.bit(x)))

def entered(+n: Nat,+q: U32,+dir: U32,+occ: U64,+acc: U64,+x: Nat,
  valid: {U32.is_lt(q,64) == True{} : Bool},off: {U32.is_eq(q,64) == False{} : Bool},
  at: {U32.to_nat(q) == x : Nat}) ->
  {advance(n,q,dir,occ,acc,U32.is_lt(q,64),U32.is_eq(q,64)) == pulse(n,q,dir,occ,acc,x) : U64}:
  Equal.trans(U64,advance(n,q,dir,occ,acc,U32.is_lt(q,64),U32.is_eq(q,64)),
    advance(n,q,dir,occ,acc,True{},U32.is_eq(q,64)),pulse(n,q,dir,occ,acc,x),
    Equal.cong(Bool,U64,b => advance(n,q,dir,occ,acc,b,U32.is_eq(q,64)),U32.is_lt(q,64),True{},valid),
    Equal.trans(U64,advance(n,q,dir,occ,acc,True{},U32.is_eq(q,64)),pulse(n,q,dir,occ,acc,U32.to_nat(q)),pulse(n,q,dir,occ,acc,x),
      Equal.cong(Bool,U64,b => advance(n,q,dir,occ,acc,True{},b),U32.is_eq(q,64),False{},off),
      Equal.cong(Nat,U64,z => pulse(n,q,dir,occ,acc,z),U32.to_nat(q),x,at)))

def exhausted(+n: Nat,+q: U32,+dir: U32,+occ: U64,+acc: U64,
  e: {Nat.is_eq(U32.to_nat(q),64n) == True{} : Bool}) ->
  {advance(n,q,dir,occ,acc,U32.is_lt(q,64),U32.is_eq(q,64)) == acc : U64}:
  Equal.trans(U64,advance(n,q,dir,occ,acc,U32.is_lt(q,64),U32.is_eq(q,64)),
    advance(n,64,dir,occ,acc,False{},True{}),acc,
    Equal.cong(U32,U64,z => advance(n,z,dir,occ,acc,U32.is_lt(z,64),U32.is_eq(z,64)),q,64,
      Domain.injective(q,64,nat_eq(U32.to_nat(q),64n,e))),
    Equal.trans(U64,T.ray(n,True{},64,dir,occ,False{},U64.or(acc,U64.zero())),U64.or(acc,U64.zero()),acc,
      stopped(n,64,dir,occ,U64.or(acc,U64.zero())),A.zero(acc)))

def actual(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+occ: U64,+acc: U64,
  +e: {S.trace(n,sq,dir,xs) == True{} : Bool}) ->
  {T.ray(n,False{},sq,dir,occ,False{},acc) == S.scan(xs,occ,acc) : U64}:
  match n:
    case 0n:
      match xs:
        case Nil{}: {==}
        case x <> tail: Old.false_true({acc == S.scan(x <> tail,occ,acc) : U64},e)
    case 1n+p:
      match xs:
        case Nil{}: exhausted(p,T.jump(sq,T.delta(dir)),dir,occ,acc,e)
        case +x <> +tail:
          +q = T.jump(sq,T.delta(dir))
          +good = U32.is_lt(q,64)
          +not_off = Bool.not(U32.is_eq(q,64))
          +same = Nat.is_eq(U32.to_nat(q),x)
          +rest = S.trace(p,q,dir,tail)
          +tailcert = B.right(good,Bool.and(not_off,Bool.and(same,rest)),e)
          +lastcert = B.right(not_off,Bool.and(same,rest),tailcert)
          Equal.trans(U64,advance(p,q,dir,occ,acc,good,U32.is_eq(q,64)),pulse(p,q,dir,occ,acc,x),S.scan(x <> tail,occ,acc),
            entered(p,q,dir,occ,acc,x,B.left(good,Bool.and(not_off,Bool.and(same,rest)),e),
              not_false(U32.is_eq(q,64),B.left(not_off,Bool.and(same,rest),tailcert)),
              nat_eq(U32.to_nat(q),x,B.left(same,rest,lastcert))),
            choose(p,U64.test_bit(occ,x),q,dir,occ,U64.or(acc,U64.bit(x)),S.scan(tail,occ,U64.or(acc,U64.bit(x))),
              actual(p,q,dir,tail,occ,U64.or(acc,U64.bit(x)),B.right(same,rest,lastcert))))
