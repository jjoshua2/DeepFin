import Base
import ../layout/Low.bend as Low
import ./Arithmetic.bend as A
import ../prefix/Nats.bend as Nats
import ../prefix/Order.bend as PrefixOrder
import ../relative/Order.bend as Order
import ../u64/Words.bend as Old

def suc_le(+a: Nat,+b: Nat) -> {Nat.is_le(1n+a,b) == Nat.is_lt(a,b) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: Low.zero_le(q)
    case 1n+p 0n: {==}
    case 1n+p 1n+q: suc_le(p,q)

def current(+n: Nat,+at: U32,+end: U32,
  budget: {Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {U32.is_lt(at,end) == True{} : Bool}:
  Order.from_nat_lt(at,end,Nats.strip(n,U32.to_nat(at),U32.to_nat(end),
    Equal.trans(Bool,Nat.is_lt(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)),
      Nat.is_le(1n+Nat.add(n,U32.to_nat(at)),U32.to_nat(end)),True{},
      Equal.sym(Bool,Nat.is_le(1n+Nat.add(n,U32.to_nat(at)),U32.to_nat(end)),
        Nat.is_lt(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)),suc_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end))),budget)))

def tail_sum(+n: Nat,+at: U32,ib: {U32.is_lt(at,131072) == True{} : Bool}) ->
  {Nat.add(n,U32.to_nat(U32.inc(at))) == Nat.add(1n+n,U32.to_nat(at)) : Nat}:
  Equal.trans(Nat,Nat.add(n,U32.to_nat(U32.inc(at))),Nat.add(n,1n+U32.to_nat(at)),1n+Nat.add(n,U32.to_nat(at)),
    Equal.cong(Nat,Nat,x => Nat.add(n,x),U32.to_nat(U32.inc(at)),1n+U32.to_nat(at),A.inc_value(at,ib)),
    Nats.add_suc(n,U32.to_nat(at)))

def tail_budget(+n: Nat,+at: U32,+end: U32,ib: {U32.is_lt(at,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(n,U32.to_nat(U32.inc(at))),U32.to_nat(end)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(Nat.add(n,U32.to_nat(U32.inc(at))),U32.to_nat(end)),
    Nat.is_le(Nat.add(1n+n,U32.to_nat(at)),U32.to_nat(end)),True{},
    Equal.cong(Nat,Bool,x => Nat.is_le(x,U32.to_nat(end)),Nat.add(n,U32.to_nat(U32.inc(at))),
      Nat.add(1n+n,U32.to_nat(at)),tail_sum(n,at,ib)),budget)

def below_next(+q: U32,+at: U32,ib: {U32.is_lt(at,131072) == True{} : Bool},
  before: {U32.is_lt(q,at) == True{} : Bool}) -> {U32.is_lt(q,U32.inc(at)) == True{} : Bool}:
  Order.from_nat_lt(q,U32.inc(at),Equal.trans(Bool,Nat.is_lt(U32.to_nat(q),U32.to_nat(U32.inc(at))),
    Nat.is_lt(U32.to_nat(q),1n+U32.to_nat(at)),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(U32.to_nat(q),n),U32.to_nat(U32.inc(at)),1n+U32.to_nat(at),A.inc_value(at,ib)),
    Nats.weaken(U32.to_nat(q),U32.to_nat(at),Equal.trans(Bool,Nat.is_lt(U32.to_nat(q),U32.to_nat(at)),U32.is_lt(q,at),True{},
      Equal.sym(Bool,U32.is_lt(q,at),Nat.is_lt(U32.to_nat(q),U32.to_nat(at)),A.lt_observed(q,at)),before))))

def commute(+a: Nat,+b: Nat) -> {Nat.add(a,b) == Nat.add(b,a) : Nat}:
  match a:
    case 0n: Equal.sym(Nat,Nat.add(b,0n),b,Nats.right_zero(b))
    case 1n+p:
      Equal.trans(Nat,1n+Nat.add(p,b),1n+Nat.add(b,p),Nat.add(b,1n+p),
        Equal.cong(Nat,Nat,x => 1n+x,Nat.add(p,b),Nat.add(b,p),commute(p,b)),
        Equal.sym(Nat,Nat.add(b,1n+p),1n+Nat.add(b,p),Nats.add_suc(b,p)))
