# Bounded Nat offsets round-trip through the actual U32 conversion.
import Base
import ../fill_interval/Naturals.bend as NatProof
import ../fill_interval/Arithmetic.bend as Arithmetic
import ../relative/Order.bend as Order
import ../relative/Certified.bend as Relative
import ../relative/Bounds.bend as Bounds
import ../relative/Spec.bend as R
import ../prefix/Spec.bend as P
import ../prefix/LAWS.bend as PL
import ../prefix/PROOF.bend as PP
import ../prefix/Nats.bend as N
import ../prefix/Order.bend as PO
import ../layout/Domain.bend as Domain
import ../layout/Spec.bend as Layout
import ../layout/LAWS.bend as LL
import ../layout/PROOF.bend as LP

def predecessor(+n: Nat,+b: Nat,e: {Nat.is_le(1n+n,b) == True{} : Bool}) ->
  {Nat.is_lt(n,b) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(n,b),Nat.is_le(1n+n,b),True{},
    Equal.sym(Bool,Nat.is_le(1n+n,b),Nat.is_lt(n,b),NatProof.suc_le(n,b)),e)

def exact(+i: Nat,+b: U32,+ib: {Nat.is_le(i,U32.to_nat(b)) == True{} : Bool},
  +bb: {U32.is_lt(b,131072) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(i)) == i : Nat}:
  match i:
    case 0n: {==}
    case 1n+p:
      +prev = exact(p,b,N.lt_to_le(p,U32.to_nat(b),predecessor(p,U32.to_nat(b),ib)),bb)
      +bound = Order.from_nat_lt(U32.from_nat(p),b,
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(U32.from_nat(p)),U32.to_nat(b)),Nat.is_lt(p,U32.to_nat(b)),True{},
          Equal.cong(Nat,Bool,n => Nat.is_lt(n,U32.to_nat(b)),U32.to_nat(U32.from_nat(p)),p,prev),predecessor(p,U32.to_nat(b),ib)))
      Equal.trans(Nat,U32.to_nat(U32.from_nat(1n+p)),1n+U32.to_nat(U32.from_nat(p)),1n+p,
        Arithmetic.inc_value(U32.from_nat(p),Order.lt_lt(U32.from_nat(p),b,131072,bound,bb)),
        Equal.cong(Nat,Nat,n => 1n+n,U32.to_nat(U32.from_nat(p)),p,prev))

def max_small(+bishop: Bool) -> {U32.is_lt(Bool.pick(U32,bishop,512,4096),131072) == True{} : Bool}:
  match bishop:
    case False{}: {==}
    case True{}: {==}

def key_valid(+k: Nat,+e: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(k)),128n) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(U32.to_nat(U32.from_nat(k)),128n),Nat.is_lt(k,128n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,128n),U32.to_nat(U32.from_nat(k)),k,Domain.exact(k,e)),e)

def size_bound(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.is_lt(Layout.size(U32.from_nat(k)),131072) == True{} : Bool}:
  Order.le_lt(Layout.size(U32.from_nat(k)),Layout.max_size(U32.from_nat(k)),131072,
    LL.tight_size_bound(U32.from_nat(k),key_valid(k,valid)),max_small(U32.is_ge(U32.from_nat(k),64)))

def interval(+k: Nat,+i: Nat,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) -> R.interval(k,U32.from_nat(i)):
  +b = Layout.size(U32.from_nat(k))
  +same = exact(i,b,N.lt_to_le(i,U32.to_nat(b),ib),size_bound(k,valid))
  Bounds.interval(k,U32.from_nat(i),Relative.certificate(k,U32.from_nat(i),valid,
    Order.from_nat_lt(U32.from_nat(i),b,
      Equal.trans(Bool,Nat.is_lt(U32.to_nat(U32.from_nat(i)),U32.to_nat(b)),Nat.is_lt(i,U32.to_nat(b)),True{},
        Equal.cong(Nat,Bool,n => Nat.is_lt(n,U32.to_nat(b)),U32.to_nat(U32.from_nat(i)),i,same),ib))))

def lower_prefix(+k: Nat,pair: {U32.is_le(512,P.prefix(k)) == True{} : Bool}
  & {U32.is_lt(P.prefix(k),131072) == True{} : Bool}) -> {U32.is_le(512,P.prefix(k)) == True{} : Bool}:
  (lo,hi) = pair
  lo

def lower(+k: Nat,+i: Nat,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  {U32.is_le(512,R.address(k,U32.from_nat(i))) == True{} : Bool}:
  PO.trans(512,P.prefix(k),R.address(k,U32.from_nat(i)),
    lower_prefix(k,PL.prefix_in_allocation(k,N.weaken(k,128n,valid))),
    Bounds.lower(k,U32.from_nat(i),interval(k,i,valid,ib)))
