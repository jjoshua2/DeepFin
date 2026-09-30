# The first computed value survives every later write in a bounded fill.
import Base
import ../../Tables.bend as Tables
import ../../Subsets.bend as Subsets
import ../fill_interval/Actual.bend as Frame
import ../fill_interval/Arithmetic.bend as A
import ../fill_interval/Naturals.bend as N
import ../relative/Order.bend as Order
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../storage/Roundtrip.bend as Round
import ../storage/Write.bend as Write
import ../separation/Spec.bend as S

def self_next(+n: Nat) -> {Nat.is_lt(n,1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: self_next(p)

def next_after(+at: U32,ab: {U32.is_lt(at,131072) == True{} : Bool}) ->
  {U32.is_lt(at,U32.inc(at)) == True{} : Bool}:
  Order.from_nat_lt(at,U32.inc(at),Equal.trans(Bool,
    Nat.is_lt(U32.to_nat(at),U32.to_nat(U32.inc(at))),Nat.is_lt(U32.to_nat(at),1n+U32.to_nat(at)),True{},
    Equal.cong(Nat,Bool,x => Nat.is_lt(U32.to_nat(at),x),U32.to_nat(U32.inc(at)),1n+U32.to_nat(at),A.inc_value(at,ab)),
    self_next(U32.to_nat(at))))

def model(+c: O.Cells,+p: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +eb: {U32.is_lt(end,131072) == True{} : Bool},
  +budget: {Nat.is_le(Nat.add(1n+p,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(1n+p,at,sq,bishop,mask,subset,O.pack(c)),at)) == Tables.slider(sq,bishop,subset,False{}) : U64}:
  +v = Tables.slider(sq,bishop,subset,False{})
  +ab = Order.lt_lt(at,end,131072,N.current(p,at,end,budget),eb)
  Equal.trans(U64,
    S.value(Array.get(U64,Tables.fill(1n+p,at,sq,bishop,mask,subset,O.pack(c)),at)),
    S.value(Array.get(U64,Array.set(U64,O.pack(c),at,v),at)),v,
    Frame.frame_before(Array.set(U64,O.pack(c),at,v),p,U32.inc(at),end,at,sq,bishop,mask,Subsets.next(subset,mask),
      Equal.trans(O.Shape,O.shape(Array.set(U64,O.pack(c),at,v)),O.shape(O.pack(c)),C.full(17n),Write.write(O.pack(c),at,v),shape),
      eb,next_after(at,ab),N.tail_budget(p,at,end,ab,budget)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,Array.set(U64,O.pack(c),at,v),at),
      (Array.set(U64,O.pack(c),at,v),v),Round.model(c,at,v)))

def lift(-a: Array<U64>,+p: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(1n+p,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,Tables.fill(1n+p,at,sq,bishop,mask,subset,a),at)) == Tables.slider(sq,bishop,subset,False{}) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,Tables.fill(1n+p,at,sq,bishop,mask,subset,_),at)) == Tables.slider(sq,bishop,subset,False{}) : U64}
  model(c,p,at,end,sq,bishop,mask,subset,
    Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),Equal.cong(Array<U64>,O.Shape,t => O.shape(t),O.pack(c),a,same),shape),eb,budget)

def first(a: Array<U64>,+p: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(1n+p,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(1n+p,at,sq,bishop,mask,subset,a),at)) == Tables.slider(sq,bishop,subset,False{}) : U64}:
  lift(a,p,at,end,sq,bishop,mask,subset,shape,eb,budget,R.reify(a))
