# Every interior write survives its suffix of actual bounded fill operations.
import Base
import ./First.bend as First
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../../Subsets.bend as Subsets
import ../fill_interval/Naturals.bend as N
import ../relative/Order.bend as Order
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Model.bend as M
import ../storage/Representation.bend as R
import ../storage/Roundtrip.bend as Round
import ../storage/Write.bend as Write
import ../separation/Spec.bend as S
import ../prefix/Nats.bend as Nats

def stored_shape(+c: O.Cells,+at: U32,+v: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(O.pack(M.updated(c,at,v))) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(O.pack(M.updated(c,at,v))),O.shape(Array.set(U64,O.pack(c),at,v)),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,a => O.shape(a),O.pack(M.updated(c,at,v)),Array.set(U64,O.pack(c),at,v),
      Equal.sym(Array<U64>,Array.set(U64,O.pack(c),at,v),O.pack(M.updated(c,at,v)),Round.set(c,at,v))),
    Equal.trans(O.Shape,O.shape(Array.set(U64,O.pack(c),at,v)),O.shape(O.pack(c)),C.full(17n),Write.write(O.pack(c),at,v),shape))

def model(+i: Nat,+p: Nat,+c: O.Cells,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +eb: {U32.is_lt(end,131072) == True{} : Bool},
  +budget: {Nat.is_le(Nat.add(Nat.add(i,1n+p),U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(Nat.add(i,1n+p),at,sq,bishop,mask,subset,O.pack(c)),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}:
  match i:
    case 0n: First.model(c,p,at,end,sq,bishop,mask,subset,shape,eb,budget)
    case 1n+t:
      +v = Tables.slider(sq,bishop,subset,False{})
      +ab = Order.lt_lt(at,end,131072,N.current(Nat.add(t,1n+p),at,end,budget),eb)
      %Equal.sym(Array<U64>,Array.set(U64,O.pack(c),at,v),O.pack(M.updated(c,at,v)),Round.set(c,at,v)) :
        {S.value(Array.get(U64,Tables.fill(Nat.add(t,1n+p),U32.inc(at),sq,bishop,mask,Subsets.next(subset,mask),_),Spec.address(t,U32.inc(at)))) ==
          Tables.slider(sq,bishop,Spec.state(t,mask,Subsets.next(subset,mask)),False{}) : U64}
      model(t,p,M.updated(c,at,v),U32.inc(at),end,sq,bishop,mask,Subsets.next(subset,mask),
        stored_shape(c,at,v,shape),eb,N.tail_budget(Nat.add(t,1n+p),at,end,ab,budget))

def lift(-a: Array<U64>,+i: Nat,+p: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(Nat.add(i,1n+p),U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,Tables.fill(Nat.add(i,1n+p),at,sq,bishop,mask,subset,a),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,Tables.fill(Nat.add(i,1n+p),at,sq,bishop,mask,subset,_),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}
  model(i,p,c,at,end,sq,bishop,mask,subset,
    Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),Equal.cong(Array<U64>,O.Shape,t => O.shape(t),O.pack(c),a,same),shape),eb,budget)

def selected(a: Array<U64>,+i: Nat,+p: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(Nat.add(i,1n+p),U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(Nat.add(i,1n+p),at,sq,bishop,mask,subset,a),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}:
  lift(a,i,p,at,end,sq,bishop,mask,subset,shape,eb,budget,R.reify(a))

def split_count(+n: Nat,+i: Nat,ib: {Nat.is_lt(i,n) == True{} : Bool}) ->
  {Nat.add(i,1n+Nat.sub(n,1n+i)) == n : Nat}:
  Equal.trans(Nat,Nat.add(i,1n+Nat.sub(n,1n+i)),Nat.add(1n+i,Nat.sub(n,1n+i)),n,
    Nats.add_suc(i,Nat.sub(n,1n+i)),
    Equal.trans(Nat,Nat.add(1n+i,Nat.sub(n,1n+i)),Nat.add(Nat.sub(n,1n+i),1n+i),n,
      N.commute(1n+i,Nat.sub(n,1n+i)),Nats.subtract_back(1n+i,n,Nats.lt_suc(i,n,ib))))

def bounded(a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}:
  +count = split_count(n,i,ib)
  %count : {S.value(Array.get(U64,Tables.fill(_,at,sq,bishop,mask,subset,a),Spec.address(i,at))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}
  selected(a,i,Nat.sub(n,1n+i),at,end,sq,bishop,mask,subset,shape,eb,
    Equal.trans(Bool,Nat.is_le(Nat.add(Nat.add(i,1n+Nat.sub(n,1n+i)),U32.to_nat(at)),U32.to_nat(end)),
      Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)),True{},
      Equal.cong(Nat,Bool,x => Nat.is_le(Nat.add(x,U32.to_nat(at)),U32.to_nat(end)),Nat.add(i,1n+Nat.sub(n,1n+i)),n,count),budget))
