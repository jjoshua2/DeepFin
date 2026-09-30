import Base
import ./Spec.bend as Spec
import ./Schedule.bend as Schedule
import ./Entries.bend as Entries
import ../../Tables.bend as Tables
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../storage/Roundtrip.bend as Round
import ../storage/Model.bend as M
import ../separation/Spec.bend as S

def entry(a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),U32.add(at,U32.from_nat(i)))) ==
    Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}:
  %Schedule.address(i,at) :
    {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),_)) == Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}) : U64}
  Entries.bounded(a,n,i,at,end,sq,bishop,mask,subset,shape,ib,eb,budget)

def zero_start(a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),U32.add(at,U32.from_nat(i)))) ==
    Tables.slider(sq,bishop,Sequence.at(i,mask),False{}) : U64}:
  %Schedule.from_zero(i,mask) :
    {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),U32.add(at,U32.from_nat(i)))) == Tables.slider(sq,bishop,_,False{}) : U64}
  entry(a,n,i,at,end,sq,bishop,mask,U64.zero(),shape,ib,eb,budget)

# Lift a proved queried value to the complete affine read pair. The first
# component is the final updated array, never the original input buffer.
def pair_model(+c: O.Cells,+i: U32,+v: U64,
  e: {S.value(Array.get(U64,O.pack(c),i)) == v : U64}) ->
  {Array.get(U64,O.pack(c),i) == (O.pack(c),v) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Array.get(U64,O.pack(c),i),(O.pack(c),M.read_sized(c,O.size(c),i)),(O.pack(c),v),
    Round.get(c,i),
    Equal.cong(U64,Array<U64> & U64,x => (O.pack(c),x),M.read_sized(c,O.size(c),i),v,
      Equal.trans(U64,M.read_sized(c,O.size(c),i),S.value(Array.get(U64,O.pack(c),i)),v,
        Equal.sym(U64,S.value(Array.get(U64,O.pack(c),i)),M.read_sized(c,O.size(c),i),
          Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,O.pack(c),i),(O.pack(c),M.read_sized(c,O.size(c),i)),Round.get(c,i))),e)))

def pair_lift(-a: Array<U64>,+i: U32,+v: U64,
  e: {S.value(Array.get(U64,a,i)) == v : U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,a,i) == (a,v) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,_,i) == (a,v) : Array<U64> & U64}
  %same : {Array.get(U64,O.pack(c),i) == (_,v) : Array<U64> & U64}
  pair_model(c,i,v,Equal.trans(U64,S.value(Array.get(U64,O.pack(c),i)),S.value(Array.get(U64,a,i)),v,
    Equal.cong(Array<U64>,U64,t => S.value(Array.get(U64,t,i)),O.pack(c),a,same),e))

def read_pair(a: Array<U64>,+i: U32,+v: U64,e: {S.value(Array.get(U64,a,i)) == v : U64}) ->
  {Array.get(U64,a,i) == (a,v) : Array<U64> & U64}:
  pair_lift(a,i,v,e,R.reify(a))

def pair_entry_model(+c: O.Cells,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),O.pack(c)),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),O.pack(c)),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}:
  read_pair(Tables.fill(n,at,sq,bishop,mask,U64.zero(),O.pack(c)),U32.add(at,U32.from_nat(i)),Tables.slider(sq,bishop,Sequence.at(i,mask),False{}),
    zero_start(O.pack(c),n,i,at,end,sq,bishop,mask,shape,ib,eb,budget))

def pair_entry_lift(-a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),_),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),O.pack(c)),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),_),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}
  pair_entry_model(c,n,i,at,end,sq,bishop,mask,
    Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),Equal.cong(Array<U64>,O.Shape,t => O.shape(t),O.pack(c),a,same),shape),ib,eb,budget)

def complete_pair(a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,U64.zero(),a),Tables.slider(sq,bishop,Sequence.at(i,mask),False{})) : Array<U64> & U64}:
  pair_entry_lift(a,n,i,at,end,sq,bishop,mask,shape,ib,eb,budget,R.reify(a))
