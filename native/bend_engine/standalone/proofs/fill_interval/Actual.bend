import Base
import ./Clear.bend as Clear
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../separation/Fill.bend as Fill
import ../../Tables.bend as Tables

def before(-a: Array<U64>,+n: Nat,+at: U32,+end: U32,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  qb: {U32.is_lt(q,at) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.clear(O.shape(a),n,at,q) == True{} : Bool}:
  Equal.trans(Bool,S.clear(O.shape(a),n,at,q),S.clear(C.full(17n),n,at,q),True{},
    Equal.cong(O.Shape,Bool,t => S.clear(t,n,at,q),O.shape(a),C.full(17n),shape),
    Clear.before(17n,n,at,end,q,{==},{==},{==},eb,qb,budget))

def after(-a: Array<U64>,+n: Nat,+at: U32,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  qb: {U32.is_lt(q,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(q)) == True{} : Bool}) ->
  {S.clear(O.shape(a),n,at,q) == True{} : Bool}:
  Equal.trans(Bool,S.clear(O.shape(a),n,at,q),S.clear(C.full(17n),n,at,q),True{},
    Equal.cong(O.Shape,Bool,t => S.clear(t,n,at,q),O.shape(a),C.full(17n),shape),
    Clear.after(17n,n,at,q,{==},{==},{==},qb,budget))

def frame_before(a: Array<U64>,+n: Nat,+at: U32,+end: U32,+q: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  eb: {U32.is_lt(end,131072) == True{} : Bool},qb: {U32.is_lt(q,at) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  Fill.frame(a,n,at,sq,bishop,mask,subset,q,before(a,n,at,end,q,shape,eb,qb,budget))

def frame_after(a: Array<U64>,+n: Nat,+at: U32,+q: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},qb: {U32.is_lt(q,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(q)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  Fill.frame(a,n,at,sq,bishop,mask,subset,q,after(a,n,at,q,shape,qb,budget))
