# Induction over the actual four-write extras loop, not a replacement builder.
import Base
import ./Facts.bend as Facts
import ./Spec.bend as S
import ./Steps.bend as Steps
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../storage/Build.bend as Build
import ../complete/Spec.bend as C
import ../separation/Spec.bend as Sep
import ../contents/Actual.bend as Pair

def next(+k: Nat,facts: Facts.step(k)) -> {U32.inc(U32.from_nat(k)) == U32.from_nat(1n+k) : U32}:
  (inc,rest) = facts
  inc

def transfer(-a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  rule: @+c: O.Cells -> {O.shape(O.pack(c)) == C.full(17n) : O.Shape} ->
    {Sep.value(Array.get(U64,Tables.extras(n,U32.from_nat(k),O.pack(c)),q)) == Sep.value(Array.get(U64,O.pack(c),q)) : U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Sep.value(Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q)) == Sep.value(Array.get(U64,a,q)) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Sep.value(Array.get(U64,Tables.extras(n,U32.from_nat(k),_),q)) == Sep.value(Array.get(U64,a,q)) : U64}
  %same : {Sep.value(Array.get(U64,Tables.extras(n,U32.from_nat(k),O.pack(c)),q)) == Sep.value(Array.get(U64,_,q)) : U64}
  rule(c)(Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))

def value(+n: Nat,a: Array<U64>,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  +qb: {U32.is_lt(q,131072) == True{} : Bool},+e: {S.outside(q) == True{} : Bool}) ->
  {Sep.value(Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q)) == Sep.value(Array.get(U64,a,q)) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      +valid = Facts.current(p,k,budget)
      +sq = U32.from_nat(k)
      transfer(a,1n+p,k,q,shape,+c => +sh =>
        %Equal.sym(U32,U32.inc(sq),U32.from_nat(1n+k),next(k,Facts.square(k,valid))) :
          {Sep.value(Array.get(U64,Tables.extras(p,_,Build.black(c,sq)),q)) == Sep.value(Array.get(U64,O.pack(c),q)) : U64}
        Equal.trans(U64,Sep.value(Array.get(U64,Tables.extras(p,U32.from_nat(1n+k),Build.black(c,sq)),q)),
          Sep.value(Array.get(U64,Build.black(c,sq),q)),Sep.value(Array.get(U64,O.pack(c),q)),
          value(p,Build.black(c,sq),1n+k,q,
            Equal.trans(O.Shape,O.shape(Build.black(c,sq)),O.shape(O.pack(c)),C.full(17n),Build.extra_shape(c,sq),sh),Facts.tail(p,k,budget),qb,e),
          Steps.frame(c,k,q,sh,Facts.square(k,valid),qb,e)),R.reify(a))

def pair_model(+c: O.Cells,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool}) ->
  {Array.get(U64,Tables.extras(n,U32.from_nat(k),O.pack(c)),q) ==
    (Tables.extras(n,U32.from_nat(k),O.pack(c)),Sep.value(Array.get(U64,O.pack(c),q))) : Array<U64> & U64}:
  Pair.read_pair(Tables.extras(n,U32.from_nat(k),O.pack(c)),q,Sep.value(Array.get(U64,O.pack(c),q)),value(n,O.pack(c),k,q,shape,budget,qb,e))

def pair_lift(-a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q) ==
    (Tables.extras(n,U32.from_nat(k),a),Sep.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.extras(n,U32.from_nat(k),_),q) ==
    (Tables.extras(n,U32.from_nat(k),a),Sep.value(Array.get(U64,a,q))) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.extras(n,U32.from_nat(k),O.pack(c)),q) ==
    (Tables.extras(n,U32.from_nat(k),_),Sep.value(Array.get(U64,a,q))) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.extras(n,U32.from_nat(k),O.pack(c)),q) ==
    (Tables.extras(n,U32.from_nat(k),O.pack(c)),Sep.value(Array.get(U64,_,q))) : Array<U64> & U64}
  pair_model(c,n,k,q,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget,qb,e)

def pair(a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},e: {S.outside(q) == True{} : Bool}) ->
  {Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q) ==
    (Tables.extras(n,U32.from_nat(k),a),Sep.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  pair_lift(a,n,k,q,shape,budget,qb,e,R.reify(a))
