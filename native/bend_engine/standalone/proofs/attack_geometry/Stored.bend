# Induction through every subsequent write, not assumed initialized entry values.
import Base
import ./Slots.bend as S
import ./Geometry.bend as G
import ./One.bend as One
import ./Loop.bend as L
import ./Memory.bend as M
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../complete/Spec.bend as C
import ../extras/Facts.bend as E
import ../headers/Addresses.bend as A
import ../prefix/Nats.bend as N

def self_next(+k: Nat) -> {Nat.is_lt(k,1n+k) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n+p: self_next(p)

def count(before: Nat,after: Nat) -> Nat:
  Nat.add(before,1n+after)

def later(+n: Nat,a: Array<U64>,+k: Nat,+kind: G.Leaper,+q: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool},
  +earlier: {Nat.is_lt(q,k) == True{} : Bool},+qb: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {One.read(L.run(n,k,a),S.index(kind,q)) == One.read(a,S.index(kind,q)) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      M.shaped(U64,a,x => One.read(L.run(1n+p,k,x),S.index(kind,q)),x => One.read(x,S.index(kind,q)),shape,+c => +sh =>
        Equal.trans(U64,One.read(L.run(p,1n+k,S.block(O.pack(c),k)),S.index(kind,q)),
          One.read(S.block(O.pack(c),k),S.index(kind,q)),One.read(O.pack(c),S.index(kind,q)),
          later(p,S.block(O.pack(c),k),1n+k,kind,q,One.sh4(c,k,sh),E.tail(p,k,budget),N.weaken(q,k,earlier),qb),
          One.other(c,k,kind,q,sh,E.current(p,k,budget),qb,A.nat_ne(q,k,earlier))),R.reify(a))

def entry(+before: Nat,+after: Nat,a: Array<U64>,+k: Nat,+kind: G.Leaper,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_le(Nat.add(count(before,after),k),64n) == True{} : Bool}) ->
  {One.read(L.run(count(before,after),k,a),S.index(kind,Nat.add(before,k))) == S.value(kind,Nat.add(before,k)) : U64}:
  match before:
    case 0n:
      M.shaped(U64,a,x => One.read(L.run(1n+after,k,x),S.index(kind,k)),x => S.value(kind,k),shape,+c => +sh =>
        Equal.trans(U64,One.read(L.run(after,1n+k,S.block(O.pack(c),k)),S.index(kind,k)),
          One.read(S.block(O.pack(c),k),S.index(kind,k)),S.value(kind,k),
          later(after,S.block(O.pack(c),k),1n+k,kind,k,One.sh4(c,k,sh),E.tail(after,k,budget),self_next(k),E.current(after,k,budget)),
          One.own(c,k,kind,sh,E.current(after,k,budget))),R.reify(a))
    case 1n+p:
      M.shaped(U64,a,x => One.read(L.run(count(1n+p,after),k,x),S.index(kind,1n+Nat.add(p,k))),x => S.value(kind,1n+Nat.add(p,k)),shape,+c => +sh =>
        %N.add_suc(p,k) :
          {One.read(L.run(count(p,after),1n+k,S.block(O.pack(c),k)),S.index(kind,_)) == S.value(kind,1n+Nat.add(p,k)) : U64}
        %N.add_suc(p,k) :
          {One.read(L.run(count(p,after),1n+k,S.block(O.pack(c),k)),S.index(kind,Nat.add(p,1n+k))) == S.value(kind,_) : U64}
        entry(p,after,S.block(O.pack(c),k),1n+k,kind,One.sh4(c,k,sh),E.tail(count(p,after),k,budget)),R.reify(a))
