# Bind each selected square into the same complete extras call.
import Base
import ./Stored.bend as Stored
import ./Loop.bend as L
import ./Slots.bend as S
import ./Geometry.bend as G
import ./Computed.bend as Computed
import ./One.bend as One
import ./Memory.bend as Memory
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../complete/Spec.bend as C
import ../contents/Actual.bend as Pair
import ../layout/Domain.bend as Domain
import ../prefix/Nats.bend as N

def suffix(q: Nat) -> Nat:
  Nat.sub(63n,q)

def full_count(+q: Nat,e: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {Stored.count(q,suffix(q)) == 64n : Nat}:
  match q:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: Domain.impossible({Stored.count(64n+p,suffix(64n+p)) == 64n : Nat},p,e)

def budget(+n: Nat,full: {n == 64n : Nat}) ->
  {Nat.is_le(Nat.add(n,0n),64n) == True{} : Bool}:
  Equal.cong(Nat,Bool,x => Nat.is_le(Nat.add(x,0n),64n),n,64n,full)

def image(+n: Nat,a: Array<U64>,+q: Nat,+qb: {Nat.is_lt(q,64n) == True{} : Bool},+full: {n == 64n : Nat}) ->
  {L.run(Stored.count(q,suffix(q)),0n,a) == Tables.extras(n,0,a) : Array<U64>}:
  Equal.trans(Array<U64>,L.run(Stored.count(q,suffix(q)),0n,a),L.run(n,0n,a),Tables.extras(n,0,a),
    Equal.cong(Nat,Array<U64>,k => L.run(k,0n,a),Stored.count(q,suffix(q)),n,
      Equal.trans(Nat,Stored.count(q,suffix(q)),64n,n,full_count(q,qb),Equal.sym(Nat,n,64n,full))),
    L.actual(n,a,0n,budget(n,full)))

def selected(a: Array<U64>,+kind: G.Leaper,+q: Nat,
  sh: {O.shape(a) == C.full(17n) : O.Shape},+qb: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {One.read(L.run(Stored.count(q,suffix(q)),0n,a),S.index(kind,q)) == S.value(kind,q) : U64}:
  %N.right_zero(q) : {One.read(L.run(Stored.count(q,suffix(q)),0n,a),S.index(kind,_)) == S.value(kind,q) : U64}
  %N.right_zero(q) : {One.read(L.run(Stored.count(q,suffix(q)),0n,a),S.index(kind,Nat.add(q,0n))) == S.value(kind,_) : U64}
  Stored.entry(q,suffix(q),a,0n,kind,sh,budget(Stored.count(q,suffix(q)),full_count(q,qb)))

def value(+n: Nat,a: Array<U64>,+kind: G.Leaper,+q: Nat,
  sh: {O.shape(a) == C.full(17n) : O.Shape},+qb: {Nat.is_lt(q,64n) == True{} : Bool},+full: {n == 64n : Nat}) ->
  {One.read(Tables.extras(n,0,a),S.index(kind,q)) == G.mask(kind,q) : U64}:
  Memory.shaped(U64,a,x => One.read(Tables.extras(n,0,x),S.index(kind,q)),x => G.mask(kind,q),sh,+c => +shape =>
    Equal.trans(U64,One.read(Tables.extras(n,0,O.pack(c)),S.index(kind,q)),
      One.read(L.run(Stored.count(q,suffix(q)),0n,O.pack(c)),S.index(kind,q)),G.mask(kind,q),
      Equal.cong(Array<U64>,U64,x => One.read(x,S.index(kind,q)),Tables.extras(n,0,O.pack(c)),L.run(Stored.count(q,suffix(q)),0n,O.pack(c)),
        Equal.sym(Array<U64>,L.run(Stored.count(q,suffix(q)),0n,O.pack(c)),Tables.extras(n,0,O.pack(c)),image(n,O.pack(c),q,qb,full))),
      Equal.trans(U64,One.read(L.run(Stored.count(q,suffix(q)),0n,O.pack(c)),S.index(kind,q)),S.value(kind,q),G.mask(kind,q),
        selected(O.pack(c),kind,q,shape,qb),Computed.actual(kind,q,qb))),R.reify(a))

def pair(+n: Nat,a: Array<U64>,+kind: G.Leaper,+q: Nat,
  sh: {O.shape(a) == C.full(17n) : O.Shape},+qb: {Nat.is_lt(q,64n) == True{} : Bool},+full: {n == 64n : Nat}) ->
  {Array.get(U64,Tables.extras(n,0,a),S.index(kind,q)) == (Tables.extras(n,0,a),G.mask(kind,q)) : Array<U64> & U64}:
  Memory.shaped(Array<U64> & U64,a,x => Array.get(U64,Tables.extras(n,0,x),S.index(kind,q)),x => (Tables.extras(n,0,x),G.mask(kind,q)),sh,+c => shape =>
    Pair.read_pair(Tables.extras(n,0,O.pack(c)),S.index(kind,q),G.mask(kind,q),value(n,O.pack(c),kind,q,shape,qb,full)),R.reify(a))
