# Actual non-slider query route, including pawn operand order.
import Base
import ./Geometry.bend as G
import ./Slots.bend as S
import ./Full.bend as Full
import ./Memory.bend as Memory
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../complete/Spec.bend as C

def tag(kind: G.Leaper) -> U32:
  match kind:
    case G.Knight{}: 1
    case G.King{}: 5
    case G.WhitePawn{}: 0
    case G.BlackPawn{}: 0

def side(kind: G.Leaper) -> U32:
  match kind:
    case G.WhitePawn{}: 1
    case _: 0

def route(kind: G.Leaper,table: Array<U64>,+q: Nat,occ: U64) ->
  {Chess.attack(tag(kind),table,U32.from_nat(q),side(kind),occ) == Array.get(U64,table,S.index(kind,q)) : Array<U64> & U64}:
  match kind:
    case G.Knight{}: {==}
    case G.King{}: {==}
    case G.WhitePawn{}:
      Equal.cong(U32,Array<U64> & U64,i => Array.get(U64,table,i),U32.add(U32.from_nat(q),384),U32.add(384,U32.from_nat(q)),U32.add_comm(U32.from_nat(q),384))
    case G.BlackPawn{}:
      Equal.cong(U32,Array<U64> & U64,i => Array.get(U64,table,i),U32.add(U32.from_nat(q),448),U32.add(448,U32.from_nat(q)),U32.add_comm(U32.from_nat(q),448))

def actual(+n: Nat,a: Array<U64>,+kind: G.Leaper,+q: Nat,+occ: U64,
  sh: {O.shape(a) == C.full(17n) : O.Shape},+qb: {Nat.is_lt(q,64n) == True{} : Bool},+full: {n == 64n : Nat}) ->
  {Chess.attack(tag(kind),Tables.extras(n,0,a),U32.from_nat(q),side(kind),occ) == (Tables.extras(n,0,a),G.mask(kind,q)) : Array<U64> & U64}:
  Memory.shaped(Array<U64> & U64,a,x => Chess.attack(tag(kind),Tables.extras(n,0,x),U32.from_nat(q),side(kind),occ),x => (Tables.extras(n,0,x),G.mask(kind,q)),sh,+c => shape =>
    Equal.trans(Array<U64> & U64,
      Chess.attack(tag(kind),Tables.extras(n,0,O.pack(c)),U32.from_nat(q),side(kind),occ),
      Array.get(U64,Tables.extras(n,0,O.pack(c)),S.index(kind,q)),(Tables.extras(n,0,O.pack(c)),G.mask(kind,q)),
      route(kind,Tables.extras(n,0,O.pack(c)),q,occ),Full.pair(n,O.pack(c),kind,q,shape,qb,full)),R.reify(a))
