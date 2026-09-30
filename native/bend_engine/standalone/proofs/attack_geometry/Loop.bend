# Natural-index schedule with an explicit equality to the actual extras loop.
import Base
import ./Slots.bend as S
import ./Memory.bend as M
import ../../Tables.bend as Tables
import ../extras/Facts.bend as E
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R

def run(n: Nat,+k: Nat,a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p: run(p,1n+k,S.block(a,k))

def inc(+k: Nat,f: E.step(k)) ->
  {U32.inc(U32.from_nat(k)) == U32.from_nat(1n+k) : U32}:
  (i,rest) = f
  i

def actual(+n: Nat,a: Array<U64>,+k: Nat,
  +budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool}) ->
  {run(n,k,a) == Tables.extras(n,U32.from_nat(k),a) : Array<U64>}:
  match n:
    case 0n: {==}
    case 1n+p:
      M.equal(Array<U64>,a,x => run(1n+p,k,x),x => Tables.extras(1n+p,U32.from_nat(k),x),+c =>
        Equal.trans(Array<U64>,run(p,1n+k,S.block(O.pack(c),k)),
          Tables.extras(p,U32.from_nat(1n+k),S.block(O.pack(c),k)),
          Tables.extras(1n+p,U32.from_nat(k),O.pack(c)),
          actual(p,S.block(O.pack(c),k),1n+k,E.tail(p,k,budget)),
          Equal.cong(U32,Array<U64>,j => Tables.extras(p,j,S.block(O.pack(c),k)),
            U32.from_nat(1n+k),U32.inc(U32.from_nat(k)),
            Equal.sym(U32,U32.inc(U32.from_nat(k)),U32.from_nat(1n+k),inc(k,E.square(k,E.current(p,k,budget)))))),R.reify(a))
