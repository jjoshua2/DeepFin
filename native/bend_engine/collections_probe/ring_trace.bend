# Deterministic bounded-FIFO oracle trace with noncopyable Array<U64> values.
import Base
import ./Ring.bend as R

def value(+x: U32) -> Array<U64>:
  Array.new(U64, 1n, U64{x, U32.xor(x, 2779096485)})

def shown(prefix: String, r: Array<U64> & U64) -> IO(Unit):
  (a, +x) = r
  IO.print(prefix ++ " " ++ U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x)))

def sized(r: R.Ring<Array<U64>> & U32) -> IO(R.Ring<Array<U64>>):
  (q, n) = r
  do IO<R.Ring<Array<U64>>>:
    IO.print("size " ++ U32.show(n))
    return q

def pushed(r: R.Ring<Array<U64>> & Maybe<&1, Array<U64>>) -> IO(R.Ring<Array<U64>>):
  (q, item) = r
  match item:
    case None{}:
      do IO<R.Ring<Array<U64>>>:
        IO.print("accept")
        sized(R.length(Array<U64>, q))
    case Some{a}:
      do IO<R.Ring<Array<U64>>>:
        s : Unit <- shown("reject", Array.get(U64, a, 0))
        sized(R.length(Array<U64>, q))

def popped(r: R.Ring<Array<U64>> & Maybe<&1, Array<U64>>) -> IO(R.Ring<Array<U64>>):
  (q, item) = r
  match item:
    case None{}:
      do IO<R.Ring<Array<U64>>>:
        IO.print("empty")
        sized(R.length(Array<U64>, q))
    case Some{a}:
      do IO<R.Ring<Array<U64>>>:
        s : Unit <- shown("value", Array.get(U64, a, 0))
        sized(R.length(Array<U64>, q))

def fill(n: Nat, +x: U32, q: R.Ring<Array<U64>>) -> IO(R.Ring<Array<U64>>):
  match n:
    case 0n: IO.pure(R.Ring<Array<U64>>, q)
    case 1n+p:
      do IO<R.Ring<Array<U64>>>:
        q : R.Ring<Array<U64>> <- pushed(R.push(Array<U64>, q, value(x)))
        fill(p, U32.inc(x), q)

def drain(n: Nat, q: R.Ring<Array<U64>>) -> IO(R.Ring<Array<U64>>):
  match n:
    case 0n: IO.pure(R.Ring<Array<U64>>, q)
    case 1n+p:
      do IO<R.Ring<Array<U64>>>:
        q : R.Ring<Array<U64>> <- popped(R.pop(Array<U64>, q))
        drain(p, q)

def mixed_step(x: U32, q: R.Ring<Array<U64>>, enqueue: Bool) -> IO(R.Ring<Array<U64>>):
  match enqueue:
    case True{}: pushed(R.push(Array<U64>, q, value(x)))
    case False{}: popped(R.pop(Array<U64>, q))

def mixed(n: Nat, +x: U32, q: R.Ring<Array<U64>>) -> IO(R.Ring<Array<U64>>):
  match n:
    case 0n: IO.pure(R.Ring<Array<U64>>, q)
    case 1n+p:
      do IO<R.Ring<Array<U64>>>:
        q : R.Ring<Array<U64>> <- mixed_step(x, q, U32.is_lt(U32.and(U32.shrn(x, 16n), 7), 4))
        mixed(p, U32.add(U32.mul(x, 1664525), 1013904223), q)

def trace(+capacity: U32, r: Maybe<&1, R.Ring<Array<U64>>>) -> IO(Unit):
  match r:
    case None{}: IO.print("invalid " ++ U32.show(capacity))
    case Some{q}:
      do IO<Unit>:
        IO.print("capacity " ++ U32.show(capacity))
        q : R.Ring<Array<U64>> <- drain(2n, q)
        q : R.Ring<Array<U64>> <- fill(U32.to_nat(U32.add(capacity, 2)), 4294967280, q)
        q : R.Ring<Array<U64>> <- drain(U32.to_nat(U32.inc(U32.div(capacity, 2))), q)
        q : R.Ring<Array<U64>> <- fill(U32.to_nat(U32.add(capacity, 2)), 100, q)
        q : R.Ring<Array<U64>> <- mixed(512n, U32.add(305419896, capacity), q)
        q : R.Ring<Array<U64>> <- drain(U32.to_nat(U32.add(capacity, 2)), q)
        q : R.Ring<Array<U64>> <- fill(2n, 42, q)
        q : R.Ring<Array<U64>> <- drain(3n, q)
        return Unit{}

def cases(xs: List<&2, U32>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{+c, tail}:
      do IO<Unit>:
        a : Unit <- trace(c, R.new(Array<U64>, c))
        cases(tail)

def main() -> IO(Unit):
  cases([0, 1, 2, 3, 7, 16, 17, 4096, 4097, 4294967295])
