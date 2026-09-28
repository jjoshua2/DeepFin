# Owning-array FIFO trace. The Python oracle receives every pop and final size.
import Base
import ./Queue.bend as Q

def next(+x: U32) -> U32:
  U32.add(U32.mul(x, 1664525), 1013904223)

def word(+x: U32) -> U64:
  U64.or(U64.shln(U64.from_u32(U32.xor(x, 2779096485)), 32n), U64.from_u32(x))

def emit_word(q: Q.Queue<&1, Array<U64>>, r: Array<U64> & U64) -> IO(Q.Queue<&1, Array<U64>>):
  (arr, +x) = r
  do IO<Q.Queue<&1, Array<U64>>>:
    IO.print("value " ++ U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x)))
    return q

def emit(q: Q.Queue<&1, Array<U64>>, item: Maybe<&1, Array<U64>>) -> IO(Q.Queue<&1, Array<U64>>):
  match item:
    case None{}:
      do IO<Q.Queue<&1, Array<U64>>>:
        IO.print("empty")
        return q
    case Some{arr}:
      emit_word(q, Array.get(U64, arr, 0))

def popped(r: Q.Queue<&1, Array<U64>> & Maybe<&1, Array<U64>>) -> IO(Q.Queue<&1, Array<U64>>):
  (q, item) = r
  emit(q, item)

def step(put: Bool, +x: U32, q: Q.Queue<&1, Array<U64>>) -> IO(Q.Queue<&1, Array<U64>>):
  match put:
    case True{}:
      IO.pure(Q.Queue<&1, Array<U64>>, Q.push(&1, Array<U64>, q, Array.new(U64, 0n, word(x))))
    case False{}:
      popped(Q.pop(&1, Array<U64>, q))

def trace(fuel: Nat, +seed: U32, q: Q.Queue<&1, Array<U64>>) -> IO(Q.Queue<&1, Array<U64>>):
  match fuel:
    case 0n: IO.pure(Q.Queue<&1, Array<U64>>, q)
    case 1n+p:
      do IO<Q.Queue<&1, Array<U64>>>:
        q : Q.Queue<&1, Array<U64>> <- step(U32.is_lt(U32.mod(U32.div(seed, 65536), 8), 4), seed, q)
        trace(p, next(seed), q)

def fill(fuel: Nat, +i: U32, q: Q.Queue<&1, Array<U64>>) -> Q.Queue<&1, Array<U64>>:
  match fuel:
    case 0n: q
    case 1n+p: fill(p, U32.inc(i), Q.push(&1, Array<U64>, q, Array.new(U64, 0n, word(i))))

def drain(fuel: Nat, q: Q.Queue<&1, Array<U64>>) -> IO(Q.Queue<&1, Array<U64>>):
  match fuel:
    case 0n: IO.pure(Q.Queue<&1, Array<U64>>, q)
    case 1n+p:
      do IO<Q.Queue<&1, Array<U64>>>:
        q : Q.Queue<&1, Array<U64>> <- popped(Q.pop(&1, Array<U64>, q))
        drain(p, q)

def finish(r: Q.Queue<&1, Array<U64>> & Nat) -> IO(Unit):
  (q, n) = r
  IO.print("length " ++ Nat.show(n))

def main() -> IO(Unit):
  do IO<Unit>:
    q : Q.Queue<&1, Array<U64>> <- drain(3n, Q.new(&1, Array<U64>))
    q : Q.Queue<&1, Array<U64>> <- drain(18n, fill(16n, 4294967280, q))
    q : Q.Queue<&1, Array<U64>> <- trace(4096n, 305419896, q)
    q : Q.Queue<&1, Array<U64>> <- drain(4098n, q)
    finish(Q.length(&1, Array<U64>, q))
