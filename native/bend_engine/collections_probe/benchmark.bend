# CPU microbenchmark only: rotating ID queues, not owning search roots.
import Base
import ./Queue.bend as Q

def mix(+sum: U32, x: U32) -> U32:
  U32.xor(U32.add(U32.mul(sum, 1664525), 1013904223), x)

def ids(n: Nat, +i: U32, acc: List<&2, U32>) -> List<&2, U32>:
  match n:
    case 0n: List.reverse(&2, U32, acc)
    case 1n+p: ids(p, U32.inc(i), Con{i, acc})

def queued(n: Nat, +i: U32, q: Q.Queue<&2, U32>) -> Q.Queue<&2, U32>:
  match n:
    case 0n: q
    case 1n+p: queued(p, U32.inc(i), Q.push(&2, U32, q, i))

def list_loop(n: Nat, xs: List<&2, U32>, sum: U32) -> List<&2, U32> & U32:
  match n xs:
    case 0n xs: (xs, sum)
    case 1n+p Nil{}: (Nil{}, 0)
    case 1n+p Con{+x, tail}:
      list_loop(p, List.append(&2, U32, tail, Con{U32.xor(x, 2654435769), Nil{}}), mix(sum, x))

def rotate(sum: U32, r: Q.Queue<&2, U32> & Maybe<&2, U32>) -> Q.Queue<&2, U32> & U32:
  (q, item) = r
  match item:
    case None{}: (q, 0)
    case Some{+x}: (Q.push(&2, U32, q, U32.xor(x, 2654435769)), mix(sum, x))

def queue_loop(n: Nat, r: Q.Queue<&2, U32> & U32) -> Q.Queue<&2, U32> & U32:
  match n:
    case 0n: r
    case 1n+p:
      (q, sum) = r
      queue_loop(p, rotate(sum, Q.pop(&2, U32, q)))

def list_done(size: Nat, sample: Nat, started: Nat, r: List<&2, U32> & U32) -> IO(Unit):
  (xs, sum) = r
  do IO<Unit>:
    ended : Nat <- IO.now()
    IO.print("list " ++ Nat.show(size) ++ " " ++ Nat.show(sample) ++ " " ++ Nat.show(Nat.sub(ended, started)) ++ " " ++ U32.show(sum)
      ++ " " ++ Nat.show(List.length(&2, U32, xs)))

def queue_sized(size: Nat, sample: Nat, started: Nat, sum: U32, r: Q.Queue<&2, U32> & Nat) -> IO(Unit):
  (q, length) = r
  do IO<Unit>:
    ended : Nat <- IO.now()
    IO.print("queue " ++ Nat.show(size) ++ " " ++ Nat.show(sample) ++ " " ++ Nat.show(Nat.sub(ended, started)) ++ " " ++ U32.show(sum)
      ++ " " ++ Nat.show(length))

def queue_done(size: Nat, sample: Nat, started: Nat, r: Q.Queue<&2, U32> & U32) -> IO(Unit):
  (q, sum) = r
  queue_sized(size, sample, started, sum, Q.length(&2, U32, q))

def list_run(size: Nat, sample: Nat, xs: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    started : Nat <- IO.now()
    list_done(size, sample, started, list_loop(20000n, xs, 0))

def queue_run(size: Nat, sample: Nat, q: Q.Queue<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    started : Nat <- IO.now()
    queue_done(size, sample, started, queue_loop(20000n, (q, 0)))

def pair(first: Bool, +size: Nat, +sample: Nat) -> IO(Unit):
  match first:
    case True{}:
      do IO<Unit>:
        a : Unit <- list_run(size, sample, ids(size, 0, Nil{}))
        queue_run(size, sample, queued(size, 0, Q.new(&2, U32)))
    case False{}:
      do IO<Unit>:
        a : Unit <- queue_run(size, sample, queued(size, 0, Q.new(&2, U32)))
        list_run(size, sample, ids(size, 0, Nil{}))

def samples(n: Nat, +first: Bool, +size: Nat, +sample: Nat) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Unit <- pair(first, size, sample)
        samples(p, Bool.not(first), size, 1n+sample)

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- samples(6n, True{}, 16n, 0n)
    b : Unit <- samples(6n, True{}, 256n, 0n)
    samples(6n, True{}, 4096n, 0n)
