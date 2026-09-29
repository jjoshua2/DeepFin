# Exercise the actual queue adapters, mask scan and cancellation transform.
import Base
import ../collections_probe/Queue.bend as Q
import ./Cohort.bend as C
import ./DeadlineWork.bend as DW
import ./AsyncRun.bend as A
import ../standalone/Position.bend as Pos
import ../session_probe/Search.bend as S

def roots(n: Nat, +ep: U32) -> List<C.Root>:
  match n:
    case 0n: Nil{}
    case 1n+p:
      +g = Pos.start()
      Con{C.Root{g, S.start(Pos.board(g), 4096, ep, 2), 0, 0}, roots(p, U32.inc(ep))}

type Taken is Type:
  Taken{tail: Q.Queue<&1, C.Root>, reversed: List<C.Root>}

def taken(reversed: List<C.Root>, r: Q.Queue<&1, C.Root> & Maybe<&1, C.Root>) -> Taken:
  (q, item) = r
  match item:
    case None{}: Taken{q, reversed}
    case Some{root}: Taken{q, Con{root, reversed}}

def take(n: Nat, state: Taken) -> Taken:
  match n:
    case 0n: state
    case 1n+p:
      Taken{q, reversed} = state
      take(p, taken(reversed, Q.pop(&1, C.Root, q)))

def returned(state: Taken) -> Q.Queue<&1, C.Root>:
  Taken{q, reversed} = state
  C.requeue(List.reverse(&1, C.Root, reversed), q)

def rotate(n: Nat, +batch: Nat, q: Q.Queue<&1, C.Root>) -> Q.Queue<&1, C.Root>:
  match n:
    case 0n: q
    case 1n+p: rotate(p, batch, returned(take(batch, Taken{q, Nil{}})))

def emit(xs: List<C.Root>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{C.Root{g, S.Tree{pool, used, cap, n, pending, ep, req, stop, limit}, neural, draws}, rest}:
      do IO<Unit>:
        IO.print("root " ++ U32.show(ep) ++ " " ++ U32.show(n) ++ " " ++ U32.show(pending)
          ++ " " ++ U32.show(req) ++ " " ++ U32.show(stop) ++ " " ++ U32.show(neural))
        emit(rest)

def sized(mask: U32, r: Q.Queue<&1, C.Root> & Nat) -> IO(Unit):
  (q, n) = r
  do IO<Unit>:
    IO.print("state " ++ U32.show(mask) ++ " " ++ Nat.show(n))
    emit(C.queue_list(q))

def scanned(r: Q.Queue<&1, C.Root> & U32) -> IO(Unit):
  (q, mask) = r
  sized(mask, Q.length(&1, C.Root, q))

def masks(xs: List<U32>, +size: Nat, +batch: Nat) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{+mask, rest}:
      do IO<Unit>:
        IO.print("case " ++ Nat.show(size) ++ " " ++ Nat.show(batch) ++ " " ++ U32.show(mask))
        scanned(DW.scan_queue(C.Config{4,2,0,False{}},
          A.mark_queue(mask, rotate(7n, batch, C.queued(roots(size, 1)))), 1))
        masks(rest, size, batch)

def batches(xs: List<Nat>, +size: Nat) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{+batch, rest}:
      do IO<Unit>:
        masks([0,2,8,65536,65546], size, batch)
        batches(rest, size)

def sizes(xs: List<Nat>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit, Unit{})
    case Con{size, rest}:
      do IO<Unit>:
        batches([0n,1n,2n,4n,16n,17n], size)
        sizes(rest)

def main() -> IO(Unit):
  sizes([0n,1n,3n,16n])
