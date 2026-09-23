# Collection microbenchmark with actual Search.Tree allocations and owned history.
# Not a search, legal-position, async scheduler or neural throughput benchmark.
import Base
import ./Queue.bend as Q
import ./Ring.bend as R
import ../session_probe/Search.bend as S
import ../legal_probe/Chess.bend as C

type Work is Type:
  Work{id: U32, tree: S.Tree, history: Array<U64>}

type Bag is Type:
  Broken{}
  Linked{items: List<&1, Work>}
  Queued{items: Q.Queue<&1, Work>}
  Ringed{items: R.Ring<Work>}

def board() -> C.Board:
  C.Board{U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), U64.zero(), 0, 0, 64}

def work(+id: U32) -> Work:
  Work{id, S.start(board(), 4096, U32.inc(id), 20),
    Array.new(U64, 1n, U64{id, U32.xor(id, 2779096485)})}

def new_ring(r: Maybe<&1, R.Ring<Work>>) -> Bag:
  match r:
    case None{}: Broken{}
    case Some{a}: Ringed{a}

def bag(arm: U32, size: U32) -> Bag:
  match arm:
    case 0: Linked{Nil{}}
    case 1: Queued{Q.new(&1, Work)}
    case _: new_ring(R.new(Work, size))

def pushed(r: R.Ring<Work> & Maybe<&1, Work>) -> Bag:
  (q, rejected) = r
  match rejected:
    case None{}: Ringed{q}
    case Some{x}: Broken{}

def push(b: Bag, w: Work) -> Bag:
  match b:
    case Broken{}: Broken{}
    case Linked{xs}: Linked{List.append(&1, Work, xs, Con{w, Nil{}})}
    case Queued{q}: Queued{Q.push(&1, Work, q, w)}
    case Ringed{r}: pushed(R.push(Work, r, w))

def queue_pop(r: Q.Queue<&1, Work> & Maybe<&1, Work>) -> Bag & Maybe<&1, Work>:
  (q, x) = r
  (Queued{q}, x)

def ring_pop(r: R.Ring<Work> & Maybe<&1, Work>) -> Bag & Maybe<&1, Work>:
  (q, x) = r
  (Ringed{q}, x)

def pop(b: Bag) -> Bag & Maybe<&1, Work>:
  match b:
    case Broken{}: (Broken{}, None{})
    case Linked{Nil{}}: (Linked{Nil{}}, None{})
    case Linked{Con{x, tail}}: (Linked{tail}, Some{x})
    case Queued{q}: queue_pop(Q.pop(&1, Work, q))
    case Ringed{r}: ring_pop(R.pop(Work, r))

def fill(n: Nat, +id: U32, b: Bag) -> Bag:
  match n:
    case 0n: b
    case 1n+p: fill(p, U32.inc(id), push(b, work(id)))

def touch_node(id: U32, history: Array<U64>, used: U32, cap: U32, completed: U32,
  pending: U32, epoch: U32, request: U32, stop: U32, limit: U32,
  r: Array<S.Node> & S.Node) -> Work & U32:
  (nodes, +node) = r
  (Work{id, S.Tree{Array.set(S.Node, nodes, 0, S.bump(node, 0.0)), used, cap,
    U32.inc(completed), pending, epoch, request, stop, limit}, history}, S.node_n(node))

def touch(w: Work) -> Work & U32:
  Work{id, tree, history} = w
  S.Tree{nodes, used, cap, completed, pending, epoch, request, stop, limit} = tree
  touch_node(id, history, used, cap, completed, pending, epoch, request, stop, limit,
    Array.get(S.Node, nodes, 0))

def rotated(b: Bag, sum: U32, id: U32, r: Work & U32) -> Bag & U32:
  (w, count) = r
  (push(b, w), U32.xor(U32.add(U32.mul(sum, 1664525), 1013904223), U32.xor(id, count)))

def taken(sum: U32, r: Bag & Maybe<&1, Work>) -> Bag & U32:
  (b, got) = r
  match got:
    case None{}: (Broken{}, sum)
    case Some{Work{+id, tree, history}}:
      rotated(b, sum, id, touch(Work{id, tree, history}))

def loop(n: Nat, r: Bag & U32) -> Bag & U32:
  match n:
    case 0n: r
    case 1n+p:
      (b, sum) = r
      loop(p, taken(sum, pop(b)))

def history_text(prefix: String, r: Array<U64> & U64) -> IO(Unit):
  (history, +value) = r
  IO.print(prefix ++ " " ++ U32.show(U64.high(value)) ++ " " ++ U32.show(U64.low(value)))

def node_text(prefix: String, history: Array<U64>, r: Array<S.Node> & S.Node) -> IO(Unit):
  (nodes, +node) = r
  history_text(prefix ++ " " ++ U32.show(S.node_n(node)), Array.get(U64, history, 0))

def show_work(w: Work) -> IO(Unit):
  Work{id, tree, history} = w
  S.Tree{nodes, used, cap, completed, pending, epoch, request, stop, limit} = tree
  node_text("root " ++ U32.show(id) ++ " " ++ U32.show(used) ++ " " ++ U32.show(cap)
    ++ " " ++ U32.show(completed) ++ " " ++ U32.show(pending) ++ " " ++ U32.show(epoch)
    ++ " " ++ U32.show(request) ++ " " ++ U32.show(stop) ++ " " ++ U32.show(limit),
    history, Array.get(S.Node, nodes, 0))

def queue_length(r: Q.Queue<&1, Work> & Nat) -> IO(Unit):
  (q, n) = r
  IO.print("length " ++ Nat.show(n))

def ring_length(r: R.Ring<Work> & U32) -> IO(Unit):
  (q, n) = r
  IO.print("length " ++ U32.show(n))

def drained(b: Bag) -> IO(Unit):
  match b:
    case Broken{}: IO.die(Unit, 2, "broken owning collection")
    case Linked{xs}: IO.print("length " ++ Nat.show(List.length(&1, Work, xs)))
    case Queued{q}: queue_length(Q.length(&1, Work, q))
    case Ringed{r}: ring_length(R.length(Work, r))

def dump_one(r: Bag & Maybe<&1, Work>) -> IO(Bag):
  (b, got) = r
  match got:
    case None{}: IO.die(Bag, 2, "short owning collection")
    case Some{w}:
      do IO<Bag>:
        a : Unit <- show_work(w)
        return b

def dump(n: Nat, b: Bag) -> IO(Unit):
  match n:
    case 0n: drained(b)
    case 1n+p:
      do IO<Unit>:
        b : Bag <- dump_one(pop(b))
        dump(p, b)

def finish(arm: U32, +size: U32, steps: U32, started: Nat, r: Bag & U32) -> IO(Unit):
  (b, sum) = r
  do IO<Unit>:
    ended : Nat <- IO.now()
    IO.print("sample " ++ U32.show(arm) ++ " " ++ U32.show(size) ++ " " ++ U32.show(steps)
      ++ " " ++ Nat.show(Nat.sub(ended, started)) ++ " " ++ U32.show(sum))
    dump(U32.to_nat(size), b)

def run(arm: U32, size: U32, +steps: U32, b: Bag) -> IO(Unit):
  do IO<Unit>:
    started : Nat <- IO.now()
    finish(arm, size, steps, started, loop(U32.to_nat(steps), (b, 0)))

def valid(+arm: U32, +size: U32, steps: U32, ok: Bool) -> IO(Unit):
  match ok:
    case False{}: IO.die(Unit, 2, "invalid owning benchmark config")
    case True{}: run(arm, size, steps, fill(U32.to_nat(size), 0, bag(arm, size)))

def parsed(a: Maybe<&2, U32>, s: Maybe<&2, U32>, n: Maybe<&2, U32>) -> IO(Unit):
  match a s n:
    case Some{+arm} Some{+size} Some{+steps}:
      valid(arm, size, steps, Bool.and(U32.is_le(arm, 2), Bool.and(U32.is_ge(size, 1),
        Bool.and(U32.is_le(size, 128), U32.is_le(steps, 20000000)))))
    case _ _ _: IO.die(Unit, 2, "invalid owning benchmark config")

def config(xs: List<&2, String>) -> IO(Unit):
  match xs:
    case Con{a, Con{s, Con{n, Nil{}}}}: parsed(U32.read(a), U32.read(s), U32.read(n))
    case _: IO.die(Unit, 2, "invalid owning benchmark config")

def environment(r: Result<&1, &1, U32 & String, String>) -> IO(Unit):
  match r:
    case Fail{err}: IO.die(Unit, 2, "missing DEEPFIN_COLLECTION_BENCH")
    case Done{text}: config(String.split(text, ' '))

def main() -> IO(Unit):
  do IO<Unit>:
    input : Result<&1, &1, U32 & String, String> <- IO.get_env("DEEPFIN_COLLECTION_BENCH")
    environment(input)
