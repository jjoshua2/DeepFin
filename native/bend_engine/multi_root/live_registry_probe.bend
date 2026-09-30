# Compile and execute the actual metadata functions; no Python reimplementation.
import Base
import ./LiveRegistry.bend as G
import ./Deadlines.bend as D

def require(ok: Bool) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit,Unit{})
    case False{}: IO.die(Unit,2,"live registry invariant failed")

def reserved(slot: U32,book: G.Book & U32) -> G.Book:
  (book,gen) = book
  G.install(slot,gen,book)

def fill(fuel: Nat,+slot: U32,book: G.Book) -> G.Book:
  match fuel:
    case 0n: book
    case 1n+p: fill(p,U32.inc(slot),reserved(slot,G.reserve(book)))

def cycle_one(+slot: U32,+old: U32,r: G.Book & U32) -> IO(G.Book):
  (+book,+gen) = r
  do IO<G.Book>:
    require(Bool.and(U32.is_gt(gen,old),Bool.not(G.matches(slot,gen,book))))
    return G.install(slot,gen,book)

def cycle(fuel: Nat,+book: G.Book) -> IO(Unit):
  match fuel:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        +next : G.Book <- cycle_one(1,G.generation(1,book),G.reserve(G.remove(1,book)))
        require(Bool.not(G.matches(1,G.generation(1,book),next)))
        cycle(p,next)

def exhausted(r: G.Book & U32) -> IO(Unit):
  (G.Book{bindings,next},gen) = r
  do IO<Unit>:
    require(U32.is_zero(gen))
    require(U32.is_eq(next,65536))

def last_reserved(r: G.Book & U32) -> IO(Unit):
  (+book,gen) = r
  do IO<Unit>:
    require(U32.is_eq(gen,65535))
    require(Bool.not(G.room(book)))
    exhausted(G.reserve(book))

def timer_cleared(+timers: D.Timers) -> IO(Unit):
  D.Timers{+limits,+expired} = timers
  do IO<Unit>:
    require(Bool.not(D.marked(expired,1)))
    require(D.marked(expired,16))
    require(Bool.not(D.allowed(16,0n,timers)))
    require(D.allowed(1,999n,timers))
    require(Nat.is_eq(List.length(&2,D.Limit,limits),1n))

def full(+book: G.Book) -> IO(Unit):
  do IO<Unit>:
    require(U32.is_zero(G.available(book)))
    require(G.matches(1,1,book))
    require(G.matches(16,16,book))
    require(Bool.not(G.matches(1,0,book)))
    require(Bool.not(G.matches(0,1,book)))
    require(Bool.not(G.matches(17,1,book)))
    require(U32.is_eq(G.available(G.remove(8,book)),8))
    require(G.matches(16,16,G.remove(8,book)))
    cycle(128n,book)
    last_reserved(G.reserve(G.Book{Nil{},65535}))
    timer_cleared(G.clear_timer(1,D.Timers{Con{D.Limit{1,10n},Con{D.Limit{16,20n},Nil{}}},U32.or(D.bit(1),D.bit(16))}))
    IO.print("live registry: capacity, stale identity, 128 replacements, exhaustion and timer clearing passed")

def main() -> IO(Unit):
  full(fill(16n,1,G.Book{Nil{},1}))
