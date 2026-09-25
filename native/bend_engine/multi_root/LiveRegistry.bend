# Persistent controls address (slot,generation), never an ambiguous recycled slot.
# A generation is reserved once; abandoned reservations are not reused.
import Base
import ./Deadlines.bend as D

type Binding is Data:
  Binding{slot: U32, generation: U32}
type Book is Data:
  Book{bindings: List<&2,Binding>, next: U32}

def lookup(+slot: U32, bindings: List<&2,Binding>) -> U32:
  match bindings:
    case Nil{}: 0
    case Con{Binding{id,gen},rest}: Bool.pick(U32,U32.is_eq(slot,id),gen,lookup(slot,rest))

def generation(slot: U32,book: Book) -> U32:
  Book{bindings,next} = book
  lookup(slot,bindings)

def matches(+slot: U32,+gen: U32,book: Book) -> Bool:
  Bool.and(U32.is_gt(gen,0),Bool.and(U32.is_gt(slot,0),Bool.and(U32.is_le(slot,16),U32.is_eq(gen,generation(slot,book)))))

def erase(+slot: U32,bindings: List<&2,Binding>) -> List<&2,Binding>:
  match bindings:
    case Nil{}: Nil{}
    case Con{+b,+rest}:
      Binding{id,gen} = b
      Bool.pick(List<&2,Binding>,U32.is_eq(slot,id),erase(slot,rest),Con{b,erase(slot,rest)})

def remove(slot: U32,book: Book) -> Book:
  Book{bindings,next} = book
  Book{erase(slot,bindings),next}

def install(+slot: U32,gen: U32,book: Book) -> Book:
  Book{bindings,next} = book
  Book{Con{Binding{slot,gen},erase(slot,bindings)},next}

def vacant(fuel: Nat,+slot: U32,+book: Book) -> U32:
  match fuel:
    case 0n: 0
    case 1n+p: Bool.pick(U32,U32.is_zero(generation(slot,book)),slot,vacant(p,U32.inc(slot),book))

def available(book: Book) -> U32:
  vacant(16n,1,book)

# A lifetime admission bound also bounds cumulative U32 work counters:
# <=65535 generations * 256 real rows * 16 physical rows per forward.
def room(book: Book) -> Bool:
  Book{bindings,+next} = book
  Bool.and(U32.is_gt(next,0),U32.is_le(next,65535))

def reserve_checked(ok: Bool,book: Book) -> Book & U32:
  match ok:
    case False{}: (book,0)
    case True{}:
      Book{bindings,+next} = book
      (Book{bindings,U32.inc(next)},next)

def reserve(+book: Book) -> Book & U32:
  reserve_checked(room(book),book)

def clear(+mask: U32,slot: U32) -> U32:
  U32.and(mask,U32.xor(D.bit(slot),4294967295))

def limits_without(+slot: U32,limits: List<&2,D.Limit>) -> List<&2,D.Limit>:
  match limits:
    case Nil{}: Nil{}
    case Con{+limit,+rest}:
      D.Limit{id,due} = limit
      Bool.pick(List<&2,D.Limit>,U32.is_eq(slot,id),limits_without(slot,rest),Con{limit,limits_without(slot,rest)})

def clear_timer(+slot: U32,timers: D.Timers) -> D.Timers:
  D.Timers{limits,expired} = timers
  D.Timers{limits_without(slot,limits),clear(expired,slot)}
