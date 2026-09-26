# Bounded single-owner FIFO. Independent implementation, no upstream source.
# Use new(), not the exposed representation: 0 <= size <= capacity <= 4096.
import Base

type Ring<-T: Type> is Type:
  Ring{slots: Array<List<&1, T>>, head: U32, tail: U32, size: U32, capacity: U32}

# Box slots as zero/one-element lists. Recursive list carriers keep a uniform
# layout when the pinned backend erases generic T in this recursive builder.
# Array<Maybe<T>> is not safe here: creation and access can choose different
# strides for composite T. This adds allocation; measure rather than assume.
# Allocate each empty owning slot separately; Array.new requires Data.
def slots(-T: Type, depth: Nat) -> Array<List<&1, T>>:
  match depth:
    case 0n: ALeaf{Nil{}}
    case 1n+p:
      +d = p
      ANode{slots(T, d), slots(T, d)}

def depth(fuel: Nat, +capacity: U32) -> Nat:
  match fuel:
    case 0n: 0n
    case 1n+p:
      Bool.pick(Nat, U32.is_le(capacity, 1), 0n,
        1n+depth(p, U32.div(U32.inc(capacity), 2)))

def create(-T: Type, +capacity: U32, valid: Bool) -> Maybe<&1, Ring<T>>:
  match valid:
    case False{}: None{}
    case True{}:
      Some{Ring{slots(T, depth(12n, capacity)), 0, 0, 0, capacity}}

def new(-T: Type, +capacity: U32) -> Maybe<&1, Ring<T>>:
  create(T, capacity, U32.is_le(capacity, 4096))

def next(+i: U32, capacity: U32) -> U32:
  Bool.pick(U32, U32.is_eq(U32.inc(i), capacity), 0, U32.inc(i))

def length(-T: Type, r: Ring<T>) -> Ring<T> & U32:
  Ring{a, h, t, +n, c} = r
  (Ring{a, h, t, n, c}, n)

# Success returns None; full returns the incoming value, retaining ownership.
def pushed(-T: Type, h: U32, t: U32, n: U32, c: U32,
  result: Array<List<&1, T>> & List<&1, T>) -> Ring<T> & Maybe<&1, T>:
  (a, previous) = result
  match previous:
    case Nil{}: (Ring{a, h, t, n, c}, None{})
    case Con{x, tail}: (Ring{a, h, t, n, c}, Some{x})

def push_open(-T: Type, r: Ring<T>, value: T) -> Ring<T> & Maybe<&1, T>:
  Ring{a, h, +t, n, +c} = r
  pushed(T, h, next(t, c), U32.inc(n), c,
    Array.swap(List<&1, T>, a, t, Con{value, Nil{}}))

def push_if(-T: Type, r: Ring<T>, value: T, full: Bool) -> Ring<T> & Maybe<&1, T>:
  match full:
    case True{}: (r, Some{value})
    case False{}: push_open(T, r, value)

def push(-T: Type, r: Ring<T>, value: T) -> Ring<T> & Maybe<&1, T>:
  Ring{a, h, t, +n, +c} = r
  push_if(T, Ring{a, h, t, n, c}, value, U32.is_eq(n, c))

def pop_open(-T: Type, r: Ring<T>) -> Ring<T> & Maybe<&1, T>:
  Ring{a, +h, t, n, +c} = r
  pushed(T, next(h, c), t, U32.sub(n, 1), c,
    Array.swap(List<&1, T>, a, h, Nil{}))

def pop_if(-T: Type, r: Ring<T>, empty: Bool) -> Ring<T> & Maybe<&1, T>:
  match empty:
    case True{}: (r, None{})
    case False{}: pop_open(T, r)

def pop(-T: Type, r: Ring<T>) -> Ring<T> & Maybe<&1, T>:
  Ring{a, h, t, +n, c} = r
  pop_if(T, Ring{a, h, t, n, c}, U32.is_zero(n))
