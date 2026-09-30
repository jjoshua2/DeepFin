# Single-owner FIFO. New implementation of the standard two-list algorithm.
# No bend-collections source is copied or imported.
import Base

type Queue<a, -T: Kind(a)> is Type:
  Queue{front: List<a, T>, rear: List<a, T>, size: Nat}

def new(a, -T: Kind(a)) -> Queue<a, T>:
  Queue{Nil{}, Nil{}, 0n}

def length(a, -T: Kind(a), q: Queue<a, T>) -> Queue<a, T> & Nat:
  Queue{front, rear, +size} = q
  (Queue{front, rear, size}, size)

def push(a, -T: Kind(a), q: Queue<a, T>, value: T) -> Queue<a, T>:
  Queue{front, rear, size} = q
  Queue{front, Con{value, rear}, 1n+size}

def take(a, -T: Kind(a), front: List<a, T>, rear: List<a, T>, size: Nat) -> Queue<a, T> & Maybe<a, T>:
  match front:
    case Nil{}:
      (Queue{Nil{}, rear, size}, None{})
    case Con{value, tail}:
      (Queue{tail, rear, Nat.sub(size, 1n)}, Some{value})

def pop(a, -T: Kind(a), q: Queue<a, T>) -> Queue<a, T> & Maybe<a, T>:
  Queue{front, rear, size} = q
  match front:
    case Nil{}:
      take(a, T, List.reverse(a, T, rear), Nil{}, size)
    case Con{value, tail}:
      (Queue{tail, rear, Nat.sub(size, 1n)}, Some{value})
