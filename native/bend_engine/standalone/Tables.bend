# Attack tables constructed entirely in Bend. Same layout as Chess.bend.
# Coordinates use wrapping U32 deltas; off-board coordinates are rejected.
import Base
import ./Subsets.bend as Subsets

def delta(dir: U32) -> U32 & U32:
  match dir:
    case 0: (1, 0)
    case 1: (4294967295, 0)
    case 2: (0, 1)
    case 3: (0, 4294967295)
    case 4: (1, 1)
    case 5: (1, 4294967295)
    case 6: (4294967295, 1)
    case _: (4294967295, 4294967295)

def square(+x: U32, +y: U32) -> U32:
  Bool.pick(U32, Bool.and(U32.is_lt(x, 8), U32.is_lt(y, 8)), U32.add(x, U32.mul(y, 8)), 64)

def jump(+sq: U32, d: U32 & U32) -> U32:
  (dx, dy) = d
  square(U32.add(U32.mod(sq, 8), dx), U32.add(U32.div(sq, 8), dy))

def ray(fuel: Nat, stopped: Bool, +sq: U32, +dir: U32, +occ: U64, +mask: Bool, acc: U64) -> U64:
  match fuel:
    case 0n: acc
    case 1n+p:
      match stopped:
        case True{}: acc
        case False{}:
          +next = jump(sq, delta(dir))
          later = jump(next, delta(dir))
          add = Bool.and(U32.is_lt(next, 64), Bool.not(Bool.and(mask, U32.is_eq(later, 64))))
          acc = U64.or(acc, Bool.pick(U64, add, U64.bit(U32.to_nat(next)), U64.zero()))
          stop = Bool.or(U32.is_eq(next, 64), U64.test_bit(occ, U32.to_nat(next)))
          ray(p, stop, next, dir, occ, mask, acc)

def slider(+sq: U32, bishop: Bool, +occ: U64, +mask: Bool) -> U64:
  +d = Bool.pick(U32, bishop, 4, 0)
  U64.or(U64.or(ray(7n, False{}, sq, d, occ, mask, U64.zero()), ray(7n, False{}, sq, U32.inc(d), occ, mask, U64.zero())),
    U64.or(ray(7n, False{}, sq, U32.add(d, 2), occ, mask, U64.zero()), ray(7n, False{}, sq, U32.add(d, 3), occ, mask, U64.zero())))

def fill(n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      a = Array.set(U64, a, at, slider(sq, bishop, subset, False{}))
      fill(p, U32.inc(at), sq, bishop, mask, Subsets.next(subset, mask), a)

def tables(n: Nat, +key: U32, +at: U32, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      +sq = U32.mod(key, 64)
      +bishop = U32.is_ge(key, 64)
      +mask = slider(sq, bishop, U64.zero(), True{})
      +size = U32.shln(1, U32.to_nat(U64.popcount(mask)))
      a = Array.set(U64, Array.set(U64, a, key, mask), U32.add(128, key), U64.from_u32(at))
      tables(p, U32.inc(key), U32.add(at, size), fill(U32.to_nat(size), at, sq, bishop, mask, U64.zero(), a))

def knight(dir: U32) -> U32 & U32:
  match dir:
    case 0: (1, 2)
    case 1: (2, 1)
    case 2: (4294967295, 2)
    case 3: (4294967294, 1)
    case 4: (1, 4294967294)
    case 5: (2, 4294967295)
    case 6: (4294967295, 4294967294)
    case _: (4294967294, 4294967295)

def leaps(n: Nat, +sq: U32, +dir: U32, +king: Bool, acc: U64) -> U64:
  match n:
    case 0n: acc
    case 1n+p:
      xy = Bool.pick(U32 & U32, king, delta(dir), knight(dir))
      leaps(p, sq, U32.inc(dir), king, U64.or(acc, U64.bit(U32.to_nat(jump(sq, xy)))))

def pawns(+sq: U32, +dy: U32) -> U64:
  U64.or(U64.bit(U32.to_nat(jump(sq, (1, dy)))), U64.bit(U32.to_nat(jump(sq, (4294967295, dy)))))

def extras(n: Nat, +sq: U32, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      a = Array.set(U64, a, U32.add(256, sq), leaps(8n, sq, 0, False{}, U64.zero()))
      a = Array.set(U64, a, U32.add(320, sq), leaps(8n, sq, 0, True{}, U64.zero()))
      a = Array.set(U64, a, U32.add(384, sq), pawns(sq, 1))
      a = Array.set(U64, a, U32.add(448, sq), pawns(sq, 4294967295))
      extras(p, U32.inc(sq), a)

def build() -> Array<U64>:
  extras(64n, 0, tables(128n, 0, 512, Array.new(U64, 17n, U64.zero())))
