import Base
import ./Fast.bend as Fast
import ../../bitboard_probe/Sliders.bend as Sliders
import ../Tables.bend as Tables

# These specifications are checked against the unmodified production loops by
# Core.fill_agrees/Core.tables_agrees and the public lookup-address theorem.
def next_subset(subset: U64, +mask: U64) -> U64:
  U64.and(U64.sub(subset, mask), mask)

def address(offset: U32, index: U32) -> U32:
  U32.add(offset, index)

def slider_mask(+key: U32) -> U64:
  Tables.slider(U32.mod(key, 64), U32.is_ge(key, 64), U64.zero(), True{})

def segment_size(mask: U64) -> U32:
  U32.shln(1, U32.to_nat(U64.popcount(mask)))

# Full-width index, position in segment, non-wrapping address, and used range.
def production_step_ok(+mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32) -> Bool:
  +wide = U64.pext(subset, mask)
  +index = Sliders.pext_index(subset, mask)
  +wide_address = U64.add(U64.from_u32(offset), U64.from_u32(i))
  Bool.and(U32.is_eq(address(offset, index), at),
    Bool.and(U32.is_zero(U64.high(wide)),
    Bool.and(U32.is_eq(index, i),
      Bool.and(U32.is_lt(i, size),
        Bool.and(U32.is_zero(U64.high(wide_address)),
          Bool.and(U32.is_eq(U64.low(wide_address), at),
            Bool.and(U32.is_ge(at, 512), U32.is_lt(at, 108160))))))))

def step_values(+wide: U64, +i: U32, +at: U32, +offset: U32, +size: U32) -> Bool:
  +index = U64.low(wide)
  +wide_address = U64.add(U64.from_u32(offset), U64.from_u32(i))
  Bool.and(U32.is_eq(address(offset, index), at),
    Bool.and(U32.is_zero(U64.high(wide)),
    Bool.and(U32.is_eq(index, i),
      Bool.and(U32.is_lt(i, size),
        Bool.and(U32.is_zero(U64.high(wide_address)),
          Bool.and(U32.is_eq(U64.low(wide_address), at),
            Bool.and(U32.is_ge(at, 512), U32.is_lt(at, 108160))))))))

# Same eight conditions, with a proved-equivalent, linear-time PEXT evaluator.
# Core.step_agrees certifies this substitution; it is not an external oracle.
def step_ok(mask: U64, subset: U64, i: U32, at: U32, offset: U32, size: U32) -> Bool:
  step_values(Fast.pext(subset, mask), i, at, offset, size)

# Finite conjunction over EVERY state visited by the actual subset step.
# The final state must cycle back to the empty subset, exactly at segment size.
def walk(n: Nat, +mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32) -> Bool:
  match n:
    case 0n: U64.is_zero(subset)
    case 1n+p:
      Bool.and(step_ok(mask, subset, i, at, offset, size),
        walk(p, mask, next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size))

# The descriptors are calculated by the engine, not a table of test constants.
def tables(n: Nat, +key: U32, +at: U32) -> Bool:
  match n:
    case 0n: U32.is_eq(at, 108160)
    case 1n+p:
      +mask = slider_mask(key)
      +size = segment_size(mask)
      Bool.and(U32.is_le(U64.popcount(mask), 12),
        Bool.and(walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
          tables(p, U32.inc(key), U32.add(at, size))))


# Specification: use the lookup index, rather than a sequential write cursor.
# It is compared with the ACTUAL Tables.fill by induction in Core.bend.
def indexed_fill(n: Nat, +offset: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      a = Array.set(U64, a, address(offset, Sliders.pext_index(subset, mask)), Tables.slider(sq, bishop, subset, False{}))
      indexed_fill(p, offset, sq, bishop, mask, next_subset(subset, mask), a)

# Metadata layout, and two explicitly different ways to fill one segment.
def metadata(+key: U32, +at: U32, a: Array<U64>) -> Array<U64>:
  Array.set(U64, Array.set(U64, a, key, slider_mask(key)), U32.add(128, key), U64.from_u32(at))

def segment(+key: U32, +at: U32, a: Array<U64>) -> Array<U64>:
  +mask = slider_mask(key)
  Tables.fill(U32.to_nat(segment_size(mask)), at, U32.mod(key, 64), U32.is_ge(key, 64), mask, U64.zero(), metadata(key, at, a))

def indexed_segment(+key: U32, +at: U32, a: Array<U64>) -> Array<U64>:
  +mask = slider_mask(key)
  indexed_fill(U32.to_nat(segment_size(mask)), at, U32.mod(key, 64), U32.is_ge(key, 64), mask, U64.zero(), metadata(key, at, a))

def indexed_tables(n: Nat, +key: U32, +at: U32, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p:
      indexed_tables(p, U32.inc(key), U32.add(at, segment_size(slider_mask(key))), indexed_segment(key, at, a))

# Select a visited state without duplicating the production transition.
def slot_at(skip: Nat, +mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32) -> Bool:
  match skip:
    case 0n: step_ok(mask, subset, i, at, offset, size)
    case 1n+p: slot_at(p, mask, next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size)

def segment_ok(+key: U32, +at: U32) -> Bool:
  +mask = slider_mask(key)
  +size = segment_size(mask)
  Bool.and(U32.is_le(U64.popcount(mask), 12), walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size))

def segment_at(skip: Nat, +key: U32, +at: U32) -> Bool:
  match skip:
    case 0n: segment_ok(key, at)
    case 1n+p: segment_at(p, U32.inc(key), U32.add(at, segment_size(slider_mask(key))))
