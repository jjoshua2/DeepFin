# Fixed-budget U64 -> U32 exact map, independently implemented.
# Single owner. Use new(); exposed representation is not an input-validation API.
import Base

type Slot is Data:
  Empty{}
  Entry{key: U64, value: U32}

type Table is Type:
  Table{slots: Array<Slot>, mask: U32, size: U32}

type Place is Data:
  Seeking{index: U32}
  Vacant{index: U32}
  Present{index: U32, value: U32}

type Walk is Type:
  Walk{slots: Array<Slot>, place: Place}

type Put is Data:
  Added{}
  Replaced{previous: U32}
  Full{}

# Noncryptographic bucket selection. Equality below always checks both halves.
def bucket(+key: U64, mask: U32) -> U32:
  +fold = U32.xor(U64.low(key), U32.mul(U64.high(key), 2654435761))
  +mixed = U32.mul(U32.xor(fold, U32.shrn(fold, 16n)), 2246822519)
  U32.and(U32.xor(mixed, U32.shrn(mixed, 13n)), mask)

def create(+bits: U32, valid: Bool) -> Maybe<&1, Table>:
  match valid:
    case False{}: None{}
    case True{}:
      Some{Table{Array.new(Slot, U32.to_nat(bits), Empty{}),
        U32.sub(U32.shln(1, U32.to_nat(bits)), 1), 0}}

# Physical buckets = 2^bits, max entries = half that. Invalid sizes do not allocate.
def new(+bits: U32) -> Maybe<&1, Table>:
  create(bits, Bool.and(U32.is_ge(bits, 1), U32.is_le(bits, 16)))

def length(t: Table) -> Table & U32:
  Table{slots, mask, +size} = t
  (Table{slots, mask, size}, size)

def next(index: U32, mask: U32) -> U32:
  U32.and(U32.inc(index), mask)

def seen(key: U64, +index: U32, mask: U32, r: Array<Slot> & Slot) -> Walk:
  (slots, slot) = r
  match slot:
    case Empty{}: Walk{slots, Vacant{index}}
    case Entry{stored, value}:
      Walk{slots, Bool.pick(Place, U64.is_eq(key, stored),
        Present{index, value}, Seeking{next(index, mask)})}

def walk(fuel: Nat, +key: U64, +mask: U32, state: Walk) -> Walk:
  match fuel state:
    case 0n Walk{slots, place}: Walk{slots, place}
    case 1n+p Walk{slots, Vacant{i}}: Walk{slots, Vacant{i}}
    case 1n+p Walk{slots, Present{i, v}}: Walk{slots, Present{i, v}}
    case 1n+p Walk{slots, Seeking{+i}}:
      walk(p, key, mask, seen(key, i, mask, Array.get(Slot, slots, i)))

def located(mask: U32, size: U32, w: Walk) -> Table & Place:
  Walk{slots, place} = w
  (Table{slots, mask, size}, place)

def locate(t: Table, +key: U64) -> Table & Place:
  Table{slots, +mask, size} = t
  located(mask, size, walk(U32.to_nat(U32.inc(mask)), key, mask,
    Walk{slots, Seeking{bucket(key, mask)}}))

def found(r: Table & Place) -> Table & Maybe<&2, U32>:
  (t, place) = r
  match place:
    case Present{i, value}: (t, Some{value})
    case Vacant{i}: (t, None{})
    case Seeking{i}: (t, None{})

def get(t: Table, key: U64) -> Table & Maybe<&2, U32>:
  found(locate(t, key))

def insert(t: Table, index: U32, key: U64, value: U32, full: Bool) -> Table & Put:
  Table{slots, mask, size} = t
  match full:
    case True{}: (Table{slots, mask, size}, Full{})
    case False{}:
      (Table{Array.set(Slot, slots, index, Entry{key, value}), mask, U32.inc(size)}, Added{})

def put_at(key: U64, value: U32, r: Table & Place) -> Table & Put:
  (t, place) = r
  Table{slots, +mask, +size} = t
  match place:
    case Seeking{i}: (Table{slots, mask, size}, Full{})
    case Present{i, previous}:
      (Table{Array.set(Slot, slots, i, Entry{key, value}), mask, size}, Replaced{previous})
    case Vacant{i}:
      insert(Table{slots, mask, size}, i, key, value,
        U32.is_ge(size, U32.shr(U32.inc(mask))))

# Search before capacity check: replacement must still succeed at the entry limit.
def put(t: Table, key: U64, value: U32) -> Table & Put:
  +k = key
  put_at(k, value, locate(t, k))

# Backward-shift deletion, including clusters crossing the last bucket.
# hole and cursor are bucket indices, not key/value sentinels.
type Shifting is Type:
  Shifting{slots: Array<Slot>, hole: U32, cursor: U32, active: Bool}

def shifted(+hole: U32, +cursor: U32, mask: U32, +entry: Slot,
  slots: Array<Slot>, move: Bool) -> Shifting:
  match move:
    case False{}: Shifting{slots, hole, next(cursor, mask), True{}}
    case True{}:
      Shifting{Array.set(Slot, Array.set(Slot, slots, hole, entry), cursor, Empty{}),
        cursor, next(cursor, mask), True{}}

def shift_seen(+hole: U32, +cursor: U32, +mask: U32, r: Array<Slot> & Slot) -> Shifting:
  (slots, slot) = r
  match slot:
    case Empty{}: Shifting{slots, hole, cursor, False{}}
    case Entry{+key, value}:
      +home = bucket(key, mask)
      shifted(hole, cursor, mask, Entry{key, value}, slots,
        U32.is_lt(U32.and(U32.sub(hole, home), mask), U32.and(U32.sub(cursor, home), mask)))

def shift(fuel: Nat, +mask: U32, state: Shifting) -> Array<Slot>:
  match fuel state:
    case 0n Shifting{slots, h, c, active}: slots
    case 1n+p Shifting{slots, h, c, False{}}: slots
    case 1n+p Shifting{slots, h, +c, True{}}:
      shift(p, mask, shift_seen(h, c, mask, Array.get(Slot, slots, c)))

def erased(r: Table & Place) -> Table & Maybe<&2, U32>:
  (t, place) = r
  Table{slots, +mask, size} = t
  match place:
    case Vacant{i}: (Table{slots, mask, size}, None{})
    case Seeking{i}: (Table{slots, mask, size}, None{})
    case Present{+i, previous}:
      (Table{shift(U32.to_nat(U32.inc(mask)), mask,
        Shifting{Array.set(Slot, slots, i, Empty{}), i, next(i, mask), True{}}),
        mask, U32.sub(size, 1)}, Some{previous})

def remove(t: Table, key: U64) -> Table & Maybe<&2, U32>:
  erased(locate(t, key))
