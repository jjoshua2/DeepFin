# Bounded structural-identity interning. Hashes select chains, never identity.
# Inputs are already-canonical fields; this module does not validate chess boards.
import Base
import ./U64Map.bend as M

type Identity is Data:
  Identity{pawns: U64, knights: U64, bishops: U64, rooks: U64,
    queens: U64, kings: U64, white: U64, black: U64,
    turn: U32, castling: U32, ep: U32}

def same(a: Identity, b: Identity) -> Bool:
  Identity{ap, an, ab, ar, aq, ak, aw, ax, at, ac, ae} = a
  Identity{bp, bn, bb, br, bq, bk, bw, bx, bt, bc, be} = b
  Bool.and(U64.is_eq(ap, bp), Bool.and(U64.is_eq(an, bn),
    Bool.and(U64.is_eq(ab, bb), Bool.and(U64.is_eq(ar, br),
    Bool.and(U64.is_eq(aq, bq), Bool.and(U64.is_eq(ak, bk),
    Bool.and(U64.is_eq(aw, bw), Bool.and(U64.is_eq(ax, bx),
    Bool.and(U32.is_eq(at, bt), Bool.and(U32.is_eq(ac, bc), U32.is_eq(ae, be)))))))))))

type Cell is Data:
  Empty{}
  Stored{identity: Identity, next: Maybe<&2, U32>}

type Index is Type:
  Index{heads: M.Table, cells: Array<Cell>, size: U32, capacity: U32}

type Outcome is Data:
  Assigned{id: U32}
  Known{id: U32}
  Full{}

def created(+bits: U32, result: Maybe<&1, M.Table>) -> Maybe<&1, Index>:
  match result:
    case None{}: None{}
    case Some{heads}:
      Some{Index{heads, Array.new(Cell, U32.to_nat(U32.sub(bits, 1)), Empty{}),
        0, U32.shln(1, U32.to_nat(U32.sub(bits, 1)))}}

# Exactly half as many identity records as hash buckets, at most 32,768 records.
# Invalid bits reach None before deriving array size or allocating record storage.
def new(+bits: U32) -> Maybe<&1, Index>:
  created(bits, M.new(bits))

def head(place: M.Place) -> Maybe<&2, U32>:
  match place:
    case M.Present{index, value}: Some{value}
    case M.Vacant{index}: None{}
    case M.Seeking{index}: None{}

type Walking is Type:
  Walking{cells: Array<Cell>, cursor: Maybe<&2, U32>, found: Maybe<&2, U32>}

def compared(id: U32, next: Maybe<&2, U32>, cells: Array<Cell>, equal: Bool) -> Walking:
  match equal:
    case True{}: Walking{cells, None{}, Some{id}}
    case False{}: Walking{cells, next, None{}}

def examined(identity: Identity, id: U32, result: Array<Cell> & Cell) -> Walking:
  (cells, cell) = result
  match cell:
    case Empty{}: Walking{cells, None{}, None{}}
    case Stored{stored, next}: compared(id, next, cells, same(identity, stored))

def follow(fuel: Nat, +identity: Identity, walk: Walking) -> Array<Cell> & Maybe<&2, U32>:
  match fuel walk:
    case 0n Walking{cells, cursor, found}: (cells, found)
    case 1n+p Walking{cells, None{}, found}: (cells, found)
    case 1n+p Walking{cells, Some{+id}, found}:
      follow(p, identity, examined(identity, id, Array.get(Cell, cells, id)))

type Located is Type:
  Located{heads: M.Table, cells: Array<Cell>, size: U32, capacity: U32,
    place: M.Place, found: Maybe<&2, U32>}

def located(heads: M.Table, size: U32, capacity: U32, place: M.Place,
  result: Array<Cell> & Maybe<&2, U32>) -> Located:
  (cells, found) = result
  Located{heads, cells, size, capacity, place, found}

def searched(identity: Identity, cells: Array<Cell>, +size: U32, capacity: U32,
  result: M.Table & M.Place) -> Located:
  (heads, +place) = result
  located(heads, size, capacity, place, follow(U32.to_nat(U32.inc(size)), identity, Walking{cells, head(place), None{}}))

def search(index: Index, hash: U64, identity: Identity) -> Located:
  Index{heads, cells, size, capacity} = index
  searched(identity, cells, size, capacity, M.locate(heads, hash))

def found(result: Located) -> Index & Maybe<&2, U32>:
  Located{heads, cells, size, capacity, place, value} = result
  (Index{heads, cells, size, capacity}, value)

def get(index: Index, hash: U64, identity: Identity) -> Index & Maybe<&2, U32>:
  found(search(index, hash, identity))

def stored(identity: Identity, next: Maybe<&2, U32>, cells: Array<Cell>, +size: U32,
  capacity: U32, heads: M.Table) -> Index & Outcome:
  (Index{heads, Array.set(Cell, cells, size, Stored{identity, next}), U32.inc(size), capacity}, Assigned{size})

def published(identity: Identity, next: Maybe<&2, U32>, cells: Array<Cell>, size: U32,
  capacity: U32, result: M.Table & M.Put) -> Index & Outcome:
  (heads, outcome) = result
  match outcome:
    case M.Added{}: stored(identity, next, cells, size, capacity, heads)
    case M.Replaced{old}: stored(identity, next, cells, size, capacity, heads)
    case M.Full{}: (Index{heads, cells, size, capacity}, Full{})

def admit(hash: U64, identity: Identity, heads: M.Table, cells: Array<Cell>, +size: U32,
  capacity: U32, +place: M.Place, room: Bool) -> Index & Outcome:
  match room:
    case False{}: (Index{heads, cells, size, capacity}, Full{})
    case True{}:
      published(identity, head(place), cells, size, capacity, M.put_at(hash, size, (heads, place)))

def interned(hash: U64, identity: Identity, result: Located) -> Index & Outcome:
  Located{heads, cells, +size, +capacity, place, previous} = result
  match previous:
    case Some{id}: (Index{heads, cells, size, capacity}, Known{id})
    case None{}: admit(hash, identity, heads, cells, size, capacity, place, U32.is_lt(size, capacity))

# Dense IDs are scoped to this index lifetime. Hits still work at record capacity.
# Same canonical identity must be supplied with the same deterministic hash.
def intern(index: Index, +hash: U64, +identity: Identity) -> Index & Outcome:
  interned(hash, identity, search(index, hash, identity))

def length(index: Index) -> Index & U32:
  Index{heads, cells, +size, capacity} = index
  (Index{heads, cells, size, capacity}, size)
