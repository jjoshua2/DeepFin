# Append-only numeric-key interning within one registry lifetime.
# No deletion/reset, cross-registry identity, node allocation or neural caching.
import Base
import ./U64Map.bend as M
import ./InsertOnce.bend as I

type Registry is Type:
  Registry{table: M.Table, next_id: Maybe<&2, U32>}

type Outcome is Data:
  Assigned{id: U32}
  Known{id: U32}
  TableFull{}
  IdsExhausted{}

type Info is Data:
  Info{size: U32, next_id: Maybe<&2, U32>}

def created(first: U32, t: Maybe<&1, M.Table>) -> Maybe<&1, Registry>:
  match t:
    case None{}: None{}
    case Some{table}: Some{Registry{table, Some{first}}}

# first is the first ID to issue, not a record restored from an existing table.
def new(bits: U32, first: U32) -> Maybe<&1, Registry>:
  created(first, M.new(bits))

def after_id(id: U32, last: Bool) -> Maybe<&2, U32>:
  match last:
    case True{}: None{}
    case False{}: Some{U32.inc(id)}

def admitted(+candidate: U32, r: M.Table & I.Outcome) -> Registry & Outcome:
  (t, outcome) = r
  match outcome:
    case I.Inserted{+id}:
      (Registry{t, after_id(id, U32.is_eq(id, 4294967295))}, Assigned{id})
    case I.Existing{id}: (Registry{t, Some{candidate}}, Known{id})
    case I.Saturated{}: (Registry{t, Some{candidate}}, TableFull{})

def exhausted(r: M.Table & Maybe<&2, U32>) -> Registry & Outcome:
  (t, found) = r
  match found:
    case Some{id}: (Registry{t, None{}}, Known{id})
    case None{}: (Registry{t, None{}}, IdsExhausted{})

# Membership is still queried after ID exhaustion, so existing bindings work.
# At simultaneous map/ID limits a missing key reports IdsExhausted.
def intern(r: Registry, key: U64) -> Registry & Outcome:
  Registry{t, next_id} = r
  match next_id:
    case Some{+id}: admitted(id, I.get_or_insert(t, key, id))
    case None{}: exhausted(M.get(t, key))

def looked_up(next_id: Maybe<&2, U32>, r: M.Table & Maybe<&2, U32>) -> Registry & Maybe<&2, U32>:
  (t, found) = r
  (Registry{t, next_id}, found)

def get(r: Registry, key: U64) -> Registry & Maybe<&2, U32>:
  Registry{t, next_id} = r
  looked_up(next_id, M.get(t, key))

def inspected(+next_id: Maybe<&2, U32>, r: M.Table & U32) -> Registry & Info:
  (t, size) = r
  (Registry{t, next_id}, Info{size, next_id})

def inspect(r: Registry) -> Registry & Info:
  Registry{t, next_id} = r
  inspected(next_id, M.length(t))

# Two-phase admission for a caller that may fail while preparing external data.
# A reservation owns the table; no other operation can invalidate its vacant slot.
type Reservation is Type:
  Reservation{table: M.Table, key: U64, index: U32, id: U32}

type Preparation is Type:
  Prepared{reservation: Reservation}
  Settled{registry: Registry, outcome: Outcome}

def reserve_limit(t: M.Table, key: U64, index: U32, id: U32, full: Bool) -> Preparation:
  match full:
    case True{}: Settled{Registry{t, Some{id}}, TableFull{}}
    case False{}: Prepared{Reservation{t, key, index, id}}

def reserve_vacant(t: M.Table, key: U64, index: U32, id: U32) -> Preparation:
  M.Table{slots, +mask, +size} = t
  reserve_limit(M.Table{slots, mask, size}, key, index, id,
    U32.is_ge(size, U32.shr(U32.inc(mask))))

def reserve_found(id: U32, key: U64, result: M.Table & M.Place) -> Preparation:
  (t, place) = result
  match place:
    case M.Present{index, previous}: Settled{Registry{t, Some{id}}, Known{previous}}
    case M.Seeking{index}: Settled{Registry{t, Some{id}}, TableFull{}}
    case M.Vacant{index}: reserve_vacant(t, key, index, id)

def settled(result: Registry & Outcome) -> Preparation:
  (r, outcome) = result
  Settled{r, outcome}

# Prepared has not inserted the key or consumed the ID. Settled never assigns.
def reserve(r: Registry, +key: U64) -> Preparation:
  Registry{t, next_id} = r
  match next_id:
    case Some{id}: reserve_found(id, key, M.locate(t, key))
    case None{}: settled(exhausted(M.get(t, key)))

# This ID is provisional until commit. Abort permits it to be proposed again.
def candidate(r: Reservation) -> Reservation & U32:
  Reservation{t, key, index, +id} = r
  (Reservation{t, key, index, id}, id)

def abort(r: Reservation) -> Registry:
  Reservation{t, key, index, +id} = r
  Registry{t, Some{id}}

# No new lookup. The owned vacant slot cannot go stale through the public API.
# With a valid reservation this returns Assigned, exactly once.
def commit(r: Reservation) -> Registry & Outcome:
  Reservation{t, key, index, +id} = r
  admitted(id, I.stored(id, M.put_at(key, id, (t, M.Vacant{index}))))
