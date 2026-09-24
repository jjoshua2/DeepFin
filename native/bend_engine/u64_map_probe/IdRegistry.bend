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
