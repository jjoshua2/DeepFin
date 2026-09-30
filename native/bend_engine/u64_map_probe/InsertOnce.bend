# Lookup or insert, never overwrite. Single-owner operation, not a concurrent API.
# Use only a valid U64Map.Table, not the different ScanMap placement invariant.
import Base
import ./U64Map.bend as M

type Outcome is Data:
  Inserted{value: U32}
  Existing{value: U32}
  Saturated{}

# put_at receives the already-found vacant slot: no second hash/probe walk.
def stored(candidate: U32, r: M.Table & M.Put) -> M.Table & Outcome:
  (t, result) = r
  match result:
    case M.Added{}: (t, Inserted{candidate})
    case M.Replaced{previous}: (t, Existing{previous})
    case M.Full{}: (t, Saturated{})

def resolved(+key: U64, +candidate: U32, r: M.Table & M.Place) -> M.Table & Outcome:
  (t, place) = r
  match place:
    case M.Present{index, previous}: (t, Existing{previous})
    case M.Seeking{index}: (t, Saturated{})
    case M.Vacant{index}:
      stored(candidate, M.put_at(key, candidate, (t, M.Vacant{index})))

# Inserted returns the proposed value. Existing returns the stored value, even
# at capacity; Saturated means a missing key could not be admitted. Neither
# Existing nor Saturated changes the logical dictionary. All U32 values work.
# Callers allocating IDs advance their counter only for Inserted. Identity,
# counter exhaustion, collision handling and ownership of external data remain
# caller responsibilities; a U64 match alone is not a neural-cache identity.
def get_or_insert(t: M.Table, +key: U64, candidate: U32) -> M.Table & Outcome:
  resolved(key, candidate, M.locate(t, key))
