# Benchmark-only control: dense unordered entries, full-key scan, swap-last erase.
# Shares the candidate's exact Table/Slot types and physical allocation.
import Base
import ./U64Map.bend as M

type Finding is Type:
  Finding{slots: Array<M.Slot>, index: U32, value: U32, found: Bool}

def matched(index: U32, value: U32, slots: Array<M.Slot>, equal: Bool) -> Finding:
  match equal:
    case True{}: Finding{slots, index, value, True{}}
    case False{}: Finding{slots, U32.inc(index), 0, False{}}

def seen(key: U64, +index: U32, r: Array<M.Slot> & M.Slot) -> Finding:
  (slots, slot) = r
  match slot:
    case M.Empty{}: Finding{slots, U32.inc(index), 0, False{}}
    case M.Entry{stored, value}:
      matched(index, value, slots, U64.is_eq(key, stored))

def scan(fuel: Nat, +key: U64, state: Finding) -> Finding:
  match fuel state:
    case 0n Finding{a, i, v, f}: Finding{a, i, v, f}
    case 1n+p Finding{a, i, v, True{}}: Finding{a, i, v, True{}}
    case 1n+p Finding{a, +i, v, False{}}:
      scan(p, key, seen(key, i, Array.get(M.Slot, a, i)))

def located(mask: U32, size: U32, state: Finding) -> M.Table & M.Place:
  Finding{slots, +i, v, found} = state
  (M.Table{slots, mask, size}, Bool.pick(M.Place, found, M.Present{i, v}, M.Vacant{i}))

def locate(t: M.Table, key: U64) -> M.Table & M.Place:
  M.Table{slots, mask, +size} = t
  located(mask, size, scan(U32.to_nat(size), key, Finding{slots, 0, 0, False{}}))

def get(t: M.Table, key: U64) -> M.Table & Maybe<&2, U32>:
  M.found(locate(t, key))

def put(t: M.Table, +key: U64, value: U32) -> M.Table & M.Put:
  # M.put_at only interprets found/vacant indices and enforces the shared limit.
  M.put_at(key, value, locate(t, key))

def swapped(index: U32, +last: U32, mask: U32, previous: U32,
  r: Array<M.Slot> & M.Slot) -> M.Table & Maybe<&2, U32>:
  (slots, entry) = r
  (M.Table{Array.set(M.Slot, Array.set(M.Slot, slots, index, entry), last, M.Empty{}),
    mask, last}, Some{previous})

def erased(r: M.Table & M.Place) -> M.Table & Maybe<&2, U32>:
  (t, place) = r
  M.Table{slots, mask, +size} = t
  match place:
    case M.Present{i, value}:
      swapped(i, U32.sub(size, 1), mask, value, Array.get(M.Slot, slots, U32.sub(size, 1)))
    case M.Vacant{i}: (M.Table{slots, mask, size}, None{})
    case M.Seeking{i}: (M.Table{slots, mask, size}, None{})

def remove(t: M.Table, key: U64) -> M.Table & Maybe<&2, U32>:
  erased(locate(t, key))
