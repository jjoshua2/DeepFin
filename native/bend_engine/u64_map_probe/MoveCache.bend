# Single-owner structural legal-move cache. Never caches draw or neural results.
# Valid orthodox boards and the matching immutable attack tables are required.
import Base
import ../legal_probe/Chess.bend as C
import ./BoardIndex.bend as B
import ./PositionIndex.bend as P

type Entry is Data:
  Entry{moves: List<&2, C.Ply>}

type Cache is Type:
  Cache{index: P.Index, entries: Array<Entry>}

type Access is Data:
  Hit{}
  Filled{}
  Bypassed{}

type Result is Type:
  Result{tables: Array<U64>, cache: Cache, moves: List<&2, C.Ply>, access: Access}

def created(bits: U32, result: Maybe<&1, P.Index>) -> Maybe<&1, Cache>:
  match result:
    case None{}: None{}
    case Some{index}:
      Some{Cache{index, Array.new(Entry, U32.to_nat(U32.sub(bits, 1)), Entry{Nil{}})}}

# Count is bounded by PositionIndex's identity-record limit. Move-list payload
# storage is additional; this is not a bound expressed in bytes.
def new(+bits: U32) -> Maybe<&1, Cache>:
  created(bits, P.new(bits))

def cached(tables: Array<U64>, index: P.Index, result: Array<Entry> & Entry) -> Result:
  (entries, entry) = result
  Entry{moves} = entry
  Result{tables, Cache{index, entries}, moves, Hit{}}

def filled(index: P.Index, entries: Array<Entry>, id: U32,
  result: Array<U64> & List<&2, C.Ply>) -> Result:
  (tables, +moves) = result
  Result{tables, Cache{index, Array.set(Entry, entries, id, Entry{moves})}, moves, Filled{}}

def bypassed(index: P.Index, entries: Array<Entry>,
  result: Array<U64> & List<&2, C.Ply>) -> Result:
  (tables, moves) = result
  Result{tables, Cache{index, entries}, moves, Bypassed{}}

def resolved(tables: Array<U64>, entries: Array<Entry>, board: C.Board,
  result: P.Index & P.Outcome) -> Result:
  (index, outcome) = result
  match outcome:
    case P.Known{id}: cached(tables, index, Array.get(Entry, entries, id))
    case P.Assigned{id}: filled(index, entries, id, C.legal_moves(tables, board))
    case P.Full{}: bypassed(index, entries, C.legal_moves(tables, board))

# Cached empty lists are hits too. Saturation only disables admission: legal
# generation continues, without eviction or changing already-cached entries.
def legal_moves(tables: Array<U64>, cache: Cache, +board: C.Board) -> Result:
  Cache{index, entries} = cache
  resolved(tables, entries, board, B.intern(index, board))
