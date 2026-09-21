import Base
import ./Bits.bend as Bits
import ./Fast.bend as Fast
import ./Spec.bend as Spec
import ../../bitboard_probe/Sliders.bend as Sliders
import ../Tables.bend as Tables

# Prove that reflection checks the exact original eight-condition predicate.
# This includes the real Sliders.pext_index, not only a model of its result.
def step_agrees(+mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32) ->
  {Spec.production_step_ok(mask, subset, i, at, offset, size) == Spec.step_ok(mask, subset, i, at, offset, size) : Bool}:
  Equal.cong(U64, Bool, w => Spec.step_values(w, i, at, offset, size),
    U64.pext(subset, mask), Fast.pext(subset, mask), Fast.u64_agrees(subset, mask))

# The first conjunct ties the walk certificate to the real lookup's address.
def step_address(+mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32,
  e: {Spec.step_ok(mask, subset, i, at, offset, size) == True{} : Bool}) ->
  {Spec.address(offset, Sliders.pext_index(subset, mask)) == at : U32}:
  original = Equal.trans(Bool, Spec.production_step_ok(mask, subset, i, at, offset, size),
    Spec.step_ok(mask, subset, i, at, offset, size), True{},
    step_agrees(mask, subset, i, at, offset, size), e)
  Bits.u32_eq(Spec.address(offset, Sliders.pext_index(subset, mask)), at,
    Bits.and_left(U32.is_eq(Spec.address(offset, Sliders.pext_index(subset, mask)), at),
      Bool.and(U32.is_zero(U64.high(U64.pext(subset, mask))),
        Bool.and(U32.is_eq(Sliders.pext_index(subset, mask), i),
          Bool.and(U32.is_lt(i, size),
            Bool.and(U32.is_zero(U64.high(U64.add(U64.from_u32(offset), U64.from_u32(i)))),
              Bool.and(U32.is_eq(U64.low(U64.add(U64.from_u32(offset), U64.from_u32(i))), at),
                Bool.and(U32.is_ge(at, 512), U32.is_lt(at, 108160))))))), original))

def fill_step(+p: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +offset: U32,
  -a: Array<U64>, addr: {Spec.address(offset, Sliders.pext_index(subset, mask)) == at : U32},
  ih: {Tables.fill(p, U32.inc(at), sq, bishop, mask, Spec.next_subset(subset, mask), Array.set(U64, a, at, Tables.slider(sq, bishop, subset, False{})))
    == Spec.indexed_fill(p, offset, sq, bishop, mask, Spec.next_subset(subset, mask), Array.set(U64, a, at, Tables.slider(sq, bishop, subset, False{}))) : Array<U64>}) ->
  {Tables.fill(1n+p, at, sq, bishop, mask, subset, a) == Spec.indexed_fill(1n+p, offset, sq, bishop, mask, subset, a) : Array<U64>}:
  %Equal.sym(U32, Spec.address(offset, Sliders.pext_index(subset, mask)), at, addr) :
    {Tables.fill(1n+p, at, sq, bishop, mask, subset, a)
      == Spec.indexed_fill(p, offset, sq, bishop, mask, Spec.next_subset(subset, mask), Array.set(U64, a, _, Tables.slider(sq, bishop, subset, False{}))) : Array<U64>}
  ih

def fill_agrees(n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +i: U32, +offset: U32, +size: U32,
  -a: Array<U64>, e: {Spec.walk(n, mask, subset, i, at, offset, size) == True{} : Bool}) ->
  {Tables.fill(n, at, sq, bishop, mask, subset, a) == Spec.indexed_fill(n, offset, sq, bishop, mask, subset, a) : Array<U64>}:
  match n:
    case 0n: {==}
    case 1n++p:
      Bits.and_elim(
        {Tables.fill(1n+p, at, sq, bishop, mask, subset, a) == Spec.indexed_fill(1n+p, offset, sq, bishop, mask, subset, a) : Array<U64>},
        Spec.step_ok(mask, subset, i, at, offset, size),
        Spec.walk(p, mask, Spec.next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size), e,
        head => tail => fill_step(p, at, sq, bishop, mask, subset, offset, a,
          step_address(mask, subset, i, at, offset, size, head),
          fill_agrees(p, U32.inc(at), sq, bishop, mask, Spec.next_subset(subset, mask), U32.inc(i), offset, size,
            Array.set(U64, a, at, Tables.slider(sq, bishop, subset, False{})), tail)))

# Lift the proved fill equivalence through the ACTUAL metadata/table loop.
def tables_step(+p: Nat, +key: U32, +at: U32, -a: Array<U64>,
  filled: {Spec.segment(key, at, a) == Spec.indexed_segment(key, at, a) : Array<U64>},
  ih: {Tables.tables(p, U32.inc(key), U32.add(at, Spec.segment_size(Spec.slider_mask(key))), Spec.segment(key, at, a))
    == Spec.indexed_tables(p, U32.inc(key), U32.add(at, Spec.segment_size(Spec.slider_mask(key))), Spec.segment(key, at, a)) : Array<U64>}) ->
  {Tables.tables(1n+p, key, at, a) == Spec.indexed_tables(1n+p, key, at, a) : Array<U64>}:
  %filled : {Tables.tables(1n+p, key, at, a)
    == Spec.indexed_tables(p, U32.inc(key), U32.add(at, Spec.segment_size(Spec.slider_mask(key))), _) : Array<U64>}
  ih

def tables_agrees(n: Nat, +key: U32, +at: U32, -a: Array<U64>, e: {Spec.tables(n, key, at) == True{} : Bool}) ->
  {Tables.tables(n, key, at, a) == Spec.indexed_tables(n, key, at, a) : Array<U64>}:
  match n:
    case 0n: {==}
    case 1n++p:
      +mask = Spec.slider_mask(key)
      +size = Spec.segment_size(mask)
      Bits.and_elim(
        {Tables.tables(1n+p, key, at, a) == Spec.indexed_tables(1n+p, key, at, a) : Array<U64>},
        Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
        Spec.tables(p, U32.inc(key), U32.add(at, size)),
        Bits.and_right(U32.is_le(U64.popcount(mask), 12),
          Bool.and(Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
            Spec.tables(p, U32.inc(key), U32.add(at, size))), e),
        head => tail => tables_step(p, key, at, a,
          fill_agrees(U32.to_nat(size), at, U32.mod(key, 64), U32.is_ge(key, 64), mask, U64.zero(), 0, at, size, Spec.metadata(key, at, a), head),
          tables_agrees(p, U32.inc(key), U32.add(at, size), Spec.segment(key, at, a), tail)))

# Every ordinal below the walk length inherits all eight slot conditions.
def walk_slot(n: Nat, skip: Nat, +mask: U64, +subset: U64, +i: U32, +at: U32, +offset: U32, +size: U32,
  within: {Nat.is_lt(skip, n) == True{} : Bool}, e: {Spec.walk(n, mask, subset, i, at, offset, size) == True{} : Bool}) ->
  {Spec.slot_at(skip, mask, subset, i, at, offset, size) == True{} : Bool}:
  match n:
    case 0n:
      Bits.not_lt_zero({Spec.slot_at(skip, mask, subset, i, at, offset, size) == True{} : Bool}, skip, within)
    case 1n++p:
      match skip:
        case 0n:
          Bits.and_left(Spec.step_ok(mask, subset, i, at, offset, size),
            Spec.walk(p, mask, Spec.next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size), e)
        case 1n+s:
          walk_slot(p, s, mask, Spec.next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size, within,
            Bits.and_right(Spec.step_ok(mask, subset, i, at, offset, size),
              Spec.walk(p, mask, Spec.next_subset(subset, mask), U32.inc(i), U32.inc(at), offset, size), e))

# Every canonical segment inherits its count bound and complete walk certificate.
def table_slot(n: Nat, +skip: Nat, +key: U32, +at: U32,
  within: {Nat.is_lt(skip, n) == True{} : Bool}, e: {Spec.tables(n, key, at) == True{} : Bool}) ->
  {Spec.segment_at(skip, key, at) == True{} : Bool}:
  match n:
    case 0n:
      Bits.not_lt_zero({Spec.segment_at(skip, key, at) == True{} : Bool}, skip, within)
    case 1n++p:
      match skip:
        case 0n:
          +mask = Spec.slider_mask(key)
          +size = Spec.segment_size(mask)
          Bits.and_elim({Spec.segment_ok(key, at) == True{} : Bool},
            U32.is_le(U64.popcount(mask), 12),
            Bool.and(Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
              Spec.tables(p, U32.inc(key), U32.add(at, size))), e,
            count => rest => Bits.and_intro(U32.is_le(U64.popcount(mask), 12),
              Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size), count,
              Bits.and_left(Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
                Spec.tables(p, U32.inc(key), U32.add(at, size)), rest)))
        case 1n+s:
          +mask = Spec.slider_mask(key)
          +size = Spec.segment_size(mask)
          table_slot(p, s, U32.inc(key), U32.add(at, size), within,
            Bits.and_right(Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
              Spec.tables(p, U32.inc(key), U32.add(at, size)),
              Bits.and_right(U32.is_le(U64.popcount(mask), 12),
                Bool.and(Spec.walk(U32.to_nat(size), mask, U64.zero(), 0, at, at, size),
                  Spec.tables(p, U32.inc(key), U32.add(at, size))), e)))

# Unpack the certificate while n/key/at remain generic. At a canonical call,
# the checker compares the small indexed type instead of re-running tables.
def certified_table_slot(+n: Nat, +skip: Nat, +key: U32, +at: U32,
  within: {Nat.is_lt(skip, n) == True{} : Bool}, certificate: Spec.CheckedTables<n, key, at>) ->
  {Spec.segment_at(skip, key, at) == True{} : Bool}:
  match certificate:
    case Spec.TableCertificate{checked}:
      table_slot(n, skip, key, at, within, checked)

def certified_tables(+n: Nat, +key: U32, +at: U32, -a: Array<U64>,
  certificate: Spec.CheckedTables<n, key, at>) ->
  {Tables.tables(n, key, at, a) == Spec.indexed_tables(n, key, at, a) : Array<U64>}:
  match certificate:
    case Spec.TableCertificate{checked}:
      tables_agrees(n, key, at, a, checked)
