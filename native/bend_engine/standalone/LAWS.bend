# Contracts for the production table builder and lookup, not a second engine.
import Base
import ../bitboard_probe/Sliders.bend as Sliders
import ./Tables.bend as Tables
import ./proofs/Spec.bend as Spec

law index_ignores_off_mask:
  for +occupied: U64
  for +mask: U64
  {Sliders.pext_index(U64.and(occupied, mask), mask) == Sliders.pext_index(occupied, mask) : U32}

# This precondition is explicit; truncating an unrestricted mask is not safe.
law index_full_width:
  for +occupied: U64
  for +mask: U64
  for high_zero: {U64.high(U64.pext(occupied, mask)) == 0 : U32}
  {U64.from_u32(Sliders.pext_index(occupied, mask)) == U64.pext(occupied, mask) : U64}

law index_recovers_masked_input:
  for +occupied: U64
  for +mask: U64
  for high_zero: {U64.high(U64.pext(occupied, mask)) == 0 : U32}
  {U64.pdep(U64.from_u32(Sliders.pext_index(occupied, mask)), mask) == U64.and(occupied, mask) : U64}

# Reflection may run faster, but must certify exactly the original conditions.
law reflection_preserves_conditions:
  for +mask: U64
  for +subset: U64
  for +index: U32
  for +at: U32
  for +offset: U32
  for +size: U32
  {Spec.production_step_ok(mask, subset, index, at, offset, size) == Spec.step_ok(mask, subset, index, at, offset, size) : Bool}

# Generic induction over the ACTUAL Array.set loop, for any incoming array.
law fill_matches_lookup:
  for n: Nat
  for +at: U32
  for +square: U32
  for +bishop: Bool
  for +mask: U64
  for +subset: U64
  for +index: U32
  for +offset: U32
  for +size: U32
  for -a: Array<U64>
  for checked: {Spec.walk(n, mask, subset, index, at, offset, size) == True{} : Bool}
  {Tables.fill(n, at, square, bishop, mask, subset, a) == Spec.indexed_fill(n, offset, square, bishop, mask, subset, a) : Array<U64>}

law every_visited_slot_is_safe:
  for n: Nat
  for skip: Nat
  for +mask: U64
  for +subset: U64
  for +index: U32
  for +at: U32
  for +offset: U32
  for +size: U32
  for within: {Nat.is_lt(skip, n) == True{} : Bool}
  for checked: {Spec.walk(n, mask, subset, index, at, offset, size) == True{} : Bool}
  {Spec.slot_at(skip, mask, subset, index, at, offset, size) == True{} : Bool}

# Closed computational proof over every canonical segment and occupancy subset.
# Includes the real masks, count/segment/index/address bounds, and exact wrap.
law canonical_tables_checked:
  Spec.CheckedTables<128n, 0, 512>

law every_canonical_segment_is_safe:
  for +skip: Nat
  for within: {Nat.is_lt(skip, 128n) == True{} : Bool}
  {Spec.segment_at(skip, 0, 512) == True{} : Bool}

# Universal theorem over the production metadata/table loop, for any array.
# Specialize it with canonical_tables_checked; no generated table is trusted.
law tables_match_lookup:
  for n: Nat
  for +key: U32
  for +at: U32
  for -a: Array<U64>
  for checked: Spec.CheckedTables<n, key, at>
  {Tables.tables(n, key, at, a) == Spec.indexed_tables(n, key, at, a) : Array<U64>}

# Also connect the specification's scalar address to the actual reader.
law lookup_uses_checked_address:
  for -a: Array<U64>
  for +offset: U32
  for +index: U32
  {Sliders.lookup(a, offset, index) == Array.get(U64, a, Spec.address(offset, index)) : Array<U64> & U64}
