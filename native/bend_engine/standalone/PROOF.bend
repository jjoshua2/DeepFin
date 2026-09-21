import Base
import ./LAWS.bend as Laws
import ./proofs/Bits.bend as Bits
import ./proofs/Core.bend as Core
import ./proofs/Lookup.bend as Lookup
import ./proofs/Spec.bend as Spec
import ./proofs/u64/LAWS.bend as U64Laws
import ./proofs/u64/PROOF.bend as U64Proof
import ../bitboard_probe/Sliders.bend as Sliders
import ./Tables.bend as Tables

def Laws.index_ignores_off_mask(occupied, mask):
  %Bits.u64_pext_masked(occupied, mask) : {Sliders.pext_index(U64.and(occupied, mask), mask) == U64.low(_) : U32}
  {==}

def Laws.index_full_width(occupied, mask, high_zero):
  Bits.zero_high(U64.pext(occupied, mask), high_zero)

def Laws.index_recovers_masked_input(occupied, mask, high_zero):
  %Equal.sym(U64, U64.from_u32(Sliders.pext_index(occupied, mask)), U64.pext(occupied, mask), Laws.index_full_width(occupied, mask, high_zero)) : {U64.pdep(_, mask) == U64.and(occupied, mask) : U64}
  U64Laws.pext_pdep_roundtrip(occupied, mask)

def Laws.reflection_preserves_conditions(mask, subset, index, at, offset, size):
  Core.step_agrees(mask, subset, index, at, offset, size)

def Laws.fill_matches_lookup(n, at, square, bishop, mask, subset, index, offset, size, a, checked):
  Core.fill_agrees(n, at, square, bishop, mask, subset, index, offset, size, a, checked)

def Laws.every_visited_slot_is_safe(n, skip, mask, subset, index, at, offset, size, within, checked):
  Core.walk_slot(n, skip, mask, subset, index, at, offset, size, within, checked)

def Laws.canonical_tables_checked():
  Spec.TableCertificate{{==}}

def Laws.every_canonical_segment_is_safe(skip, within):
  Core.certified_table_slot(128n, skip, 0, 512, within, Laws.canonical_tables_checked())

def Laws.tables_match_lookup(n, key, at, a, checked):
  Core.certified_tables(n, key, at, a, checked)

def Laws.lookup_uses_checked_address(a, offset, index):
  Lookup.address_agrees(a, offset, index)
