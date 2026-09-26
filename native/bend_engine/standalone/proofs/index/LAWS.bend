# Accepted compact-index contracts. Production PEXT/PDEP and Subsets are imported,
# not reimplemented; no law here asserts the carry-rippler's enumeration order.
import Base
import ./Bits.bend as Bits
import ../LAWS.bend as SubsetLaws

law capacity_matches_popcount:
  for +mask: U64
  {Bits.capacity(mask) == Nat.pow(2n, U32.to_nat(U64.popcount(mask))) : Nat}

law extraction_bound:
  for +x: U64
  for +mask: U64
  {Nat.is_lt(Bits.value(U64.pext(x, mask)), Bits.capacity(mask)) == True{} : Bool}

# Without a bound, reverse composition retains exactly the low k bits.
law compact_projection:
  for +x: U64
  for +mask: U64
  {U64.pext(U64.pdep(x, mask), mask)
    == U64.from_word(Bits.clip(64n, Bits.width(mask), U64.to_word(x))) : U64}

law compact_roundtrip:
  for +x: U64
  for +mask: U64
  for bounded: {Nat.is_lt(Bits.value(x), Bits.capacity(mask)) == True{} : Bool}
  {U64.pext(U64.pdep(x, mask), mask) == x : U64}

law deposit_in_mask:
  for +x: U64
  for +mask: U64
  {U64.and(U64.pdep(x, mask), mask) == U64.pdep(x, mask) : U64}

law deposit_injective:
  for +a: U64
  for +b: U64
  for +mask: U64
  for a_bound: {Nat.is_lt(Bits.value(a), Bits.capacity(mask)) == True{} : Bool}
  for b_bound: {Nat.is_lt(Bits.value(b), Bits.capacity(mask)) == True{} : Bool}
  for same: {U64.pdep(a, mask) == U64.pdep(b, mask) : U64}
  {a == b : U64}

law masked_extract_injective:
  for +a: U64
  for +b: U64
  for +mask: U64
  for a_member: {U64.and(a, mask) == a : U64}
  for b_member: {U64.and(b, mask) == b : U64}
  for same: {U64.pext(a, mask) == U64.pext(b, mask) : U64}
  {a == b : U64}

# This is coverage by PDEP, NOT coverage by the carry-rippler's sequence.
law compact_coverage:
  for +x: U64
  for +mask: U64
  for bounded: {Nat.is_lt(Bits.value(x), Bits.capacity(mask)) == True{} : Bool}
  exs subset: U64
  {U64.and(subset, mask) == subset : U64} & {U64.pext(subset, mask) == x : U64}

law sequence_index_bound:
  for n: Nat
  for +mask: U64
  {Nat.is_lt(Bits.value(U64.pext(SubsetLaws.at(n, mask), mask)), Bits.capacity(mask)) == True{} : Bool}
