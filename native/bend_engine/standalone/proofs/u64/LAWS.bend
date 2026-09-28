# Public U64 contracts. PROOF.bend must discharge every claim.
import Base

# Extract then deposit recovers exactly the masked input, for all 128 input bits.
law pext_pdep_roundtrip:
  for +x: U64
  for +mask: U64
  {U64.pdep(U64.pext(x, mask), mask) == U64.and(x, mask) : U64}

# No overflow or bogus upper-half bits may inflate a population count.
law popcount_bound:
  for +x: U64
  {U32.is_le(U64.popcount(x), 64) == True{} : Bool}

law popcount_zero:
  {U64.popcount(U64.zero()) == 0 : U32}

law popcount_full:
  {U64.popcount(U64.from_parts(4294967295, 4294967295)) == 64 : U32}

law popcount_bit63:
  {U64.popcount(U64.bit(63n)) == 1 : U32}

# Packing/unpacking is bijective at the source level, including bit 63.
law word_roundtrip:
  for x: U64
  {U64.from_word(U64.to_word(x)) == x : U64}

law word_fullwidth:
  for w: Word(64n)
  {U64.to_word(U64.from_word(w)) == w : Word(64n)}

law parts_roundtrip:
  for +x: U64
  {U64.from_parts(U64.high(x), U64.low(x)) == x : U64}

law from_parts_low:
  for hi: U32
  for lo: U32
  {U64.low(U64.from_parts(hi, lo)) == lo : U32}

law from_parts_high:
  for hi: U32
  for lo: U32
  {U64.high(U64.from_parts(hi, lo)) == hi : U32}

law widening_low:
  for x: U32
  {U64.low(U64.from_u32(x)) == x : U32}

law widening_high_zero:
  for x: U32
  {U64.high(U64.from_u32(x)) == 0 : U32}

law pext_zero_mask:
  for +x: U64
  {U64.pext(x, U64.zero()) == U64.zero() : U64}

law pdep_zero_mask:
  for +x: U64
  {U64.pdep(x, U64.zero()) == U64.zero() : U64}

# Undoing the same bitboard/XOR-hash update restores the original.
law xor_twice:
  for +x: U64
  for +mask: U64
  {U64.xor(U64.xor(x, mask), mask) == x : U64}

law not_twice:
  for +x: U64
  {U64.not(U64.not(x)) == x : U64}
