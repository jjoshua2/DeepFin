# Exact scalar addressing, not an affine-buffer or attack-geometry theorem.
import Base
import ../index/Bits.bend as Bits
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence
import ../../../bitboard_probe/Sliders.bend as Sliders

# Symbolic k includes 32 without forcing a concrete 2^32 unary numeral.
law low_projection_exact:
  for +x: U64
  for +k: Nat
  for retained: {Nat.is_le(k,32n) == True{} : Bool}
  for bound: {Nat.is_lt(Bits.value(x),Bits.power(k)) == True{} : Bool}
  {U32.to_nat(U64.low(x)) == Bits.value(x) : Nat}

law compact_index_exact:
  for +occ: U64
  for +mask: U64
  for small: {Nat.is_le(Bits.width(mask),32n) == True{} : Bool}
  {U32.to_nat(Sliders.pext_index(occ,mask)) == Bits.value(U64.pext(occ,mask)) : Nat}

law chess_lookup_exact:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))) == Bits.value(U64.pext(occ,Layout.mask(key))) : Nat}

law chess_lookup_word:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))) == U64.pext(occ,Layout.mask(key)) : U64}

law chess_sequence_ordinal:
  for +key: U32
  for +i: Nat
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for bounded: {Nat.is_lt(i,Bits.capacity(Layout.mask(key))) == True{} : Bool}
  {U32.to_nat(Sliders.pext_index(Sequence.at(i,Layout.mask(key)),Layout.mask(key))) == i : Nat}

law chess_sequence_coverage:
  for +key: U32
  for +s: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for member: {U64.and(s,Layout.mask(key)) == s : U64}
  {Sequence.at(U32.to_nat(Sliders.pext_index(s,Layout.mask(key))),Layout.mask(key)) == s : U64}

law chess_lookup_injective:
  for +key: U32
  for +a: U64
  for +b: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for am: {U64.and(a,Layout.mask(key)) == a : U64}
  for bm: {U64.and(b,Layout.mask(key)) == b : U64}
  for same: {Sliders.pext_index(a,Layout.mask(key)) == Sliders.pext_index(b,Layout.mask(key)) : U32}
  {a == b : U64}

law chess_lookup_redeposit:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U64.pdep(U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}
