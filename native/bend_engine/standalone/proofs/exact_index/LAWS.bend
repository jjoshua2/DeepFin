# Exact scalar correspondence, not a statement about Array storage or ray values.
import Base
import ../index/Bits.bend as Bits
import ../layout/Spec.bend as Spec
import ../LAWS.bend as Sequence
import ../../../bitboard_probe/Sliders.bend as Sliders

law low_projection_exact:
  for +x: U64
  for +k: Nat
  for width: {Nat.is_le(k,32n) == True{} : Bool}
  for bounded: {Nat.is_lt(Bits.value(x),Bits.power(k)) == True{} : Bool}
  {U32.to_nat(U64.low(x)) == Bits.value(x) : Nat}

law lookup_index_exact:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))) == Bits.value(U64.pext(occ,Spec.mask(key))) : Nat}

law lookup_sequence_ordinal:
  for +key: U32
  for +i: Nat
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for bounded: {Nat.is_lt(i,U32.to_nat(Spec.size(key))) == True{} : Bool}
  {U32.to_nat(Sliders.pext_index(Sequence.at(i,Spec.mask(key)),Spec.mask(key))) == i : Nat}

law lookup_collision_sound:
  for +key: U32
  for +a: U64
  for +b: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  for same: {Sliders.pext_index(a,Spec.mask(key)) == Sliders.pext_index(b,Spec.mask(key)) : U32}
  {U64.and(a,Spec.mask(key)) == U64.and(b,Spec.mask(key)) : U64}

# This direction is valid for any U64 mask, even populations above 32.
law lookup_collision_complete:
  for +a: U64
  for +b: U64
  for +mask: U64
  for same: {U64.and(a,mask) == U64.and(b,mask) : U64}
  {Sliders.pext_index(a,mask) == Sliders.pext_index(b,mask) : U32}

law lookup_state_recovery:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),Spec.mask(key))
    == U64.and(occ,Spec.mask(key)) : U64}
