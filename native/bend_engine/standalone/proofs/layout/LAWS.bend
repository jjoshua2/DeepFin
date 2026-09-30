# Scalar size/index contracts for the actual 128 chess slider masks.
# These do not claim correctness of affine table writes or attack geometry.
import Base
import ./Spec.bend as Spec
import ./Geometry.bend as Geometry
import ../index/Bits.bend as Bits
import ../../../bitboard_probe/Sliders.bend as Sliders

law mask_population:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}

law tight_population_bound:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.is_le(U64.popcount(Spec.mask(key)),Spec.max_population(key)) == True{} : Bool}

law shift_in_range:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.is_lt(U64.popcount(Spec.mask(key)),32) == True{} : Bool}

law size_positive:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.is_gt(Spec.size(key),0) == True{} : Bool}

law tight_size_bound:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.is_le(Spec.size(key),Spec.max_size(key)) == True{} : Bool}

law size_matches_capacity:
  for +key: U32
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {U32.to_nat(Spec.size(key)) == Bits.capacity(Spec.mask(key)) : Nat}

law full_index_bound:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {Nat.is_lt(Bits.value(U64.pext(occ,Spec.mask(key))),U32.to_nat(Spec.size(key))) == True{} : Bool}

law lookup_index_bound:
  for +key: U32
  for +occ: U64
  for +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}
  {Nat.is_lt(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),U32.to_nat(Spec.size(key))) == True{} : Bool}
