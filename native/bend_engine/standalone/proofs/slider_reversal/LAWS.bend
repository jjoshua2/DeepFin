# One shared arbitrary occupancy; only the strict segment interior can block.
import Base
import ./Spec.bend as S

law slider_geometric_membership_reverses:
  for +kind: S.Slider
  for +src: Nat
  for +dst: Nat
  for +occ: U64
  for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}
  for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}
  {U64.test_bit(S.geometry(kind,src,occ),dst) == U64.test_bit(S.geometry(kind,dst,occ),src) : Bool}

law computed_slider_mask_reverses:
  for +kind: S.Slider
  for +src: Nat
  for +dst: Nat
  for +occ: U64
  for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}
  for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}
  {U64.test_bit(S.computed(kind,src,occ),dst) == U64.test_bit(S.computed(kind,dst,occ),src) : Bool}
