# Both endpoints are on the board; reversal swaps pawn color, not king/knight.
import Base
import ./Spec.bend as S
import ../attack_geometry/Geometry.bend as G
import ../attack_geometry/Slots.bend as Slots

law leaper_coordinate_membership_reverses:
  for +kind: G.Leaper
  for +src: Nat
  for +dst: Nat
  for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}
  for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}
  {S.contains(dst,G.targets(kind,src)) == S.contains(src,G.targets(G.reverse(kind),dst)) : Bool}

# Actual computation used by Tables.extras; initializer and array lifetime are separate.
law computed_leaper_mask_reverses:
  for +kind: G.Leaper
  for +src: Nat
  for +dst: Nat
  for source_bound: {Nat.is_lt(src,64n) == True{} : Bool}
  for target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}
  {U64.test_bit(Slots.value(kind,src),dst) ==
    U64.test_bit(Slots.value(G.reverse(kind),dst),src) : Bool}
