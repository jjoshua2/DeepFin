import Base
import ./Bridge.bend as Bridge
import ../bitboard_inventory/ClearStep.bend as ClearStep

def use_actual_u64_zero_word(+a: U64) ->
  {U64.is_zero(a) == ClearStep.is_zero_word(64n,U64.to_word(a)) : Bool}:
  Bridge.u64_is_zero_word(a)
