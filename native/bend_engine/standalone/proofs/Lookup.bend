import Base
import ../../bitboard_probe/Sliders.bend as Sliders
import ./Spec.bend as Spec

# No reader refactoring: check the actual Array.get call definitionally.
def address_agrees(-a: Array<U64>, +offset: U32, +index: U32) ->
  {Sliders.lookup(a, offset, index) == Array.get(U64, a, Spec.address(offset, index)) : Array<U64> & U64}:
  {==}
