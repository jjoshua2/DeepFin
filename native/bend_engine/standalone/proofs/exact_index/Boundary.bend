# Small independent source witnesses for the runtime function itself.
# No layout laws or PROOF imports: mutation failures here cannot be old proofs.
import Base
import ../../../bitboard_probe/Sliders.bend as Sliders

def high_bit_to_first() -> {Sliders.pext_index(U64.bit(63n),U64.bit(63n)) == 1 : U32}:
  {==}

def cross_half_to_second() ->
  {Sliders.pext_index(U64.bit(32n),U64.or(U64.bit(31n),U64.bit(32n))) == 2 : U32}:
  {==}

def third_compact_bit() ->
  {Sliders.pext_index(U64.bit(63n),U64.or(U64.bit(0n),U64.or(U64.bit(32n),U64.bit(63n)))) == 4 : U32}:
  {==}

def thirty_two_compact_bits() ->
  {Sliders.pext_index(U64.from_parts(4294967295,0),U64.from_parts(4294967295,0)) == 4294967295 : U32}:
  {==}

# A >32-bit mask can lose relevant information: do not generalize the inverse.
def wide_mask_collision() ->
  {Sliders.pext_index(U64.bit(32n),U64.from_parts(1,4294967295)) == 0 : U32}:
  {==}

def wide_mask_still_relevant() ->
  {U64.and(U64.bit(32n),U64.from_parts(1,4294967295)) == U64.bit(32n) : U64}:
  {==}
