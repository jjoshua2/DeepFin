# Non-vacuous theorem use on arbitrary masks and concrete cross-half/high-bit cases.
import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Bits.bend as Bits
import ./Nats.bend as Nats

def zero_for_every_mask(+mask: U64) ->
  {U64.pext(U64.pdep(U64.zero(), mask), mask) == U64.zero() : U64}:
  Laws.compact_roundtrip(U64.zero(), mask, Nats.power_positive(Bits.width(mask)))

def all_masks_have_positive_capacity(mask: U64) ->
  {Nat.is_lt(0n, Bits.capacity(mask)) == True{} : Bool}:
  Nats.power_positive(Bits.width(mask))

def cross_half() ->
  {U64.pext(U64.pdep(U64.from_u32(7), U64.from_parts(2147483649, 1)), U64.from_parts(2147483649, 1))
    == U64.from_u32(7) : U64}:
  Laws.compact_roundtrip(U64.from_u32(7), U64.from_parts(2147483649, 1), {==})

def top_bit() ->
  {U64.pext(U64.pdep(U64.one(), U64.bit(63n)), U64.bit(63n)) == U64.one() : U64}:
  Laws.compact_roundtrip(U64.one(), U64.bit(63n), {==})

# The unbounded reverse identity would be false: compact 1 is out of range
# for an empty mask, whose mathematical capacity is 1.
def out_of_range_counterexample() ->
  {U64.pext(U64.pdep(U64.one(), U64.zero()), U64.zero()) == U64.zero() : U64}:
  {==}
