import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ../LAWS.bend as Sequence
import ../index/Bits.bend as Bits
import ../index/LAWS.bend as IndexLaws
import ../index/PROOF.bend as IndexProof
import ../index/Nats.bend as Nats
import ../../Subsets.bend as Subsets

def any_mask_zero(+mask: U64) -> {Bits.value(U64.pext(Sequence.at(0n,mask),mask)) == 0n : Nat}:
  Laws.sequence_ordinal(0n,mask,Nats.power_positive(Bits.width(mask)))

def every_deposit(+x: U64, +mask: U64) ->
  Exists(Nat, i => {Nat.is_lt(i,Bits.capacity(mask)) == True{} : Bool} & {Sequence.at(i,mask) == U64.pdep(x,mask) : U64}):
  Laws.sequence_coverage(U64.pdep(x,mask),mask,IndexLaws.deposit_in_mask(x,mask))

def every_cycle(+mask: U64) -> {Sequence.at(Bits.capacity(mask),mask) == U64.zero() : U64}:
  Laws.cycle_endpoint(mask)

# Tiny closed checks exercise boundaries without enumerating large Nat periods.
def high_bit_first() -> {Subsets.next(U64.zero(),U64.bit(63n)) == U64.bit(63n) : U64}:
  {==}

def full_mask_wrap() -> {Subsets.next(U64.from_parts(4294967295,4294967295),U64.from_parts(4294967295,4294967295)) == U64.zero() : U64}:
  {==}

def strict_bound_counterexample() -> {Bits.value(U64.pext(Sequence.at(1n,U64.zero()),U64.zero())) == 0n : Nat}:
  {==}

# Without membership the successor claim is false: state=2, mask=5.
def membership_counterexample() -> {U64.pext(Subsets.next(U64.from_u32(2),U64.from_u32(5)),U64.from_u32(5)) == U64.from_u32(3) : U64}:
  {==}

def membership_expected() -> {Spec.cycle(Bits.value(U64.pext(U64.from_u32(2),U64.from_u32(5))),Bits.capacity(U64.from_u32(5))) == 1n : Nat}:
  {==}
