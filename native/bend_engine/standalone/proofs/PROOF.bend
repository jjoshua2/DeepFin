import Base
import ../Subsets.bend as Subsets
import ./LAWS.bend as Laws
import ./Mask.bend as Mask
import ./u64/LAWS.bend as U64Laws
import ./u64/PROOF.bend as U64Proof

def Laws.step_in_mask(subset, mask):
  Mask.u64_absorb(Subsets.difference(subset, mask), mask)

def Laws.sequence_in_mask(n, mask):
  match n:
    case 0n: Mask.u64_zero_left(mask)
    case 1n+p: Laws.step_in_mask(Laws.at(p, mask), mask)

def masked_roundtrip(+x: U64, +mask: U64, member: {U64.and(x, mask) == x : U64}) ->
  {U64.pdep(U64.pext(x, mask), mask) == x : U64}:
  Equal.trans(U64, U64.pdep(U64.pext(x, mask), mask), U64.and(x, mask), x,
    U64Laws.pext_pdep_roundtrip(x, mask), member)

def Laws.sequence_extract_deposit(n, mask):
  match n:
    case 0n: masked_roundtrip(U64.zero(), mask, Mask.u64_zero_left(mask))
    case 1n+p:
      +previous = Laws.at(p, mask)
      masked_roundtrip(Subsets.next(previous, mask), mask, Laws.step_in_mask(previous, mask))

def Laws.empty_mask_sequence(n):
  match n:
    case 0n: {==}
    case 1n+p: Mask.u64_zero_right(Subsets.difference(Laws.at(p, U64.zero()), U64.zero()))

def Laws.cross_low_half():
  {==}

def Laws.cross_bit63():
  {==}

def Laws.mask_cycle_end():
  {==}

def Laws.full_word_wrap():
  {==}
