# Satisfiable public uses and explicit invalid-domain counterexamples.
import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../index/Bits.bend as Bits
import ../index/Nats.bend as Nats
import ../layout/Spec.bend as Layout
import ../layout/LAWS.bend as LayoutLaws
import ../layout/PROOF.bend as LayoutProof
import ../LAWS.bend as Sequence
import ../../../bitboard_probe/Sliders.bend as Sliders

# This consumer quantifies over arbitrary occupancies, not one sampled position.
def every_key(+key: U32, +occ: U64,
  valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  {U64.pdep(U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}:
  Laws.chess_lookup_redeposit(key,occ,valid)

def first_index(+key: U32, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  {U32.to_nat(Sliders.pext_index(Sequence.at(0n,Layout.mask(key)),Layout.mask(key))) == 0n : Nat}:
  Laws.chess_sequence_ordinal(key,0n,valid,Nats.power_positive(Bits.width(Layout.mask(key))))

# Empty masks are a valid instance of the general <=32-bit compact theorem.
def empty_mask(+occ: U64) ->
  {U32.to_nat(Sliders.pext_index(occ,U64.zero())) == Bits.value(U64.pext(occ,U64.zero())) : Nat}:
  Laws.compact_index_exact(occ,U64.zero(),{==})

# High physical bit, small compact index. Cross-half source observations stay small.
def bit63() -> {Sliders.pext_index(U64.bit(63n),U64.bit(63n)) == 1 : U32}:
  {==}

def cross_halves() -> {Sliders.pext_index(U64.bit(32n),U64.or(U64.bit(31n),U64.bit(32n))) == 2 : U32}:
  {==}

# Small symbolic-width use of the generic projection theorem.
def low_three() -> {U32.to_nat(U64.low(U64.from_u32(7))) == Bits.value(U64.from_u32(7)) : Nat}:
  Laws.low_projection_exact(U64.from_u32(7),3n,{==},{==})

# 33 selected bits expose the reason for the general compact-width precondition.
# Compare U64 words, not enormous normalized Nat numerals.
def overwide_counterexample() ->
  {U64.pext(U64.bit(32n),U64.from_parts(1,4294967295)) == U64.from_parts(1,0) : U64}:
  {==}

def overwide_truncates() ->
  {Sliders.pext_index(U64.bit(32n),U64.from_parts(1,4294967295)) == 0 : U32}:
  {==}

# Mask membership is essential for recovering the original occupancy rather
# than only its projection. A rook-a1 mask excludes its starting square.
def outside_mask() -> {U64.and(U64.bit(0n),Layout.mask(0)) == U64.zero() : U64}:
  {==}
