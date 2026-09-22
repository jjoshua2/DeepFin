import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Boundary.bend as Boundary
import ../index/Bits.bend as Bits
import ../layout/Spec.bend as Spec
import ../layout/LAWS.bend as Layout
import ../layout/PROOF.bend as LayoutProof
import ../LAWS.bend as Sequence
import ../../../bitboard_probe/Sliders.bend as Sliders

def every_chess_occupancy(+key: U32, +occ: U64,
  +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  Exists(Nat,i => {Nat.is_lt(i,U32.to_nat(Spec.size(key))) == True{} : Bool}
    & {Sequence.at(i,Spec.mask(key)) == U64.and(occ,Spec.mask(key)) : U64}):
  (U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),
    (Layout.lookup_index_bound(key,occ,valid),Laws.lookup_state_recovery(key,occ,valid)))

def last_key(+occ: U64) ->
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Spec.mask(127))),Spec.mask(127)) == U64.and(occ,Spec.mask(127)) : U64}:
  Laws.lookup_state_recovery(127,occ,{==})

# The generic lossless-projection domain includes every symbolic width <=32.
def any_bounded_width(+x: U64, +k: Nat, width: {Nat.is_le(k,32n) == True{} : Bool},
  bounded: {Nat.is_lt(Bits.value(x),Bits.power(k)) == True{} : Bool}) ->
  {U32.to_nat(U64.low(x)) == Bits.value(x) : Nat}:
  Laws.low_projection_exact(x,k,width,bounded)
