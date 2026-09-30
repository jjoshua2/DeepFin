# Satisfiable uses of the public contracts, including arbitrary occupancies.
import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../successor/PROOF.bend as OrdinalProof
import ./Spec.bend as Spec
import ../../../bitboard_probe/Sliders.bend as Sliders

def any_valid_index(+key: U32, +occ: U64,
  valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),U32.to_nat(Spec.size(key))) == True{} : Bool}:
  Laws.lookup_index_bound(key,occ,valid)

def rook_corner() -> {U64.popcount(Spec.mask(0)) == 12 : U32}:
  Laws.mask_population(0,{==})

def bishop_center() -> {U64.popcount(Spec.mask(91)) == 9 : U32}:
  Laws.mask_population(91,{==})

def last_key(+occ: U64) ->
  {Nat.is_lt(U32.to_nat(Sliders.pext_index(occ,Spec.mask(127))),U32.to_nat(Spec.size(127))) == True{} : Bool}:
  Laws.lookup_index_bound(127,occ,{==})

# Invalid keys cannot be supplied to the accepted domain by reflexivity.
def invalid_key() -> {Nat.is_lt(U32.to_nat(128),128n) == False{} : Bool}:
  {==}
