# Reuse accepted representation laws; this is not a new ray-geometry definition.
import Base
import ../index/LAWS.bend as IL
import ../index/PROOF.bend as IP
import ../address/LAWS.bend as AL
import ../address/PROOF.bend as AP
import ../u64/LAWS.bend as UL
import ../u64/PROOF.bend as UP
import ../Mask.bend as Mask
import ../LAWS.bend as Sequence
import ../layout/Spec.bend as Layout
import ../../../bitboard_probe/Sliders.bend as Sliders

def masked_extract(+x: U64,+m: U64) ->
  {U64.pext(U64.and(x,m),m) == U64.pext(x,m) : U64}:
  Equal.trans(U64,U64.pext(U64.and(x,m),m),U64.pext(U64.pdep(U64.pext(x,m),m),m),U64.pext(x,m),
    Equal.cong(U64,U64,y => U64.pext(y,m),U64.and(x,m),U64.pdep(U64.pext(x,m),m),
      Equal.sym(U64,U64.pdep(U64.pext(x,m),m),U64.and(x,m),UL.pext_pdep_roundtrip(x,m))),
    IL.compact_roundtrip(U64.pext(x,m),m,IL.extraction_bound(x,m)))

def state(+key: U32,+occ: U64,valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) ->
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}:
  %Equal.cong(U64,U32,x => U64.low(x),U64.pext(U64.and(occ,Layout.mask(key)),Layout.mask(key)),
    U64.pext(occ,Layout.mask(key)),masked_extract(occ,Layout.mask(key))) :
    {Sequence.at(U32.to_nat(_),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}
  AL.chess_sequence_coverage(key,U64.and(occ,Layout.mask(key)),valid,Mask.u64_absorb(occ,Layout.mask(key)))
