# Link the runtime U32 index to the already-proved full-width ordinal.
import Base
import ./Projection.bend as Projection
import ../index/Bits.bend as Bits
import ../index/LAWS.bend as Index
import ../index/PROOF.bend as IndexProof
import ../layout/Spec.bend as Spec
import ../layout/Geometry.bend as Geometry
import ../layout/Consequences.bend as Consequences
import ../layout/Facts.bend as Facts
import ../layout/Domain.bend as Domain
import ../u64/LAWS.bend as U64Laws
import ../u64/PROOF.bend as U64Proof
import ../successor/LAWS.bend as Successor
import ../successor/PROOF.bend as SuccessorProof
import ../successor/Value.bend as Value
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../../../bitboard_probe/Sliders.bend as Sliders

def small32(+k: Nat, e: {Nat.is_lt(k,13n) == True{} : Bool}) ->
  {Nat.is_le(k,32n) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n+p: Domain.impossible({Nat.is_le(13n+p,32n) == True{} : Bool},p,e)

def width_fits(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {Nat.is_le(Bits.width(Spec.mask(key)),32n) == True{} : Bool}:
  %IndexProof.count_exact(Spec.mask(key)) : {Nat.is_le(_,32n) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U64.popcount(Spec.mask(key))),Geometry.population(U32.to_nat(key)),
    Consequences.observed(key,valid,count)) :
    {Nat.is_le(_,32n) == True{} : Bool}
  small32(Geometry.population(U32.to_nat(key)),Facts.small(U32.to_nat(key),key,Facts.actual(key,valid)))

def exact(+key: U32, +occ: U64, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))) == Bits.value(U64.pext(occ,Spec.mask(key))) : Nat}:
  Projection.low_exact(U64.pext(occ,Spec.mask(key)),Bits.width(Spec.mask(key)),
    width_fits(key,valid,count),Index.extraction_bound(occ,Spec.mask(key)))

def capacity_bound(+key: U32, +i: Nat, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  bounded: {Nat.is_lt(i,U32.to_nat(Spec.size(key))) == True{} : Bool}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {Nat.is_lt(i,Bits.capacity(Spec.mask(key))) == True{} : Bool}:
  %Consequences.capacity(key,valid,count) : {Nat.is_lt(i,_) == True{} : Bool}
  bounded

def ordinal(+key: U32, +i: Nat, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  bounded: {Nat.is_lt(i,U32.to_nat(Spec.size(key))) == True{} : Bool}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U32.to_nat(Sliders.pext_index(Sequence.at(i,Spec.mask(key)),Spec.mask(key))) == i : Nat}:
  Equal.trans(Nat,U32.to_nat(Sliders.pext_index(Sequence.at(i,Spec.mask(key)),Spec.mask(key))),
    Bits.value(U64.pext(Sequence.at(i,Spec.mask(key)),Spec.mask(key))),i,
    exact(key,Sequence.at(i,Spec.mask(key)),valid,count),
    Successor.sequence_ordinal(i,Spec.mask(key),capacity_bound(key,i,valid,bounded,count)))

# Equality of actual indices cannot collapse distinct relevant occupancies.
def collision_sound(+key: U32, +a: U64, +b: U64,
  +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  same: {Sliders.pext_index(a,Spec.mask(key)) == Sliders.pext_index(b,Spec.mask(key)) : U32}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U64.and(a,Spec.mask(key)) == U64.and(b,Spec.mask(key)) : U64}:
  %U64Laws.pext_pdep_roundtrip(a,Spec.mask(key)) : {_ == U64.and(b,Spec.mask(key)) : U64}
  %U64Laws.pext_pdep_roundtrip(b,Spec.mask(key)) : {U64.pdep(U64.pext(a,Spec.mask(key)),Spec.mask(key)) == _ : U64}
  Equal.cong(U64,U64,x => U64.pdep(x,Spec.mask(key)),U64.pext(a,Spec.mask(key)),U64.pext(b,Spec.mask(key)),
    Value.u64_injective(U64.pext(a,Spec.mask(key)),U64.pext(b,Spec.mask(key)),
      Equal.trans(Nat,Bits.value(U64.pext(a,Spec.mask(key))),
        U32.to_nat(Sliders.pext_index(a,Spec.mask(key))),Bits.value(U64.pext(b,Spec.mask(key))),
        Equal.sym(Nat,U32.to_nat(Sliders.pext_index(a,Spec.mask(key))),Bits.value(U64.pext(a,Spec.mask(key))),exact(key,a,valid,count)),
        Equal.trans(Nat,U32.to_nat(Sliders.pext_index(a,Spec.mask(key))),U32.to_nat(Sliders.pext_index(b,Spec.mask(key))),
          Bits.value(U64.pext(b,Spec.mask(key))),
          Equal.cong(U32,Nat,x => U32.to_nat(x),Sliders.pext_index(a,Spec.mask(key)),Sliders.pext_index(b,Spec.mask(key)),same),
          exact(key,b,valid,count)))))

# The converse needs no population bound: irrelevant bits never affect indexing.
def collision_complete(+a: U64, +b: U64, +mask: U64,
  same: {U64.and(a,mask) == U64.and(b,mask) : U64}) ->
  {Sliders.pext_index(a,mask) == Sliders.pext_index(b,mask) : U32}:
  Equal.cong(U64,U32,x => U64.low(x),U64.pext(a,mask),U64.pext(b,mask),
    Index.deposit_injective(U64.pext(a,mask),U64.pext(b,mask),mask,
      Index.extraction_bound(a,mask),Index.extraction_bound(b,mask),
      Equal.trans(U64,U64.pdep(U64.pext(a,mask),mask),U64.and(a,mask),U64.pdep(U64.pext(b,mask),mask),
        U64Laws.pext_pdep_roundtrip(a,mask),
        Equal.trans(U64,U64.and(a,mask),U64.and(b,mask),U64.pdep(U64.pext(b,mask),mask),same,
          Equal.sym(U64,U64.pdep(U64.pext(b,mask),mask),U64.and(b,mask),U64Laws.pext_pdep_roundtrip(b,mask))))))

def recovery(+key: U32, +occ: U64, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, +count: {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {Sequence.at(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),Spec.mask(key)) == U64.and(occ,Spec.mask(key)) : U64}:
  +i = U32.to_nat(Sliders.pext_index(occ,Spec.mask(key)))
  %Sequence.sequence_in_mask(i,Spec.mask(key)) : {_ == U64.and(occ,Spec.mask(key)) : U64}
  collision_sound(key,Sequence.at(i,Spec.mask(key)),occ,valid,
    Domain.injective(Sliders.pext_index(Sequence.at(i,Spec.mask(key)),Spec.mask(key)),Sliders.pext_index(occ,Spec.mask(key)),
      ordinal(key,i,valid,Consequences.lookup_index(key,occ,valid,count),count)),count)
