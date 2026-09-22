# General consequences from the finite-domain mask equality. No assumed laws.
import Base
import ./Spec.bend as Spec
import ./Geometry.bend as Geometry
import ./Facts.bend as Facts
import ./Low.bend as Low
import ../index/Bits.bend as Bits
import ../index/LAWS.bend as Index
import ../index/PROOF.bend as IndexProof
import ../../../bitboard_probe/Sliders.bend as Sliders

def count(+key: U32) -> Type:
  {U64.popcount(Spec.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}

def observed(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.to_nat(U64.popcount(Spec.mask(key))) == Geometry.population(U32.to_nat(key)) : Nat}:
  Equal.trans(Nat,U32.to_nat(U64.popcount(Spec.mask(key))),
    U32.to_nat(U32.from_nat(Geometry.population(U32.to_nat(key)))),Geometry.population(U32.to_nat(key)),
    Equal.cong(U32,Nat,w => U32.to_nat(w),U64.popcount(Spec.mask(key)),
      U32.from_nat(Geometry.population(U32.to_nat(key))),e),Facts.exact(U32.to_nat(key),key,Facts.actual(key,valid)))

def population(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.is_le(U64.popcount(Spec.mask(key)),Spec.max_population(key)) == True{} : Bool}:
  %Equal.sym(U32,U64.popcount(Spec.mask(key)),U32.from_nat(Geometry.population(U32.to_nat(key))),e) :
    {U32.is_le(_,Spec.max_population(key)) == True{} : Bool}
  Facts.bound(U32.to_nat(key),key,Facts.actual(key,valid))

def safe_shift(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.is_lt(U64.popcount(Spec.mask(key)),32) == True{} : Bool}:
  %Equal.sym(U32,U64.popcount(Spec.mask(key)),U32.from_nat(Geometry.population(U32.to_nat(key))),e) :
    {U32.is_lt(_,32) == True{} : Bool}
  Facts.safe(Geometry.population(U32.to_nat(key)),Facts.shift(Geometry.population(U32.to_nat(key)),
    Facts.small(U32.to_nat(key),key,Facts.actual(key,valid))))

def positive(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.is_gt(Spec.size(key),0) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U64.popcount(Spec.mask(key))),Geometry.population(U32.to_nat(key)),observed(key,valid,e)) :
    {U32.is_gt(U32.shln(1,_),0) == True{} : Bool}
  Facts.positive(Geometry.population(U32.to_nat(key)),Facts.shift(Geometry.population(U32.to_nat(key)),
    Facts.small(U32.to_nat(key),key,Facts.actual(key,valid))))

def size_bound(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.is_le(Spec.size(key),Spec.max_size(key)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U64.popcount(Spec.mask(key))),Geometry.population(U32.to_nat(key)),observed(key,valid,e)) :
    {U32.is_le(U32.shln(1,_),Spec.max_size(key)) == True{} : Bool}
  Facts.size_bound(U32.to_nat(key),key,Facts.actual(key,valid))

def capacity(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {U32.to_nat(Spec.size(key)) == Bits.capacity(Spec.mask(key)) : Nat}:
  %Equal.sym(Nat,Bits.capacity(Spec.mask(key)),Nat.pow(2n,U32.to_nat(U64.popcount(Spec.mask(key)))),
    Index.capacity_matches_popcount(Spec.mask(key))) :
    {U32.to_nat(Spec.size(key)) == _ : Nat}
  %Equal.sym(Nat,U32.to_nat(U64.popcount(Spec.mask(key))),Geometry.population(U32.to_nat(key)),observed(key,valid,e)) :
    {U32.to_nat(U32.shln(1,_)) == Nat.pow(2n,_) : Nat}
  Facts.power(Geometry.population(U32.to_nat(key)),Facts.shift(Geometry.population(U32.to_nat(key)),
    Facts.small(U32.to_nat(key),key,Facts.actual(key,valid))))

def full_index(+key: U32, +occ: U64, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {Nat.is_lt(Bits.value(U64.pext(occ,Spec.mask(key))),U32.to_nat(Spec.size(key))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Spec.size(key)),Bits.capacity(Spec.mask(key)),capacity(key,valid,e)) :
    {Nat.is_lt(Bits.value(U64.pext(occ,Spec.mask(key))),_) == True{} : Bool}
  Index.extraction_bound(occ,Spec.mask(key))

def lookup_index(+key: U32, +occ: U64, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}, e: count(key)) ->
  {Nat.is_lt(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),U32.to_nat(Spec.size(key))) == True{} : Bool}:
  Low.le_lt(U32.to_nat(Sliders.pext_index(occ,Spec.mask(key))),Bits.value(U64.pext(occ,Spec.mask(key))),
    U32.to_nat(Spec.size(key)),Low.low_le(U64.pext(occ,Spec.mask(key))),full_index(key,occ,valid,e))
