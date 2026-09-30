import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout
import ../normalization/Bound.bend as Normalize
import ../complete/Spec.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as Separation

# Satisfiable boundary facts. A complete block's end is excluded, not a valid offset.
def first_last_offset() -> {U32.is_lt(4095,Layout.size(0)) == True{} : Bool}:
  {==}

def first_endpoint_excluded() -> {U32.is_lt(4096,Layout.size(0)) == False{} : Bool}:
  {==}

def unchecked_addition_wraps() -> {U32.add(4294967295,1) == 0 : U32}:
  {==}

# Arbitrary occupancy: no caller-provided relative bound, masks, or prefix certificates.
def arbitrary_lookup(+k: Nat,+occ: U64,+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  S.interval(k,S.index(k,occ)):
  Laws.lookup_address_bounds(k,occ,valid)

def lookup_mask(+k: Nat,+occ: U64,+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.and(S.address(k,S.index(k,occ)),131071) == S.address(k,S.index(k,occ)) : U32}:
  Normalize.table_address(S.address(k,S.index(k,occ)),Bounds.allocation(k,S.index(k,occ),Laws.lookup_address_bounds(k,occ,valid)))

def actual_sum(+k: Nat,+r: U32,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  rb: {U32.is_lt(r,Layout.size(U32.from_nat(k))) == True{} : Bool}) ->
  {U64.from_u32(S.address(k,r)) == U64.add(U64.from_u32(Prefix.prefix(k)),U64.from_u32(r)) : U64}:
  Laws.relative_addition_exact(k,r,valid,rb)

def paths(a: Array<U64>,+i: Nat,+j: Nat,+ri: U32,+rj: U32,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +before: {Nat.is_lt(i,j) == True{} : Bool},+valid: {Nat.is_lt(j,128n) == True{} : Bool},
  +rib: {U32.is_lt(ri,Layout.size(U32.from_nat(i))) == True{} : Bool},
  +rjb: {U32.is_lt(rj,Layout.size(U32.from_nat(j))) == True{} : Bool}) ->
  {Separation.separate(a,S.address(i,ri),S.address(j,rj)) == True{} : Bool}:
  Laws.ordered_block_address_paths(a,i,j,ri,rj,shape,before,valid,rib,rjb)

def actual_write(a: Array<U64>,+i: Nat,+j: Nat,+occi: U64,+occj: U64,+v: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +before: {Nat.is_lt(i,j) == True{} : Bool},+valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,S.address(i,S.index(i,occi)),v),S.address(j,S.index(j,occj))) ==
    (Array.set(U64,a,S.address(i,S.index(i,occi)),v),Separation.value(Array.get(U64,a,S.address(j,S.index(j,occj))))) : Array<U64> & U64}:
  Laws.lookup_other_block_write(a,i,j,occi,occj,v,shape,before,valid)
