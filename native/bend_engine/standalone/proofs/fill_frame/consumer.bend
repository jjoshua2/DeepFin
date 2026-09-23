import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ../complete/Spec.bend as Complete
import ../complete/Actual.bend as CompleteProof
import ../prefix/Spec.bend as Prefix
import ../relative/Spec.bend as Relative
import ../separation/Spec.bend as S
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables

# Positive-count frame use, both sides of a half-open interval.
def before_two(a: Array<U64>,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Tables.fill(2n,1024,sq,bishop,mask,subset,a),1023)) == S.value(Array.get(U64,a,1023)) : U64}:
  Laws.interval_fill_preserves_query(a,2n,1024,1026,sq,bishop,mask,subset,1023,shape,{==},{==},{==},{==})

def after_two(a: Array<U64>,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Tables.fill(2n,1024,sq,bishop,mask,subset,a),1026)) == S.value(Array.get(U64,a,1026)) : U64}:
  Laws.interval_fill_preserves_query(a,2n,1024,1026,sq,bishop,mask,subset,1026,shape,{==},{==},{==},{==})

# Retain concrete counterexamples: the first written address is not outside;
# modular overflow is not certified as room by mathematical count arithmetic.
def first_address_not_outside() -> {Spec.outside(1024,1024,1026) == False{} : Bool}:
  {==}

def too_many_writes() -> {Spec.room(3n,1024,1026) == False{} : Bool}:
  {==}

def zero_count_clear(+t: O.Shape,+at: U32,+q: U32) -> {S.clear(t,0n,at,q) == True{} : Bool}:
  {==}

# A true entire block fill, arbitrary key and query, with no assumed clear
# certificate or mathematical count relation supplied by the caller.
def whole_block(a: Array<U64>,+k: Nat,+q: U32,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},
  outside: {Spec.outside(q,Prefix.prefix(k),Prefix.prefix(1n+k)) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(a,k),q)) == S.value(Array.get(U64,a,q)) : U64}:
  Laws.chess_fill_preserves_query(a,k,q,shape,valid,qb,outside)

# Keep depth symbolic during composition to avoid expanding 131072 leaves.
# The actual allocator, not the caller, supplies the complete-shape certificate.
def initialized_reserved(+d: Nat,+k: Nat,+q: U32,+seed: U64,
  eq: {d == 17n : Nat},+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  reserved: {U32.is_lt(q,512) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(Array.new(U64,d,seed),k),q)) == S.value(Array.get(U64,Array.new(U64,d,seed),q)) : U64}:
  Laws.fill_preserves_reserved_query(Array.new(U64,d,seed),k,q,
    Equal.trans(O.Shape,O.shape(Array.new(U64,d,seed)),Complete.full(d),Complete.full(17n),
      CompleteProof.allocation(d,seed),Equal.cong(Nat,O.Shape,n=>Complete.full(n),d,17n,eq)),valid,reserved)

def next_block(a: Array<U64>,+i: Nat,+j: Nat,+occ: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +before: {Nat.is_lt(i,j) == True{} : Bool},+valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {S.value(Array.get(U64,Spec.block(a,i),Relative.address(j,Relative.index(j,occ)))) ==
    S.value(Array.get(U64,a,Relative.address(j,Relative.index(j,occ)))) : U64}:
  Laws.fill_preserves_later_lookup(a,i,j,occ,shape,before,valid)
