import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../../Tables.bend as Tables

# Positive, value-polymorphic use: no empty-domain or unique-buffer assumption.
def pair(+x: U64, +y: U64) -> Array<U64>:
  ANode{ALeaf{x},ALeaf{y}}

def separate_pair(+x: U64, +y: U64, +v: U64) ->
  {Array.get(U64,Array.set(U64,pair(x,y),0,v),1) == (Array.set(U64,pair(x,y),0,v),y) : Array<U64> & U64}:
  Laws.write_preserves_other_location(pair(x,y),0,1,v,{==})

# Source semantics permit nonuniform trees: normalization follows left depth.
def ragged(+x: U64, +y: U64, +z: U64, +v: U64) ->
  {Array.get(U64,Array.set(U64,ANode{ALeaf{x},pair(y,z)},0,v),1) ==
    (Array.set(U64,ANode{ALeaf{x},pair(y,z)},0,v),z) : Array<U64> & U64}:
  Laws.write_preserves_other_location(ANode{ALeaf{x},pair(y,z)},0,1,v,{==})

# Numeric inequality is not the precondition: indices 0 and 2 alias here.
def alias_guard(+x: U64, +y: U64) -> {S.separate(pair(x,y),0,2) == False{} : Bool}:
  {==}

def alias_witness(+x: U64, +y: U64, +v: U64) ->
  {S.value(Array.get(U64,Array.set(U64,pair(x,y),2,v),0)) == v : U64}:
  {==}

# Positive-count use of actual fill; writes wrap U32_MAX to zero, while slot 2
# in the four-cell array remains untouched. All slider parameters are arbitrary.
def wrapped_fill(+seed: U64, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64) ->
  {S.value(Array.get(U64,Tables.fill(2n,4294967295,sq,bishop,mask,subset,Array.new(U64,2n,seed)),2)) == seed : U64}:
  Laws.fill_preserves_unwritten_location(Array.new(U64,2n,seed),2n,4294967295,sq,bishop,mask,subset,2,{==})

def overlapping_fill_rejected() ->
  {S.clear(O.shape(Array.new(U64,2n,U64.zero())),2n,3,0) == False{} : Bool}:
  {==}

def writes_do_not_change_paths(a: Array<U64>, +i: U32, +j: U32, at: U32, v: U64) ->
  {S.separate(Array.set(U64,a,at,v),i,j) == S.separate(a,i,j) : Bool}:
  Laws.write_preserves_separation(a,i,j,at,v)

def fill_does_not_change_paths(a: Array<U64>, +n: Nat, +at: U32, +sq: U32, +bishop: Bool, +mask: U64, +subset: U64, +i: U32, +j: U32) ->
  {S.separate(Tables.fill(n,at,sq,bishop,mask,subset,a),i,j) == S.separate(a,i,j) : Bool}:
  Laws.fill_preserves_separation(a,n,at,sq,bishop,mask,subset,i,j)
