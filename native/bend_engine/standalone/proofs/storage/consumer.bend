import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Observe.bend as O
import ../../Tables.bend as Tables

# Arbitrary affine buffers, not a duplicated or assumed reference array.
def arbitrary_read(a: Array<U64>, +i: U32) ->
  {O.storage(Array.get(U64,a,i)) == a : Array<U64>}:
  Laws.read_preserves_storage(a,i)

def arbitrary_write(a: Array<U64>, +i: U32, +v: U64) ->
  {Array.get(U64,Array.set(U64,a,i,v),i) == (Array.set(U64,a,i,v),v) : Array<U64> & U64}:
  Laws.write_read_same_location(a,i,v)

# Shape and same-location correctness do not establish non-aliasing. Array.get
# masks its index: slots 0 and 2 alias in this two-leaf example.
def wrapping_counterexample() ->
  {Array.get(U64,Array.set(U64,Array.new(U64,1n,U64.zero()),2,U64.bit(63n)),0) ==
    (ANode{ALeaf{U64.bit(63n)},ALeaf{U64.zero()}},U64.bit(63n)) : Array<U64> & U64}:
  {==}

def empty_fill(a: Array<U64>, +at: U32) ->
  {O.shape(Tables.fill(0n,at,0,False{},U64.zero(),U64.zero(),a)) == O.shape(a) : O.Shape}:
  Laws.fill_preserves_shape(0n,at,0,False{},U64.zero(),U64.zero(),a)
