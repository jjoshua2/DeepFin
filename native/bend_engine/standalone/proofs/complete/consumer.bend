import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Core.bend as Core
import ./Arithmetic.bend as A
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

# Required positive uses retain arbitrary contents and genuine affine buffers.
def allocation_shape(+d: Nat,+seed: U64) ->
  {O.shape(Array.new(U64,d,seed)) == Spec.full(d) : O.Shape}:
  Laws.allocation_complete(d,seed)

def pipeline_shape(+d: Nat,+n: Nat,+extra: Nat,+key: U32,+at: U32,+sq: U32,+seed: U64) ->
  {O.shape(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))) == Spec.full(d) : O.Shape}:
  Laws.table_pipeline_complete(d,n,extra,key,at,sq,seed)

# Shape is PRODUCED from the real pipeline, not assumed by this consumer.
def pipeline_routes(+n: Nat,+extra: Nat,+key: U32,+at: U32,+sq: U32,+seed: U64,+i: U32,+j: U32,
  ib: {U32.is_lt(i,131072) == True{} : Bool},jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) ->
  {S.separate(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),i,j) == True{} : Bool}:
  Laws.bounded_complete_routes(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,17n,seed))),i,j,
    Laws.table_pipeline_complete(17n,n,extra,key,at,sq,seed),ib,jb,ne)

def write_symbolic(a: Array<U64>,+i: U32,+j: U32,+v: U64,
  shape: {O.shape(a) == Spec.full(17n) : O.Shape},
  ib: {U32.is_lt(i,131072) == True{} : Bool},jb: {U32.is_lt(j,131072) == True{} : Bool},
  ne: {U32.is_eq(i,j) == False{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,i,v),j) == (Array.set(U64,a,i,v),S.value(Array.get(U64,a,j))) : Array<U64> & U64}:
  Laws.bounded_other_index_write(a,i,j,v,shape,ib,jb,ne)

# Small concrete route and half-subtraction boundaries, including same-side recursion.
def two_cell_route() -> {S.apart(Spec.full(1n),2,0,1) == True{} : Bool}:
  Core.routes(1n,0,1,{==},{==},{==},eq => A.different(0,1,{==},eq))

def right_half_route() -> {S.apart(Spec.full(3n),8,6,7) == True{} : Bool}:
  Core.routes(3n,6,7,{==},{==},{==},eq => A.different(6,7,{==},eq))

def last_right_half() -> {U32.is_lt(U32.sub(131071,65536),65536) == True{} : Bool}:
  A.right32(16n,131071,{==},{==})

# Numerically unequal addresses alias when the strict allocation bound is omitted.
def excluded_endpoint() -> {S.disjoint(Spec.full(1n),0,2) == False{} : Bool}:
  {==}

# In-range numerically different addresses still alias in an incomplete shape.
def ragged() -> O.Shape:
  O.Node{O.Node{O.Leaf{},O.Leaf{}},O.Leaf{}}

def ragged_capacity() -> {S.size(ragged()) == 4 : U32}:
  {==}

def ragged_alias() -> {S.disjoint(ragged(),2,3) == False{} : Bool}:
  {==}
