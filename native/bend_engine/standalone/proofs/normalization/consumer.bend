import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

# Boundary and non-endpoint uses: the entire allocation is the valid domain.
def zero_address() -> {U32.and(0,131071) == 0 : U32}:
  Laws.bounded_mask_identity(0,{==})

def interior_address() -> {U32.and(70437,131071) == 70437 : U32}:
  Laws.bounded_mask_identity(70437,{==})

def last_address() -> {U32.and(131071,131071) == 131071 : U32}:
  Laws.bounded_mask_identity(131071,{==})

# Out-of-range address normalization aliases zero. A <= capacity domain is false.
def excluded_endpoint() -> {U32.and(131072,131071) == 0 : U32}:
  {==}

# Observed capacity of the real initialized allocation. API laws use arbitrary
# live buffers with this capacity, avoiding expansion of a complete filled table.
def table_allocation_size(+seed: U64) ->
  {S.size(O.shape(Array.new(U64,17n,seed))) == 131072 : U32}:
  {==}

def equal_addresses(+i: U32, +j: U32,
  ib: {U32.is_lt(i,131072) == True{} : Bool},jb: {U32.is_lt(j,131072) == True{} : Bool},
  e: {U32.and(i,131071) == U32.and(j,131071) : U32}) -> {i == j : U32}:
  Laws.bounded_mask_injective(i,j,ib,jb,e)

# Symbolic live-buffer uses of all three public API contracts.
def read_symbolic(a: Array<U64>, +i: U32,
  c: {S.size(O.shape(a)) == 131072 : U32},b: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {Array.get(U64,a,i) == Array.get.go(U64,a,131072,i) : Array<U64> & U64}:
  Laws.bounded_read_direct(a,i,c,b)

def write_symbolic(a: Array<U64>, +i: U32, +v: U64,
  c: {S.size(O.shape(a)) == 131072 : U32},b: {U32.is_lt(i,131072) == True{} : Bool}) ->
  {Array.set(U64,a,i,v) == Array.set.fin(U64,Array.swap.go(U64,a,131072,i,v)) : Array<U64>}:
  Laws.bounded_write_direct(a,i,v,c,b)

def paths_symbolic(a: Array<U64>, +i: U32, +j: U32,
  c: {S.size(O.shape(a)) == 131072 : U32},ib: {U32.is_lt(i,131072) == True{} : Bool},jb: {U32.is_lt(j,131072) == True{} : Bool}) ->
  {S.separate(a,i,j) == S.apart(O.shape(a),131072,i,j) : Bool}:
  Laws.bounded_separation_direct(a,i,j,c,ib,jb)
