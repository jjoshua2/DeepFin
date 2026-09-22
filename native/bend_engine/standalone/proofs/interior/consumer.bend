import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../storage/Observe.bend as O

# Positive strict-bound endpoints, including the last allocatable address.
def first() -> {S.normalize(0) == 0 : U32}:
  Laws.allocation_mask_exact(0,{==})

def last() -> {S.normalize(131071) == 131071 : U32}:
  Laws.allocation_mask_exact(131071,{==})

# Both are genuine interior addresses, not endpoints passed with false premises.
def rook_inside() -> S.interior(513):
  Laws.prefix_interior_certificate(0n,513,{==},({==},{==}))

def bishop_inside() -> S.interior(108159):
  Laws.prefix_interior_certificate(127n,108159,{==},({==},{==}))

def distant() -> {U32.is_ne(S.normalize(513),S.normalize(108159)) == True{} : Bool}:
  Laws.ordered_normalized_addresses_distinct(0n,127n,513,108159,{==},{==},({==},{==}),({==},{==}))

# The actual affine array remains arbitrary; its source-size premise is explicit.
def real_read(a: Array<U64>, size: {O.count(Array.size(U64,a)) == 131072 : U32}) ->
  {Array.get(U64,a,513) == Array.get.go(U64,a,131072,513) : Array<U64> & U64}:
  Laws.interior_read_uses_unmasked_address(a,0n,513,{==},({==},{==}),size)

# Reject the inclusive allocation endpoint: masking genuinely aliases 131072 to 0.
def outside_alias() -> {S.normalize(131072) == S.normalize(0) : U32}:
  {==}

def distinct_unmasked() -> {U32.is_ne(131072,0) == True{} : Bool}:
  {==}

# The size premise is satisfiable for every seed in the actual allocation shape.
def allocation_size(+seed: U64) ->
  {O.count(Array.size(U64,Array.new(U64,17n,seed))) == 131072 : U32}:
  {==}

# Distinct normalized numbers are still not an arbitrary-tree route theorem.
# This nonuniform eight-address source tree sends both 4 and 5 to the right leaf.
def ragged_route_alias(+x: U64, +y: U64) ->
  {Array.get(U64,ANode{Array.new(U64,2n,x),ALeaf{y}},4) ==
    Array.get(U64,ANode{Array.new(U64,2n,x),ALeaf{y}},5) : Array<U64> & U64}:
  {==}
