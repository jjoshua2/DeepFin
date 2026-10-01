import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S

def use_geometry(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,
  sb: {Nat.is_lt(src,64n) == True{} : Bool},tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(S.geometry(kind,src,occ),dst) == U64.test_bit(S.geometry(kind,dst,occ),src) : Bool}:
  Laws.slider_geometric_membership_reverses(kind,src,dst,occ,sb,tb)

def use_computed(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,
  sb: {Nat.is_lt(src,64n) == True{} : Bool},tb: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(S.computed(kind,src,occ),dst) == U64.test_bit(S.computed(kind,dst,occ),src) : Bool}:
  Laws.computed_slider_mask_reverses(kind,src,dst,occ,sb,tb)

# The first occupied square is included; occupation at either endpoint is allowed.
def occupied_endpoints() ->
  {U64.test_bit(S.computed(S.Rook{},0n,U64.or(U64.bit(0n),U64.bit(7n))),7n) == True{} : Bool}:
  {==}

def occupied_middle() ->
  {U64.test_bit(S.computed(S.Rook{},0n,U64.or(U64.bit(3n),U64.bit(7n))),7n) == False{} : Bool}:
  {==}

def after_target() ->
  {U64.test_bit(S.computed(S.Rook{},0n,U64.bit(5n)),3n) == True{} : Bool}:
  {==}

def diagonal_endpoints() ->
  {U64.test_bit(S.computed(S.Bishop{},0n,U64.or(U64.bit(0n),U64.bit(63n))),63n) == True{} : Bool}:
  {==}

def diagonal_blocked() ->
  {U64.test_bit(S.computed(S.Queen{},0n,U64.bit(36n)),63n) == False{} : Bool}:
  {==}

def no_self_edge() ->
  {U64.test_bit(S.computed(S.Queen{},27n,U64.zero()),27n) == False{} : Bool}:
  {==}

# Reversal must use the same occupancy, not unrelated forward/reverse Boards.
def changed_occupancy_counterexample() ->
  {U64.test_bit(S.computed(S.Rook{},7n,U64.bit(3n)),0n) == False{} : Bool}:
  {==}
