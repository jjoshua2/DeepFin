import Base
import ./LAWS.bend as Laws
import ./Reversal.bend as R
import ./Mask.bend as M
import ./Spec.bend as S
import ../attack_geometry/Geometry.bend as G
import ../attack_geometry/Slots.bend as Slots
import ../attack_geometry/Computed.bend as Computed

def Laws.leaper_coordinate_membership_reverses(kind,src,dst,source_bound,target_bound):
  R.bools(kind,src,dst,source_bound,target_bound)

# One endpoint's exact computation, bit reflection, reciprocity, then reverse back.
def computed_edge(+kind: G.Leaper,+src: Nat,+dst: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(Slots.value(kind,src),dst) == S.contains(dst,G.targets(kind,src)) : Bool}:
  Equal.trans(Bool,U64.test_bit(Slots.value(kind,src),dst),U64.test_bit(G.mask(kind,src),dst),S.contains(dst,G.targets(kind,src)),
    Equal.cong(U64,Bool,b => U64.test_bit(b,dst),Slots.value(kind,src),G.mask(kind,src),Computed.actual(kind,src,source_bound)),
    M.test(kind,src,dst,target_bound))

def computed_reversal(+kind: G.Leaper,+src: Nat,+dst: Nat,+source_bound: {Nat.is_lt(src,64n) == True{} : Bool},+target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(Slots.value(kind,src),dst) == U64.test_bit(Slots.value(G.reverse(kind),dst),src) : Bool}:
  Equal.trans(Bool,U64.test_bit(Slots.value(kind,src),dst),S.contains(dst,G.targets(kind,src)),U64.test_bit(Slots.value(G.reverse(kind),dst),src),
    computed_edge(kind,src,dst,source_bound,target_bound),
    Equal.trans(Bool,S.contains(dst,G.targets(kind,src)),S.contains(src,G.targets(G.reverse(kind),dst)),U64.test_bit(Slots.value(G.reverse(kind),dst),src),
      R.bools(kind,src,dst,source_bound,target_bound),
      Equal.sym(Bool,U64.test_bit(Slots.value(G.reverse(kind),dst),src),S.contains(src,G.targets(G.reverse(kind),dst)),
        computed_edge(G.reverse(kind),dst,src,target_bound,source_bound))))

def Laws.computed_leaper_mask_reverses(kind,src,dst,source_bound,target_bound):
  computed_reversal(kind,src,dst,source_bound,target_bound)
