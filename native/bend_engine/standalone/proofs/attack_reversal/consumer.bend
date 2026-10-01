import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../attack_geometry/Geometry.bend as G
import ../attack_geometry/Slots.bend as Slots

def use_all(+kind: G.Leaper,+src: Nat,+dst: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {S.contains(dst,G.targets(kind,src)) == S.contains(src,G.targets(G.reverse(kind),dst)) : Bool}:
  Laws.leaper_coordinate_membership_reverses(kind,src,dst,source_bound,target_bound)

def pawn_forward() -> {S.contains(27n,G.targets(G.WhitePawn{},18n)) == True{} : Bool}: {==}
def wrong_same_color() -> {S.contains(18n,G.targets(G.WhitePawn{},27n)) == False{} : Bool}: {==}
def pawn_reversed() -> {S.contains(18n,G.targets(G.BlackPawn{},27n)) == True{} : Bool}: {==}
def self_not_an_attack() -> {S.contains(18n,G.targets(G.King{},18n)) == False{} : Bool}: {==}
def sentinel_is_not_a_square() -> {S.contains(64n,G.targets(G.Knight{},0n)) == True{} : Bool}: {==}

def use_computed(+kind: G.Leaper,+src: Nat,+dst: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(Slots.value(kind,src),dst) == U64.test_bit(Slots.value(G.reverse(kind),dst),src) : Bool}:
  Laws.computed_leaper_mask_reverses(kind,src,dst,source_bound,target_bound)

def top_edge() -> {U64.test_bit(Slots.value(G.Knight{},63n),46n) == True{} : Bool}: {==}
def outside_bit() -> {U64.test_bit(G.mask(G.Knight{},0n),64n) == False{} : Bool}: {==}
