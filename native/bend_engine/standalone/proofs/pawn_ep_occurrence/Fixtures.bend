# Actual small emissions keep full fields and every duplicate tail occurrence.
import Base
import ./consumer.bend as Public
import ./Bridge.bend as Bridge
import ./Pipeline.bend as Pipe
import ./Spec.bend as S
import ../destination_factorization/Spec.bend as F
import ../castle_emission/Spec.bend as E
import ../initialized_pawn_double/Fixtures.bend as Prior
import ../initialized_pawn_double/Spec.bend as D
import ../../../legal_probe/Chess.bend as C

def repeated() -> List<&2,C.Ply>:
  C.Ply{8,16,0,1} <> C.Ply{8,16,0,1} <> Nil{}

def tagged() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(8,True{},16,repeated(),(ALeaf{U64.bit(47n)},U64.bit(16n)))

def tagged_exact() ->
  {tagged() == (ALeaf{U64.bit(47n)},C.Ply{8,16,0,1} <> repeated()) : Array<U64> & List<&2,C.Ply>}: {==}

def duplicate_count() -> {F.count(E.moves(tagged()),C.Ply{8,16,0,1}) == 3n : Nat}: {==}
def wrong_ep_flag_absent() -> {F.count(E.moves(tagged()),C.Ply{8,16,0,0}) == 0n : Nat}: {==}

def promoted() -> Array<U64> & List<&2,C.Ply>:
  C.scan_after(48,True{},56,Nil{},(ALeaf{U64.bit(47n)},U64.or(U64.bit(56n),U64.bit(57n))))

def promotion_exact() ->
  {promoted() == (ALeaf{U64.bit(47n)},
    C.Ply{48,57,1,0} <> C.Ply{48,57,2,0} <> C.Ply{48,57,3,0} <> C.Ply{48,57,4,0} <>
    C.Ply{48,56,1,0} <> C.Ply{48,56,2,0} <> C.Ply{48,56,3,0} <> C.Ply{48,56,4,0} <> Nil{}) : Array<U64> & List<&2,C.Ply>}: {==}
def wrong_promotion_absent() -> {F.count(E.moves(promoted()),C.Ply{48,56,0,0}) == 0n : Nat}: {==}
def promotion_raw_ep_flag_absent() -> {F.count(E.moves(promoted()),C.Ply{48,56,1,1}) == 0n : Nat}: {==}

# The certificate is copied along the later occurrence, not replaced by a head.
def duplicate_later() -> E.member(C.Ply{8,16,0,1},repeated()) & E.member(C.Ply{8,16,0,1},repeated()):
  Bridge.duplicate(repeated(),C.Ply{8,16,0,1},Inr{Inl{{==}}})

# A generic full-Ply legal occurrence works even for arbitrary attack arrays.
# It proves the mask hit, not initialized geometry or the empty2 guard.
def jump_origin() -> S.occurrence(ALeaf{U64.bit(48n)},Prior.jump(),C.Ply{32,48,0,0}):
  Pipe.member(ALeaf{U64.bit(48n)},Prior.jump(),C.Ply{32,48,0,0},Inl{{==}})
def arbitrary_jump_not_double() -> {D.empty2(Prior.jump(),32) == False{} : Bool}: {==}
def arbitrary_jump_tag() -> {C.get_ep(C.make_move(Prior.jump(),C.Ply{32,48,0,0})) == 40 : U32}: {==}
def arbitrary_jump_ep_incoherent() -> {C.ep_target(C.make_move(Prior.jump(),C.Ply{32,48,0,0})) == U64.zero() : U64}: {==}

# The stale destination==ep flag counterexample also remains a legal occurrence.
def stale_origin() -> S.occurrence(ALeaf{U64.zero()},Prior.stale(),C.Ply{8,16,0,1}):
  Pipe.member(ALeaf{U64.zero()},Prior.stale(),C.Ply{8,16,0,1},Inl{{==}})
def stale_king_loss() -> {C.get_kings(C.make_move(Prior.stale(),C.Ply{8,16,0,1})) == U64.bit(0n) : U64}: {==}
