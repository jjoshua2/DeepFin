# Import the actual public consumer plus concrete multi-bit/duplicate witnesses.
import Base
import ../castle_chain/Spec.bend as Chain
import ../castle_emission/Spec.bend as E
import ./consumer.bend as Public
import ./Source.bend as Source
import ./Spec.bend as S
import ./Witnesses.bend as W
import ../child_ep_coherence/Witnesses.bend as Prior
import ../../../legal_probe/Chess.bend as C

# A malformed parent EP can be tagged by the actual legal pipeline. This is
# full membership in the given arbitrary table, not an initialized-root claim.
def stale_actual_member() -> S.facts(Prior.stale(),C.Ply{8,16,0,1}):
  Public.legal_member(ALeaf{U64.zero()},Prior.stale(),C.Ply{8,16,0,1},Inl{{==}},{==})

def preserved_duplicate_tail(-Goal: @-r: Array<U64> & List<&2,C.Ply> -> Type,
  next: @r: Array<U64> & List<&2,C.Ply> ->
    Chain.every(m => S.property(Prior.stale(),m),E.moves(r)) -> Goal(r)) ->
  Goal(C.scan_after(8,True{},16,W.repeated(),(ALeaf{U64.bit(47n)},W.targets()))):
  Source.after(Goal,Prior.stale(),8,True{},16,W.repeated(),(ALeaf{U64.bit(47n)},W.targets()),{==},{==},
    (({==},({==},({==},{==}))),(({==},({==},({==},{==}))),Unit{})),next)

def stale_actual_kind() -> {C.piece(Prior.stale(),8) == 0 : U32}:
  Public.decoded_kind(ALeaf{U64.zero()},Prior.stale(),C.Ply{8,16,0,1},Inl{{==}},{==})
