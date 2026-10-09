# Positive entry includes the whole public consumer and direct runtime witnesses.
import Base
import ./consumer.bend as Public
import ./Witnesses.bend as W
import ./Source.bend as Source
import ../../../legal_probe/Chess.bend as C

def arbitrary_jump_source() -> Source.generated(W.jump(),C.Ply{32,48,0,0}):
  Source.member(ALeaf{U64.bit(48n)},W.jump(),C.Ply{32,48,0,0},Inl{{==}})

# Preserve the stale raw EP actual occurrence without importing the whole prior
# consumer/fixture closure; the child EP theorem is not king preservation.
def stale_source() -> Source.generated(W.stale(),C.Ply{8,16,0,1}):
  Source.member(ALeaf{U64.zero()},W.stale(),C.Ply{8,16,0,1},Inl{{==}})
