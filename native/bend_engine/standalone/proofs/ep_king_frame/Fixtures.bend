# Check the public actual legal-list consumer, not just a raw update equality.
import Base
import ./consumer.bend as Public
import ./Witnesses.bend as W
import ./Frame.bend as F
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C

def white_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},W.white()))): Inl{{==}}
def black_member() -> E.member(C.Ply{27,20,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(20n)},W.black()))): Inr{Inr{Inl{{==}}}}
def actual_white() -> Public.preservation(W.white(),C.Ply{36,43,0,1}):
  Public.legal_member(ALeaf{U64.bit(43n)},W.white(),C.Ply{36,43,0,1},W.white_partition(),W.white_coherent(),white_member(),{==})
def actual_black() -> Public.preservation(W.black(),C.Ply{27,20,0,1}):
  Public.legal_member(ALeaf{U64.bit(20n)},W.black(),C.Ply{27,20,0,1},W.black_partition(),W.black_coherent(),black_member(),{==})
def actual_selected() -> {U64.ctz(F.plane(C.make_move(W.white(),C.Ply{36,43,0,1}),1)) == 0 : U32}:
  Public.selected(ALeaf{U64.bit(43n)},W.white(),C.Ply{36,43,0,1},W.white_partition(),W.white_coherent(),white_member(),{==})
