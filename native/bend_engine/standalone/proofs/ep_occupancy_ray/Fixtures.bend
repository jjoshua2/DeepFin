# Instantiate the new public theorem on both actual full-Ply EP legal occurrences.
import Base
import ./consumer.bend as Public
import ./EP.bend as EP
import ./Witnesses.bend as W
import ../ep_king_frame/Witnesses.bend as Parent
import ../ep_king_frame/Fixtures.bend as Members
import ../castle_emission/Spec.bend as E
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as C

def white_route() -> {Route.trace(7n,40,0,W.white_path()) == True{} : Bool}: {==}
def black_route() -> {Route.trace(7n,16,0,W.black_path()) == True{} : Bool}: {==}
def white_clear() -> D.disjoint(EP.removed(36,43),Path.attack(W.white_path(),C.occupied(Parent.white()))): {==}
def black_clear() -> D.disjoint(EP.removed(27,20),Path.attack(W.black_path(),C.occupied(Parent.black()))): {==}
def actual_white(+acc: U64) -> Public.result(7n,40,0,Parent.white(),C.Ply{36,43,0,1},acc):
  Public.legal_member(ALeaf{U64.bit(43n)},7n,40,0,W.white_path(),Parent.white(),C.Ply{36,43,0,1},acc,
    Parent.white_partition(),Parent.white_coherent(),white_route(),white_clear(),Members.white_member(),{==})
def actual_black(+acc: U64) -> Public.result(7n,16,0,Parent.black(),C.Ply{27,20,0,1},acc):
  Public.legal_member(ALeaf{U64.bit(20n)},7n,16,0,W.black_path(),Parent.black(),C.Ply{27,20,0,1},acc,
    Parent.black_partition(),Parent.black_coherent(),black_route(),black_clear(),Members.black_member(),{==})

# Actual legal-list membership at this arbitrary table does not supply ray disjointness.
def discovered_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},W.discovered()))): Inl{{==}}
