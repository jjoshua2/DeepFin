# Actual full-Ply legal occurrences with nonzero opponent R/B/Q rows.
import Base
import ./consumer.bend as Public
import ./Hits.bend as Hits
import ../ep_occupancy_ray/EP.bend as EP
import ../board/Spec.bend as Board
import ../castle_emission/Spec.bend as E
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as C

def white() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.bit(61n),U64.bit(47n),U64.bit(55n),U64.or(U64.bit(0n),U64.bit(63n)),
    U64.or(U64.bit(0n),U64.bit(36n)),U64.or(U64.or(U64.or(U64.or(U64.bit(63n),U64.bit(35n)),U64.bit(47n)),U64.bit(61n)),U64.bit(55n)),1,0,43}
def black() -> C.Board:
  C.Board{U64.or(U64.bit(27n),U64.bit(28n)),U64.zero(),U64.bit(2n),U64.bit(23n),U64.bit(8n),U64.or(U64.bit(0n),U64.bit(63n)),
    U64.or(U64.or(U64.or(U64.or(U64.bit(0n),U64.bit(28n)),U64.bit(23n)),U64.bit(2n)),U64.bit(8n)),U64.or(U64.bit(63n),U64.bit(27n)),0,0,20}
def white_partition() -> {Board.valid(white()) == True{} : Bool}: {==}
def white_coherent() -> {Position.valid_ep(43,white()) == True{} : Bool}: {==}
def black_partition() -> {Board.valid(black()) == True{} : Bool}: {==}
def black_coherent() -> {Position.valid_ep(20,black()) == True{} : Bool}: {==}
def white_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},white()))): Inl{{==}}
def black_member() -> E.member(C.Ply{27,20,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(20n)},black()))): Inr{Inr{Inl{{==}}}}
def white_path() -> List<&2,Nat>: 41n <> 42n <> 43n <> 44n <> 45n <> 46n <> 47n <> Nil{}
def black_path() -> List<&2,Nat>: 17n <> 18n <> 19n <> 20n <> 21n <> 22n <> 23n <> Nil{}
def white_diagonal() -> List<&2,Nat>: 34n <> 43n <> 52n <> 61n <> Nil{}
def black_diagonal() -> List<&2,Nat>: 29n <> 20n <> 11n <> 2n <> Nil{}
def white_route() -> {Route.trace(7n,40,0,white_path()) == True{} : Bool}: {==}
def black_route() -> {Route.trace(7n,16,0,black_path()) == True{} : Bool}: {==}
def white_diagonal_route() -> {Route.trace(4n,25,4,white_diagonal()) == True{} : Bool}: {==}
def black_diagonal_route() -> {Route.trace(4n,38,7,black_diagonal()) == True{} : Bool}: {==}
def white_clear() -> D.disjoint(EP.removed(36,43),Path.attack(white_path(),C.occupied(white()))): {==}
def black_clear() -> D.disjoint(EP.removed(27,20),Path.attack(black_path(),C.occupied(black()))): {==}
def white_diagonal_clear() -> D.disjoint(EP.removed(36,43),Path.attack(white_diagonal(),C.occupied(white()))): {==}
def black_diagonal_clear() -> D.disjoint(EP.removed(27,20),Path.attack(black_diagonal(),C.occupied(black()))): {==}

def actual_white(+acc: U64) -> Public.result(7n,40,0,white(),C.Ply{36,43,0,1},False{},0,acc):
  Public.legal_member(ALeaf{U64.bit(43n)},7n,40,0,white_path(),white(),C.Ply{36,43,0,1},False{},0,acc,
    white_partition(),white_coherent(),white_route(),white_clear(),white_member(),{==})
def actual_black(+acc: U64) -> Public.result(7n,16,0,black(),C.Ply{27,20,0,1},False{},1,acc):
  Public.legal_member(ALeaf{U64.bit(20n)},7n,16,0,black_path(),black(),C.Ply{27,20,0,1},False{},1,acc,
    black_partition(),black_coherent(),black_route(),black_clear(),black_member(),{==})
def actual_white_diagonal(+acc: U64) -> Public.result(4n,25,4,white(),C.Ply{36,43,0,1},True{},0,acc):
  Public.legal_member(ALeaf{U64.bit(43n)},4n,25,4,white_diagonal(),white(),C.Ply{36,43,0,1},True{},0,acc,
    white_partition(),white_coherent(),white_diagonal_route(),white_diagonal_clear(),white_member(),{==})
def actual_black_diagonal(+acc: U64) -> Public.result(4n,38,7,black(),C.Ply{27,20,0,1},True{},1,acc):
  Public.legal_member(ALeaf{U64.bit(20n)},4n,38,7,black_diagonal(),black(),C.Ply{27,20,0,1},True{},1,acc,
    black_partition(),black_coherent(),black_diagonal_route(),black_diagonal_clear(),black_member(),{==})

# An old geometric ray has no hit despite nonzero R/B/Q rows elsewhere.
def clear_path() -> List<&2,Nat>: 25n <> 26n <> 27n <> 28n <> 29n <> 30n <> 31n <> Nil{}
def clear_route() -> {Route.trace(7n,24,0,clear_path()) == True{} : Bool}: {==}
def clear_removed() -> D.disjoint(EP.removed(36,43),Path.attack(clear_path(),C.occupied(white()))): {==}
def old_clear() -> {Hits.raw(7n,24,0,white(),True{},0,U64.zero()) == U64.zero() : U64}: {==}
def actual_clear() -> Public.clear_result(7n,24,0,white(),C.Ply{36,43,0,1},True{},0,U64.zero()):
  Public.clear_member(ALeaf{U64.bit(43n)},7n,24,0,clear_path(),white(),C.Ply{36,43,0,1},True{},0,U64.zero(),
    white_partition(),white_coherent(),clear_route(),clear_removed(),white_member(),{==},old_clear())
