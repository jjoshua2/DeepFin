# Four input-only routes and OLD two-removal premises for real full-Ply EP fixtures.
import Base
import ./Rays.bend as Rays
import ./Slider.bend as Slider
import ./consumer.bend as Public
import ../ep_geometric_hits/Fixtures.bend as F
import ../ep_occupancy_ray/EP.bend as EP
import ../ray/Spec.bend as Route
import ../ray/Path.bend as Path
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as C

def white_rook_path0() -> List<&2,Nat>: 41n <> 42n <> 43n <> 44n <> 45n <> 46n <> 47n <> Nil{}
def white_rook_route0() -> {Route.trace(7n,40,0,white_rook_path0()) == True{} : Bool}: {==}
def white_rook_empty0() -> D.disjoint(EP.removed(36,43),Path.attack(white_rook_path0(),C.occupied(F.white()))): {==}
def white_rook_path1() -> List<&2,Nat>: Nil{}
def white_rook_route1() -> {Route.trace(7n,40,1,white_rook_path1()) == True{} : Bool}: {==}
def white_rook_empty1() -> D.disjoint(EP.removed(36,43),Path.attack(white_rook_path1(),C.occupied(F.white()))): {==}
def white_rook_path2() -> List<&2,Nat>: 48n <> 56n <> Nil{}
def white_rook_route2() -> {Route.trace(7n,40,2,white_rook_path2()) == True{} : Bool}: {==}
def white_rook_empty2() -> D.disjoint(EP.removed(36,43),Path.attack(white_rook_path2(),C.occupied(F.white()))): {==}
def white_rook_path3() -> List<&2,Nat>: 32n <> 24n <> 16n <> 8n <> 0n <> Nil{}
def white_rook_route3() -> {Route.trace(7n,40,3,white_rook_path3()) == True{} : Bool}: {==}
def white_rook_empty3() -> D.disjoint(EP.removed(36,43),Path.attack(white_rook_path3(),C.occupied(F.white()))): {==}
def actual_white_rook() -> Public.result(40,F.white(),C.Ply{36,43,0,1},False{},0):
  Public.legal_member(ALeaf{U64.bit(43n)},40,F.white(),C.Ply{36,43,0,1},False{},0,
    white_rook_path0(),white_rook_path1(),white_rook_path2(),white_rook_path3(),F.white_partition(),F.white_coherent(),
    white_rook_route0(),white_rook_route1(),white_rook_route2(),white_rook_route3(),
    white_rook_empty0(),white_rook_empty1(),white_rook_empty2(),white_rook_empty3(),F.white_member(),{==})

def black_rook_path0() -> List<&2,Nat>: 17n <> 18n <> 19n <> 20n <> 21n <> 22n <> 23n <> Nil{}
def black_rook_route0() -> {Route.trace(7n,16,0,black_rook_path0()) == True{} : Bool}: {==}
def black_rook_empty0() -> D.disjoint(EP.removed(27,20),Path.attack(black_rook_path0(),C.occupied(F.black()))): {==}
def black_rook_path1() -> List<&2,Nat>: Nil{}
def black_rook_route1() -> {Route.trace(7n,16,1,black_rook_path1()) == True{} : Bool}: {==}
def black_rook_empty1() -> D.disjoint(EP.removed(27,20),Path.attack(black_rook_path1(),C.occupied(F.black()))): {==}
def black_rook_path2() -> List<&2,Nat>: 24n <> 32n <> 40n <> 48n <> 56n <> Nil{}
def black_rook_route2() -> {Route.trace(7n,16,2,black_rook_path2()) == True{} : Bool}: {==}
def black_rook_empty2() -> D.disjoint(EP.removed(27,20),Path.attack(black_rook_path2(),C.occupied(F.black()))): {==}
def black_rook_path3() -> List<&2,Nat>: 8n <> 0n <> Nil{}
def black_rook_route3() -> {Route.trace(7n,16,3,black_rook_path3()) == True{} : Bool}: {==}
def black_rook_empty3() -> D.disjoint(EP.removed(27,20),Path.attack(black_rook_path3(),C.occupied(F.black()))): {==}
def actual_black_rook() -> Public.result(16,F.black(),C.Ply{27,20,0,1},False{},1):
  Public.legal_member(ALeaf{U64.bit(20n)},16,F.black(),C.Ply{27,20,0,1},False{},1,
    black_rook_path0(),black_rook_path1(),black_rook_path2(),black_rook_path3(),F.black_partition(),F.black_coherent(),
    black_rook_route0(),black_rook_route1(),black_rook_route2(),black_rook_route3(),
    black_rook_empty0(),black_rook_empty1(),black_rook_empty2(),black_rook_empty3(),F.black_member(),{==})

def white_bishop_path0() -> List<&2,Nat>: 34n <> 43n <> 52n <> 61n <> Nil{}
def white_bishop_route0() -> {Route.trace(7n,25,4,white_bishop_path0()) == True{} : Bool}: {==}
def white_bishop_empty0() -> D.disjoint(EP.removed(36,43),Path.attack(white_bishop_path0(),C.occupied(F.white()))): {==}
def white_bishop_path1() -> List<&2,Nat>: 18n <> 11n <> 4n <> Nil{}
def white_bishop_route1() -> {Route.trace(7n,25,5,white_bishop_path1()) == True{} : Bool}: {==}
def white_bishop_empty1() -> D.disjoint(EP.removed(36,43),Path.attack(white_bishop_path1(),C.occupied(F.white()))): {==}
def white_bishop_path2() -> List<&2,Nat>: 32n <> Nil{}
def white_bishop_route2() -> {Route.trace(7n,25,6,white_bishop_path2()) == True{} : Bool}: {==}
def white_bishop_empty2() -> D.disjoint(EP.removed(36,43),Path.attack(white_bishop_path2(),C.occupied(F.white()))): {==}
def white_bishop_path3() -> List<&2,Nat>: 16n <> Nil{}
def white_bishop_route3() -> {Route.trace(7n,25,7,white_bishop_path3()) == True{} : Bool}: {==}
def white_bishop_empty3() -> D.disjoint(EP.removed(36,43),Path.attack(white_bishop_path3(),C.occupied(F.white()))): {==}
def actual_white_bishop() -> Public.result(25,F.white(),C.Ply{36,43,0,1},True{},0):
  Public.legal_member(ALeaf{U64.bit(43n)},25,F.white(),C.Ply{36,43,0,1},True{},0,
    white_bishop_path0(),white_bishop_path1(),white_bishop_path2(),white_bishop_path3(),F.white_partition(),F.white_coherent(),
    white_bishop_route0(),white_bishop_route1(),white_bishop_route2(),white_bishop_route3(),
    white_bishop_empty0(),white_bishop_empty1(),white_bishop_empty2(),white_bishop_empty3(),F.white_member(),{==})

def black_bishop_path0() -> List<&2,Nat>: 47n <> Nil{}
def black_bishop_route0() -> {Route.trace(7n,38,4,black_bishop_path0()) == True{} : Bool}: {==}
def black_bishop_empty0() -> D.disjoint(EP.removed(27,20),Path.attack(black_bishop_path0(),C.occupied(F.black()))): {==}
def black_bishop_path1() -> List<&2,Nat>: 31n <> Nil{}
def black_bishop_route1() -> {Route.trace(7n,38,5,black_bishop_path1()) == True{} : Bool}: {==}
def black_bishop_empty1() -> D.disjoint(EP.removed(27,20),Path.attack(black_bishop_path1(),C.occupied(F.black()))): {==}
def black_bishop_path2() -> List<&2,Nat>: 45n <> 52n <> 59n <> Nil{}
def black_bishop_route2() -> {Route.trace(7n,38,6,black_bishop_path2()) == True{} : Bool}: {==}
def black_bishop_empty2() -> D.disjoint(EP.removed(27,20),Path.attack(black_bishop_path2(),C.occupied(F.black()))): {==}
def black_bishop_path3() -> List<&2,Nat>: 29n <> 20n <> 11n <> 2n <> Nil{}
def black_bishop_route3() -> {Route.trace(7n,38,7,black_bishop_path3()) == True{} : Bool}: {==}
def black_bishop_empty3() -> D.disjoint(EP.removed(27,20),Path.attack(black_bishop_path3(),C.occupied(F.black()))): {==}
def actual_black_bishop() -> Public.result(38,F.black(),C.Ply{27,20,0,1},True{},1):
  Public.legal_member(ALeaf{U64.bit(20n)},38,F.black(),C.Ply{27,20,0,1},True{},1,
    black_bishop_path0(),black_bishop_path1(),black_bishop_path2(),black_bishop_path3(),F.black_partition(),F.black_coherent(),
    black_bishop_route0(),black_bishop_route1(),black_bishop_route2(),black_bishop_route3(),
    black_bishop_empty0(),black_bishop_empty1(),black_bishop_empty2(),black_bishop_empty3(),F.black_member(),{==})

def empty_bishop_path0() -> List<&2,Nat>: 33n <> 42n <> 51n <> 60n <> Nil{}
def empty_bishop_route0() -> {Route.trace(7n,24,4,empty_bishop_path0()) == True{} : Bool}: {==}
def empty_bishop_empty0() -> D.disjoint(EP.removed(36,43),Path.attack(empty_bishop_path0(),C.occupied(F.white()))): {==}
def empty_bishop_path1() -> List<&2,Nat>: 17n <> 10n <> 3n <> Nil{}
def empty_bishop_route1() -> {Route.trace(7n,24,5,empty_bishop_path1()) == True{} : Bool}: {==}
def empty_bishop_empty1() -> D.disjoint(EP.removed(36,43),Path.attack(empty_bishop_path1(),C.occupied(F.white()))): {==}
def empty_bishop_path2() -> List<&2,Nat>: Nil{}
def empty_bishop_route2() -> {Route.trace(7n,24,6,empty_bishop_path2()) == True{} : Bool}: {==}
def empty_bishop_empty2() -> D.disjoint(EP.removed(36,43),Path.attack(empty_bishop_path2(),C.occupied(F.white()))): {==}
def empty_bishop_path3() -> List<&2,Nat>: Nil{}
def empty_bishop_route3() -> {Route.trace(7n,24,7,empty_bishop_path3()) == True{} : Bool}: {==}
def empty_bishop_empty3() -> D.disjoint(EP.removed(36,43),Path.attack(empty_bishop_path3(),C.occupied(F.white()))): {==}
def old_clear() -> {Slider.raw(24,F.white(),True{},0) == U64.zero() : U64}: {==}
def actual_empty_bishop() -> Public.clear_result(24,F.white(),C.Ply{36,43,0,1},True{},0):
  Public.clear_member(ALeaf{U64.bit(43n)},24,F.white(),C.Ply{36,43,0,1},True{},0,
    empty_bishop_path0(),empty_bishop_path1(),empty_bishop_path2(),empty_bishop_path3(),F.white_partition(),F.white_coherent(),
    empty_bishop_route0(),empty_bishop_route1(),empty_bishop_route2(),empty_bishop_route3(),
    empty_bishop_empty0(),empty_bishop_empty1(),empty_bishop_empty2(),empty_bishop_empty3(),F.white_member(),{==},old_clear())

