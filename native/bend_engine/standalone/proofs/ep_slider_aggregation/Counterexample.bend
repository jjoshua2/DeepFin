# Exact immutable parent board and OLD trace, separated from occurrence imports.
import Base
import ./Slider.bend as Slider
import ../ep_geometric_hits/Plane.bend as Plane
import ../board/Spec.bend as Board
import ../ray/Path.bend as Path
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as C

def discovered() -> C.Board:
  C.Board{U64.or(U64.bit(36n),U64.bit(35n)),U64.zero(),U64.zero(),U64.bit(39n),U64.zero(),U64.or(U64.bit(32n),U64.bit(63n)),
    U64.or(U64.bit(32n),U64.bit(36n)),U64.or(U64.or(U64.bit(63n),U64.bit(35n)),U64.bit(39n)),1,0,43}
def discovered_path() -> List<&2,Nat>: 33n <> 34n <> 35n <> 36n <> 37n <> 38n <> 39n <> Nil{}

# Same validated-parent, source-only observation and actual four-ray gate propositions.
def discovered_partition() -> {Board.valid(discovered()) == True{} : Bool}: {==}
def discovered_coherent() -> {Position.valid_ep(43,discovered()) == True{} : Bool}: {==}
def discovered_source_only() -> {U64.and(U64.bit(36n),Path.attack(discovered_path(),C.occupied(discovered()))) == U64.zero() : U64}: {==}
def discovered_old_hit() -> {Slider.raw(32,discovered(),False{},0) == U64.zero() : U64}: {==}
def discovered_new_hit() -> {Slider.raw(32,Plane.moved(discovered(),36,43),False{},0) == U64.bit(39n) : U64}: {==}
def discovered_not_subset() ->
  {U64.is_eq(U64.and(Slider.raw(32,discovered(),False{},0),Slider.raw(32,Plane.moved(discovered(),36,43),False{},0)),
    Slider.raw(32,Plane.moved(discovered(),36,43),False{},0)) == False{} : Bool}: {==}
