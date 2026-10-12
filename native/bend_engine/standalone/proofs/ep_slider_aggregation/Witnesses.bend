# Full four-ray gate effects, checked separately from boundary and occurrence witnesses.
import Base
import ./Rays.bend as Rays
import ./Slider.bend as Slider
import ../ep_geometric_hits/Fixtures.bend as Boards
import ../ep_geometric_hits/Plane.bend as Plane
import ../board/Spec.bend as Board
import ../../../legal_probe/Chess.bend as C

def white_rook_old() -> {Slider.raw(40,Boards.white(),False{},0) == U64.bit(47n) : U64}: {==}
def white_rook_new() -> {Slider.raw(40,Plane.moved(Boards.white(),36,43),False{},0) == U64.zero() : U64}: {==}
def black_rook_old() -> {Slider.raw(16,Boards.black(),False{},1) == U64.or(U64.bit(23n),U64.bit(8n)) : U64}: {==}
def black_rook_new() -> {Slider.raw(16,Plane.moved(Boards.black(),27,20),False{},1) == U64.bit(8n) : U64}: {==}
def black_rook_not_clear() -> {U64.is_zero(Slider.raw(16,Plane.moved(Boards.black(),27,20),False{},1)) == False{} : Bool}: {==}
def white_bishop_old() -> {Slider.raw(25,Boards.white(),True{},0) == U64.bit(61n) : U64}: {==}
def white_bishop_new() -> {Slider.raw(25,Plane.moved(Boards.white(),36,43),True{},0) == U64.zero() : U64}: {==}
def black_bishop_old() -> {Slider.raw(38,Boards.black(),True{},1) == U64.bit(2n) : U64}: {==}
def black_bishop_new() -> {Slider.raw(38,Plane.moved(Boards.black(),27,20),True{},1) == U64.zero() : U64}: {==}
def queen_old() -> {Slider.raw(54,Boards.white(),False{},0) == U64.bit(55n) : U64}: {==}
def queen_new() -> {Slider.raw(54,Plane.moved(Boards.white(),36,43),False{},0) == U64.bit(55n) : U64}: {==}
def fourth_bishop_ray() -> {Rays.mask(38,7,Boards.black()) == U64.or(U64.or(U64.bit(29n),U64.bit(20n)),U64.or(U64.bit(11n),U64.bit(2n))) : U64}: {==}

def promotion_zero_needed() -> {U64.is_eq(Plane.plane(C.make_move(Boards.white(),C.Ply{36,43,3,1}),Board.Rook{},1),U64.and_not(Plane.plane(Boards.white(),Board.Rook{},1),Plane.mask(36,43))) == False{} : Bool}: {==}
