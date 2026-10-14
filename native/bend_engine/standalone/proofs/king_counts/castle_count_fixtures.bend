# Import the full candidate theorem, including nonmoving-color deletion without insertion.
import Base
import ./CastleCount.bend as C
import ./CastleClear.bend as Clear
import ./Actual.bend as K
import ./Frame.bend as Frame
import ./castle_frame_fixtures.bend as Prior
import ../color_counts/U64Count.bend as U
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def white_king() -> K.preserved(Prior.board(1),E.move(Prior.board(1),True{})):
  C.preserved(Prior.board(1),True{},Prior.rows(1),{==})
def white_queen() -> K.preserved(Prior.board(1),E.move(Prior.board(1),False{})):
  C.preserved(Prior.board(1),False{},Prior.rows(1),{==})
def black_king() -> K.preserved(Prior.board(0),E.move(Prior.board(0),True{})):
  C.preserved(Prior.board(0),True{},Prior.rows(0),{==})
def black_queen() -> K.preserved(Prior.board(0),E.move(Prior.board(0),False{})):
  C.preserved(Prior.board(0),False{},Prior.rows(0),{==})
def raw_turn_king() -> K.preserved(Prior.board(2),E.move(Prior.board(2),True{})):
  C.preserved(Prior.board(2),True{},Prior.rows(2),{==})
def raw_turn_queen() -> K.preserved(Prior.board(2),E.move(Prior.board(2),False{})):
  C.preserved(Prior.board(2),False{},Prior.rows(2),{==})

# Counterexample to a plane theorem omitting the source-color equation.
# Pinned U64.from_parts takes high word first.
def zero_color() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.from_parts(0,128),U64.zero(),U64.from_parts(0,16),
    U64.from_parts(0,144),U64.zero(),1,1,64}
def zero_color_rows() -> {B.valid(zero_color()) == True{} : Bool}: {==}
def zero_color_guard() -> {E.guard(zero_color(),True{}) == True{} : Bool}: {==}
def zero_color_parent_count() -> {U.count(U64.and(Chess.get_kings(zero_color()),U64.zero())) == 0n : Nat}: {==}
def zero_color_wrong_model() ->
  {U.count(Frame.model(zero_color(),E.move(zero_color(),True{}),U64.zero(),True{})) == 1n : Nat}: {==}
def zero_color_source() -> {U64.test_bit(U64.zero(),U32.to_nat(E.src(zero_color()))) == False{} : Bool}: {==}
def zero_color_correct_plane() ->
  {U.count(Frame.model(zero_color(),E.move(zero_color(),True{}),U64.zero(),False{})) ==
   U.count(U64.and(Chess.get_kings(zero_color()),U64.zero())) : Nat}:
  C.plane(zero_color(),True{},U64.zero(),False{},zero_color_rows(),zero_color_guard(),{==})
def zero_color_actual() -> K.preserved(zero_color(),E.move(zero_color(),True{})):
  C.preserved(zero_color(),True{},zero_color_rows(),zero_color_guard())

# Landing f1 is clear while destination g1 contains an enemy king.
def destination_occupied() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.from_parts(0,128),U64.zero(),U64.from_parts(0,80),
    U64.from_parts(0,144),U64.from_parts(0,64),1,1,64}
def destination_rows() -> {B.valid(destination_occupied()) == True{} : Bool}: {==}
def destination_guard_false() -> {E.guard(destination_occupied(),True{}) == False{} : Bool}: {==}
def destination_landing_clear() ->
  {U64.and(Chess.get_kings(destination_occupied()),U64.bit(5n)) == U64.zero() : U64}: {==}
def destination_king_present() ->
  {U64.and(U64.bit(U32.to_nat(E.dst(destination_occupied(),True{}))),Chess.get_kings(destination_occupied())) == U64.bit(6n) : U64}: {==}

# Guard true alone allows an overlapping king/rook at h1. Row validity excludes it.
def overlapping_rook() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.from_parts(0,128),U64.zero(),U64.from_parts(0,144),
    U64.from_parts(0,144),U64.zero(),1,1,64}
def overlapping_rows_false() -> {B.valid(overlapping_rook()) == False{} : Bool}: {==}
def overlapping_guard() -> {E.guard(overlapping_rook(),True{}) == True{} : Bool}: {==}
def overlapping_rook_has_king() ->
  {U64.and(U64.bit(U32.to_nat(E.rook(overlapping_rook(),True{}))),Chess.get_kings(overlapping_rook())) == U64.bit(7n) : U64}: {==}
