# Two kings of each color; both routes and raw turn2 remain supported.
import Base
import ./CastleFrame.bend as C
import ./Actual.bend as K
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess
def board(turn: U32) -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.from_parts(129,2164260864),U64.zero(),
    U64.from_parts(1048592,268435712),U64.from_parts(1048721,0),U64.from_parts(0,2432696576),turn,15,64}
def rows(+turn: U32) -> {B.valid(board(turn)) == True{} : Bool}: {==}
def parent_counts(+turn: U32) -> {K.white(board(turn)) == 2n : Nat} & {K.black(board(turn)) == 2n : Nat}: ({==},{==})
def white_king() -> C.both(board(1),E.move(board(1),True{})):
  C.guarded(board(1),True{},rows(1),{==})
def white_queen() -> C.both(board(1),E.move(board(1),False{})):
  C.guarded(board(1),False{},rows(1),{==})
def black_king() -> C.both(board(0),E.move(board(0),True{})):
  C.guarded(board(0),True{},rows(0),{==})
def black_queen() -> C.both(board(0),E.move(board(0),False{})):
  C.guarded(board(0),False{},rows(0),{==})
def raw_turn_king() -> C.both(board(2),E.move(board(2),True{})):
  C.guarded(board(2),True{},rows(2),{==})
def raw_turn_queen() -> C.both(board(2),E.move(board(2),False{})):
  C.guarded(board(2),False{},rows(2),{==})
def white_king_counts() -> K.preserved(board(1),E.move(board(1),True{})): ({==},{==})
def white_queen_counts() -> K.preserved(board(1),E.move(board(1),False{})): ({==},{==})
def black_king_counts() -> K.preserved(board(0),E.move(board(0),True{})): ({==},{==})
def black_queen_counts() -> K.preserved(board(0),E.move(board(0),False{})): ({==},{==})
def raw_turn_king_counts() -> K.preserved(board(2),E.move(board(2),True{})): ({==},{==})
def raw_turn_queen_counts() -> K.preserved(board(2),E.move(board(2),False{})): ({==},{==})

# Valid rows, but an enemy king on the rook landing square blocks the guard.
# Unconditionally executing flag2 adds mover ownership to this surviving king.
def blocked() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.from_parts(128,0),U64.zero(),
    U64.from_parts(48,268435456),U64.from_parts(144,0),U64.from_parts(32,268435456),1,1,64}
def blocked_rows() -> {B.valid(blocked()) == True{} : Bool}: {==}
def blocked_guard() -> {E.guard(blocked(),True{}) == False{} : Bool}: {==}
def blocked_parent_counts() -> {K.white(blocked()) == 1n : Nat} & {K.black(blocked()) == 2n : Nat}: ({==},{==})
def blocked_child_counts() ->
  {K.white(Chess.make_move(blocked(),E.move(blocked(),True{}))) == 2n : Nat} &
  {K.black(Chess.make_move(blocked(),E.move(blocked(),True{}))) == 2n : Nat}: ({==},{==})
