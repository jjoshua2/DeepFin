# Closed specification fixtures for semantic corruption controls, not public laws.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as Forward

def pawn_board() -> Chess.Board:
  Chess.Board{U64.bit(27n),U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.zero(),
    U64.bit(27n),U64.zero(),1,0,64}

def upper_rook_board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.bit(63n),U64.zero(),U64.zero(),
    U64.zero(),U64.bit(63n),0,0,64}

def queen_board() -> Chess.Board:
  Chess.Board{U64.zero(),U64.zero(),U64.zero(),U64.zero(),U64.bit(2n),U64.zero(),
    U64.bit(2n),U64.zero(),1,0,64}

def blocked_rook_board() -> Chess.Board:
  Chess.Board{U64.bit(1n),U64.zero(),U64.zero(),U64.bit(0n),U64.zero(),U64.zero(),
    U64.bit(0n),U64.bit(1n),1,0,64}

def white_pawn_forward() -> {Forward.attacked(pawn_board(),36n,True{}) == True{} : Bool}:
  {==}

def white_pawn_backward() -> {Forward.attacked(pawn_board(),20n,True{}) == False{} : Bool}:
  {==}

def wrong_color() -> {Forward.attacked(pawn_board(),36n,False{}) == False{} : Bool}:
  {==}

def upper_rook() -> {Forward.attacked(upper_rook_board(),7n,False{}) == True{} : Bool}:
  {==}

def queen_diagonal() -> {Forward.attacked(queen_board(),11n,True{}) == True{} : Bool}:
  {==}

def blocked_rook() -> {Forward.attacked(blocked_rook_board(),2n,True{}) == False{} : Bool}:
  {==}
