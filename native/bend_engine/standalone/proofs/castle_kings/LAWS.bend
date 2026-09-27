# Actual selected-king propagation; these are not historical-rights or attack-reversal proofs.
import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

law transit_preserves_single_moving_king:
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}
  {S.kings(S.mid(b,white,ks),white) == U64.bit(U32.to_nat(S.transit(white,ks))) : U64}

law castling_preserves_single_moving_king:
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {S.kings(S.final(b,white,ks),white) == U64.bit(U32.to_nat(S.dst(white,ks))) : U64}

law move_flips_valid_side:
  for +b: Chess.Board
  for +move: Chess.Ply
  for +white: Bool
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  {Chess.get_turn(Chess.make_move(b,move)) == Bool.to_u32(Bool.not(white)) : U32}

law castling_stage_check_targets_moving_king:
  for table: Array<U64>
  for +b: Chess.Board
  for +white: Bool
  for +ks: Bool
  for +stage: S.Stage
  for good: {B.valid(b) == True{} : Bool}
  for turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}
  for one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}
  for guard: {E.guard(b,ks) == True{} : Bool}
  {Chess.in_check(table,S.stage_board(b,white,ks,stage),Bool.to_u32(white)) ==
   Chess.attacked(table,S.stage_board(b,white,ks,stage),S.stage_square(white,ks,stage),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}
