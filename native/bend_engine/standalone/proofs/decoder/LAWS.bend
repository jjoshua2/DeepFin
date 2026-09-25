import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess

law square_projection_valid:
  for +b: Chess.Board
  for +sq: U32
  for good: {B.valid(b) == True{} : Bool}
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  {B.row_valid(S.row(b,sq)) == True{} : Bool}

law abstract_square_roundtrip:
  for +b: Chess.Board
  for +sq: U32
  for good: {B.valid(b) == True{} : Bool}
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  {S.encode(S.square(b,sq)) == S.row(b,sq) : B.Row}

law guarded_decoder_matches_square:
  for +b: Chess.Board
  for +sq: U32
  for good: {B.valid(b) == True{} : Bool}
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  {S.decoded(b,sq) == S.square(b,sq) : S.Cell}

law occupied_decoder_matches_square:
  for +b: Chess.Board
  for +sq: U32
  for good: {B.valid(b) == True{} : Bool}
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  for present: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == True{} : Bool}
  {S.square(b,sq) == S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))} : S.Cell}
