import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../../../legal_probe/Chess.bend as Chess

law whole_board_roundtrip:
  for +b: Chess.Board
  for good: {B.valid(b) == True{} : Bool}
  {S.restore(S.observe(b)) == b : Chess.Board}

law snapshot_has_64_squares:
  for +b: Chess.Board
  {S.lengths(S.observe(b)) == (32n,32n) : Nat & Nat}

law equal_snapshots_identify_board:
  for +a: Chess.Board
  for +b: Chess.Board
  for ga: {B.valid(a) == True{} : Bool}
  for gb: {B.valid(b) == True{} : Bool}
  for same: {S.observe(a) == S.observe(b) : S.Snapshot}
  {a == b : Chess.Board}

law empty_square_has_empty_observation:
  for +b: Chess.Board
  for +sq: U32
  for good: {B.valid(b) == True{} : Bool}
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  for absent: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == False{} : Bool}
  {D.row(b,sq) == D.encode(D.Empty{}) : B.Row}

law snapshot_square_correspondence:
  for +b: Chess.Board
  for +sq: U32
  for bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}
  {S.square(S.observe(b),sq) == D.square(b,sq) : D.Cell}
