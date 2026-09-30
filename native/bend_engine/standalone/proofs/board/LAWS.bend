import Base
import ./Spec.bend as S
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

law empty_representation:
  {S.valid(Position.empty()) == True{} : Bool}

law initial_representation:
  {S.valid(Position.board(Position.start())) == True{} : Bool}

law fresh_put_preserves_representation:
  for +b: Chess.Board
  for +sq: U32
  for +piece: S.Piece
  for +white: Bool
  for good: {S.valid(b) == True{} : Bool}
  for fresh: {S.fresh(b,sq) == True{} : Bool}
  {S.valid(Position.put(S.tag(piece),white,sq,b)) == True{} : Bool}

law put_preserves_metadata:
  for +b: Chess.Board
  for kind: U32
  for white: Bool
  for sq: U32
  {S.metadata(Position.put(kind,white,sq,b)) == S.metadata(b) : U32 & (U32 & U32)}

law metadata_preserves_representation:
  for +b: Chess.Board
  for turn: U32
  for rights: U32
  for ep: U32
  {S.valid(Position.metadata(b,turn,rights,ep)) == S.valid(b) : Bool}

law put_exact_bitboard_update:
  for +b: Chess.Board
  for +sq: U32
  for +piece: S.Piece
  for +white: Bool
  {Position.put(S.tag(piece),white,sq,b) == S.insert_mask(b,U64.bit(U32.to_nat(sq)),piece,white) : Chess.Board}
