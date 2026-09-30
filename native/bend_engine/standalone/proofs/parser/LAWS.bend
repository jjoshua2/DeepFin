# Actual layout traversal and accepted-prefix safety; cursor freshness remains separate.
import Base
import ./Spec.bend as S
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B

law layout_append_exact:
  for +prefix: String
  for +suffix: String
  for +l: Position.Layout
  {Position.layout(String.append(prefix,suffix),l) == Position.layout(suffix,Position.layout(prefix,l)) : Position.Layout}

law invalid_layout_rejected:
  for +s: String
  for +l: Position.Layout
  for e: {S.live(l) == False{} : Bool}
  {Position.layout_result(Position.layout(s,l)) == None{} : Maybe<&2,Chess.Board>}

law accepted_layout_has_valid_prefix:
  for +prefix: String
  for +suffix: String
  for +l: Position.Layout
  for e: {S.accepted(Position.layout_result(Position.layout(String.append(prefix,suffix),l))) == True{} : Bool}
  {S.live(Position.layout(prefix,l)) == True{} : Bool}

law layout_preserves_metadata:
  for +s: String
  for +l: Position.Layout
  {S.metadata(Position.layout(s,l)) == S.metadata(l) : U32 & (U32 & U32)}

law typed_piece_transition:
  for +p: B.Piece
  for +white: Bool
  for +b: Chess.Board
  for +file: U32
  for +rank: U32
  for +valid: Bool
  {Position.layout(SCon{Chr{S.letter(p,white)},SNil{}},Position.Layout{b,file,rank,valid}) ==
    Position.Layout{Position.put(B.tag(p),white,U32.add(file,U32.mul(rank,8)),b),U32.inc(file),rank,
      Bool.and(valid,U32.is_lt(file,8))} : Position.Layout}
