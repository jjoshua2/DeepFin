# Exact typed-character transition, including actual insertion and cursor fields.
import Base
import ./Spec.bend as S
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as BS

def step(+p: BS.Piece,+white: Bool,+b: Chess.Board,+file: U32,+rank: U32,+valid: Bool) ->
  {Position.layout(SCon{Chr{S.letter(p,white)},SNil{}},Position.Layout{b,file,rank,valid}) ==
    Position.Layout{Position.put(BS.tag(p),white,U32.add(file,U32.mul(rank,8)),b),U32.inc(file),rank,
      Bool.and(valid,U32.is_lt(file,8))} : Position.Layout}:
  match p white:
    case BS.Pawn{} False{}: {==}
    case BS.Pawn{} True{}: {==}
    case BS.Knight{} False{}: {==}
    case BS.Knight{} True{}: {==}
    case BS.Bishop{} False{}: {==}
    case BS.Bishop{} True{}: {==}
    case BS.Rook{} False{}: {==}
    case BS.Rook{} True{}: {==}
    case BS.Queen{} False{}: {==}
    case BS.Queen{} True{}: {==}
    case BS.King{} False{}: {==}
    case BS.King{} True{}: {==}
