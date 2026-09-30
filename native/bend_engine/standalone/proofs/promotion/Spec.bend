# Representation update for the four promotion choices, not move legality.
import Base
import ../board/Spec.bend as Board
import ../move_update/Spec.bend as Move
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as Ordinary
import ../../../legal_probe/Chess.bend as Chess

type Choice is Data:
  Knight{}
  Bishop{}
  Rook{}
  Queen{}

def piece(c: Choice) -> Board.Piece:
  match c:
    case Knight{}: Board.Knight{}
    case Bishop{}: Board.Bishop{}
    case Rook{}: Board.Rook{}
    case Queen{}: Board.Queen{}

def tag(c: Choice) -> U32:
  Board.tag(piece(c))

def from_tag(v: U32) -> Maybe<&2,Choice>:
  match v:
    case 1: Some{Knight{}}
    case 2: Some{Bishop{}}
    case 3: Some{Rook{}}
    case 4: Some{Queen{}}
    case _: None{}

def ply(src: U32,dst: U32,c: Choice) -> Chess.Ply:
  Chess.Ply{src,dst,tag(c),0}

def choices(+src: U32,+dst: U32) -> List<&2,Chess.Ply>:
  ply(src,dst,Knight{}) <> ply(src,dst,Bishop{}) <> ply(src,dst,Rook{}) <> ply(src,dst,Queen{}) <> Nil{}

def expected(+b: Chess.Board,+src: U32,+dst: U32,c: Choice) -> Chess.Board:
  Core.assembled(b,Move.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),piece(c),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    Ordinary.rights(b,src,dst,Chess.piece(b,src)),Ordinary.ep(src,dst,Chess.piece(b,src)))
