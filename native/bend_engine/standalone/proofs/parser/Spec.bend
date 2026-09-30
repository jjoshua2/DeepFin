# Observations of the actual layout state and optional result, not replacement parsing.
import Base
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B

def live(l: Position.Layout) -> Bool:
  Position.Layout{b,file,rank,valid} = l
  valid

def board(l: Position.Layout) -> Chess.Board:
  Position.Layout{b,file,rank,valid} = l
  b

def metadata(l: Position.Layout) -> U32 & (U32 & U32):
  B.metadata(board(l))

def accepted(result: Maybe<&2,Chess.Board>) -> Bool:
  match result:
    case None{}: False{}
    case Some{b}: True{}

# Independent six-kind character encoding used only by the one-character contract.
def lowercase(p: B.Piece) -> U32:
  match p:
    case B.Pawn{}: 112
    case B.Knight{}: 110
    case B.Bishop{}: 98
    case B.Rook{}: 114
    case B.Queen{}: 113
    case B.King{}: 107

def letter(p: B.Piece,white: Bool) -> U32:
  +c = lowercase(p)
  Bool.pick(U32,white,U32.sub(c,32),c)
