import Base
import ./Spec.bend as S
import ../u64/Words.bend as Old

def keep(+row: S.Row,+piece: S.Piece,+white: Bool,+bit: Bool,
  good: {S.row_valid(row) == True{} : Bool},fresh: {S.row_fresh(row,bit) == True{} : Bool}) ->
  {S.row_valid(S.row_add(row,piece,white,bit)) == True{} : Bool}:
  match row:
    case S.Row{False{},False{},False{},False{},False{},False{},False{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: {==}
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: {==}
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: {==}
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: {==}
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: {==}
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: {==}
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: {==}
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: {==}
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: {==}
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: {==}
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: {==}
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: {==}
    case S.Row{False{},False{},False{},False{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},False{},True{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},False{},False{},True{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},False{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},False{},True{},False{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},False{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},True{},False{},False{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},False{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},False{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},False{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},False{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},True{},False{},False{},False{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},False{},True{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},False{},True{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},False{},True{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},False{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},True{},False{},False{},False{},False{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{False{},True{},False{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},False{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},False{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{False{},True{},True{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{False{},True{},True{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},False{},False{},True{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},False{},True{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{True{},False{},False{},False{},False{},False{},True{},False{}}:
      match piece white bit:
        case S.Pawn{} False{} False{}: {==}
        case S.Pawn{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Pawn{},False{},True{})) == True{} : Bool},fresh)
        case S.Pawn{} True{} False{}: {==}
        case S.Pawn{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Pawn{},True{},True{})) == True{} : Bool},fresh)
        case S.Knight{} False{} False{}: {==}
        case S.Knight{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Knight{},False{},True{})) == True{} : Bool},fresh)
        case S.Knight{} True{} False{}: {==}
        case S.Knight{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Knight{},True{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} False{} False{}: {==}
        case S.Bishop{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Bishop{},False{},True{})) == True{} : Bool},fresh)
        case S.Bishop{} True{} False{}: {==}
        case S.Bishop{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Bishop{},True{},True{})) == True{} : Bool},fresh)
        case S.Rook{} False{} False{}: {==}
        case S.Rook{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Rook{},False{},True{})) == True{} : Bool},fresh)
        case S.Rook{} True{} False{}: {==}
        case S.Rook{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Rook{},True{},True{})) == True{} : Bool},fresh)
        case S.Queen{} False{} False{}: {==}
        case S.Queen{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Queen{},False{},True{})) == True{} : Bool},fresh)
        case S.Queen{} True{} False{}: {==}
        case S.Queen{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.Queen{},True{},True{})) == True{} : Bool},fresh)
        case S.King{} False{} False{}: {==}
        case S.King{} False{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.King{},False{},True{})) == True{} : Bool},fresh)
        case S.King{} True{} False{}: {==}
        case S.King{} True{} True{}: Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},False{}},S.King{},True{},True{})) == True{} : Bool},fresh)
    case S.Row{True{},False{},False{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},False{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},False{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},False{},True{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},False{},True{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},False{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},False{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},False{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},False{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},False{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},False{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},False{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},False{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},False{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},False{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},False{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},False{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},False{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},False{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},True{},False{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},True{},False{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},True{},False{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},True{},False{},True{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},True{},True{},False{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},True{},True{},False{}},piece,white,bit)) == True{} : Bool},good)
    case S.Row{True{},True{},True{},True{},True{},True{},True{},True{}}:
      Old.false_true({S.row_valid(S.row_add(S.Row{True{},True{},True{},True{},True{},True{},True{},True{}},piece,white,bit)) == True{} : Bool},good)
