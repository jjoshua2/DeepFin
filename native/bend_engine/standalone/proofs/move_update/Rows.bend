# Finite Boolean facts checked by Bend, not external acceptance tables.
import Base
import ./Spec.bend as S
import ../board/Spec.bend as B

def clear(+row: B.Row,+mask: Bool,good: {B.row_valid(row) == True{} : Bool}) -> {B.row_valid(S.row_clear(row,mask)) == True{} : Bool}:
  match row:
    case B.Row{False{},False{},False{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},False{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},False{},True{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},False{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{False{},True{},True{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},False{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},False{},True{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},False{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},False{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},False{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},False{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},False{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},False{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},False{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},True{},False{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},True{},False{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},True{},True{},False{}}:
      match mask:
        case False{}: good
        case True{}: {==}
    case B.Row{True{},True{},True{},True{},True{},True{},True{},True{}}:
      match mask:
        case False{}: good
        case True{}: {==}

def fresh(+w: Bool,+b: Bool,+s: Bool,+t: Bool) ->
  {Bool.not(Bool.and(t,Bool.or(Bool.and(w,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))),Bool.and(b,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{}))))))) == True{} : Bool}:
  match w b s t:
    case False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{}: {==}
    case False{} True{} True{} True{}: {==}
    case True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{}: {==}
    case True{} False{} True{} True{}: {==}
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}
