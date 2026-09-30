import Base
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../castling/Route.bend as R
import ../castling/Actual.bend as C

def Laws.producer_source_is_home_king(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}

def Laws.producer_destination_matches_side(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}

def Laws.producer_rook_source_matches_update(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}

def Laws.producer_transit_is_rook_landing(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}

def Laws.producer_between_contains_transit(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}

def Laws.producer_between_contains_destination(r):
  match r:
    case R.WhiteKing{}: {==}
    case R.WhiteQueen{}: {==}
    case R.BlackKing{}: {==}
    case R.BlackQueen{}: {==}
