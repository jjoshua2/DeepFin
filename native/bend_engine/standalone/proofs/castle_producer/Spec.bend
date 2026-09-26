# Independent geometry for the four orthodox castling producer routes.
# This describes coordinates/masks only; it is not a legal-move certificate.
import Base
import ../castling/Route.bend as R
import ../castling/Actual.bend as C

def king_side(r: R.Route) -> Bool:
  match r:
    case R.WhiteKing{}: True{}
    case R.WhiteQueen{}: False{}
    case R.BlackKing{}: True{}
    case R.BlackQueen{}: False{}

def home(r: R.Route) -> U32:
  match r:
    case R.WhiteKing{}: 0
    case R.WhiteQueen{}: 0
    case R.BlackKing{}: 56
    case R.BlackQueen{}: 56

def transit(r: R.Route) -> U32:
  U32.add(home(r),Bool.pick(U32,king_side(r),5,3))

def rook_source(r: R.Route) -> U32:
  U32.add(home(r),Bool.pick(U32,king_side(r),7,0))

def right(r: R.Route) -> U32:
  match r:
    case R.WhiteKing{}: 1
    case R.WhiteQueen{}: 2
    case R.BlackKing{}: 4
    case R.BlackQueen{}: 8

def between(r: R.Route) -> U64:
  U64.shln(U64.from_u32(Bool.pick(U32,king_side(r),96,14)),U32.to_nat(home(r)))
