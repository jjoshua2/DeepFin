# Four orthodox coordinate routes, not a certificate of legal castling.
import Base
import ./Actual.bend as C
import ./Fresh.bend as F
import ../move_update/Spec.bend as S

type Route is Data:
  WhiteKing{}
  WhiteQueen{}
  BlackKing{}
  BlackQueen{}

def src(r: Route) -> U32:
  match r:
    case WhiteKing{}: 4
    case WhiteQueen{}: 4
    case BlackKing{}: 60
    case BlackQueen{}: 60

def dst(r: Route) -> U32:
  match r:
    case WhiteKing{}: 6
    case WhiteQueen{}: 2
    case BlackKing{}: 62
    case BlackQueen{}: 58

def source(+r: Route) -> U64:
  U64.or(U64.bit(U32.to_nat(src(r))),U64.bit(U32.to_nat(C.rook_from(dst(r)))))

def target(r: Route) -> U64:
  U64.bit(U32.to_nat(dst(r)))

def rook(+r: Route) -> U64:
  C.rook_to(src(r),dst(r))

def masks(+r: Route) -> {C.mask(src(r),dst(r)) == S.removal(source(r),target(r)) : U64}:
  match r:
    case WhiteKing{}: {==}
    case WhiteQueen{}: {==}
    case BlackKing{}: {==}
    case BlackQueen{}: {==}

def distinct(+r: Route) -> {F.disjoint(target(r),rook(r)) == True{} : Bool}:
  match r:
    case WhiteKing{}: {==}
    case WhiteQueen{}: {==}
    case BlackKing{}: {==}
    case BlackQueen{}: {==}
