# The four FEN castling rights and their fixed home geometry.
import Base
import ../castle_rights/Words.bend as U
import ../../../legal_probe/Chess.bend as Chess

type Right is Data:
  Wk{}
  Wq{}
  Bk{}
  Bq{}

def bit(r: Right) -> U32:
  match r:
    case Wk{}: 1
    case Wq{}: 2
    case Bk{}: 4
    case Bq{}: 8

def rook(r: Right) -> U32:
  match r:
    case Wk{}: 7
    case Wq{}: 0
    case Bk{}: 63
    case Bq{}: 56

def king(r: Right) -> U32:
  match r:
    case Wk{}: 4
    case Wq{}: 4
    case Bk{}: 60
    case Bq{}: 60

def side(r: Right) -> U32:
  match r:
    case Wk{}: 1
    case Wq{}: 1
    case Bk{}: 0
    case Bq{}: 0

def white(r: Right) -> Bool:
  match r:
    case Wk{}: True{}
    case Wq{}: True{}
    case Bk{}: False{}
    case Bq{}: False{}

def own(r: Right) -> U32:
  match r:
    case Wk{}: 3
    case Wq{}: 3
    case Bk{}: 12
    case Bq{}: 12

def rook_bound(+r: Right) -> {Nat.is_lt(U32.to_nat(rook(r)),64n) == True{} : Bool}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def king_bound(+r: Right) -> {Nat.is_lt(U32.to_nat(king(r)),64n) == True{} : Bool}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def corner(+r: Right) -> {Chess.corner_right(rook(r)) == bit(r) : U32}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def own_bit(+r: Right) -> U.subset(own(r),bit(r)):
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def canonical_side(+r: Right) -> {side(r) == Bool.to_u32(white(r)) : U32}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def own_pick(+r: Right) -> {own(r) == Bool.pick(U32,white(r),3,12) : U32}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}

def white_side(+r: Right) -> {white(r) == U32.is_eq(side(r),1) : Bool}:
  match r:
    case Wk{}: {==}
    case Wq{}: {==}
    case Bk{}: {==}
    case Bq{}: {==}
