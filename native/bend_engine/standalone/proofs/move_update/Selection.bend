# A typed image of the actual decoder; not an independent occupied-square spec.
import Base
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess

def choose(p: Bool,n: Bool,b: Bool,r: Bool,q: Bool) -> B.Piece:
  match p n b r q:
    case False{} False{} False{} False{} False{}: B.King{}
    case False{} False{} False{} False{} True{}: B.Queen{}
    case False{} False{} False{} True{} False{}: B.Rook{}
    case False{} False{} False{} True{} True{}: B.Rook{}
    case False{} False{} True{} False{} False{}: B.Bishop{}
    case False{} False{} True{} False{} True{}: B.Bishop{}
    case False{} False{} True{} True{} False{}: B.Bishop{}
    case False{} False{} True{} True{} True{}: B.Bishop{}
    case False{} True{} False{} False{} False{}: B.Knight{}
    case False{} True{} False{} False{} True{}: B.Knight{}
    case False{} True{} False{} True{} False{}: B.Knight{}
    case False{} True{} False{} True{} True{}: B.Knight{}
    case False{} True{} True{} False{} False{}: B.Knight{}
    case False{} True{} True{} False{} True{}: B.Knight{}
    case False{} True{} True{} True{} False{}: B.Knight{}
    case False{} True{} True{} True{} True{}: B.Knight{}
    case True{} False{} False{} False{} False{}: B.Pawn{}
    case True{} False{} False{} False{} True{}: B.Pawn{}
    case True{} False{} False{} True{} False{}: B.Pawn{}
    case True{} False{} False{} True{} True{}: B.Pawn{}
    case True{} False{} True{} False{} False{}: B.Pawn{}
    case True{} False{} True{} False{} True{}: B.Pawn{}
    case True{} False{} True{} True{} False{}: B.Pawn{}
    case True{} False{} True{} True{} True{}: B.Pawn{}
    case True{} True{} False{} False{} False{}: B.Pawn{}
    case True{} True{} False{} False{} True{}: B.Pawn{}
    case True{} True{} False{} True{} False{}: B.Pawn{}
    case True{} True{} False{} True{} True{}: B.Pawn{}
    case True{} True{} True{} False{} False{}: B.Pawn{}
    case True{} True{} True{} False{} True{}: B.Pawn{}
    case True{} True{} True{} True{} False{}: B.Pawn{}
    case True{} True{} True{} True{} True{}: B.Pawn{}

def tag(+p: Bool,+n: Bool,+b: Bool,+r: Bool,+q: Bool) ->
  {B.tag(choose(p,n,b,r,q)) == Bool.pick(U32,p,0,Bool.pick(U32,n,1,Bool.pick(U32,b,2,Bool.pick(U32,r,3,Bool.pick(U32,q,4,5))))) : U32}:
  match p n b r q:
    case False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} True{}: {==}
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} True{}: {==}
    case True{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{} True{}: {==}

def piece(+board: Chess.Board,+sq: U32) -> B.Piece:
  choose(U64.test_bit(Chess.get_pawns(board),U32.to_nat(sq)),U64.test_bit(Chess.get_knights(board),U32.to_nat(sq)),U64.test_bit(Chess.get_bishops(board),U32.to_nat(sq)),U64.test_bit(Chess.get_rooks(board),U32.to_nat(sq)),U64.test_bit(Chess.get_queens(board),U32.to_nat(sq)))

def actual(+board: Chess.Board,+sq: U32) -> {B.tag(piece(board,sq)) == Chess.piece(board,sq) : U32}:
  tag(U64.test_bit(Chess.get_pawns(board),U32.to_nat(sq)),U64.test_bit(Chess.get_knights(board),U32.to_nat(sq)),U64.test_bit(Chess.get_bishops(board),U32.to_nat(sq)),U64.test_bit(Chess.get_rooks(board),U32.to_nat(sq)),U64.test_bit(Chess.get_queens(board),U32.to_nat(sq)))
