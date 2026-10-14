# Row partition plus an actual rook bit excludes a king at the rook source.
import Base
import ../board/Spec.bend as B
import ../decoder/Spec.bend as Row
import ../decoder/Projection.bend as Projection
import ../candidate_king_destination/Algebra.bend as A
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def rook(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl}=row
  r
def king(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl}=row
  k
def rook_clear(+row: B.Row,good: {B.row_valid(row) == True{} : Bool},
  yes: {rook(row) == True{} : Bool}) -> {king(row) == False{} : Bool}:
  match row:
    case B.Row{p,n,b,r,q,False{},w,bl}: {==}
    case B.Row{p,n,b,False{},q,True{},w,bl}: Old.false_true({True{} == False{} : Bool},yes)
    case B.Row{False{},False{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},False{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{False{},True{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},False{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},False{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},False{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},False{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},False{},True{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},True{},False{}}: Old.false_true({True{} == False{} : Bool},good)
    case B.Row{True{},True{},True{},True{},True{},True{},True{},True{}}: Old.false_true({True{} == False{} : Bool},good)

def square(+b: Chess.Board,+s: U32,good: {B.valid(b) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  yes: {U64.test_bit(Chess.get_rooks(b),U32.to_nat(s)) == True{} : Bool}) ->
  {U64.and(U64.bit(U32.to_nat(s)),Chess.get_kings(b)) == U64.zero() : U64}:
  A.test_zero(Chess.get_kings(b),U32.to_nat(s),rook_clear(Row.row(b,s),Projection.valid(b,s,good,bound),yes))
