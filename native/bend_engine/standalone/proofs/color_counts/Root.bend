import Base
import ./Bounds.bend as B
import ../validated_root_kings/Kings.bend as K
import ../ray/Frame.bend as F
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def structural(+b: Chess.Board,+good: {Pos.structural(b) == True{} : Bool}) -> B.valid(b):
  +kw=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))),1)
  +kb=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))),1)
  +w=U32.is_le(U64.popcount(Chess.get_white(b)),16)
  +bl=U32.is_le(U64.popcount(Chess.get_black(b)),16)
  +pw=U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_white(b))),8)
  +pb=U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_black(b))),8)
  +clear=U64.is_zero(U64.and(Chess.get_pawns(b),U64.from_parts(4278190080,255)))
  +rest=Bool.and(pw,Bool.and(pb,clear))
  +material=F.right(kb,K.material(b),F.right(kw,Bool.and(kb,K.material(b)),good))
  (F.left(w,Bool.and(bl,rest),material),F.left(bl,rest,F.right(w,Bool.and(bl,rest),material)))
def valid(+b: Chess.Board,good: {Pos.valid(b) == True{} : Bool}) -> B.valid(b):
  structural(b,K.structural(b,good))
