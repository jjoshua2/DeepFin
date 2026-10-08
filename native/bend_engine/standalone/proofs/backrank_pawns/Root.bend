import Base
import ./Mask.bend as M
import ../validated_root_kings/Kings.bend as K
import ../ray/Frame.bend as F
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

# Extract the final, actual structural conjunction; no new FEN assumptions.
def structural(+b: Chess.Board,good: {Pos.structural(b) == True{} : Bool}) -> M.safe(b):
  +kw=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_white(b))),1)
  +kb=U32.is_eq(U64.popcount(U64.and(Chess.get_kings(b),Chess.get_black(b))),1)
  +w=U32.is_le(U64.popcount(Chess.get_white(b)),16)
  +bl=U32.is_le(U64.popcount(Chess.get_black(b)),16)
  +pw=U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_white(b))),8)
  +pb=U32.is_le(U64.popcount(U64.and(Chess.get_pawns(b),Chess.get_black(b))),8)
  +clear=U64.is_zero(U64.and(Chess.get_pawns(b),M.mask()))
  F.right(pb,clear,F.right(pw,Bool.and(pb,clear),
    F.right(bl,Bool.and(pw,Bool.and(pb,clear)),
      F.right(w,Bool.and(bl,Bool.and(pw,Bool.and(pb,clear))),
        F.right(kb,K.material(b),F.right(kw,Bool.and(kb,K.material(b)),good))))))
def valid(+b: Chess.Board,good: {Pos.valid(b) == True{} : Bool}) -> M.safe(b):
  structural(b,K.structural(b,good))
