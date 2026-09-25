# Representation-level deletion and non-castling move model, not legal chess.
import Base
import ../board/Spec.bend as B
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def row_clear(row: B.Row,+mask: Bool) -> B.Row:
  B.Row{p,n,b,r,q,k,w,bl} = row
  B.Row{Bool.and(p,Bool.not(mask)),Bool.and(n,Bool.not(mask)),Bool.and(b,Bool.not(mask)),Bool.and(r,Bool.not(mask)),Bool.and(q,Bool.not(mask)),Bool.and(k,Bool.not(mask)),Bool.and(w,Bool.not(mask)),Bool.and(bl,Bool.not(mask))}

def erase(b: Chess.Board,+mask: U64) -> Chess.Board:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  Chess.Board{U64.and_not(p,mask),U64.and_not(n,mask),U64.and_not(bi,mask),U64.and_not(r,mask),U64.and_not(q,mask),U64.and_not(k,mask),U64.and_not(w,mask),U64.and_not(bl,mask),turn,rights,ep}

def removal(+source: U64,+target: U64) -> U64:
  U64.or(U64.or(source,target),U64.or(target,U64.zero()))

def replace(b: Chess.Board,+source: U64,+target: U64,piece: B.Piece,white: Bool) -> Chess.Board:
  B.insert_mask(erase(b,removal(source,target)),target,piece,white)
