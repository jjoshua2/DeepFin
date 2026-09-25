# Representation-level promotion and en-passant updates; no legal-move claim.
import Base
import ../board/Spec.bend as B
import ../move_update/Spec.bend as M
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../move_update/Selection.bend as Selection
import ../../../legal_probe/Chess.bend as Chess

type Promotion is Data:
  Knight{}
  Bishop{}
  Rook{}
  Queen{}

def piece(p: Promotion) -> B.Piece:
  match p:
    case Knight{}: B.Knight{}
    case Bishop{}: B.Bishop{}
    case Rook{}: B.Rook{}
    case Queen{}: B.Queen{}

def tag(p: Promotion) -> U32:
  B.tag(piece(p))

def target(dst: U32) -> U64:
  U64.bit(U32.to_nat(dst))

def promotion_expected(+b: Chess.Board,+src: U32,+dst: U32,p: Promotion) -> Chess.Board:
  Core.assembled(b,M.removal(target(src),target(dst)),target(dst),piece(p),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src)))

# Source and the off-destination victim are both removed. The destination is
# cleared independently, matching the actual flag1 branch even on raw inputs.
def removal(+src: U32,+dst: U32) -> U64:
  U64.or(U64.or(target(src),target(U32.xor(dst,8))),U64.or(target(dst),U64.zero()))

def ep_expected(+b: Chess.Board,+src: U32,+dst: U32) -> Chess.Board:
  Core.assembled(b,removal(src,dst),target(dst),Selection.piece(b,src),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src)))
