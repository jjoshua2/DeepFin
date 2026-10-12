import Base
import ../ordinary_slider_preservation/Algebra.bend as A
import ../ordinary_slider_preservation/Check.bend as C
import ../ordinary_king_frame/Enemy.bend as Enemy
import ../ordinary_king_frame/Frame.bend as F
import ../board/Spec.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../prepared_ray_coverage/Coverage.bend as G
import ../ray/Algebra.bend as R
import ../ray/Slider.bend as Slider
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def hits(+c: O.Cells,+b: Chess.Board,+q: Nat,+by: U32) -> U64:
  U64.and(C.prefix(c,b,U32.from_nat(q),by),Chess.color(b,by))
def pawn(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32) -> U64:
  U64.and(C.pawn(c,U32.from_nat(q),U32.xor(side,1)),Enemy.enemy_kind(b,side,B.Pawn{}))
def knight(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32) -> U64:
  U64.and(C.knight(c,U32.from_nat(q)),Enemy.enemy_kind(b,side,B.Knight{}))
def king(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32) -> U64:
  U64.and(C.king(c,U32.from_nat(q)),Enemy.enemy_kind(b,side,B.King{}))
def group(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32) -> U64:
  U64.or(U64.or(pawn(c,b,q,side),knight(c,b,q,side)),king(c,b,q,side))

# Distribute only the enemy mask. The raw prefix is retained and may be nonzero.
def shape(+c: O.Cells,+b: Chess.Board,+q: Nat,+side: U32) ->
  {hits(c,b,q,U32.xor(side,1)) == group(c,b,q,side) : U64}:
  +by = U32.xor(side,1)
  +enemy = Chess.color(b,by)
  +pa = C.pawn(c,U32.from_nat(q),by)
  +na = C.knight(c,U32.from_nat(q))
  +ka = C.king(c,U32.from_nat(q))
  +pr = U64.and(pa,Chess.get_pawns(b))
  +nr = U64.and(na,Chess.get_knights(b))
  +kr = U64.and(ka,Chess.get_kings(b))
  Equal.trans(U64,hits(c,b,q,by),
    U64.or(U64.and(U64.or(pr,nr),enemy),U64.and(kr,enemy)),group(c,b,q,side),
    L.and_or(U64.or(pr,nr),kr,enemy),
    Slider.combine(U64.and(U64.or(pr,nr),enemy),U64.or(pawn(c,b,q,side),knight(c,b,q,side)),
      U64.and(kr,enemy),king(c,b,q,side),
      Equal.trans(U64,U64.and(U64.or(pr,nr),enemy),U64.or(U64.and(pr,enemy),U64.and(nr,enemy)),
        U64.or(pawn(c,b,q,side),knight(c,b,q,side)),L.and_or(pr,nr,enemy),
        Slider.combine(U64.and(pr,enemy),pawn(c,b,q,side),U64.and(nr,enemy),knight(c,b,q,side),
          R.and_assoc(pa,Chess.get_pawns(b),enemy),R.and_assoc(na,Chess.get_knights(b),enemy))),
      R.and_assoc(ka,Chess.get_kings(b),enemy)))

def piece_subset(+mask: U64,+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  L.subset(U64.and(mask,Enemy.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece)),
    U64.and(mask,Enemy.enemy_kind(b,Chess.get_turn(b),piece))):
  A.and_mono(mask,mask,Enemy.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),
    Enemy.enemy_kind(b,Chess.get_turn(b),piece),L.reflexive(mask),Enemy.kind_subset(b,src,dst,piece,white,canonical))

def preserved(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  L.subset(hits(c,F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)),hits(c,b,q,U32.xor(Chess.get_turn(b),1))):
  +post = F.moved(b,src,dst)
  +side = Chess.get_turn(b)
  L.rewrite(group(c,post,q,side),group(c,b,q,side),hits(c,post,q,U32.xor(side,1)),hits(c,b,q,U32.xor(side,1)),
    shape(c,post,q,side),shape(c,b,q,side),
    A.or_mono(U64.or(pawn(c,post,q,side),knight(c,post,q,side)),U64.or(pawn(c,b,q,side),knight(c,b,q,side)),
      king(c,post,q,side),king(c,b,q,side),
      A.or_mono(pawn(c,post,q,side),pawn(c,b,q,side),knight(c,post,q,side),knight(c,b,q,side),
        piece_subset(C.pawn(c,U32.from_nat(q),U32.xor(side,1)),b,src,dst,B.Pawn{},white,canonical),
        piece_subset(C.knight(c,U32.from_nat(q)),b,src,dst,B.Knight{},white,canonical)),
      piece_subset(C.king(c,U32.from_nat(q)),b,src,dst,B.King{},white,canonical)))

def old_clear(+c: O.Cells,+b: Chess.Board,+q: Nat,
  initial: {C.hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}) ->
  {hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}:
  +by = U32.xor(Chess.get_turn(b),1)
  +prefix = C.prefix(c,b,U32.from_nat(q),by)
  +enemy = Chess.color(b,by)
  A.zero_subset(hits(c,b,q,by),C.hits(c,b,q,by),
    A.and_mono(prefix,C.raw(c,b,q,by),enemy,enemy,
      G.lift_left(prefix,U64.or(prefix,C.rook_raw(b,q)),C.bishop_raw(b,q),
        G.lift_left(prefix,prefix,C.rook_raw(b,q),L.reflexive(prefix))),L.reflexive(enemy)),initial)

def clear(+c: O.Cells,+b: Chess.Board,+q: Nat,+src: U32,+dst: U32,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  initial: {C.hits(c,b,q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}) ->
  {hits(c,F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)) == U64.zero() : U64}:
  A.zero_subset(hits(c,F.moved(b,src,dst),q,U32.xor(Chess.get_turn(b),1)),hits(c,b,q,U32.xor(Chess.get_turn(b),1)),
    preserved(c,b,q,src,dst,white,canonical),old_clear(c,b,q,initial))
