# Certified connection to the actual parser helper, not a replacement engine.
import Base
import ./Spec.bend as S
import ./Words.bend as W
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J

# General mask insertion is a proof-side intermediate. The public helper uses
# the actual single-square mask and the bridge below includes the complete Board.
def bridge(+b: Chess.Board,+sq: U32,+piece: S.Piece,+white: Bool) ->
  {Position.put(S.tag(piece),white,sq,b) == S.insert_mask(b,U64.bit(U32.to_nat(sq)),piece,white) : Chess.Board}:
  match b:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep}: {==}

def mask_valid(+b: Chess.Board,+mask: U64,+piece: S.Piece,+white: Bool,
  +good: {S.valid(b) == True{} : Bool},+fresh: {S.fresh_mask(b,mask) == True{} : Bool}) ->
  {S.valid(S.insert_mask(b,mask,piece,white)) == True{} : Bool}:
  match b mask piece white:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Pawn{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,ml),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,mh),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Pawn{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Pawn{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Pawn{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,ml),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,mh),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Pawn{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Pawn{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Knight{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,ml),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,mh),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Knight{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Knight{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Knight{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,ml),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,mh),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Knight{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Knight{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Bishop{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,ml),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,mh),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Bishop{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Bishop{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Bishop{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,ml),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,mh),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Bishop{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Bishop{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Rook{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,ml),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,mh),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Rook{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Rook{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Rook{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,ml),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,mh),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Rook{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Rook{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Queen{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,ml),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,mh),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Queen{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Queen{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.Queen{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,ml),Word.or(32n,kl,Word.zero(32n)),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,mh),Word.or(32n,kh,Word.zero(32n)),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.Queen{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.Queen{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.King{} False{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,ml),Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bll,ml)),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,mh),Word.or(32n,wh,Word.zero(32n)),Word.or(32n,blh,mh)),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.King{},False{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.King{},False{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} S.King{} True{}:
      J.and_true(S.words(32n,Word.or(32n,pl,Word.zero(32n)),Word.or(32n,nl,Word.zero(32n)),Word.or(32n,bl,Word.zero(32n)),Word.or(32n,rl,Word.zero(32n)),Word.or(32n,ql,Word.zero(32n)),Word.or(32n,kl,ml),Word.or(32n,wl,ml),Word.or(32n,bll,Word.zero(32n))),S.words(32n,Word.or(32n,ph,Word.zero(32n)),Word.or(32n,nh,Word.zero(32n)),Word.or(32n,bh,Word.zero(32n)),Word.or(32n,rh,Word.zero(32n)),Word.or(32n,qh,Word.zero(32n)),Word.or(32n,kh,mh),Word.or(32n,wh,mh),Word.or(32n,blh,Word.zero(32n))),
        W.insert(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,S.King{},True{},F.left(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)),
        W.insert(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,S.King{},True{},F.right(S.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),S.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(S.fresh_words(32n,wl,bll,ml),S.fresh_words(32n,wh,blh,mh),fresh)))

def insert(+b: Chess.Board,+sq: U32,+piece: S.Piece,+white: Bool,
  good: {S.valid(b) == True{} : Bool},fresh: {S.fresh(b,sq) == True{} : Bool}) ->
  {S.valid(Position.put(S.tag(piece),white,sq,b)) == True{} : Bool}:
  Equal.trans(Bool,S.valid(Position.put(S.tag(piece),white,sq,b)),S.valid(S.insert_mask(b,U64.bit(U32.to_nat(sq)),piece,white)),True{},
    Equal.cong(Chess.Board,Bool,x => S.valid(x),Position.put(S.tag(piece),white,sq,b),S.insert_mask(b,U64.bit(U32.to_nat(sq)),piece,white),bridge(b,sq,piece,white)),
    mask_valid(b,U64.bit(U32.to_nat(sq)),piece,white,good,fresh))

def put_metadata(+b: Chess.Board,kind: U32,white: Bool,sq: U32) ->
  {S.metadata(Position.put(kind,white,sq,b)) == S.metadata(b) : U32 & (U32 & U32)}:
  match b:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep}: {==}

def metadata(+b: Chess.Board,turn: U32,rights: U32,ep: U32) ->
  {S.valid(Position.metadata(b,turn,rights,ep)) == S.valid(b) : Bool}:
  match b:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bil},U32{bih}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},U64{U32{wl},U32{wh}},U64{U32{bll},U32{blh}},oldturn,oldrights,oldep}: {==}
