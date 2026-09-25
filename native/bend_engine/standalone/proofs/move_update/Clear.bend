import Base
import ./Spec.bend as S
import ./Words.bend as W
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J
import ../../../legal_probe/Chess.bend as Chess

def clear(+board: Chess.Board,+mask: U64,+good: {B.valid(board) == True{} : Bool}) -> {B.valid(S.erase(board,mask)) == True{} : Bool}:
  match board mask:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bil},U32{+bih}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}}:
      J.and_true(B.words(32n,Word.and(32n,pl,Word.not(32n,ml)),Word.and(32n,nl,Word.not(32n,ml)),Word.and(32n,bil,Word.not(32n,ml)),Word.and(32n,rl,Word.not(32n,ml)),Word.and(32n,ql,Word.not(32n,ml)),Word.and(32n,kl,Word.not(32n,ml)),Word.and(32n,wl,Word.not(32n,ml)),Word.and(32n,bll,Word.not(32n,ml))),B.words(32n,Word.and(32n,ph,Word.not(32n,mh)),Word.and(32n,nh,Word.not(32n,mh)),Word.and(32n,bih,Word.not(32n,mh)),Word.and(32n,rh,Word.not(32n,mh)),Word.and(32n,qh,Word.not(32n,mh)),Word.and(32n,kh,Word.not(32n,mh)),Word.and(32n,wh,Word.not(32n,mh)),Word.and(32n,blh,Word.not(32n,mh))),
        W.clear(32n,pl,nl,bil,rl,ql,kl,wl,bll,ml,F.left(B.words(32n,pl,nl,bil,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bih,rh,qh,kh,wh,blh),good)),
        W.clear(32n,ph,nh,bih,rh,qh,kh,wh,blh,mh,F.right(B.words(32n,pl,nl,bil,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bih,rh,qh,kh,wh,blh),good)))

def fresh(+board: Chess.Board,+source: U64,+target: U64) -> {B.fresh_mask(S.erase(board,S.removal(source,target)),target) == True{} : Bool}:
  match board source target:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      J.and_true(B.fresh_words(32n,Word.and(32n,wl,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),Word.and(32n,bl,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),tl),B.fresh_words(32n,Word.and(32n,wh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),Word.and(32n,bh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th),
        W.fresh(32n,wl,bl,sl,tl),W.fresh(32n,wh,bh,sh,th))

def replace(+board: Chess.Board,+source: U64,+target: U64,piece: B.Piece,white: Bool,good: {B.valid(board) == True{} : Bool}) ->
  {B.valid(S.replace(board,source,target,piece,white)) == True{} : Bool}:
  BA.mask_valid(S.erase(board,S.removal(source,target)),target,piece,white,clear(board,S.removal(source,target),good),fresh(board,source,target))
