# Fresh rook destination survives clearing and a disjoint first insertion.
import Base
import ../board/Spec.bend as B
import ../move_update/Spec.bend as S
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J
import ../../../legal_probe/Chess.bend as Chess

def row(+w: Bool,+b: Bool,+m: Bool,+t: Bool,+r: Bool,+white: Bool,
  e: {Bool.not(Bool.and(r,Bool.or(w,b))) == True{} : Bool},
  d: {Bool.not(Bool.and(r,Bool.or(t,False{}))) == True{} : Bool}) ->
  {Bool.not(Bool.and(r,Bool.or(Bool.or(Bool.and(w,Bool.not(m)),Bool.pick(Bool,white,t,False{})),Bool.or(Bool.and(b,Bool.not(m)),Bool.pick(Bool,white,False{},t))))) == True{} : Bool}:
  match w b m t r white:
    case False{} False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} False{} True{}: {==}
    case False{} False{} False{} True{} True{} False{}: d
    case False{} False{} False{} True{} True{} True{}: d
    case False{} False{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} False{} False{} True{}: {==}
    case False{} False{} True{} False{} True{} False{}: {==}
    case False{} False{} True{} False{} True{} True{}: {==}
    case False{} False{} True{} True{} False{} False{}: {==}
    case False{} False{} True{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} True{} False{}: d
    case False{} False{} True{} True{} True{} True{}: d
    case False{} True{} False{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} True{} False{}: e
    case False{} True{} False{} False{} True{} True{}: e
    case False{} True{} False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} False{} True{}: {==}
    case False{} True{} False{} True{} True{} False{}: e
    case False{} True{} False{} True{} True{} True{}: e
    case False{} True{} True{} False{} False{} False{}: {==}
    case False{} True{} True{} False{} False{} True{}: {==}
    case False{} True{} True{} False{} True{} False{}: {==}
    case False{} True{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} True{} False{}: e
    case False{} True{} True{} True{} True{} True{}: e
    case True{} False{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} True{} False{}: e
    case True{} False{} False{} False{} True{} True{}: e
    case True{} False{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} False{} True{}: {==}
    case True{} False{} False{} True{} True{} False{}: e
    case True{} False{} False{} True{} True{} True{}: e
    case True{} False{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{} True{} False{}: {==}
    case True{} False{} True{} False{} True{} True{}: {==}
    case True{} False{} True{} True{} False{} False{}: {==}
    case True{} False{} True{} True{} False{} True{}: {==}
    case True{} False{} True{} True{} True{} False{}: e
    case True{} False{} True{} True{} True{} True{}: e
    case True{} True{} False{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} True{} False{}: e
    case True{} True{} False{} False{} True{} True{}: e
    case True{} True{} False{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} False{} True{}: {==}
    case True{} True{} False{} True{} True{} False{}: e
    case True{} True{} False{} True{} True{} True{}: e
    case True{} True{} True{} False{} False{} False{}: {==}
    case True{} True{} True{} False{} False{} True{}: {==}
    case True{} True{} True{} False{} True{} False{}: {==}
    case True{} True{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} True{} True{} False{}: e
    case True{} True{} True{} True{} True{} True{}: e

def word(+n: Nat,+w: Word(n),+b: Word(n),+m: Word(n),+t: Word(n),+r: Word(n),+white: Bool,
  +e: {B.fresh_words(n,w,b,r) == True{} : Bool},
  +d: {B.fresh_words(n,t,Word.zero(n),r) == True{} : Bool}) ->
  {B.fresh_words(n,Word.or(n,Word.and(n,w,Word.not(n,m)),Bool.pick(Word(n),white,t,Word.zero(n))),Word.or(n,Word.and(n,b,Word.not(n,m)),Bool.pick(Word(n),white,Word.zero(n),t)),r) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b m t r white:
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+mh,+mt} WCon{+th,+tt} WCon{+rh,+rt} False{}:
          J.and_true(Bool.not(Bool.and(rh,Bool.or(Bool.or(Bool.and(wh,Bool.not(mh)),Bool.pick(Bool,False{},th,False{})),Bool.or(Bool.and(bh,Bool.not(mh)),Bool.pick(Bool,False{},False{},th))))),B.fresh_words(p,Word.or(p,Word.and(p,wt,Word.not(p,mt)),Bool.pick(Word(p),False{},tt,Word.zero(p))),Word.or(p,Word.and(p,bt,Word.not(p,mt)),Bool.pick(Word(p),False{},Word.zero(p),tt)),rt),
            row(wh,bh,mh,th,rh,False{},F.left(Bool.not(Bool.and(rh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,rt),e),F.left(Bool.not(Bool.and(rh,Bool.or(th,False{}))),B.fresh_words(p,tt,Word.zero(p),rt),d)),
            word(p,wt,bt,mt,tt,rt,False{},F.right(Bool.not(Bool.and(rh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,rt),e),F.right(Bool.not(Bool.and(rh,Bool.or(th,False{}))),B.fresh_words(p,tt,Word.zero(p),rt),d)))
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+mh,+mt} WCon{+th,+tt} WCon{+rh,+rt} True{}:
          J.and_true(Bool.not(Bool.and(rh,Bool.or(Bool.or(Bool.and(wh,Bool.not(mh)),Bool.pick(Bool,True{},th,False{})),Bool.or(Bool.and(bh,Bool.not(mh)),Bool.pick(Bool,True{},False{},th))))),B.fresh_words(p,Word.or(p,Word.and(p,wt,Word.not(p,mt)),Bool.pick(Word(p),True{},tt,Word.zero(p))),Word.or(p,Word.and(p,bt,Word.not(p,mt)),Bool.pick(Word(p),True{},Word.zero(p),tt)),rt),
            row(wh,bh,mh,th,rh,True{},F.left(Bool.not(Bool.and(rh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,rt),e),F.left(Bool.not(Bool.and(rh,Bool.or(th,False{}))),B.fresh_words(p,tt,Word.zero(p),rt),d)),
            word(p,wt,bt,mt,tt,rt,True{},F.right(Bool.not(Bool.and(rh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,rt),e),F.right(Bool.not(Bool.and(rh,Bool.or(th,False{}))),B.fresh_words(p,tt,Word.zero(p),rt),d)))

def disjoint(target: U64,rook: U64) -> Bool:
  match target rook:
    case U64{U32{tl},U32{th}} U64{U32{rl},U32{rh}}:
      Bool.and(B.fresh_words(32n,tl,Word.zero(32n),rl),B.fresh_words(32n,th,Word.zero(32n),rh))

def after(+board: Chess.Board,+mask: U64,+target: U64,+rook: U64,piece: B.Piece,+white: Bool,
  +fresh: {B.fresh_mask(board,rook) == True{} : Bool},
  +apart: {disjoint(target,rook) == True{} : Bool}) ->
  {B.fresh_mask(B.insert_mask(S.erase(board,mask),target,piece,white),rook) == True{} : Bool}:
  match board mask target rook white:
    case Chess.Board{p,n,b,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} U64{U32{+tl},U32{+th}} U64{U32{+rl},U32{+rh}} False{}:
      J.and_true(B.fresh_words(32n,Word.or(32n,Word.and(32n,wl,Word.not(32n,ml)),Bool.pick(Word(32n),False{},tl,Word.zero(32n))),Word.or(32n,Word.and(32n,bl,Word.not(32n,ml)),Bool.pick(Word(32n),False{},Word.zero(32n),tl)),rl),B.fresh_words(32n,Word.or(32n,Word.and(32n,wh,Word.not(32n,mh)),Bool.pick(Word(32n),False{},th,Word.zero(32n))),Word.or(32n,Word.and(32n,bh,Word.not(32n,mh)),Bool.pick(Word(32n),False{},Word.zero(32n),th)),rh),
        word(32n,wl,bl,ml,tl,rl,False{},F.left(B.fresh_words(32n,wl,bl,rl),B.fresh_words(32n,wh,bh,rh),fresh),F.left(B.fresh_words(32n,tl,Word.zero(32n),rl),B.fresh_words(32n,th,Word.zero(32n),rh),apart)),
        word(32n,wh,bh,mh,th,rh,False{},F.right(B.fresh_words(32n,wl,bl,rl),B.fresh_words(32n,wh,bh,rh),fresh),F.right(B.fresh_words(32n,tl,Word.zero(32n),rl),B.fresh_words(32n,th,Word.zero(32n),rh),apart)))
    case Chess.Board{p,n,b,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+ml},U32{+mh}} U64{U32{+tl},U32{+th}} U64{U32{+rl},U32{+rh}} True{}:
      J.and_true(B.fresh_words(32n,Word.or(32n,Word.and(32n,wl,Word.not(32n,ml)),Bool.pick(Word(32n),True{},tl,Word.zero(32n))),Word.or(32n,Word.and(32n,bl,Word.not(32n,ml)),Bool.pick(Word(32n),True{},Word.zero(32n),tl)),rl),B.fresh_words(32n,Word.or(32n,Word.and(32n,wh,Word.not(32n,mh)),Bool.pick(Word(32n),True{},th,Word.zero(32n))),Word.or(32n,Word.and(32n,bh,Word.not(32n,mh)),Bool.pick(Word(32n),True{},Word.zero(32n),th)),rh),
        word(32n,wl,bl,ml,tl,rl,True{},F.left(B.fresh_words(32n,wl,bl,rl),B.fresh_words(32n,wh,bh,rh),fresh),F.left(B.fresh_words(32n,tl,Word.zero(32n),rl),B.fresh_words(32n,th,Word.zero(32n),rh),apart)),
        word(32n,wh,bh,mh,th,rh,True{},F.right(B.fresh_words(32n,wl,bl,rl),B.fresh_words(32n,wh,bh,rh),fresh),F.right(B.fresh_words(32n,tl,Word.zero(32n),rl),B.fresh_words(32n,th,Word.zero(32n),rh),apart)))
