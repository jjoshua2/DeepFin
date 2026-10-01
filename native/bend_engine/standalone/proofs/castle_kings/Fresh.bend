import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Rows.bend as Rows
import ../board/Spec.bend as B
import ../ray/Frame.bend as F

def word(+width: Nat,+p: Word(width),+n: Word(width),+b: Word(width),+r: Word(width),+q: Word(width),+k: Word(width),+w: Word(width),+bl: Word(width),+mask: Word(width),
  +good: {B.words(width,p,n,b,r,q,k,w,bl) == True{} : Bool},
  +fresh: {B.fresh_words(width,w,bl,mask) == True{} : Bool}) ->
  {Word.and(width,k,mask) == Word.zero(width) : Word(width)}:
  match width:
    case 0n: {==}
    case 1n+t:
      match p n b r q k w bl mask:
        case WCon{+ph,+pt} WCon{+nh,+nt} WCon{+bh,+bt} WCon{+rh,+rt} WCon{+qh,+qt} WCon{+kh,+kt} WCon{+wh,+wt} WCon{+blh,+blt} WCon{+mh,+mt}:
          Equal.trans(Word(1n+t),WCon{Bool.and(kh,mh),Word.and(t,kt,mt)},WCon{False{},Word.and(t,kt,mt)},Word.zero(1n+t),
            Equal.cong(Bool,Word(1n+t),h => WCon{h,Word.and(t,kt,mt)},Bool.and(kh,mh),False{},
              Rows.fresh_king(B.Row{ph,nh,bh,rh,qh,kh,wh,blh},mh,
                F.left(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good),
                F.left(Bool.not(Bool.and(mh,Bool.or(wh,blh))),B.fresh_words(t,wt,blt,mt),fresh))),
            Equal.cong(Word(t),Word(1n+t),v => WCon{False{},v},Word.and(t,kt,mt),Word.zero(t),
              word(t,pt,nt,bt,rt,qt,kt,wt,blt,mt,
                F.right(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good),
                F.right(Bool.not(Bool.and(mh,Bool.or(wh,blh))),B.fresh_words(t,wt,blt,mt),fresh))))


def board(+board: Chess.Board,+mask: U64,+good: {B.valid(board) == True{} : Bool},+fresh: {B.fresh_mask(board,mask) == True{} : Bool}) ->
  {U64.and(Chess.get_kings(board),mask) == U64.zero() : U64}:
  match board mask:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},U64{U32{+wl},U32{+wh}},U64{U32{+bll},U32{+blh}},turn,rights,ep} U64{U32{+ml},U32{+mh}}:
      Equal.trans(U64,
        U64{U32{Word.and(32n,kl,ml)},U32{Word.and(32n,kh,mh)}},
        U64{U32{Word.zero(32n)},U32{Word.and(32n,kh,mh)}},U64.zero(),
        Equal.cong(Word(32n),U64,x => U64{U32{x},U32{Word.and(32n,kh,mh)}},Word.and(32n,kl,ml),Word.zero(32n),
          word(32n,pl,nl,bl,rl,ql,kl,wl,bll,ml,F.left(B.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.left(B.fresh_words(32n,wl,bll,ml),B.fresh_words(32n,wh,blh,mh),fresh))),
        Equal.cong(Word(32n),U64,x => U64{U32{Word.zero(32n)},U32{x}},Word.and(32n,kh,mh),Word.zero(32n),
          word(32n,ph,nh,bh,rh,qh,kh,wh,blh,mh,F.right(B.words(32n,pl,nl,bl,rl,ql,kl,wl,bll),B.words(32n,ph,nh,bh,rh,qh,kh,wh,blh),good),F.right(B.fresh_words(32n,wl,bll,ml),B.fresh_words(32n,wh,blh,mh),fresh))))
