# Generic mask containment and insertion preservation, lifted to actual color planes.
import Base
import ./Logic.bend as L
import ../board/Spec.bend as B
import ../board/Actual.bend as Actual
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess

def narrow_row(w: Bool,b: Bool,s: Bool,t: Bool,
  old: {Bool.not(Bool.and(t,Bool.or(w,b))) == True{} : Bool},
  sub: {L.covers(s,t) == True{} : Bool}) ->
  {Bool.not(Bool.and(s,Bool.or(w,b))) == True{} : Bool}:
  match w b s t:
    case False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{}: sub
    case False{} True{} True{} True{}: old
    case True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{}: sub
    case True{} False{} True{} True{}: old
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{}: sub
    case True{} True{} True{} True{}: old

def narrow_word(+n: Nat,+w: Word(n),+b: Word(n),+s: Word(n),+t: Word(n),
  +old: {B.fresh_words(n,w,b,t) == True{} : Bool},+sub: {L.sub(n,s,t) == True{} : Bool}) ->
  {B.fresh_words(n,w,b,s) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b s t:
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+sh,+st} WCon{+th,+tt}:
          L.both(Bool.not(Bool.and(sh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,st),
            narrow_row(wh,bh,sh,th,
              L.left(Bool.not(Bool.and(th,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,tt),old),
              L.left(L.covers(sh,th),L.sub(p,st,tt),sub)),
            narrow_word(p,wt,bt,st,tt,
              L.right(Bool.not(Bool.and(th,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,tt),old),
              L.right(L.covers(sh,th),L.sub(p,st,tt),sub)))

def add_row(w: Bool,b: Bool,i: Bool,j: Bool,m: Bool,
  old: {Bool.not(Bool.and(m,Bool.or(w,b))) == True{} : Bool},
  di: {Bool.not(Bool.and(i,m)) == True{} : Bool},
  dj: {Bool.not(Bool.and(j,m)) == True{} : Bool}) ->
  {Bool.not(Bool.and(m,Bool.or(Bool.or(w,i),Bool.or(b,j)))) == True{} : Bool}:
  match w b i j m:
    case False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} True{}: dj
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} True{}: di
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{}: di
    case False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} True{}: old
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} True{}: old
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{}: old
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} True{}: old
    case True{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} True{}: old
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} True{}: old
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} True{}: old
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} True{}: old
    case True{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} True{}: old
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} True{}: old
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{}: old
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{} True{}: old

def add_word(+n: Nat,+w: Word(n),+b: Word(n),+i: Word(n),+j: Word(n),+m: Word(n),
  +old: {B.fresh_words(n,w,b,m) == True{} : Bool},
  +di: {L.apart(n,i,m) == True{} : Bool},+dj: {L.apart(n,j,m) == True{} : Bool}) ->
  {B.fresh_words(n,Word.or(n,w,i),Word.or(n,b,j),m) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b i j m:
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+ih,+it} WCon{+jh,+jt} WCon{+mh,+mt}:
          L.both(Bool.not(Bool.and(mh,Bool.or(Bool.or(wh,ih),Bool.or(bh,jh)))),
            B.fresh_words(p,Word.or(p,wt,it),Word.or(p,bt,jt),mt),
            add_row(wh,bh,ih,jh,mh,
              L.left(Bool.not(Bool.and(mh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,mt),old),
              L.left(Bool.not(Bool.and(ih,mh)),L.apart(p,it,mt),di),
              L.left(Bool.not(Bool.and(jh,mh)),L.apart(p,jt,mt),dj)),
            add_word(p,wt,bt,it,jt,mt,
              L.right(Bool.not(Bool.and(mh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,mt),old),
              L.right(Bool.not(Bool.and(ih,mh)),L.apart(p,it,mt),di),
              L.right(Bool.not(Bool.and(jh,mh)),L.apart(p,jt,mt),dj)))

def zero_apart(+n: Nat,m: Word(n)) -> {L.apart(n,Word.zero(n),m) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match m:
        case WCon{h,t}: zero_apart(p,t)

def narrow(+b: Chess.Board,+small: U64,+large: U64,
  +fresh: {B.fresh_mask(b,large) == True{} : Bool},+sub: {L.subset(small,large) == True{} : Bool}) ->
  {B.fresh_mask(b,small) == True{} : Bool}:
  match b small large:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      L.both(B.fresh_words(32n,wl,bl,sl),B.fresh_words(32n,wh,bh,sh),
        narrow_word(32n,wl,bl,sl,tl,L.left(B.fresh_words(32n,wl,bl,tl),B.fresh_words(32n,wh,bh,th),fresh),L.left(L.sub(32n,sl,tl),L.sub(32n,sh,th),sub)),
        narrow_word(32n,wh,bh,sh,th,L.right(B.fresh_words(32n,wl,bl,tl),B.fresh_words(32n,wh,bh,th),fresh),L.right(L.sub(32n,sl,tl),L.sub(32n,sh,th),sub)))

def after_struct(+b: Chess.Board,+bit: U64,+pc: B.Piece,+white: Bool,+future: U64,
  +fresh: {B.fresh_mask(b,future) == True{} : Bool},
  +apart: {L.disjoint(bit,future) == True{} : Bool}) ->
  {B.fresh_mask(B.insert_mask(b,bit,pc,white),future) == True{} : Bool}:
  match b bit white future:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+il},U32{+ih}} False{} U64{U32{+ml},U32{+mh}}:
      L.both(B.fresh_words(32n,Word.or(32n,wl,Word.zero(32n)),Word.or(32n,bl,il),ml),B.fresh_words(32n,Word.or(32n,wh,Word.zero(32n)),Word.or(32n,bh,ih),mh),
        add_word(32n,wl,bl,Word.zero(32n),il,ml,L.left(B.fresh_words(32n,wl,bl,ml),B.fresh_words(32n,wh,bh,mh),fresh),zero_apart(32n,ml),L.left(L.apart(32n,il,ml),L.apart(32n,ih,mh),apart)),
        add_word(32n,wh,bh,Word.zero(32n),ih,mh,L.right(B.fresh_words(32n,wl,bl,ml),B.fresh_words(32n,wh,bh,mh),fresh),zero_apart(32n,mh),L.right(L.apart(32n,il,ml),L.apart(32n,ih,mh),apart)))
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+il},U32{+ih}} True{} U64{U32{+ml},U32{+mh}}:
      L.both(B.fresh_words(32n,Word.or(32n,wl,il),Word.or(32n,bl,Word.zero(32n)),ml),B.fresh_words(32n,Word.or(32n,wh,ih),Word.or(32n,bh,Word.zero(32n)),mh),
        add_word(32n,wl,bl,il,Word.zero(32n),ml,L.left(B.fresh_words(32n,wl,bl,ml),B.fresh_words(32n,wh,bh,mh),fresh),L.left(L.apart(32n,il,ml),L.apart(32n,ih,mh),apart),zero_apart(32n,ml)),
        add_word(32n,wh,bh,ih,Word.zero(32n),mh,L.right(B.fresh_words(32n,wl,bl,ml),B.fresh_words(32n,wh,bh,mh),fresh),L.right(L.apart(32n,il,ml),L.apart(32n,ih,mh),apart),zero_apart(32n,mh)))

def after_mask(+b: Chess.Board,+sq: U32,+pc: B.Piece,+white: Bool,+future: U64,
  fresh: {B.fresh_mask(b,future) == True{} : Bool},
  apart: {L.disjoint(U64.bit(U32.to_nat(sq)),future) == True{} : Bool}) ->
  {B.fresh_mask(P.put(B.tag(pc),white,sq,b),future) == True{} : Bool}:
  Equal.trans(Bool,B.fresh_mask(P.put(B.tag(pc),white,sq,b),future),
    B.fresh_mask(B.insert_mask(b,U64.bit(U32.to_nat(sq)),pc,white),future),True{},
    Equal.cong(Chess.Board,Bool,x => B.fresh_mask(x,future),P.put(B.tag(pc),white,sq,b),
      B.insert_mask(b,U64.bit(U32.to_nat(sq)),pc,white),Actual.bridge(b,sq,pc,white)),
    after_struct(b,U64.bit(U32.to_nat(sq)),pc,white,future,fresh,apart))
