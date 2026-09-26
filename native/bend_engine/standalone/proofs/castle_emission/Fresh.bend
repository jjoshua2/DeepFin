# Convert the actual zero-occupancy guard to the existing independent fresh-mask predicate.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J
import ../u64/Words.bend as Old

def zero_step(+h: Bool,+t: Cmp) ->
  {Cmp.is_eq(Word.cmp.fin(h,False{},t)) == Bool.and(Bool.not(h),Cmp.is_eq(t)) : Bool}:
  match h t:
    case False{} LT{}: {==}
    case False{} EQ{}: {==}
    case False{} GT{}: {==}
    case True{} LT{}: {==}
    case True{} EQ{}: {==}
    case True{} GT{}: {==}

def head(+a: Bool,+b: Bool) -> {Bool.not(Bool.and(a,b)) == Bool.not(Bool.and(b,a)) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def zero_word(+n: Nat,+w: Word(n),+b: Word(n),+m: Word(n)) ->
  {Cmp.is_eq(Word.cmp(n,Word.and(n,Word.or(n,w,b),m),Word.zero(n))) == B.fresh_words(n,w,b,m) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b m:
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+mh,+mt}:
          Equal.trans(Bool,
            Cmp.is_eq(Word.cmp(1n+p,Word.and(1n+p,Word.or(1n+p,WCon{wh,wt},WCon{bh,bt}),WCon{mh,mt}),Word.zero(1n+p))),
            Bool.and(Bool.not(Bool.and(Bool.or(wh,bh),mh)),Cmp.is_eq(Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p)))),
            B.fresh_words(1n+p,WCon{wh,wt},WCon{bh,bt},WCon{mh,mt}),
            zero_step(Bool.and(Bool.or(wh,bh),mh),Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p))),
            Equal.trans(Bool,
              Bool.and(Bool.not(Bool.and(Bool.or(wh,bh),mh)),Cmp.is_eq(Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p)))),
              Bool.and(Bool.not(Bool.and(mh,Bool.or(wh,bh))),Cmp.is_eq(Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p)))),
              B.fresh_words(1n+p,WCon{wh,wt},WCon{bh,bt},WCon{mh,mt}),
              Equal.cong(Bool,Bool,x => Bool.and(x,Cmp.is_eq(Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p)))),
                Bool.not(Bool.and(Bool.or(wh,bh),mh)),Bool.not(Bool.and(mh,Bool.or(wh,bh))),head(Bool.or(wh,bh),mh)),
              Equal.cong(Bool,Bool,x => Bool.and(Bool.not(Bool.and(mh,Bool.or(wh,bh))),x),
                Cmp.is_eq(Word.cmp(p,Word.and(p,Word.or(p,wt,bt),mt),Word.zero(p))),B.fresh_words(p,wt,bt,mt),zero_word(p,wt,bt,mt))))

def zero_board(+b: Chess.Board,+mask: U64) ->
  {U64.is_zero(U64.and(Chess.occupied(b),mask)) == B.fresh_mask(b,mask) : Bool}:
  match b mask:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+ml},U32{+mh}}:
      Equal.trans(Bool,
        Bool.and(Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wl,bl),ml),Word.zero(32n))),Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wh,bh),mh),Word.zero(32n)))),
        Bool.and(B.fresh_words(32n,wl,bl,ml),Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wh,bh),mh),Word.zero(32n)))),
        Bool.and(B.fresh_words(32n,wl,bl,ml),B.fresh_words(32n,wh,bh,mh)),
        Equal.cong(Bool,Bool,x => Bool.and(x,Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wh,bh),mh),Word.zero(32n)))),
          Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wl,bl),ml),Word.zero(32n))),B.fresh_words(32n,wl,bl,ml),zero_word(32n,wl,bl,ml)),
        Equal.cong(Bool,Bool,x => Bool.and(B.fresh_words(32n,wl,bl,ml),x),
          Cmp.is_eq(Word.cmp(32n,Word.and(32n,Word.or(32n,wh,bh),mh),Word.zero(32n))),B.fresh_words(32n,wh,bh,mh),zero_word(32n,wh,bh,mh)))

def sub_words(+n: Nat,small: Word(n),large: Word(n)) -> Bool:
  match n:
    case 0n: True{}
    case 1n+p:
      match small large:
        case WCon{sh,st} WCon{lh,lt}:
          Bool.and(Bool.or(Bool.not(sh),lh),sub_words(p,st,lt))

def subset(small: U64,large: U64) -> Bool:
  match small large:
    case U64{U32{sl},U32{sh}} U64{U32{ll},U32{lh}}:
      Bool.and(sub_words(32n,sl,ll),sub_words(32n,sh,lh))

def shrink_bit(+w: Bool,+b: Bool,+small: Bool,+large: Bool,
  fresh: {Bool.not(Bool.and(large,Bool.or(w,b))) == True{} : Bool},
  sub: {Bool.or(Bool.not(small),large) == True{} : Bool}) ->
  {Bool.not(Bool.and(small,Bool.or(w,b))) == True{} : Bool}:
  match small large:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: Old.false_true({Bool.not(Bool.and(True{},Bool.or(w,b))) == True{} : Bool},sub)
    case True{} True{}: fresh

def shrink_words(+n: Nat,+w: Word(n),+b: Word(n),+small: Word(n),+large: Word(n),
  +fresh: {B.fresh_words(n,w,b,large) == True{} : Bool},+sub: {sub_words(n,small,large) == True{} : Bool}) ->
  {B.fresh_words(n,w,b,small) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b small large:
        case WCon{+wh,+wt} WCon{+bh,+bt} WCon{+sh,+st} WCon{+lh,+lt}:
          J.and_true(Bool.not(Bool.and(sh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,st),
            shrink_bit(wh,bh,sh,lh,
              F.left(Bool.not(Bool.and(lh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,lt),fresh),
              F.left(Bool.or(Bool.not(sh),lh),sub_words(p,st,lt),sub)),
            shrink_words(p,wt,bt,st,lt,
              F.right(Bool.not(Bool.and(lh,Bool.or(wh,bh))),B.fresh_words(p,wt,bt,lt),fresh),
              F.right(Bool.or(Bool.not(sh),lh),sub_words(p,st,lt),sub)))

def shrink(+b: Chess.Board,+small: U64,+large: U64,
  +fresh: {B.fresh_mask(b,large) == True{} : Bool},+sub: {subset(small,large) == True{} : Bool}) ->
  {B.fresh_mask(b,small) == True{} : Bool}:
  match b small large:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+sl},U32{+sh}} U64{U32{+ll},U32{+lh}}:
      J.and_true(B.fresh_words(32n,wl,bl,sl),B.fresh_words(32n,wh,bh,sh),
        shrink_words(32n,wl,bl,sl,ll,F.left(B.fresh_words(32n,wl,bl,ll),B.fresh_words(32n,wh,bh,lh),fresh),F.left(sub_words(32n,sl,ll),sub_words(32n,sh,lh),sub)),
        shrink_words(32n,wh,bh,sh,lh,F.right(B.fresh_words(32n,wl,bl,ll),B.fresh_words(32n,wh,bh,lh),fresh),F.right(sub_words(32n,sl,ll),sub_words(32n,sh,lh),sub)))
