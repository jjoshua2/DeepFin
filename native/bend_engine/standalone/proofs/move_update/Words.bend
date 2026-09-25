import Base
import ./Rows.bend as Rows
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../ray/Frame.bend as F
import ../ray/Masking.bend as J

def clear(+width: Nat,+p: Word(width),+n: Word(width),+b: Word(width),+r: Word(width),+q: Word(width),+k: Word(width),+w: Word(width),+bl: Word(width),+mask: Word(width),+good: {B.words(width,p,n,b,r,q,k,w,bl) == True{} : Bool}) ->
  {B.words(width,Word.and(width,p,Word.not(width,mask)),Word.and(width,n,Word.not(width,mask)),Word.and(width,b,Word.not(width,mask)),Word.and(width,r,Word.not(width,mask)),Word.and(width,q,Word.not(width,mask)),Word.and(width,k,Word.not(width,mask)),Word.and(width,w,Word.not(width,mask)),Word.and(width,bl,Word.not(width,mask))) == True{} : Bool}:
  match width:
    case 0n: {==}
    case 1n+t:
      match p n b r q k w bl mask:
        case WCon{+ph,+pt} WCon{+nh,+nt} WCon{+bh,+bt} WCon{+rh,+rt} WCon{+qh,+qt} WCon{+kh,+kt} WCon{+wh,+wt} WCon{+blh,+blt} WCon{+maskh,+maskt}:
          J.and_true(B.row_valid(S.row_clear(B.Row{ph,nh,bh,rh,qh,kh,wh,blh},maskh)),B.words(t,Word.and(t,pt,Word.not(t,maskt)),Word.and(t,nt,Word.not(t,maskt)),Word.and(t,bt,Word.not(t,maskt)),Word.and(t,rt,Word.not(t,maskt)),Word.and(t,qt,Word.not(t,maskt)),Word.and(t,kt,Word.not(t,maskt)),Word.and(t,wt,Word.not(t,maskt)),Word.and(t,blt,Word.not(t,maskt))),
            Rows.clear(B.Row{ph,nh,bh,rh,qh,kh,wh,blh},maskh,F.left(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good)),
            clear(t,pt,nt,bt,rt,qt,kt,wt,blt,maskt,F.right(B.row_valid(B.Row{ph,nh,bh,rh,qh,kh,wh,blh}),B.words(t,pt,nt,bt,rt,qt,kt,wt,blt),good)))

def fresh(+width: Nat,+w: Word(width),+b: Word(width),+source: Word(width),+target: Word(width)) ->
  {B.fresh_words(width,Word.and(width,w,Word.not(width,Word.or(width,Word.or(width,source,target),Word.or(width,target,Word.zero(width))))),Word.and(width,b,Word.not(width,Word.or(width,Word.or(width,source,target),Word.or(width,target,Word.zero(width))))),target) == True{} : Bool}:
  match width:
    case 0n: {==}
    case 1n+p:
      match w b source target:
        case WCon{+wh,wt} WCon{+bh,bt} WCon{+sh,st} WCon{+th,tt}:
          J.and_true(Bool.not(Bool.and(th,Bool.or(Bool.and(wh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),Bool.and(bh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{}))))))),
            B.fresh_words(p,Word.and(p,wt,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),Word.and(p,bt,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt),Rows.fresh(wh,bh,sh,th),fresh(p,wt,bt,st,tt))
