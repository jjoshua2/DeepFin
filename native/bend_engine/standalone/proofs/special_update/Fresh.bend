# The flag1 removal mask contains the destination; no empty-target premise.
import Base
import ../board/Spec.bend as B
import ../move_update/Spec.bend as M
import ../ray/Masking.bend as J
import ../../../legal_probe/Chess.bend as Chess

def bit(+w: Bool,+b: Bool,+s: Bool,+t: Bool) ->
  {Bool.not(Bool.and(t,Bool.or(Bool.and(w,Bool.not(Bool.or(s,Bool.or(t,False{})))),Bool.and(b,Bool.not(Bool.or(s,Bool.or(t,False{}))))))) == True{} : Bool}:
  match w b s t:
    case False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{}: {==}
    case False{} True{} True{} True{}: {==}
    case True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{}: {==}
    case True{} False{} True{} True{}: {==}
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}


def words(+n: Nat,+w: Word(n),+b: Word(n),+s: Word(n),+t: Word(n)) ->
  {B.fresh_words(n,Word.and(n,w,Word.not(n,Word.or(n,s,Word.or(n,t,Word.zero(n))))),Word.and(n,b,Word.not(n,Word.or(n,s,Word.or(n,t,Word.zero(n))))),t) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w b s t:
        case WCon{+wh,wt} WCon{+bh,bt} WCon{+sh,st} WCon{+th,tt}:
          J.and_true(Bool.not(Bool.and(th,Bool.or(Bool.and(wh,Bool.not(Bool.or(sh,Bool.or(th,False{})))),Bool.and(bh,Bool.not(Bool.or(sh,Bool.or(th,False{}))))))),
            B.fresh_words(p,Word.and(p,wt,Word.not(p,Word.or(p,st,Word.or(p,tt,Word.zero(p))))),Word.and(p,bt,Word.not(p,Word.or(p,st,Word.or(p,tt,Word.zero(p))))),tt),
            bit(wh,bh,sh,th),words(p,wt,bt,st,tt))

def target(+board: Chess.Board,+source: U64,+dest: U64) ->
  {B.fresh_mask(M.erase(board,U64.or(source,U64.or(dest,U64.zero()))),dest) == True{} : Bool}:
  match board source dest:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{+wl},U32{+wh}},U64{U32{+bl},U32{+bh}},turn,rights,ep} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      J.and_true(B.fresh_words(32n,Word.and(32n,wl,Word.not(32n,Word.or(32n,sl,Word.or(32n,tl,Word.zero(32n))))),Word.and(32n,bl,Word.not(32n,Word.or(32n,sl,Word.or(32n,tl,Word.zero(32n))))),tl),
        B.fresh_words(32n,Word.and(32n,wh,Word.not(32n,Word.or(32n,sh,Word.or(32n,th,Word.zero(32n))))),Word.and(32n,bh,Word.not(32n,Word.or(32n,sh,Word.or(32n,th,Word.zero(32n))))),th),
        words(32n,wl,bl,sl,tl),words(32n,wh,bh,sh,th))
