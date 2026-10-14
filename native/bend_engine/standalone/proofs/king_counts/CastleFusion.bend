# Candidate: extra rook insertion does not alter the king/color intersection.
import Base
import ../pawn_counts/Fusion.bend as F
import ../u64/Words.bend as Old
import ../castle_kings/Algebra.bend as Parts
import ../../../legal_probe/Chess.bend as Chess

def extra_bool(+k: Bool,+c: Bool,+m: Bool,+t: Bool,+r: Bool,+put: Bool,+side: Bool,
  empty: {Bool.and(k,r) == False{} : Bool}) ->
  {Bool.and(F.update_bool(k,m,t,put),F.update_bool(c,m,Bool.or(t,r),side)) ==
   Bool.and(F.update_bool(k,m,t,put),F.update_bool(c,m,t,side)) : Bool}:
  match k c m t r put side:
    case False{} False{} False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} False{} True{} False{} True{}: {==}
    case False{} False{} False{} False{} True{} True{} False{}: {==}
    case False{} False{} False{} False{} True{} True{} True{}: {==}
    case False{} False{} False{} True{} False{} False{} False{}: {==}
    case False{} False{} False{} True{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{} False{} True{}: {==}
    case False{} False{} False{} True{} True{} True{} False{}: {==}
    case False{} False{} False{} True{} True{} True{} True{}: {==}
    case False{} False{} True{} False{} False{} False{} False{}: {==}
    case False{} False{} True{} False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{} False{} True{} False{}: {==}
    case False{} False{} True{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} False{} False{} True{}: {==}
    case False{} False{} True{} True{} False{} True{} False{}: {==}
    case False{} False{} True{} True{} False{} True{} True{}: {==}
    case False{} False{} True{} True{} True{} False{} False{}: {==}
    case False{} False{} True{} True{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{} True{} False{} False{}: {==}
    case False{} True{} False{} False{} True{} False{} True{}: {==}
    case False{} True{} False{} False{} True{} True{} False{}: {==}
    case False{} True{} False{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{} False{} True{}: {==}
    case False{} True{} False{} True{} True{} True{} False{}: {==}
    case False{} True{} False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} False{} False{} False{} True{}: {==}
    case False{} True{} True{} False{} False{} True{} False{}: {==}
    case False{} True{} True{} False{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} False{} False{} True{}: {==}
    case False{} True{} True{} True{} False{} True{} False{}: {==}
    case False{} True{} True{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} False{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(False{},False{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(False{},False{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} False{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(False{},False{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(False{},False{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} False{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(False{},False{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(False{},False{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} False{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(False{},False{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(False{},False{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{} False{} False{} True{}: {==}
    case True{} False{} False{} True{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(False{},False{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(False{},False{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} True{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(False{},False{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(False{},False{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} True{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(False{},False{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(False{},False{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} True{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(False{},False{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(False{},False{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} False{} False{} False{} False{}: {==}
    case True{} False{} True{} False{} False{} False{} True{}: {==}
    case True{} False{} True{} False{} False{} True{} False{}: {==}
    case True{} False{} True{} False{} False{} True{} True{}: {==}
    case True{} False{} True{} False{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(False{},True{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(False{},True{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} False{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(False{},True{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(False{},True{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} False{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(False{},True{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(False{},True{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} False{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(False{},True{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(False{},True{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} False{} False{} True{}: {==}
    case True{} False{} True{} True{} False{} True{} False{}: {==}
    case True{} False{} True{} True{} False{} True{} True{}: {==}
    case True{} False{} True{} True{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(False{},True{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(False{},True{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} True{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(False{},True{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(False{},True{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} True{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(False{},True{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(False{},True{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} True{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(False{},True{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(False{},True{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} False{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} False{} True{} True{}: {==}
    case True{} True{} False{} False{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(True{},False{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(True{},False{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} False{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(True{},False{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},False{},False{}),F.update_bool(True{},False{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} False{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(True{},False{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(True{},False{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} False{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(True{},False{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},False{},True{}),F.update_bool(True{},False{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(True{},False{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(True{},False{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} True{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(True{},False{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},True{},False{}),F.update_bool(True{},False{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} True{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(True{},False{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(True{},False{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} True{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(True{},False{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},False{},True{},True{}),F.update_bool(True{},False{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} False{} False{} False{} False{}: {==}
    case True{} True{} True{} False{} False{} False{} True{}: {==}
    case True{} True{} True{} False{} False{} True{} False{}: {==}
    case True{} True{} True{} False{} False{} True{} True{}: {==}
    case True{} True{} True{} False{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(True{},True{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(True{},True{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} False{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(True{},True{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},False{},False{}),F.update_bool(True{},True{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} False{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(True{},True{},Bool.or(False{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(True{},True{},False{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} False{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(True{},True{},Bool.or(False{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},False{},True{}),F.update_bool(True{},True{},False{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} True{} False{} False{} False{}: {==}
    case True{} True{} True{} True{} False{} False{} True{}: {==}
    case True{} True{} True{} True{} False{} True{} False{}: {==}
    case True{} True{} True{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} True{} True{} False{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(True{},True{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(True{},True{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} True{} True{} False{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(True{},True{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},True{},False{}),F.update_bool(True{},True{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} True{} True{} True{} False{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(True{},True{},Bool.or(True{},True{}),False{})) == Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(True{},True{},True{},False{})) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} True{} True{} True{} True{}: Old.false_true({Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(True{},True{},Bool.or(True{},True{}),True{})) == Bool.and(F.update_bool(True{},True{},True{},True{}),F.update_bool(True{},True{},True{},True{})) : Bool},Equal.sym(Bool,True{},False{},empty))

def left(+n: Nat,+k: Word(n),+c: Word(n),+m: Word(n),+t: Word(n),+r: Word(n),+put: Bool,side: Bool) -> Word(n):
  Word.and(n,F.update_word(n,k,m,t,put),F.update_word(n,c,m,Word.or(n,t,r),side))
def right(+n: Nat,+k: Word(n),+c: Word(n),+m: Word(n),+t: Word(n),+put: Bool,side: Bool) -> Word(n):
  Word.and(n,F.update_word(n,k,m,t,put),F.update_word(n,c,m,t,side))
def word(+n: Nat,+k: Word(n),+c: Word(n),+m: Word(n),+t: Word(n),+r: Word(n),+put: Bool,+side: Bool,
  +empty: {Word.and(n,k,r) == Word.zero(n) : Word(n)}) ->
  {left(n,k,c,m,t,r,put,side) == right(n,k,c,m,t,put,side) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+j:
      match k c m t r:
        case WCon{+kh,+kt} WCon{+ch,+ct} WCon{+mh,+mt} WCon{+th,+tt} WCon{+rh,+rt}:
          match put side:
            case False{} False{}:
              Equal.trans(Word(1n+j),WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,Bool.or(th,rh),False{})),left(j,kt,ct,mt,tt,rt,False{},False{})},WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,False{})),left(j,kt,ct,mt,tt,rt,False{},False{})},WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,False{})),right(j,kt,ct,mt,tt,False{},False{})},
                Equal.cong(Bool,Word(1n+j),h => WCon{h,left(j,kt,ct,mt,tt,rt,False{},False{})},Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,Bool.or(th,rh),False{})),Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,False{})),
                  extra_bool(kh,ch,mh,th,rh,False{},False{},Equal.cong(Word(1n+j),Bool,x => Parts.head(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))),
                Equal.cong(Word(j),Word(1n+j),v => WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,False{})),v},left(j,kt,ct,mt,tt,rt,False{},False{}),right(j,kt,ct,mt,tt,False{},False{}),
                  word(j,kt,ct,mt,tt,rt,False{},False{},Equal.cong(Word(1n+j),Word(j),x => Parts.tail(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))))
            case False{} True{}:
              Equal.trans(Word(1n+j),WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,Bool.or(th,rh),True{})),left(j,kt,ct,mt,tt,rt,False{},True{})},WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,True{})),left(j,kt,ct,mt,tt,rt,False{},True{})},WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,True{})),right(j,kt,ct,mt,tt,False{},True{})},
                Equal.cong(Bool,Word(1n+j),h => WCon{h,left(j,kt,ct,mt,tt,rt,False{},True{})},Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,Bool.or(th,rh),True{})),Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,True{})),
                  extra_bool(kh,ch,mh,th,rh,False{},True{},Equal.cong(Word(1n+j),Bool,x => Parts.head(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))),
                Equal.cong(Word(j),Word(1n+j),v => WCon{Bool.and(F.update_bool(kh,mh,th,False{}),F.update_bool(ch,mh,th,True{})),v},left(j,kt,ct,mt,tt,rt,False{},True{}),right(j,kt,ct,mt,tt,False{},True{}),
                  word(j,kt,ct,mt,tt,rt,False{},True{},Equal.cong(Word(1n+j),Word(j),x => Parts.tail(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))))
            case True{} False{}:
              Equal.trans(Word(1n+j),WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,Bool.or(th,rh),False{})),left(j,kt,ct,mt,tt,rt,True{},False{})},WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,False{})),left(j,kt,ct,mt,tt,rt,True{},False{})},WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,False{})),right(j,kt,ct,mt,tt,True{},False{})},
                Equal.cong(Bool,Word(1n+j),h => WCon{h,left(j,kt,ct,mt,tt,rt,True{},False{})},Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,Bool.or(th,rh),False{})),Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,False{})),
                  extra_bool(kh,ch,mh,th,rh,True{},False{},Equal.cong(Word(1n+j),Bool,x => Parts.head(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))),
                Equal.cong(Word(j),Word(1n+j),v => WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,False{})),v},left(j,kt,ct,mt,tt,rt,True{},False{}),right(j,kt,ct,mt,tt,True{},False{}),
                  word(j,kt,ct,mt,tt,rt,True{},False{},Equal.cong(Word(1n+j),Word(j),x => Parts.tail(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))))
            case True{} True{}:
              Equal.trans(Word(1n+j),WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,Bool.or(th,rh),True{})),left(j,kt,ct,mt,tt,rt,True{},True{})},WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,True{})),left(j,kt,ct,mt,tt,rt,True{},True{})},WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,True{})),right(j,kt,ct,mt,tt,True{},True{})},
                Equal.cong(Bool,Word(1n+j),h => WCon{h,left(j,kt,ct,mt,tt,rt,True{},True{})},Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,Bool.or(th,rh),True{})),Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,True{})),
                  extra_bool(kh,ch,mh,th,rh,True{},True{},Equal.cong(Word(1n+j),Bool,x => Parts.head(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))),
                Equal.cong(Word(j),Word(1n+j),v => WCon{Bool.and(F.update_bool(kh,mh,th,True{}),F.update_bool(ch,mh,th,True{})),v},left(j,kt,ct,mt,tt,rt,True{},True{}),right(j,kt,ct,mt,tt,True{},True{}),
                  word(j,kt,ct,mt,tt,rt,True{},True{},Equal.cong(Word(1n+j),Word(j),x => Parts.tail(j,x),Word.and(1n+j,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+j),empty))))

def extra(+k: U64,+c: U64,+m: U64,+t: U64,+r: U64,+put: Bool,+side: Bool,
  +empty: {U64.and(k,r) == U64.zero() : U64}) ->
  {U64.and(Chess.update_piece(k,m,t,put),Chess.update_piece(c,m,U64.or(t,r),side)) ==
   U64.and(Chess.update_piece(k,m,t,put),Chess.update_piece(c,m,t,side)) : U64}:
  match k c m t r:
    case U64{U32{+kl},U32{+kh}} U64{U32{+cl},U32{+ch}} U64{U32{+ml},U32{+mh}} U64{U32{+tl},U32{+th}} U64{U32{+rl},U32{+rh}}:
      match put side:
        case False{} False{}:
          Equal.trans(U64,U64{U32{left(32n,kl,cl,ml,tl,rl,False{},False{})},U32{left(32n,kh,ch,mh,th,rh,False{},False{})}},U64{U32{right(32n,kl,cl,ml,tl,False{},False{})},U32{left(32n,kh,ch,mh,th,rh,False{},False{})}},U64{U32{right(32n,kl,cl,ml,tl,False{},False{})},U32{right(32n,kh,ch,mh,th,False{},False{})}},
            Equal.cong(Word(32n),U64,x => U64{U32{x},U32{left(32n,kh,ch,mh,th,rh,False{},False{})}},left(32n,kl,cl,ml,tl,rl,False{},False{}),right(32n,kl,cl,ml,tl,False{},False{}),
              word(32n,kl,cl,ml,tl,rl,False{},False{},Equal.cong(U64,Word(32n),Parts.lo,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))),
            Equal.cong(Word(32n),U64,x => U64{U32{right(32n,kl,cl,ml,tl,False{},False{})},U32{x}},left(32n,kh,ch,mh,th,rh,False{},False{}),right(32n,kh,ch,mh,th,False{},False{}),
              word(32n,kh,ch,mh,th,rh,False{},False{},Equal.cong(U64,Word(32n),Parts.hi,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))))
        case False{} True{}:
          Equal.trans(U64,U64{U32{left(32n,kl,cl,ml,tl,rl,False{},True{})},U32{left(32n,kh,ch,mh,th,rh,False{},True{})}},U64{U32{right(32n,kl,cl,ml,tl,False{},True{})},U32{left(32n,kh,ch,mh,th,rh,False{},True{})}},U64{U32{right(32n,kl,cl,ml,tl,False{},True{})},U32{right(32n,kh,ch,mh,th,False{},True{})}},
            Equal.cong(Word(32n),U64,x => U64{U32{x},U32{left(32n,kh,ch,mh,th,rh,False{},True{})}},left(32n,kl,cl,ml,tl,rl,False{},True{}),right(32n,kl,cl,ml,tl,False{},True{}),
              word(32n,kl,cl,ml,tl,rl,False{},True{},Equal.cong(U64,Word(32n),Parts.lo,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))),
            Equal.cong(Word(32n),U64,x => U64{U32{right(32n,kl,cl,ml,tl,False{},True{})},U32{x}},left(32n,kh,ch,mh,th,rh,False{},True{}),right(32n,kh,ch,mh,th,False{},True{}),
              word(32n,kh,ch,mh,th,rh,False{},True{},Equal.cong(U64,Word(32n),Parts.hi,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))))
        case True{} False{}:
          Equal.trans(U64,U64{U32{left(32n,kl,cl,ml,tl,rl,True{},False{})},U32{left(32n,kh,ch,mh,th,rh,True{},False{})}},U64{U32{right(32n,kl,cl,ml,tl,True{},False{})},U32{left(32n,kh,ch,mh,th,rh,True{},False{})}},U64{U32{right(32n,kl,cl,ml,tl,True{},False{})},U32{right(32n,kh,ch,mh,th,True{},False{})}},
            Equal.cong(Word(32n),U64,x => U64{U32{x},U32{left(32n,kh,ch,mh,th,rh,True{},False{})}},left(32n,kl,cl,ml,tl,rl,True{},False{}),right(32n,kl,cl,ml,tl,True{},False{}),
              word(32n,kl,cl,ml,tl,rl,True{},False{},Equal.cong(U64,Word(32n),Parts.lo,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))),
            Equal.cong(Word(32n),U64,x => U64{U32{right(32n,kl,cl,ml,tl,True{},False{})},U32{x}},left(32n,kh,ch,mh,th,rh,True{},False{}),right(32n,kh,ch,mh,th,True{},False{}),
              word(32n,kh,ch,mh,th,rh,True{},False{},Equal.cong(U64,Word(32n),Parts.hi,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))))
        case True{} True{}:
          Equal.trans(U64,U64{U32{left(32n,kl,cl,ml,tl,rl,True{},True{})},U32{left(32n,kh,ch,mh,th,rh,True{},True{})}},U64{U32{right(32n,kl,cl,ml,tl,True{},True{})},U32{left(32n,kh,ch,mh,th,rh,True{},True{})}},U64{U32{right(32n,kl,cl,ml,tl,True{},True{})},U32{right(32n,kh,ch,mh,th,True{},True{})}},
            Equal.cong(Word(32n),U64,x => U64{U32{x},U32{left(32n,kh,ch,mh,th,rh,True{},True{})}},left(32n,kl,cl,ml,tl,rl,True{},True{}),right(32n,kl,cl,ml,tl,True{},True{}),
              word(32n,kl,cl,ml,tl,rl,True{},True{},Equal.cong(U64,Word(32n),Parts.lo,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))),
            Equal.cong(Word(32n),U64,x => U64{U32{right(32n,kl,cl,ml,tl,True{},True{})},U32{x}},left(32n,kh,ch,mh,th,rh,True{},True{}),right(32n,kh,ch,mh,th,True{},True{}),
              word(32n,kh,ch,mh,th,rh,True{},True{},Equal.cong(U64,Word(32n),Parts.hi,U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))))
