import Base
import ../u64/Words.bend as Old

def head(n: Nat,w: Word(1n+n)) -> Bool:
  WCon{h,t} = w
  h
def tail(n: Nat,w: Word(1n+n)) -> Word(n):
  WCon{h,t} = w
  t
def low(a: U64) -> Word(32n):
  U64{U32{l},U32{h}} = a
  l
def high(a: U64) -> Word(32n):
  U64{U32{l},U32{h}} = a
  h

def or_commute_bool(+a: Bool,+b: Bool) -> {Bool.or(a,b) == Bool.or(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def or_commute_word(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.or(n,a,b) == Word.or(n,b,a) : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{+ah,+at} WCon{+bh,+bt}:
          Equal.trans(Word(1n+p),WCon{Bool.or(ah,bh),Word.or(p,at,bt)},WCon{Bool.or(bh,ah),Word.or(p,at,bt)},WCon{Bool.or(bh,ah),Word.or(p,bt,at)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,at,bt)},Bool.or(ah,bh),Bool.or(bh,ah),or_commute_bool(ah,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(bh,ah),t},Word.or(p,at,bt),Word.or(p,bt,at),or_commute_word(p,at,bt)))

def or_commute(+a: U64,+b: U64) -> {U64.or(a,b) == U64.or(b,a) : U64}:
  match a b:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}}:
      Equal.trans(U64,U64{U32{Word.or(32n,al,bl)},U32{Word.or(32n,ah,bh)}},U64{U32{Word.or(32n,bl,al)},U32{Word.or(32n,ah,bh)}},U64{U32{Word.or(32n,bl,al)},U32{Word.or(32n,bh,ah)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,ah,bh)}},Word.or(32n,al,bl),Word.or(32n,bl,al),or_commute_word(32n,al,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,bl,al)},U32{h}},Word.or(32n,ah,bh),Word.or(32n,bh,ah),or_commute_word(32n,ah,bh)))
def removal_clear_bool(+s: Bool,+t: Bool,+k: Bool,e0: {Bool.and(s,k) == False{} : Bool},e1: {Bool.and(t,k) == False{} : Bool}) -> {Bool.and(Bool.or(Bool.or(s,t),Bool.or(t,False{})),k) == False{} : Bool}:
  match s t k:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.or(False{},True{}),Bool.or(True{},False{})),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e1))
    case True{} False{} False{}: {==}
    case True{} False{} True{}: Old.false_true({Bool.and(Bool.or(Bool.or(True{},False{}),Bool.or(False{},False{})),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e0))
    case True{} True{} False{}: {==}
    case True{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.or(True{},True{}),Bool.or(True{},False{})),True{}) == False{} : Bool},Equal.sym(Bool,True{},False{},e0))

def removal_clear_word(+n: Nat,+s: Word(n),+t: Word(n),+k: Word(n),+e0: {Word.and(n,s,k) == Word.zero(n) : Word(n)},+e1: {Word.and(n,t,k) == Word.zero(n) : Word(n)}) -> {Word.and(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))),k) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match s t k:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match s t k:
        case WCon{+sh,+st} WCon{+th,+tt} WCon{+kh,+kt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(Bool.or(sh,th),Bool.or(th,False{})),kh),Word.and(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))),kt)},WCon{False{},Word.and(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))),kt)},WCon{False{},Word.zero(p)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))),kt)},Bool.and(Bool.or(Bool.or(sh,th),Bool.or(th,False{})),kh),False{},removal_clear_bool(sh,th,kh,Equal.cong(Word(1n+p),Bool,w => head(p,w),Word.and(1n+p,s,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Bool,w => head(p,w),Word.and(1n+p,t,k),Word.zero(1n+p),e1))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.and(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))),kt),Word.zero(p),removal_clear_word(p,st,tt,kt,Equal.cong(Word(1n+p),Word(p),w => tail(p,w),Word.and(1n+p,s,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Word(p),w => tail(p,w),Word.and(1n+p,t,k),Word.zero(1n+p),e1))))

def removal_clear(+s: U64,+t: U64,+k: U64,+e0: {U64.and(s,k) == U64.zero() : U64},+e1: {U64.and(t,k) == U64.zero() : U64}) -> {U64.and(U64.or(U64.or(s,t),U64.or(t,U64.zero())),k) == U64.zero() : U64}:
  match s t k:
    case U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}} U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))),kl)},U32{Word.and(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))),kh)}},U64{U32{Word.zero(32n)},U32{Word.and(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))),kh)}},U64{U32{Word.zero(32n)},U32{Word.zero(32n)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))),kh)}},Word.and(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))),kl),Word.zero(32n),removal_clear_word(32n,sl,tl,kl,Equal.cong(U64,Word(32n),low,U64.and(U64{U32{sl},U32{sh}},U64{U32{kl},U32{kh}}),U64.zero(),e0),Equal.cong(U64,Word(32n),low,U64.and(U64{U32{tl},U32{th}},U64{U32{kl},U32{kh}}),U64.zero(),e1))),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.zero(32n)},U32{h}},Word.and(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))),kh),Word.zero(32n),removal_clear_word(32n,sh,th,kh,Equal.cong(U64,Word(32n),high,U64.and(U64{U32{sl},U32{sh}},U64{U32{kl},U32{kh}}),U64.zero(),e0),Equal.cong(U64,Word(32n),high,U64.and(U64{U32{tl},U32{th}},U64{U32{kl},U32{kh}}),U64.zero(),e1))))
def keep_bool(+k: Bool,+m: Bool,e0: {Bool.and(m,k) == False{} : Bool}) -> {Bool.and(k,Bool.not(m)) == k : Bool}:
  match k m:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: Old.false_true({Bool.and(True{},Bool.not(True{})) == True{} : Bool},Equal.sym(Bool,True{},False{},e0))

def keep_word(+n: Nat,+k: Word(n),+m: Word(n),+e0: {Word.and(n,m,k) == Word.zero(n) : Word(n)}) -> {Word.and(n,k,Word.not(n,m)) == k : Word(n)}:
  match n:
    case 0n:
      match k m:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match k m:
        case WCon{+kh,+kt} WCon{+mh,+mt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(kh,Bool.not(mh)),Word.and(p,kt,Word.not(p,mt))},WCon{kh,Word.and(p,kt,Word.not(p,mt))},WCon{kh,kt},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,kt,Word.not(p,mt))},Bool.and(kh,Bool.not(mh)),kh,keep_bool(kh,mh,Equal.cong(Word(1n+p),Bool,w => head(p,w),Word.and(1n+p,m,k),Word.zero(1n+p),e0))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{kh,t},Word.and(p,kt,Word.not(p,mt)),kt,keep_word(p,kt,mt,Equal.cong(Word(1n+p),Word(p),w => tail(p,w),Word.and(1n+p,m,k),Word.zero(1n+p),e0))))

def keep(+k: U64,+m: U64,+e0: {U64.and(m,k) == U64.zero() : U64}) -> {U64.and(k,U64.not(m)) == k : U64}:
  match k m:
    case U64{U32{+kl},U32{+kh}} U64{U32{+ml},U32{+mh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,kl,Word.not(32n,ml))},U32{Word.and(32n,kh,Word.not(32n,mh))}},U64{U32{kl},U32{Word.and(32n,kh,Word.not(32n,mh))}},U64{U32{kl},U32{kh}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,kh,Word.not(32n,mh))}},Word.and(32n,kl,Word.not(32n,ml)),kl,keep_word(32n,kl,ml,Equal.cong(U64,Word(32n),low,U64.and(U64{U32{ml},U32{mh}},U64{U32{kl},U32{kh}}),U64.zero(),e0))),
        Equal.cong(Word(32n),U64,h => U64{U32{kl},U32{h}},Word.and(32n,kh,Word.not(32n,mh)),kh,keep_word(32n,kh,mh,Equal.cong(U64,Word(32n),high,U64.and(U64{U32{ml},U32{mh}},U64{U32{kl},U32{kh}}),U64.zero(),e0))))
def frame_bool(+k: Bool,+o: Bool,+s: Bool,+t: Bool,e0: {Bool.and(s,k) == False{} : Bool},e1: {Bool.and(t,k) == False{} : Bool}) -> {Bool.and(k,Bool.or(Bool.and(o,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))),t)) == Bool.and(k,o) : Bool}:
  match k o s t:
    case False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{}: {==}
    case False{} True{} True{} True{}: {==}
    case True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(False{},Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(True{},False{})))),True{})) == Bool.and(True{},False{}) : Bool},Equal.sym(Bool,True{},False{},e1))
    case True{} False{} True{} False{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(False{},Bool.not(Bool.or(Bool.or(True{},False{}),Bool.or(False{},False{})))),False{})) == Bool.and(True{},False{}) : Bool},Equal.sym(Bool,True{},False{},e0))
    case True{} False{} True{} True{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(False{},Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(True{},False{})))),True{})) == Bool.and(True{},False{}) : Bool},Equal.sym(Bool,True{},False{},e0))
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(True{},Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(True{},False{})))),True{})) == Bool.and(True{},True{}) : Bool},Equal.sym(Bool,True{},False{},e1))
    case True{} True{} True{} False{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(True{},Bool.not(Bool.or(Bool.or(True{},False{}),Bool.or(False{},False{})))),False{})) == Bool.and(True{},True{}) : Bool},Equal.sym(Bool,True{},False{},e0))
    case True{} True{} True{} True{}: Old.false_true({Bool.and(True{},Bool.or(Bool.and(True{},Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(True{},False{})))),True{})) == Bool.and(True{},True{}) : Bool},Equal.sym(Bool,True{},False{},e0))

def frame_word(+n: Nat,+k: Word(n),+o: Word(n),+s: Word(n),+t: Word(n),+e0: {Word.and(n,s,k) == Word.zero(n) : Word(n)},+e1: {Word.and(n,t,k) == Word.zero(n) : Word(n)}) -> {Word.and(n,k,Word.or(n,Word.and(n,o,Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))))),t)) == Word.and(n,k,o) : Word(n)}:
  match n:
    case 0n:
      match k o s t:
        case WNil{} WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match k o s t:
        case WCon{+kh,+kt} WCon{+oh,+ot} WCon{+sh,+st} WCon{+th,+tt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(kh,Bool.or(Bool.and(oh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),th)),Word.and(p,kt,Word.or(p,Word.and(p,ot,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt))},WCon{Bool.and(kh,oh),Word.and(p,kt,Word.or(p,Word.and(p,ot,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt))},WCon{Bool.and(kh,oh),Word.and(p,kt,ot)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,kt,Word.or(p,Word.and(p,ot,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt))},Bool.and(kh,Bool.or(Bool.and(oh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),th)),Bool.and(kh,oh),frame_bool(kh,oh,sh,th,Equal.cong(Word(1n+p),Bool,w => head(p,w),Word.and(1n+p,s,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Bool,w => head(p,w),Word.and(1n+p,t,k),Word.zero(1n+p),e1))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(kh,oh),t},Word.and(p,kt,Word.or(p,Word.and(p,ot,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt)),Word.and(p,kt,ot),frame_word(p,kt,ot,st,tt,Equal.cong(Word(1n+p),Word(p),w => tail(p,w),Word.and(1n+p,s,k),Word.zero(1n+p),e0),Equal.cong(Word(1n+p),Word(p),w => tail(p,w),Word.and(1n+p,t,k),Word.zero(1n+p),e1))))

def frame(+k: U64,+o: U64,+s: U64,+t: U64,+e0: {U64.and(s,k) == U64.zero() : U64},+e1: {U64.and(t,k) == U64.zero() : U64}) -> {U64.and(k,U64.or(U64.and(o,U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero())))),t)) == U64.and(k,o) : U64}:
  match k o s t:
    case U64{U32{+kl},U32{+kh}} U64{U32{+ol},U32{+oh}} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      Equal.trans(U64,U64{U32{Word.and(32n,kl,Word.or(32n,Word.and(32n,ol,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),tl))},U32{Word.and(32n,kh,Word.or(32n,Word.and(32n,oh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th))}},U64{U32{Word.and(32n,kl,ol)},U32{Word.and(32n,kh,Word.or(32n,Word.and(32n,oh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th))}},U64{U32{Word.and(32n,kl,ol)},U32{Word.and(32n,kh,oh)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,kh,Word.or(32n,Word.and(32n,oh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th))}},Word.and(32n,kl,Word.or(32n,Word.and(32n,ol,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),tl)),Word.and(32n,kl,ol),frame_word(32n,kl,ol,sl,tl,Equal.cong(U64,Word(32n),low,U64.and(U64{U32{sl},U32{sh}},U64{U32{kl},U32{kh}}),U64.zero(),e0),Equal.cong(U64,Word(32n),low,U64.and(U64{U32{tl},U32{th}},U64{U32{kl},U32{kh}}),U64.zero(),e1))),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,kl,ol)},U32{h}},Word.and(32n,kh,Word.or(32n,Word.and(32n,oh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th)),Word.and(32n,kh,oh),frame_word(32n,kh,oh,sh,th,Equal.cong(U64,Word(32n),high,U64.and(U64{U32{sl},U32{sh}},U64{U32{kl},U32{kh}}),U64.zero(),e0),Equal.cong(U64,Word(32n),high,U64.and(U64{U32{tl},U32{th}},U64{U32{kl},U32{kh}}),U64.zero(),e1))))
def enemy_none_bool(+a: Bool,+e: Bool,+s: Bool,+t: Bool) -> {Bool.and(Bool.or(Bool.and(a,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))),False{}),Bool.and(e,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{}))))) == Bool.and(Bool.and(a,e),Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))) : Bool}:
  match a e s t:
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

def enemy_none_word(+n: Nat,+a: Word(n),+e: Word(n),+s: Word(n),+t: Word(n)) -> {Word.and(n,Word.or(n,Word.and(n,a,Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))))),Word.zero(n)),Word.and(n,e,Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n)))))) == Word.and(n,Word.and(n,a,e),Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))))) : Word(n)}:
  match n:
    case 0n:
      match a e s t:
        case WNil{} WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a e s t:
        case WCon{+ah,+at} WCon{+eh,+et} WCon{+sh,+st} WCon{+th,+tt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),False{}),Bool.and(eh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{}))))),Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),Word.zero(p)),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),Word.zero(p)),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),Word.and(p,Word.and(p,at,et),Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p)))))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),Word.zero(p)),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},Bool.and(Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),False{}),Bool.and(eh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{}))))),Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),enemy_none_bool(ah,eh,sh,th)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),t},Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),Word.zero(p)),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p)))))),Word.and(p,Word.and(p,at,et),Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),enemy_none_word(p,at,et,st,tt)))

def enemy_none(+a: U64,+e: U64,+s: U64,+t: U64) -> {U64.and(U64.or(U64.and(a,U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero())))),U64.zero()),U64.and(e,U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero()))))) == U64.and(U64.and(a,e),U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero())))) : U64}:
  match a e s t:
    case U64{U32{+al},U32{+ah}} U64{U32{+el},U32{+eh}} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,el,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))))},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{Word.and(32n,Word.and(32n,ah,eh),Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n)))))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},Word.and(32n,Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,el,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))),Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),enemy_none_word(32n,al,el,sl,tl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{h}},Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),Word.zero(32n)),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n)))))),Word.and(32n,Word.and(32n,ah,eh),Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),enemy_none_word(32n,ah,eh,sh,th)))
def enemy_put_bool(+a: Bool,+e: Bool,+s: Bool,+t: Bool) -> {Bool.and(Bool.or(Bool.and(a,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))),t),Bool.and(e,Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{}))))) == Bool.and(Bool.and(a,e),Bool.not(Bool.or(Bool.or(s,t),Bool.or(t,False{})))) : Bool}:
  match a e s t:
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

def enemy_put_word(+n: Nat,+a: Word(n),+e: Word(n),+s: Word(n),+t: Word(n)) -> {Word.and(n,Word.or(n,Word.and(n,a,Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))))),t),Word.and(n,e,Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n)))))) == Word.and(n,Word.and(n,a,e),Word.not(n,Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))))) : Word(n)}:
  match n:
    case 0n:
      match a e s t:
        case WNil{} WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a e s t:
        case WCon{+ah,+at} WCon{+eh,+et} WCon{+sh,+st} WCon{+th,+tt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),th),Bool.and(eh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{}))))),Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),Word.and(p,Word.and(p,at,et),Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p)))))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))))},Bool.and(Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),th),Bool.and(eh,Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{}))))),Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),enemy_put_bool(ah,eh,sh,th)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(Bool.and(ah,eh),Bool.not(Bool.or(Bool.or(sh,th),Bool.or(th,False{})))),t},Word.and(p,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),tt),Word.and(p,et,Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p)))))),Word.and(p,Word.and(p,at,et),Word.not(p,Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))))),enemy_put_word(p,at,et,st,tt)))

def enemy_put(+a: U64,+e: U64,+s: U64,+t: U64) -> {U64.and(U64.or(U64.and(a,U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero())))),t),U64.and(e,U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero()))))) == U64.and(U64.and(a,e),U64.not(U64.or(U64.or(s,t),U64.or(t,U64.zero())))) : U64}:
  match a e s t:
    case U64{U32{+al},U32{+ah}} U64{U32{+el},U32{+eh}} U64{U32{+sl},U32{+sh}} U64{U32{+tl},U32{+th}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),tl),Word.and(32n,el,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))))},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{Word.and(32n,Word.and(32n,ah,eh),Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n)))))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))))}},Word.and(32n,Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),tl),Word.and(32n,el,Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))),Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))))),enemy_put_word(32n,al,el,sl,tl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,Word.and(32n,al,el),Word.not(32n,Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))))},U32{h}},Word.and(32n,Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),th),Word.and(32n,eh,Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n)))))),Word.and(32n,Word.and(32n,ah,eh),Word.not(32n,Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))))),enemy_put_word(32n,ah,eh,sh,th)))
def erased_subset_bool(+a: Bool,+m: Bool) -> {Bool.and(a,Bool.and(a,Bool.not(m))) == Bool.and(a,Bool.not(m)) : Bool}:
  match a m:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def erased_subset_word(+n: Nat,+a: Word(n),+m: Word(n)) -> {Word.and(n,a,Word.and(n,a,Word.not(n,m))) == Word.and(n,a,Word.not(n,m)) : Word(n)}:
  match n:
    case 0n:
      match a m:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a m:
        case WCon{+ah,+at} WCon{+mh,+mt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(ah,Bool.and(ah,Bool.not(mh))),Word.and(p,at,Word.and(p,at,Word.not(p,mt)))},WCon{Bool.and(ah,Bool.not(mh)),Word.and(p,at,Word.and(p,at,Word.not(p,mt)))},WCon{Bool.and(ah,Bool.not(mh)),Word.and(p,at,Word.not(p,mt))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,at,Word.and(p,at,Word.not(p,mt)))},Bool.and(ah,Bool.and(ah,Bool.not(mh))),Bool.and(ah,Bool.not(mh)),erased_subset_bool(ah,mh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(ah,Bool.not(mh)),t},Word.and(p,at,Word.and(p,at,Word.not(p,mt))),Word.and(p,at,Word.not(p,mt)),erased_subset_word(p,at,mt)))

def erased_subset(+a: U64,+m: U64) -> {U64.and(a,U64.and(a,U64.not(m))) == U64.and(a,U64.not(m)) : U64}:
  match a m:
    case U64{U32{+al},U32{+ah}} U64{U32{+ml},U32{+mh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,al,Word.and(32n,al,Word.not(32n,ml)))},U32{Word.and(32n,ah,Word.and(32n,ah,Word.not(32n,mh)))}},U64{U32{Word.and(32n,al,Word.not(32n,ml))},U32{Word.and(32n,ah,Word.and(32n,ah,Word.not(32n,mh)))}},U64{U32{Word.and(32n,al,Word.not(32n,ml))},U32{Word.and(32n,ah,Word.not(32n,mh))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,ah,Word.and(32n,ah,Word.not(32n,mh)))}},Word.and(32n,al,Word.and(32n,al,Word.not(32n,ml))),Word.and(32n,al,Word.not(32n,ml)),erased_subset_word(32n,al,ml)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,al,Word.not(32n,ml))},U32{h}},Word.and(32n,ah,Word.and(32n,ah,Word.not(32n,mh))),Word.and(32n,ah,Word.not(32n,mh)),erased_subset_word(32n,ah,mh)))
