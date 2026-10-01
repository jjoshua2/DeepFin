import Base
import ../u64/Words.bend as Old

def head(n: Nat,w: Word(1n+n)) -> Bool:
  WCon{h,t} = w
  h

def tail(n: Nat,w: Word(1n+n)) -> Word(n):
  WCon{h,t} = w
  t

def project_bool(+k: Bool,+c: Bool,+m: Bool,+t: Bool,+r: Bool,
  empty: {Bool.and(k,r) == False{} : Bool}) ->
  {Bool.and(Bool.or(Bool.and(k,Bool.not(m)),t),Bool.or(Bool.and(c,Bool.not(m)),Bool.or(t,r))) ==
   Bool.or(Bool.and(Bool.and(k,c),Bool.not(m)),t) : Bool}:
  match k c m t r:
    case False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(False{})),False{}),Bool.or(Bool.and(False{},Bool.not(False{})),Bool.or(False{},True{}))) == Bool.or(Bool.and(Bool.and(True{},False{}),Bool.not(False{})),False{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(False{})),True{}),Bool.or(Bool.and(False{},Bool.not(False{})),Bool.or(True{},True{}))) == Bool.or(Bool.and(Bool.and(True{},False{}),Bool.not(False{})),True{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(True{})),False{}),Bool.or(Bool.and(False{},Bool.not(True{})),Bool.or(False{},True{}))) == Bool.or(Bool.and(Bool.and(True{},False{}),Bool.not(True{})),False{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(True{})),True{}),Bool.or(Bool.and(False{},Bool.not(True{})),Bool.or(True{},True{}))) == Bool.or(Bool.and(Bool.and(True{},False{}),Bool.not(True{})),True{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(False{})),False{}),Bool.or(Bool.and(True{},Bool.not(False{})),Bool.or(False{},True{}))) == Bool.or(Bool.and(Bool.and(True{},True{}),Bool.not(False{})),False{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(False{})),True{}),Bool.or(Bool.and(True{},Bool.not(False{})),Bool.or(True{},True{}))) == Bool.or(Bool.and(Bool.and(True{},True{}),Bool.not(False{})),True{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(True{})),False{}),Bool.or(Bool.and(True{},Bool.not(True{})),Bool.or(False{},True{}))) == Bool.or(Bool.and(Bool.and(True{},True{}),Bool.not(True{})),False{}) : Bool},Equal.sym(Bool,True{},False{},empty))
    case True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{} True{}: Old.false_true({Bool.and(Bool.or(Bool.and(True{},Bool.not(True{})),True{}),Bool.or(Bool.and(True{},Bool.not(True{})),Bool.or(True{},True{}))) == Bool.or(Bool.and(Bool.and(True{},True{}),Bool.not(True{})),True{}) : Bool},Equal.sym(Bool,True{},False{},empty))

def left(+n: Nat,+k: Word(n),+c: Word(n),+m: Word(n),+t: Word(n),+r: Word(n)) -> Word(n):
  Word.and(n,Word.or(n,Word.and(n,k,Word.not(n,m)),t),Word.or(n,Word.and(n,c,Word.not(n,m)),Word.or(n,t,r)))

def right(+n: Nat,k: Word(n),c: Word(n),m: Word(n),t: Word(n)) -> Word(n):
  Word.or(n,Word.and(n,Word.and(n,k,c),Word.not(n,m)),t)

def word(+n: Nat,+k: Word(n),+c: Word(n),+m: Word(n),+t: Word(n),+r: Word(n),
  +empty: {Word.and(n,k,r) == Word.zero(n) : Word(n)}) ->
  {left(n,k,c,m,t,r) == right(n,k,c,m,t) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match k c m t r:
        case WCon{+kh,+kt} WCon{+ch,+ct} WCon{+mh,+mt} WCon{+th,+tt} WCon{+rh,+rt}:
          Equal.trans(Word(1n+p),
            WCon{Bool.and(Bool.or(Bool.and(kh,Bool.not(mh)),th),Bool.or(Bool.and(ch,Bool.not(mh)),Bool.or(th,rh))),left(p,kt,ct,mt,tt,rt)},
            WCon{Bool.or(Bool.and(Bool.and(kh,ch),Bool.not(mh)),th),left(p,kt,ct,mt,tt,rt)},
            WCon{Bool.or(Bool.and(Bool.and(kh,ch),Bool.not(mh)),th),right(p,kt,ct,mt,tt)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,left(p,kt,ct,mt,tt,rt)},
              Bool.and(Bool.or(Bool.and(kh,Bool.not(mh)),th),Bool.or(Bool.and(ch,Bool.not(mh)),Bool.or(th,rh))),
              Bool.or(Bool.and(Bool.and(kh,ch),Bool.not(mh)),th),
              project_bool(kh,ch,mh,th,rh,Equal.cong(Word(1n+p),Bool,x => head(p,x),Word.and(1n+p,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+p),empty))),
            Equal.cong(Word(p),Word(1n+p),v => WCon{Bool.or(Bool.and(Bool.and(kh,ch),Bool.not(mh)),th),v},
              left(p,kt,ct,mt,tt,rt),right(p,kt,ct,mt,tt),
              word(p,kt,ct,mt,tt,rt,Equal.cong(Word(1n+p),Word(p),x => tail(p,x),Word.and(1n+p,WCon{kh,kt},WCon{rh,rt}),Word.zero(1n+p),empty))))

def lo(a: U64) -> Word(32n):
  U64{U32{l},U32{h}} = a
  l

def hi(a: U64) -> Word(32n):
  U64{U32{l},U32{h}} = a
  h

def projection(+k: U64,+c: U64,+m: U64,+t: U64,+r: U64,+empty: {U64.and(k,r) == U64.zero() : U64}) ->
  {U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(c,m),U64.or(t,r))) == U64.or(U64.and_not(U64.and(k,c),m),t) : U64}:
  match k c m t r:
    case U64{U32{+kl},U32{+kh}} U64{U32{+cl},U32{+ch}} U64{U32{+ml},U32{+mh}} U64{U32{+tl},U32{+th}} U64{U32{+rl},U32{+rh}}:
      Equal.trans(U64,
        U64{U32{left(32n,kl,cl,ml,tl,rl)},U32{left(32n,kh,ch,mh,th,rh)}},
        U64{U32{right(32n,kl,cl,ml,tl)},U32{left(32n,kh,ch,mh,th,rh)}},
        U64{U32{right(32n,kl,cl,ml,tl)},U32{right(32n,kh,ch,mh,th)}},
        Equal.cong(Word(32n),U64,x => U64{U32{x},U32{left(32n,kh,ch,mh,th,rh)}},left(32n,kl,cl,ml,tl,rl),right(32n,kl,cl,ml,tl),
          word(32n,kl,cl,ml,tl,rl,Equal.cong(U64,Word(32n),x => lo(x),U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))),
        Equal.cong(Word(32n),U64,x => U64{U32{right(32n,kl,cl,ml,tl)},U32{x}},left(32n,kh,ch,mh,th,rh),right(32n,kh,ch,mh,th),
          word(32n,kh,ch,mh,th,rh,Equal.cong(U64,Word(32n),x => hi(x),U64.and(U64{U32{kl},U32{kh}},U64{U32{rl},U32{rh}}),U64.zero(),empty))))
