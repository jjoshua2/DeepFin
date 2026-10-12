import Base
import ../ordinary_king_frame/Algebra.bend as P
import ../u64/Words.bend as Old

def erase_word(+n: Nat,+k: Word(n),+c: Word(n),+s: Word(n),v: Word(n),d: Word(n)) -> Word(n):
  Word.and(n,Word.and(n,k,c),Word.not(n,Word.or(n,Word.or(n,s,v),Word.or(n,d,Word.zero(n)))))
def single_word(+n: Nat,k: Word(n),c: Word(n),s: Word(n)) -> Word(n):
  Word.and(n,Word.and(n,k,c),Word.not(n,s))
def erase(+k: U64,+c: U64,+s: U64,v: U64,d: U64) -> U64:
  U64.and_not(U64.and(k,c),U64.or(U64.or(s,v),U64.or(d,U64.zero())))
def single(+k: U64,c: U64,s: U64) -> U64:
  U64.and_not(U64.and(k,c),s)

def strip_bool(+k: Bool,+c: Bool,+s: Bool,+v: Bool,+d: Bool,
  vc: {Bool.and(v,k) == False{} : Bool},dc: {Bool.and(d,k) == False{} : Bool}) ->
  {Bool.and(Bool.and(k,c),Bool.not(Bool.or(Bool.or(s,v),Bool.or(d,False{})))) == Bool.and(Bool.and(k,c),Bool.not(s)) : Bool}:
  match k c s v d:
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
    case True{} False{} False{} False{} True{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(False{},False{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},dc))
    case True{} False{} False{} True{} False{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(False{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} False{} False{} True{} True{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} True{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(True{},False{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},dc))
    case True{} False{} True{} True{} False{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(False{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} False{} True{} True{} True{}: Old.false_true({Bool.and(Bool.and(True{},False{}),Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},False{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} True{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(False{},False{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},dc))
    case True{} True{} False{} True{} False{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(False{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} True{} False{} True{} True{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(False{},True{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(False{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(True{},False{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},dc))
    case True{} True{} True{} True{} False{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(False{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},vc))
    case True{} True{} True{} True{} True{}: Old.false_true({Bool.and(Bool.and(True{},True{}),Bool.not(Bool.or(Bool.or(True{},True{}),Bool.or(True{},False{})))) == Bool.and(Bool.and(True{},True{}),Bool.not(True{})) : Bool},Equal.sym(Bool,True{},False{},vc))

def strip_word(+n: Nat,+k: Word(n),+c: Word(n),+s: Word(n),+v: Word(n),+d: Word(n),
  +vc: {Word.and(n,v,k) == Word.zero(n) : Word(n)},+dc: {Word.and(n,d,k) == Word.zero(n) : Word(n)}) ->
  {erase_word(n,k,c,s,v,d) == single_word(n,k,c,s) : Word(n)}:
  match n:
    case 0n:
      match k c s v d:
        case WNil{} WNil{} WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match k c s v d:
        case WCon{+kh,+kt} WCon{+ch,+ct} WCon{+sh,+st} WCon{+vh,+vt} WCon{+dh,+dt}:
          Equal.trans(Word(1n+p),erase_word(1n+p,k,c,s,v,d),
            WCon{Bool.and(Bool.and(kh,ch),Bool.not(sh)),erase_word(p,kt,ct,st,vt,dt)},single_word(1n+p,k,c,s),
            Equal.cong(Bool,Word(1n+p),h => WCon{h,erase_word(p,kt,ct,st,vt,dt)},
              Bool.and(Bool.and(kh,ch),Bool.not(Bool.or(Bool.or(sh,vh),Bool.or(dh,False{})))),
              Bool.and(Bool.and(kh,ch),Bool.not(sh)),
              strip_bool(kh,ch,sh,vh,dh,
                Equal.cong(Word(1n+p),Bool,x => P.head(p,x),Word.and(1n+p,v,k),Word.zero(1n+p),vc),
                Equal.cong(Word(1n+p),Bool,x => P.head(p,x),Word.and(1n+p,d,k),Word.zero(1n+p),dc))),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(Bool.and(kh,ch),Bool.not(sh)),t},
              erase_word(p,kt,ct,st,vt,dt),single_word(p,kt,ct,st),
              strip_word(p,kt,ct,st,vt,dt,
                Equal.cong(Word(1n+p),Word(p),x => P.tail(p,x),Word.and(1n+p,v,k),Word.zero(1n+p),vc),
                Equal.cong(Word(1n+p),Word(p),x => P.tail(p,x),Word.and(1n+p,d,k),Word.zero(1n+p),dc))))

def strip(+k: U64,+c: U64,+s: U64,+v: U64,+d: U64,
  +vc: {U64.and(v,k) == U64.zero() : U64},+dc: {U64.and(d,k) == U64.zero() : U64}) ->
  {erase(k,c,s,v,d) == single(k,c,s) : U64}:
  match k c s v d:
    case U64{U32{+kl},U32{+kh}} U64{U32{+cl},U32{+ch}} U64{U32{+sl},U32{+sh}} U64{U32{+vl},U32{+vh}} U64{U32{+dl},U32{+dh}}:
      Equal.trans(U64,erase(k,c,s,v,d),
        U64{U32{single_word(32n,kl,cl,sl)},U32{erase_word(32n,kh,ch,sh,vh,dh)}},single(k,c,s),
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{erase_word(32n,kh,ch,sh,vh,dh)}},
          erase_word(32n,kl,cl,sl,vl,dl),single_word(32n,kl,cl,sl),
          strip_word(32n,kl,cl,sl,vl,dl,
            Equal.cong(U64,Word(32n),P.low,U64.and(v,k),U64.zero(),vc),
            Equal.cong(U64,Word(32n),P.low,U64.and(d,k),U64.zero(),dc))),
        Equal.cong(Word(32n),U64,h => U64{U32{single_word(32n,kl,cl,sl)},U32{h}},
          erase_word(32n,kh,ch,sh,vh,dh),single_word(32n,kh,ch,sh),
          strip_word(32n,kh,ch,sh,vh,dh,
            Equal.cong(U64,Word(32n),P.high,U64.and(v,k),U64.zero(),vc),
            Equal.cong(U64,Word(32n),P.high,U64.and(d,k),U64.zero(),dc))))
