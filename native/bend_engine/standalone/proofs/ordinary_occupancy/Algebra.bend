import Base
import ../ray/Algebra.bend as A
import ../../../legal_probe/Chess.bend as Chess

def shuffle_bool(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.or(Bool.or(a,c),b) == Bool.or(Bool.or(a,b),c) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def shuffle_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.or(n,Word.or(n,a,c),b) == Word.or(n,Word.or(n,a,b),c) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.or(Bool.or(ah,ch),bh),Word.or(p,Word.or(p,at,ct),bt)},WCon{Bool.or(Bool.or(ah,bh),ch),Word.or(p,Word.or(p,at,ct),bt)},WCon{Bool.or(Bool.or(ah,bh),ch),Word.or(p,Word.or(p,at,bt),ct)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,Word.or(p,at,ct),bt)},Bool.or(Bool.or(ah,ch),bh),Bool.or(Bool.or(ah,bh),ch),shuffle_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(Bool.or(ah,bh),ch),t},Word.or(p,Word.or(p,at,ct),bt),Word.or(p,Word.or(p,at,bt),ct),shuffle_word(p,at,bt,ct)))

def shuffle(+a: U64,+b: U64,+c: U64) ->
  {U64.or(U64.or(a,c),b) == U64.or(U64.or(a,b),c) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.or(32n,Word.or(32n,al,cl),bl)},U32{Word.or(32n,Word.or(32n,ah,ch),bh)}},U64{U32{Word.or(32n,Word.or(32n,al,bl),cl)},U32{Word.or(32n,Word.or(32n,ah,ch),bh)}},U64{U32{Word.or(32n,Word.or(32n,al,bl),cl)},U32{Word.or(32n,Word.or(32n,ah,bh),ch)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.or(32n,ah,ch),bh)}},Word.or(32n,Word.or(32n,al,cl),bl),Word.or(32n,Word.or(32n,al,bl),cl),shuffle_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,Word.or(32n,al,bl),cl)},U32{h}},Word.or(32n,Word.or(32n,ah,ch),bh),Word.or(32n,Word.or(32n,ah,bh),ch),shuffle_word(32n,ah,bh,ch)))

def merge_bool(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.or(Bool.and(a,Bool.not(c)),Bool.and(b,Bool.not(c))) == Bool.and(Bool.or(a,b),Bool.not(c)) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def merge_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.or(n,Word.and(n,a,Word.not(n,c)),Word.and(n,b,Word.not(n,c))) == Word.and(n,Word.or(n,a,b),Word.not(n,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.or(Bool.and(ah,Bool.not(ch)),Bool.and(bh,Bool.not(ch))),Word.or(p,Word.and(p,at,Word.not(p,ct)),Word.and(p,bt,Word.not(p,ct)))},WCon{Bool.and(Bool.or(ah,bh),Bool.not(ch)),Word.or(p,Word.and(p,at,Word.not(p,ct)),Word.and(p,bt,Word.not(p,ct)))},WCon{Bool.and(Bool.or(ah,bh),Bool.not(ch)),Word.and(p,Word.or(p,at,bt),Word.not(p,ct))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,Word.and(p,at,Word.not(p,ct)),Word.and(p,bt,Word.not(p,ct)))},Bool.or(Bool.and(ah,Bool.not(ch)),Bool.and(bh,Bool.not(ch))),Bool.and(Bool.or(ah,bh),Bool.not(ch)),merge_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(Bool.or(ah,bh),Bool.not(ch)),t},Word.or(p,Word.and(p,at,Word.not(p,ct)),Word.and(p,bt,Word.not(p,ct))),Word.and(p,Word.or(p,at,bt),Word.not(p,ct)),merge_word(p,at,bt,ct)))

def merge(+a: U64,+b: U64,+c: U64) ->
  {U64.or(U64.and(a,U64.not(c)),U64.and(b,U64.not(c))) == U64.and(U64.or(a,b),U64.not(c)) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.or(32n,Word.and(32n,al,Word.not(32n,cl)),Word.and(32n,bl,Word.not(32n,cl)))},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,ch)),Word.and(32n,bh,Word.not(32n,ch)))}},U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.not(32n,cl))},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,ch)),Word.and(32n,bh,Word.not(32n,ch)))}},U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.not(32n,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),Word.not(32n,ch))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,ch)),Word.and(32n,bh,Word.not(32n,ch)))}},Word.or(32n,Word.and(32n,al,Word.not(32n,cl)),Word.and(32n,bl,Word.not(32n,cl))),Word.and(32n,Word.or(32n,al,bl),Word.not(32n,cl)),merge_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.not(32n,cl))},U32{h}},Word.or(32n,Word.and(32n,ah,Word.not(32n,ch)),Word.and(32n,bh,Word.not(32n,ch))),Word.and(32n,Word.or(32n,ah,bh),Word.not(32n,ch)),merge_word(32n,ah,bh,ch)))

def reinsert_bool(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.or(Bool.and(a,Bool.not(Bool.or(Bool.or(b,c),Bool.or(c,False{})))),c) == Bool.or(Bool.and(a,Bool.not(b)),c) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def reinsert_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.or(n,Word.and(n,a,Word.not(n,Word.or(n,Word.or(n,b,c),Word.or(n,c,Word.zero(n))))),c) == Word.or(n,Word.and(n,a,Word.not(n,b)),c) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(bh,ch),Bool.or(ch,False{})))),ch),Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,bt,ct),Word.or(p,ct,Word.zero(p))))),ct)},WCon{Bool.or(Bool.and(ah,Bool.not(bh)),ch),Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,bt,ct),Word.or(p,ct,Word.zero(p))))),ct)},WCon{Bool.or(Bool.and(ah,Bool.not(bh)),ch),Word.or(p,Word.and(p,at,Word.not(p,bt)),ct)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,bt,ct),Word.or(p,ct,Word.zero(p))))),ct)},Bool.or(Bool.and(ah,Bool.not(Bool.or(Bool.or(bh,ch),Bool.or(ch,False{})))),ch),Bool.or(Bool.and(ah,Bool.not(bh)),ch),reinsert_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(Bool.and(ah,Bool.not(bh)),ch),t},Word.or(p,Word.and(p,at,Word.not(p,Word.or(p,Word.or(p,bt,ct),Word.or(p,ct,Word.zero(p))))),ct),Word.or(p,Word.and(p,at,Word.not(p,bt)),ct),reinsert_word(p,at,bt,ct)))

def reinsert(+a: U64,+b: U64,+c: U64) ->
  {U64.or(U64.and(a,U64.not(U64.or(U64.or(b,c),U64.or(c,U64.zero())))),c) == U64.or(U64.and(a,U64.not(b)),c) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,bl,cl),Word.or(32n,cl,Word.zero(32n))))),cl)},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,bh,ch),Word.or(32n,ch,Word.zero(32n))))),ch)}},U64{U32{Word.or(32n,Word.and(32n,al,Word.not(32n,bl)),cl)},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,bh,ch),Word.or(32n,ch,Word.zero(32n))))),ch)}},U64{U32{Word.or(32n,Word.and(32n,al,Word.not(32n,bl)),cl)},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,bh)),ch)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,bh,ch),Word.or(32n,ch,Word.zero(32n))))),ch)}},Word.or(32n,Word.and(32n,al,Word.not(32n,Word.or(32n,Word.or(32n,bl,cl),Word.or(32n,cl,Word.zero(32n))))),cl),Word.or(32n,Word.and(32n,al,Word.not(32n,bl)),cl),reinsert_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,Word.and(32n,al,Word.not(32n,bl)),cl)},U32{h}},Word.or(32n,Word.and(32n,ah,Word.not(32n,Word.or(32n,Word.or(32n,bh,ch),Word.or(32n,ch,Word.zero(32n))))),ch),Word.or(32n,Word.and(32n,ah,Word.not(32n,bh)),ch),reinsert_word(32n,ah,bh,ch)))

# One of complementary colors inserts target; neither color requires consistency.
def colors(+a: U64,+b: U64,+target: U64,+white: Bool) ->
  {U64.or(U64.or(a,Chess.select_u64(white,target,U64.zero())),
    U64.or(b,Chess.select_u64(Bool.not(white),target,U64.zero()))) == U64.or(U64.or(a,b),target) : U64}:
  match white:
    case False{}:
      Equal.trans(U64,U64.or(U64.or(a,U64.zero()),U64.or(b,target)),U64.or(a,U64.or(b,target)),U64.or(U64.or(a,b),target),
        Equal.cong(U64,U64,x => U64.or(x,U64.or(b,target)),U64.or(a,U64.zero()),a,A.zero(a)),
        Equal.sym(U64,U64.or(U64.or(a,b),target),U64.or(a,U64.or(b,target)),A.or_assoc(a,b,target)))
    case True{}:
      Equal.trans(U64,U64.or(U64.or(a,target),U64.or(b,U64.zero())),U64.or(U64.or(a,target),b),U64.or(U64.or(a,b),target),
        Equal.cong(U64,U64,x => U64.or(U64.or(a,target),x),U64.or(b,U64.zero()),b,A.zero(b)),
        shuffle(a,b,target))

def color_updates(+w: U64,+b: U64,+mask: U64,+target: U64,+white: Bool) ->
  {U64.or(Chess.update_piece(w,mask,U64.or(target,U64.zero()),white),
    Chess.update_piece(b,mask,U64.or(target,U64.zero()),Bool.not(white))) ==
    U64.or(U64.and_not(U64.or(w,b),mask),target) : U64}:
  Equal.trans(U64,
    U64.or(Chess.update_piece(w,mask,U64.or(target,U64.zero()),white),Chess.update_piece(b,mask,U64.or(target,U64.zero()),Bool.not(white))),
    U64.or(Chess.update_piece(w,mask,target,white),Chess.update_piece(b,mask,target,Bool.not(white))),
    U64.or(U64.and_not(U64.or(w,b),mask),target),
    Equal.cong(U64,U64,t => U64.or(Chess.update_piece(w,mask,t,white),Chess.update_piece(b,mask,t,Bool.not(white))),
      U64.or(target,U64.zero()),target,A.zero(target)),
    Equal.trans(U64,U64.or(Chess.update_piece(w,mask,target,white),Chess.update_piece(b,mask,target,Bool.not(white))),
      U64.or(U64.or(U64.and_not(w,mask),U64.and_not(b,mask)),target),
      U64.or(U64.and_not(U64.or(w,b),mask),target),
      colors(U64.and_not(w,mask),U64.and_not(b,mask),target,white),
      Equal.cong(U64,U64,x => U64.or(x,target),U64.or(U64.and_not(w,mask),U64.and_not(b,mask)),
        U64.and_not(U64.or(w,b),mask),merge(w,b,mask))))
