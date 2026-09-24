import Base

# Structural bitwise algebra; no assumed machine-word equalities.
def or_bool_assoc(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.or(Bool.or(a,b),c) == Bool.or(a,Bool.or(b,c)) : Bool}:
  match a:
    case True{}: {==}
    case False{}: {==}

def or_word_assoc(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.or(n,Word.or(n,a,b),c) == Word.or(n,a,Word.or(n,b,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),
            WCon{Bool.or(Bool.or(ah,bh),ch),Word.or(p,Word.or(p,at,bt),ct)},
            WCon{Bool.or(ah,Bool.or(bh,ch)),Word.or(p,Word.or(p,at,bt),ct)},
            WCon{Bool.or(ah,Bool.or(bh,ch)),Word.or(p,at,Word.or(p,bt,ct))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,Word.or(p,at,bt),ct)},
              Bool.or(Bool.or(ah,bh),ch),Bool.or(ah,Bool.or(bh,ch)),or_bool_assoc(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(ah,Bool.or(bh,ch)),t},
              Word.or(p,Word.or(p,at,bt),ct),Word.or(p,at,Word.or(p,bt,ct)),or_word_assoc(p,at,bt,ct)))

def or_assoc(+a: U64,+b: U64,+c: U64) -> {U64.or(U64.or(a,b),c) == U64.or(a,U64.or(b,c)) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,
        U64{U32{Word.or(32n,Word.or(32n,al,bl),cl)},U32{Word.or(32n,Word.or(32n,ah,bh),ch)}},
        U64{U32{Word.or(32n,al,Word.or(32n,bl,cl))},U32{Word.or(32n,Word.or(32n,ah,bh),ch)}},
        U64{U32{Word.or(32n,al,Word.or(32n,bl,cl))},U32{Word.or(32n,ah,Word.or(32n,bh,ch))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.or(32n,ah,bh),ch)}},
          Word.or(32n,Word.or(32n,al,bl),cl),Word.or(32n,al,Word.or(32n,bl,cl)),or_word_assoc(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,al,Word.or(32n,bl,cl))},U32{h}},
          Word.or(32n,Word.or(32n,ah,bh),ch),Word.or(32n,ah,Word.or(32n,bh,ch)),or_word_assoc(32n,ah,bh,ch)))

def and_bool_assoc(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.and(a,b),c) == Bool.and(a,Bool.and(b,c)) : Bool}:
  match a:
    case True{}: {==}
    case False{}: {==}

def and_word_assoc(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.and(n,Word.and(n,a,b),c) == Word.and(n,a,Word.and(n,b,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),
            WCon{Bool.and(Bool.and(ah,bh),ch),Word.and(p,Word.and(p,at,bt),ct)},
            WCon{Bool.and(ah,Bool.and(bh,ch)),Word.and(p,Word.and(p,at,bt),ct)},
            WCon{Bool.and(ah,Bool.and(bh,ch)),Word.and(p,at,Word.and(p,bt,ct))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.and(p,at,bt),ct)},
              Bool.and(Bool.and(ah,bh),ch),Bool.and(ah,Bool.and(bh,ch)),and_bool_assoc(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(ah,Bool.and(bh,ch)),t},
              Word.and(p,Word.and(p,at,bt),ct),Word.and(p,at,Word.and(p,bt,ct)),and_word_assoc(p,at,bt,ct)))

def and_assoc(+a: U64,+b: U64,+c: U64) -> {U64.and(U64.and(a,b),c) == U64.and(a,U64.and(b,c)) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,
        U64{U32{Word.and(32n,Word.and(32n,al,bl),cl)},U32{Word.and(32n,Word.and(32n,ah,bh),ch)}},
        U64{U32{Word.and(32n,al,Word.and(32n,bl,cl))},U32{Word.and(32n,Word.and(32n,ah,bh),ch)}},
        U64{U32{Word.and(32n,al,Word.and(32n,bl,cl))},U32{Word.and(32n,ah,Word.and(32n,bh,ch))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.and(32n,ah,bh),ch)}},
          Word.and(32n,Word.and(32n,al,bl),cl),Word.and(32n,al,Word.and(32n,bl,cl)),and_word_assoc(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,al,Word.and(32n,bl,cl))},U32{h}},
          Word.and(32n,Word.and(32n,ah,bh),ch),Word.and(32n,ah,Word.and(32n,bh,ch)),and_word_assoc(32n,ah,bh,ch)))

def word_zero(+n: Nat,+a: Word(n)) -> {Word.or(n,a,Word.zero(n)) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a:
        case WCon{ah,at}:
          match ah:
            case False{}:
              Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.or(p,at,Word.zero(p)),at,word_zero(p,at))
            case True{}:
              Equal.cong(Word(p),Word(1n+p),t => WCon{True{},t},Word.or(p,at,Word.zero(p)),at,word_zero(p,at))

def zero(+a: U64) -> {U64.or(a,U64.zero()) == a : U64}:
  match a:
    case U64{U32{al},U32{ah}}:
      Equal.trans(U64,U64{U32{Word.or(32n,al,Word.zero(32n))},U32{Word.or(32n,ah,Word.zero(32n))}},
        U64{U32{al},U32{Word.or(32n,ah,Word.zero(32n))}},U64{U32{al},U32{ah}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,ah,Word.zero(32n))}},Word.or(32n,al,Word.zero(32n)),al,word_zero(32n,al)),
        Equal.cong(Word(32n),U64,h => U64{U32{al},U32{h}},Word.or(32n,ah,Word.zero(32n)),ah,word_zero(32n,ah)))

def word_left_zero(+n: Nat,+a: Word(n)) -> {Word.or(n,Word.zero(n),a) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a:
        case WCon{h,t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{h,x},Word.or(p,Word.zero(p),t),t,word_left_zero(p,t))

def left_zero(+a: U64) -> {U64.or(U64.zero(),a) == a : U64}:
  match a:
    case U64{U32{al},U32{ah}}:
      Equal.trans(U64,U64{U32{Word.or(32n,Word.zero(32n),al)},U32{Word.or(32n,Word.zero(32n),ah)}},
        U64{U32{al},U32{Word.or(32n,Word.zero(32n),ah)}},U64{U32{al},U32{ah}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.zero(32n),ah)}},Word.or(32n,Word.zero(32n),al),al,word_left_zero(32n,al)),
        Equal.cong(Word(32n),U64,h => U64{U32{al},U32{h}},Word.or(32n,Word.zero(32n),ah),ah,word_left_zero(32n,ah)))

# Lattice identities used to retain interior membership under mask unions.
def absorb_left_bool(+a: Bool,+b: Bool) -> {Bool.and(Bool.or(a,b),a) == a : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def absorb_left_word(+n: Nat,+a: Word(n),+b: Word(n)) -> {Word.and(n,Word.or(n,a,b),a) == a : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),ah),Word.and(p,Word.or(p,at,bt),at)},WCon{ah,Word.and(p,Word.or(p,at,bt),at)},WCon{ah,at},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),at)},Bool.and(Bool.or(ah,bh),ah),ah,absorb_left_bool(ah,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{ah,t},Word.and(p,Word.or(p,at,bt),at),at,absorb_left_word(p,at,bt)))

def absorb_left(+a: U64,+b: U64) -> {U64.and(U64.or(a,b),a) == a : U64}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),al)},U32{Word.and(32n,Word.or(32n,ah,bh),ah)}},U64{U32{al},U32{Word.and(32n,Word.or(32n,ah,bh),ah)}},U64{U32{al},U32{ah}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),ah)}},Word.and(32n,Word.or(32n,al,bl),al),al,absorb_left_word(32n,al,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{al},U32{h}},Word.and(32n,Word.or(32n,ah,bh),ah),ah,absorb_left_word(32n,ah,bh)))

def extend_left_bool(+a: Bool,+b: Bool,+c: Bool) -> {Bool.and(Bool.or(a,b),Bool.and(b,c)) == Bool.and(b,c) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def extend_left_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) -> {Word.and(n,Word.or(n,a,b),Word.and(n,b,c)) == Word.and(n,b,c) : Word(n)}:
  match n:
    case 0n:
      match a b c:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),Bool.and(bh,ch)),Word.and(p,Word.or(p,at,bt),Word.and(p,bt,ct))},WCon{Bool.and(bh,ch),Word.and(p,Word.or(p,at,bt),Word.and(p,bt,ct))},WCon{Bool.and(bh,ch),Word.and(p,bt,ct)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),Word.and(p,bt,ct))},Bool.and(Bool.or(ah,bh),Bool.and(bh,ch)),Bool.and(bh,ch),extend_left_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(bh,ch),t},Word.and(p,Word.or(p,at,bt),Word.and(p,bt,ct)),Word.and(p,bt,ct),extend_left_word(p,at,bt,ct)))

def extend_left(+a: U64,+b: U64,+c: U64) -> {U64.and(U64.or(a,b),U64.and(b,c)) == U64.and(b,c) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.and(32n,bl,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,bh,ch))}},U64{U32{Word.and(32n,bl,cl)},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,bh,ch))}},U64{U32{Word.and(32n,bl,cl)},U32{Word.and(32n,bh,ch)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,bh,ch))}},Word.and(32n,Word.or(32n,al,bl),Word.and(32n,bl,cl)),Word.and(32n,bl,cl),extend_left_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,bl,cl)},U32{h}},Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,bh,ch)),Word.and(32n,bh,ch),extend_left_word(32n,ah,bh,ch)))

def extend_right_bool(+a: Bool,+b: Bool,+c: Bool) -> {Bool.and(Bool.or(a,b),Bool.and(a,c)) == Bool.and(a,c) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def extend_right_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) -> {Word.and(n,Word.or(n,a,b),Word.and(n,a,c)) == Word.and(n,a,c) : Word(n)}:
  match n:
    case 0n:
      match a b c:
        case WNil{} WNil{} WNil{}: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),Bool.and(ah,ch)),Word.and(p,Word.or(p,at,bt),Word.and(p,at,ct))},WCon{Bool.and(ah,ch),Word.and(p,Word.or(p,at,bt),Word.and(p,at,ct))},WCon{Bool.and(ah,ch),Word.and(p,at,ct)},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),Word.and(p,at,ct))},Bool.and(Bool.or(ah,bh),Bool.and(ah,ch)),Bool.and(ah,ch),extend_right_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.and(ah,ch),t},Word.and(p,Word.or(p,at,bt),Word.and(p,at,ct)),Word.and(p,at,ct),extend_right_word(p,at,bt,ct)))

def extend_right(+a: U64,+b: U64,+c: U64) -> {U64.and(U64.or(a,b),U64.and(a,c)) == U64.and(a,c) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.and(32n,al,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,ah,ch))}},U64{U32{Word.and(32n,al,cl)},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,ah,ch))}},U64{U32{Word.and(32n,al,cl)},U32{Word.and(32n,ah,ch)}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,ah,ch))}},Word.and(32n,Word.or(32n,al,bl),Word.and(32n,al,cl)),Word.and(32n,al,cl),extend_right_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.and(32n,al,cl)},U32{h}},Word.and(32n,Word.or(32n,ah,bh),Word.and(32n,ah,ch)),Word.and(32n,ah,ch),extend_right_word(32n,ah,bh,ch)))
