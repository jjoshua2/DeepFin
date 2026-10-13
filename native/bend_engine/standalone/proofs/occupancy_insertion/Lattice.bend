import Base
import ../ray/Algebra.bend as A
import ../decoder/Bits.bend as D

# Equality-valued inclusion, with the smaller word in the second AND position.
def subset(small: U64,big: U64) -> Type:
  {U64.and(big,small) == small : U64}

def shared_bool(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.or(a,b),Bool.or(a,c)) == Bool.or(a,Bool.and(b,c)) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def shared_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.and(n,Word.or(n,a,b),Word.or(n,a,c)) == Word.or(n,a,Word.and(n,b,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),Bool.or(ah,ch)),Word.and(p,Word.or(p,at,bt),Word.or(p,at,ct))},WCon{Bool.or(ah,Bool.and(bh,ch)),Word.and(p,Word.or(p,at,bt),Word.or(p,at,ct))},WCon{Bool.or(ah,Bool.and(bh,ch)),Word.or(p,at,Word.and(p,bt,ct))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),Word.or(p,at,ct))},Bool.and(Bool.or(ah,bh),Bool.or(ah,ch)),Bool.or(ah,Bool.and(bh,ch)),shared_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(ah,Bool.and(bh,ch)),t},Word.and(p,Word.or(p,at,bt),Word.or(p,at,ct)),Word.or(p,at,Word.and(p,bt,ct)),shared_word(p,at,bt,ct)))

def shared(+a: U64,+b: U64,+c: U64) ->
  {U64.and(U64.or(a,b),U64.or(a,c)) == U64.or(a,U64.and(b,c)) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),Word.or(32n,al,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),Word.or(32n,ah,ch))}},U64{U32{Word.or(32n,al,Word.and(32n,bl,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),Word.or(32n,ah,ch))}},U64{U32{Word.or(32n,al,Word.and(32n,bl,cl))},U32{Word.or(32n,ah,Word.and(32n,bh,ch))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),Word.or(32n,ah,ch))}},Word.and(32n,Word.or(32n,al,bl),Word.or(32n,al,cl)),Word.or(32n,al,Word.and(32n,bl,cl)),shared_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,al,Word.and(32n,bl,cl))},U32{h}},Word.and(32n,Word.or(32n,ah,bh),Word.or(32n,ah,ch)),Word.or(32n,ah,Word.and(32n,bh,ch)),shared_word(32n,ah,bh,ch)))

def or_subset(+acc: U64,+small: U64,+big: U64,e: subset(small,big)) ->
  subset(U64.or(acc,small),U64.or(acc,big)):
  Equal.trans(U64,U64.and(U64.or(acc,big),U64.or(acc,small)),U64.or(acc,U64.and(big,small)),U64.or(acc,small),
    shared(acc,big,small),Equal.cong(U64,U64,w => U64.or(acc,w),U64.and(big,small),small,e))

def reflexive(+a: U64) -> subset(a,a):
  Equal.trans(U64,U64.and(a,a),U64.and(U64.or(a,U64.zero()),a),a,
    Equal.cong(U64,U64,w => U64.and(w,a),a,U64.or(a,U64.zero()),
      Equal.sym(U64,U64.or(a,U64.zero()),a,A.zero(a))),
    A.absorb_left(a,U64.zero()))

def rewrite(+small: U64,+big: U64,+out_small: U64,+out_big: U64,
  +es: {out_small == small : U64},eb: {out_big == big : U64},e: subset(small,big)) ->
  subset(out_small,out_big):
  Equal.trans(U64,U64.and(out_big,out_small),U64.and(big,out_small),out_small,
    Equal.cong(U64,U64,w => U64.and(w,out_small),out_big,big,eb),
    Equal.trans(U64,U64.and(big,out_small),U64.and(big,small),out_small,
      Equal.cong(U64,U64,w => U64.and(big,w),out_small,small,es),
      Equal.trans(U64,U64.and(big,small),small,out_small,e,Equal.sym(U64,out_small,small,es))))

def or_equal(+a: Bool,+b: Bool,+c: Bool,+d: Bool,
  ea: {a == c : Bool},eb: {b == d : Bool}) -> {Bool.or(a,b) == Bool.or(c,d) : Bool}:
  Equal.trans(Bool,Bool.or(a,b),Bool.or(c,b),Bool.or(c,d),
    Equal.cong(Bool,Bool,x => Bool.or(x,b),a,c,ea),
    Equal.cong(Bool,Bool,x => Bool.or(c,x),b,d,eb))

def medial(+a: Bool,+b: Bool,+c: Bool,+d: Bool) ->
  {Bool.or(Bool.or(a,b),Bool.or(c,d)) == Bool.or(Bool.or(a,c),Bool.or(b,d)) : Bool}:
  match a b c d:
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

def word_any_or(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {D.any(n,Word.or(n,a,b)) == Bool.or(D.any(n,a),D.any(n,b)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{ah,at} WCon{bh,bt}:
          Equal.trans(Bool,D.any(1n+p,Word.or(1n+p,WCon{ah,at},WCon{bh,bt})),
            Bool.or(Bool.or(ah,bh),Bool.or(D.any(p,at),D.any(p,bt))),
            Bool.or(D.any(1n+p,WCon{ah,at}),D.any(1n+p,WCon{bh,bt})),
            Equal.cong(Bool,Bool,x => Bool.or(Bool.or(ah,bh),x),D.any(p,Word.or(p,at,bt)),
              Bool.or(D.any(p,at),D.any(p,bt)),word_any_or(p,at,bt)),
            medial(ah,bh,D.any(p,at),D.any(p,bt)))

def any(a: U64) -> Bool:
  match a:
    case U64{U32{lo},U32{hi}}: Bool.or(D.any(32n,hi),D.any(32n,lo))

def nonzero_any(+a: U64) -> {U64.is_ne(a,U64.zero()) == any(a) : Bool}:
  match a:
    case U64{U32{lo},U32{hi}}:
      Equal.trans(Bool,U64.is_ne(U64{U32{lo},U32{hi}},U64.zero()),
        Bool.or(D.nonzero(32n,hi),D.nonzero(32n,lo)),any(U64{U32{lo},U32{hi}}),
        D.cmp_limbs(U32{lo},Word.cmp(32n,hi,Word.zero(32n))),
        or_equal(D.nonzero(32n,hi),D.nonzero(32n,lo),D.any(32n,hi),D.any(32n,lo),
          D.nonzero_any(32n,hi),D.nonzero_any(32n,lo)))

def any_or(+a: U64,+b: U64) -> {any(U64.or(a,b)) == Bool.or(any(a),any(b)) : Bool}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      Equal.trans(Bool,any(U64.or(U64{U32{al},U32{ah}},U64{U32{bl},U32{bh}})),
        Bool.or(Bool.or(D.any(32n,ah),D.any(32n,bh)),Bool.or(D.any(32n,al),D.any(32n,bl))),
        Bool.or(any(U64{U32{al},U32{ah}}),any(U64{U32{bl},U32{bh}})),
        or_equal(D.any(32n,Word.or(32n,ah,bh)),D.any(32n,Word.or(32n,al,bl)),
          Bool.or(D.any(32n,ah),D.any(32n,bh)),Bool.or(D.any(32n,al),D.any(32n,bl)),
          word_any_or(32n,ah,bh),word_any_or(32n,al,bl)),
        medial(D.any(32n,ah),D.any(32n,bh),D.any(32n,al),D.any(32n,bl)))

def nonzero_or(+a: U64,+b: U64) ->
  {U64.is_ne(U64.or(a,b),U64.zero()) == Bool.or(U64.is_ne(a,U64.zero()),U64.is_ne(b,U64.zero())) : Bool}:
  Equal.trans(Bool,U64.is_ne(U64.or(a,b),U64.zero()),any(U64.or(a,b)),
    Bool.or(U64.is_ne(a,U64.zero()),U64.is_ne(b,U64.zero())),nonzero_any(U64.or(a,b)),
    Equal.trans(Bool,any(U64.or(a,b)),Bool.or(any(a),any(b)),
      Bool.or(U64.is_ne(a,U64.zero()),U64.is_ne(b,U64.zero())),any_or(a,b),
      or_equal(any(a),any(b),U64.is_ne(a,U64.zero()),U64.is_ne(b,U64.zero()),
        Equal.sym(Bool,U64.is_ne(a,U64.zero()),any(a),nonzero_any(a)),
        Equal.sym(Bool,U64.is_ne(b,U64.zero()),any(b),nonzero_any(b)))))

def and_or_bool(+a: Bool,+b: Bool,+c: Bool) ->
  {Bool.and(Bool.or(a,b),c) == Bool.or(Bool.and(a,c),Bool.and(b,c)) : Bool}:
  match a b c:
    case False{} False{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} True{} True{}: {==}
    case True{} False{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def and_or_word(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {Word.and(n,Word.or(n,a,b),c) == Word.or(n,Word.and(n,a,c),Word.and(n,b,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          Equal.trans(Word(1n+p),WCon{Bool.and(Bool.or(ah,bh),ch),Word.and(p,Word.or(p,at,bt),ct)},WCon{Bool.or(Bool.and(ah,ch),Bool.and(bh,ch)),Word.and(p,Word.or(p,at,bt),ct)},WCon{Bool.or(Bool.and(ah,ch),Bool.and(bh,ch)),Word.or(p,Word.and(p,at,ct),Word.and(p,bt,ct))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.and(p,Word.or(p,at,bt),ct)},Bool.and(Bool.or(ah,bh),ch),Bool.or(Bool.and(ah,ch),Bool.and(bh,ch)),and_or_bool(ah,bh,ch)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{Bool.or(Bool.and(ah,ch),Bool.and(bh,ch)),t},Word.and(p,Word.or(p,at,bt),ct),Word.or(p,Word.and(p,at,ct),Word.and(p,bt,ct)),and_or_word(p,at,bt,ct)))

def and_or(+a: U64,+b: U64,+c: U64) ->
  {U64.and(U64.or(a,b),c) == U64.or(U64.and(a,c),U64.and(b,c)) : U64}:
  match a b c:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}} U64{U32{cl},U32{ch}}:
      Equal.trans(U64,U64{U32{Word.and(32n,Word.or(32n,al,bl),cl)},U32{Word.and(32n,Word.or(32n,ah,bh),ch)}},U64{U32{Word.or(32n,Word.and(32n,al,cl),Word.and(32n,bl,cl))},U32{Word.and(32n,Word.or(32n,ah,bh),ch)}},U64{U32{Word.or(32n,Word.and(32n,al,cl),Word.and(32n,bl,cl))},U32{Word.or(32n,Word.and(32n,ah,ch),Word.and(32n,bh,ch))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.or(32n,ah,bh),ch)}},Word.and(32n,Word.or(32n,al,bl),cl),Word.or(32n,Word.and(32n,al,cl),Word.and(32n,bl,cl)),and_or_word(32n,al,bl,cl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,Word.and(32n,al,cl),Word.and(32n,bl,cl))},U32{h}},Word.and(32n,Word.or(32n,ah,bh),ch),Word.or(32n,Word.and(32n,ah,ch),Word.and(32n,bh,ch)),and_or_word(32n,ah,bh,ch)))

# The mask is the actual U64.bit(x), including its out-of-range behavior.
def test_or(+occ: U64,+added: U64,+x: Nat) ->
  {U64.test_bit(U64.or(occ,added),x) == Bool.or(U64.test_bit(occ,x),U64.test_bit(added,x)) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.or(occ,added),x),
    U64.is_ne(U64.or(U64.and(occ,U64.bit(x)),U64.and(added,U64.bit(x))),U64.zero()),
    Bool.or(U64.test_bit(occ,x),U64.test_bit(added,x)),
    Equal.cong(U64,Bool,w => U64.is_ne(w,U64.zero()),U64.and(U64.or(occ,added),U64.bit(x)),
      U64.or(U64.and(occ,U64.bit(x)),U64.and(added,U64.bit(x))),and_or(occ,added,U64.bit(x))),
    nonzero_or(U64.and(occ,U64.bit(x)),U64.and(added,U64.bit(x))))

def insert_preserves(+occ: U64,+added: U64,+x: Nat,e: {U64.test_bit(occ,x) == True{} : Bool}) ->
  {U64.test_bit(U64.or(occ,added),x) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.or(occ,added),x),
    Bool.or(U64.test_bit(occ,x),U64.test_bit(added,x)),True{},test_or(occ,added,x),
    Equal.cong(Bool,Bool,b => Bool.or(b,U64.test_bit(added,x)),U64.test_bit(occ,x),True{},e))
