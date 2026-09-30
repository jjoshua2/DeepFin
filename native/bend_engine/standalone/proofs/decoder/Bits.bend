# Independent bounded Word projection and its correspondence to real test_bit.
import Base
import ../u64/Words.bend as Old

def get(+n: Nat,w: Word(n),i: Nat) -> Bool:
  match n:
    case 0n: False{}
    case 1n+p:
      match w i:
        case WCon{b,t} 0n: b
        case WCon{b,t} 1n+j: get(p,t,j)

def bit(n: Nat,i: Nat) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match i:
        case 0n: WCon{True{},Word.zero(p)}
        case 1n+j: WCon{False{},bit(p,j)}

def any(+n: Nat,w: Word(n)) -> Bool:
  match n:
    case 0n: False{}
    case 1n+p:
      match w:
        case WCon{b,t}: Bool.or(b,any(p,t))

def nonzero(+n: Nat,w: Word(n)) -> Bool:
  Bool.not(Cmp.is_eq(Word.cmp(n,w,Word.zero(n))))

def cmp_step(+b: Bool,+c: Cmp) ->
  {Bool.not(Cmp.is_eq(Word.cmp.fin(b,False{},c))) == Bool.or(b,Bool.not(Cmp.is_eq(c))) : Bool}:
  match b c:
    case False{} LT{}: {==}
    case False{} EQ{}: {==}
    case False{} GT{}: {==}
    case True{} LT{}: {==}
    case True{} EQ{}: {==}
    case True{} GT{}: {==}

def nonzero_any(+n: Nat,+w: Word(n)) -> {nonzero(n,w) == any(n,w) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{+b,+t}:
          Equal.trans(Bool,nonzero(1n+p,WCon{b,t}),Bool.or(b,nonzero(p,t)),Bool.or(b,any(p,t)),
            cmp_step(b,Word.cmp(p,t,Word.zero(p))),
            Equal.cong(Bool,Bool,z => Bool.or(b,z),nonzero(p,t),any(p,t),nonzero_any(p,t)))

def and_zero(+n: Nat,w: Word(n)) -> {Word.and(n,w,Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{b,t}:
          match b:
            case False{}: Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))
            case True{}: Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))

def any_zero(+n: Nat) -> {any(n,Word.zero(n)) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: any_zero(p)

def any_and_bit(+n: Nat,+w: Word(n),+i: Nat) ->
  {any(n,Word.and(n,w,bit(n,i))) == get(n,w,i) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w i:
        case WCon{False{},+t} 0n:
          Equal.trans(Bool,any(p,Word.and(p,t,Word.zero(p))),any(p,Word.zero(p)),False{},
            Equal.cong(Word(p),Bool,x => any(p,x),Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t)),any_zero(p))
        case WCon{True{},t} 0n: {==}
        case WCon{False{},t} 1n+j: any_and_bit(p,t,j)
        case WCon{True{},t} 1n+j: any_and_bit(p,t,j)

def test(+n: Nat,+w: Word(n),+i: Nat) ->
  {nonzero(n,Word.and(n,w,bit(n,i))) == get(n,w,i) : Bool}:
  Equal.trans(Bool,nonzero(n,Word.and(n,w,bit(n,i))),any(n,Word.and(n,w,bit(n,i))),get(n,w,i),
    nonzero_any(n,Word.and(n,w,bit(n,i))),any_and_bit(n,w,i))

def test_zero(+n: Nat,+w: Word(n)) -> {nonzero(n,Word.and(n,w,Word.zero(n))) == False{} : Bool}:
  Equal.trans(Bool,nonzero(n,Word.and(n,w,Word.zero(n))),any(n,Word.and(n,w,Word.zero(n))),False{},
    nonzero_any(n,Word.and(n,w,Word.zero(n))),
    Equal.trans(Bool,any(n,Word.and(n,w,Word.zero(n))),any(n,Word.zero(n)),False{},
      Equal.cong(Word(n),Bool,x => any(n,x),Word.and(n,w,Word.zero(n)),Word.zero(n),and_zero(n,w)),any_zero(n)))

def cmp_limbs(+l: U32,+h: Cmp) ->
  {Bool.not(Cmp.is_eq(U64.cmp.fin(l,0,h))) == Bool.or(Bool.not(Cmp.is_eq(h)),U32.is_ne(l,0)) : Bool}:
  match h:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

def or_false(+x: Bool) -> {Bool.or(x,False{}) == x : Bool}:
  match x:
    case False{}: {==}
    case True{}: {==}

def mask_low(+lo: Word(32n),+hi: Word(32n),+i: Nat) ->
  {U64.is_ne(U64.and(U64{U32{lo},U32{hi}},U64{U32{bit(32n,i)},0}),U64.zero()) == get(32n,lo,i) : Bool}:
  Equal.trans(Bool,U64.is_ne(U64.and(U64{U32{lo},U32{hi}},U64{U32{bit(32n,i)},0}),U64.zero()),
    Bool.or(nonzero(32n,Word.and(32n,hi,Word.zero(32n))),nonzero(32n,Word.and(32n,lo,bit(32n,i)))),get(32n,lo,i),
    cmp_limbs(U32{Word.and(32n,lo,bit(32n,i))},Word.cmp(32n,Word.and(32n,hi,Word.zero(32n)),Word.zero(32n))),
    Equal.trans(Bool,Bool.or(nonzero(32n,Word.and(32n,hi,Word.zero(32n))),nonzero(32n,Word.and(32n,lo,bit(32n,i)))),
      nonzero(32n,Word.and(32n,lo,bit(32n,i))),get(32n,lo,i),
      Equal.cong(Bool,Bool,z => Bool.or(z,nonzero(32n,Word.and(32n,lo,bit(32n,i)))),
        nonzero(32n,Word.and(32n,hi,Word.zero(32n))),False{},test_zero(32n,hi)),test(32n,lo,i)))

def mask_high(+lo: Word(32n),+hi: Word(32n),+i: Nat) ->
  {U64.is_ne(U64.and(U64{U32{lo},U32{hi}},U64{0,U32{bit(32n,i)}}),U64.zero()) == get(32n,hi,i) : Bool}:
  Equal.trans(Bool,U64.is_ne(U64.and(U64{U32{lo},U32{hi}},U64{0,U32{bit(32n,i)}}),U64.zero()),
    Bool.or(nonzero(32n,Word.and(32n,hi,bit(32n,i))),nonzero(32n,Word.and(32n,lo,Word.zero(32n)))),get(32n,hi,i),
    cmp_limbs(U32{Word.and(32n,lo,Word.zero(32n))},Word.cmp(32n,Word.and(32n,hi,bit(32n,i)),Word.zero(32n))),
    Equal.trans(Bool,Bool.or(nonzero(32n,Word.and(32n,hi,bit(32n,i))),nonzero(32n,Word.and(32n,lo,Word.zero(32n)))),
      Bool.or(nonzero(32n,Word.and(32n,hi,bit(32n,i))),False{}),get(32n,hi,i),
      Equal.cong(Bool,Bool,z => Bool.or(nonzero(32n,Word.and(32n,hi,bit(32n,i))),z),
        nonzero(32n,Word.and(32n,lo,Word.zero(32n))),False{},test_zero(32n,lo)),
      Equal.trans(Bool,Bool.or(nonzero(32n,Word.and(32n,hi,bit(32n,i))),False{}),nonzero(32n,Word.and(32n,hi,bit(32n,i))),get(32n,hi,i),
        or_false(nonzero(32n,Word.and(32n,hi,bit(32n,i)))),test(32n,hi,i))))

def mask64(+i: Nat) -> U64:
  Bool.pick(U64,Nat.is_lt(i,32n),U64{U32{bit(32n,i)},0},U64{0,U32{bit(32n,Nat.sub(i,32n))}})

def get64(a: U64,+i: Nat) -> Bool:
  U64{U32{lo},U32{hi}} = a
  Bool.pick(Bool,Nat.is_lt(i,32n),get(32n,lo,i),get(32n,hi,Nat.sub(i,32n)))

def impossible(T: Type,n: Nat,e: {Nat.is_lt(n,0n) == True{} : Bool}) -> T:
  match n:
    case 0n: Old.false_true(T,e)
    case 1n+p: Old.false_true(T,e)

def bit_encoding(+i: Nat,valid: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.bit(i) == mask64(i) : U64}:
  match i:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+j:
      impossible({U64.bit(64n+j) == mask64(64n+j) : U64},j,valid)

def mask_choice(+lo: Word(32n),+hi: Word(32n),+i: Nat,+choice: Bool) ->
  {U64.is_ne(U64.and(U64{U32{lo},U32{hi}},Bool.pick(U64,choice,U64{U32{bit(32n,i)},0},U64{0,U32{bit(32n,Nat.sub(i,32n))}})),U64.zero()) ==
    Bool.pick(Bool,choice,get(32n,lo,i),get(32n,hi,Nat.sub(i,32n))) : Bool}:
  match choice:
    case True{}: mask_low(lo,hi,i)
    case False{}: mask_high(lo,hi,Nat.sub(i,32n))

def test64(+a: U64,+i: Nat,valid: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(a,i) == get64(a,i) : Bool}:
  match a:
    case U64{U32{+lo},U32{+hi}}:
      Equal.trans(Bool,U64.test_bit(U64{U32{lo},U32{hi}},i),U64.is_ne(U64.and(U64{U32{lo},U32{hi}},mask64(i)),U64.zero()),get64(U64{U32{lo},U32{hi}},i),
        Equal.cong(U64,Bool,m => U64.is_ne(U64.and(U64{U32{lo},U32{hi}},m),U64.zero()),U64.bit(i),mask64(i),bit_encoding(i,valid)),
        mask_choice(lo,hi,i,Nat.is_lt(i,32n)))

def get_or(+n: Nat,+a: Word(n),+b: Word(n),+i: Nat) ->
  {get(n,Word.or(n,a,b),i) == Bool.or(get(n,a,i),get(n,b,i)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b i:
        case WCon{ah,at} WCon{bh,bt} 0n: {==}
        case WCon{ah,at} WCon{bh,bt} 1n+j: get_or(p,at,bt,j)

def or_pick(+c: Bool,+al: Bool,+ah: Bool,+bl: Bool,+bh: Bool) ->
  {Bool.pick(Bool,c,Bool.or(al,bl),Bool.or(ah,bh)) == Bool.or(Bool.pick(Bool,c,al,ah),Bool.pick(Bool,c,bl,bh)) : Bool}:
  match c:
    case False{}: {==}
    case True{}: {==}

def get64_or(+a: U64,+b: U64,+i: Nat) ->
  {get64(U64.or(a,b),i) == Bool.or(get64(a,i),get64(b,i)) : Bool}:
  match a b:
    case U64{U32{+al},U32{+ah}} U64{U32{+bl},U32{+bh}}:
      Equal.trans(Bool,get64(U64.or(U64{U32{al},U32{ah}},U64{U32{bl},U32{bh}}),i),
        Bool.pick(Bool,Nat.is_lt(i,32n),Bool.or(get(32n,al,i),get(32n,bl,i)),get(32n,Word.or(32n,ah,bh),Nat.sub(i,32n))),
        Bool.or(get64(U64{U32{al},U32{ah}},i),get64(U64{U32{bl},U32{bh}},i)),
        Equal.cong(Bool,Bool,x => Bool.pick(Bool,Nat.is_lt(i,32n),x,get(32n,Word.or(32n,ah,bh),Nat.sub(i,32n))),
          get(32n,Word.or(32n,al,bl),i),Bool.or(get(32n,al,i),get(32n,bl,i)),get_or(32n,al,bl,i)),
        Equal.trans(Bool,Bool.pick(Bool,Nat.is_lt(i,32n),Bool.or(get(32n,al,i),get(32n,bl,i)),get(32n,Word.or(32n,ah,bh),Nat.sub(i,32n))),
          Bool.pick(Bool,Nat.is_lt(i,32n),Bool.or(get(32n,al,i),get(32n,bl,i)),Bool.or(get(32n,ah,Nat.sub(i,32n)),get(32n,bh,Nat.sub(i,32n)))),
          Bool.or(get64(U64{U32{al},U32{ah}},i),get64(U64{U32{bl},U32{bh}},i)),
          Equal.cong(Bool,Bool,x => Bool.pick(Bool,Nat.is_lt(i,32n),Bool.or(get(32n,al,i),get(32n,bl,i)),x),
            get(32n,Word.or(32n,ah,bh),Nat.sub(i,32n)),Bool.or(get(32n,ah,Nat.sub(i,32n)),get(32n,bh,Nat.sub(i,32n))),get_or(32n,ah,bh,Nat.sub(i,32n))),
          or_pick(Nat.is_lt(i,32n),get(32n,al,i),get(32n,ah,Nat.sub(i,32n)),get(32n,bl,i),get(32n,bh,Nat.sub(i,32n)))))

def test64_or(+a: U64,+b: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(U64.or(a,b),i) == Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.or(a,b),i),get64(U64.or(a,b),i),Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)),
    test64(U64.or(a,b),i,bound),
    Equal.trans(Bool,get64(U64.or(a,b),i),Bool.or(get64(a,i),get64(b,i)),Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)),get64_or(a,b,i),
      Equal.trans(Bool,Bool.or(get64(a,i),get64(b,i)),Bool.or(U64.test_bit(a,i),get64(b,i)),Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)),
        Equal.cong(Bool,Bool,x => Bool.or(x,get64(b,i)),get64(a,i),U64.test_bit(a,i),Equal.sym(Bool,U64.test_bit(a,i),get64(a,i),test64(a,i,bound))),
        Equal.cong(Bool,Bool,x => Bool.or(U64.test_bit(a,i),x),get64(b,i),U64.test_bit(b,i),Equal.sym(Bool,U64.test_bit(b,i),get64(b,i),test64(b,i,bound))))))
