# Structural bit reflection, including saturated off-board bit indices.
import Base
import ./Spec.bend as S
import ../attack_geometry/Geometry.bend as G
import ../decoder/Bits.bend as B
import ../layout/Domain.bend as D
import ../ray/Algebra.bend as A

type Predicate is Data:
  Single{at: Nat}
  Many{xs: List<&2,Nat>}
  Union{a: Predicate,b: Predicate}

def eval(p: Predicate,+i: Nat) -> Bool:
  match p:
    case Single{h}: Nat.is_eq(i,h)
    case Many{xs}: S.contains(i,xs)
    case Union{a,b}: Bool.or(eval(a,i),eval(b,i))

def word(n: Nat,+start: Nat,+p: Predicate) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+nxt: WCon{eval(p,start),word(nxt,1n+start,p)}

def packed(+xs: List<&2,Nat>) -> U64:
  U64{U32{word(32n,0n,Many{xs})},U32{word(32n,32n,Many{xs})}}

def singleton(+h: Nat) -> U64:
  U64{U32{word(32n,0n,Single{h})},U32{word(32n,32n,Single{h})}}

def add_succ(+a: Nat,+b: Nat) -> {Nat.add(a,1n+b) == 1n+Nat.add(a,b) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,x => 1n+x,Nat.add(p,1n+b),1n+Nat.add(p,b),add_succ(p,b))

def add_zero(+a: Nat) -> {a == Nat.add(a,0n) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p: Equal.cong(Nat,Nat,x => 1n+x,p,Nat.add(p,0n),add_zero(p))

def word_get(+n: Nat,+start: Nat,+p: Predicate,+i: Nat,bound: {Nat.is_lt(i,n) == True{} : Bool}) ->
  {B.get(n,word(n,start,p),i) == eval(p,Nat.add(start,i)) : Bool}:
  match n:
    case 0n: D.impossible({False{} == eval(p,Nat.add(start,i)) : Bool},i,bound)
    case 1n+nxt:
      match i:
        case 0n:
          Equal.cong(Nat,Bool,x => eval(p,x),start,Nat.add(start,0n),add_zero(start))
        case 1n+j:
          Equal.trans(Bool,B.get(nxt,word(nxt,1n+start,p),j),eval(p,1n+Nat.add(start,j)),eval(p,Nat.add(start,1n+j)),
            word_get(nxt,1n+start,p,j,bound),
            Equal.cong(Nat,Bool,x => eval(p,x),1n+Nat.add(start,j),Nat.add(start,1n+j),Equal.sym(Nat,Nat.add(start,1n+j),1n+Nat.add(start,j),add_succ(start,j))))

def word_or(+n: Nat,+start: Nat,+f: Predicate,+g: Predicate) ->
  {Word.or(n,word(n,start,f),word(n,start,g)) == word(n,start,Union{f,g}) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+nxt:
      Equal.cong(Word(nxt),Word(1n+nxt),tail => WCon{Bool.or(eval(f,start),eval(g,start)),tail},
        Word.or(nxt,word(nxt,1n+start,f),word(nxt,1n+start,g)),word(nxt,1n+start,Union{f,g}),word_or(nxt,1n+start,f,g))

def word_cons(+n: Nat,+start: Nat,+h: Nat,+xs: List<&2,Nat>) ->
  {word(n,start,Union{Single{h},Many{xs}}) == word(n,start,Many{h <> xs}) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Word(p),Word(1n+p),tail => WCon{S.contains(start,h <> xs),tail},
        word(p,1n+start,Union{Single{h},Many{xs}}),word(p,1n+start,Many{h <> xs}),word_cons(p,1n+start,h,xs))

def or_cons(+n: Nat,+start: Nat,+h: Nat,+xs: List<&2,Nat>) ->
  {Word.or(n,word(n,start,Single{h}),word(n,start,Many{xs})) == word(n,start,Many{h <> xs}) : Word(n)}:
  Equal.trans(Word(n),Word.or(n,word(n,start,Single{h}),word(n,start,Many{xs})),word(n,start,Union{Single{h},Many{xs}}),word(n,start,Many{h <> xs}),
    word_or(n,start,Single{h},Many{xs}),word_cons(n,start,h,xs))

def bit_encoding(+h: Nat) -> {U64.bit(h) == singleton(h) : U64}:
  match h:
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
    case 64n+p:
      match p:
        case 0n: {==}
        case 1n+q: {==}

def head(+h: Nat,+tail: List<&2,Nat>) ->
  {U64.or(U64.bit(h),packed(tail)) == packed(h <> tail) : U64}:
  Equal.trans(U64,U64.or(U64.bit(h),packed(tail)),U64.or(singleton(h),packed(tail)),packed(h <> tail),
    Equal.cong(U64,U64,b => U64.or(b,packed(tail)),U64.bit(h),singleton(h),bit_encoding(h)),
    Equal.trans(U64,U64.or(singleton(h),packed(tail)),
      U64{U32{word(32n,0n,Many{h <> tail})},U32{Word.or(32n,word(32n,32n,Single{h}),word(32n,32n,Many{tail}))}},
      packed(h <> tail),
      Equal.cong(Word(32n),U64,lo => U64{U32{lo},U32{Word.or(32n,word(32n,32n,Single{h}),word(32n,32n,Many{tail}))}},
        Word.or(32n,word(32n,0n,Single{h}),word(32n,0n,Many{tail})),word(32n,0n,Many{h <> tail}),or_cons(32n,0n,h,tail)),
      Equal.cong(Word(32n),U64,hi => U64{U32{word(32n,0n,Many{h <> tail})},U32{hi}},
        Word.or(32n,word(32n,32n,Single{h}),word(32n,32n,Many{tail})),word(32n,32n,Many{h <> tail}),or_cons(32n,32n,h,tail))))

def accumulation(+xs: List<&2,Nat>,+acc: U64) ->
  {G.fold(xs,acc) == U64.or(acc,packed(xs)) : U64}:
  match xs:
    case Nil{}: Equal.sym(U64,U64.or(acc,U64.zero()),acc,A.zero(acc))
    case +h <> +tail:
      Equal.trans(U64,G.fold(h <> tail,acc),U64.or(U64.or(acc,U64.bit(h)),packed(tail)),U64.or(acc,packed(h <> tail)),
        accumulation(tail,U64.or(acc,U64.bit(h))),
        Equal.trans(U64,U64.or(U64.or(acc,U64.bit(h)),packed(tail)),U64.or(acc,U64.or(U64.bit(h),packed(tail))),U64.or(acc,packed(h <> tail)),
          A.or_assoc(acc,U64.bit(h),packed(tail)),
          Equal.cong(U64,U64,b => U64.or(acc,b),U64.or(U64.bit(h),packed(tail)),packed(h <> tail),head(h,tail))))

def fold(+xs: List<&2,Nat>) -> {G.fold(xs,U64.zero()) == packed(xs) : U64}:
  Equal.trans(U64,G.fold(xs,U64.zero()),U64.or(U64.zero(),packed(xs)),packed(xs),accumulation(xs,U64.zero()),A.left_zero(packed(xs)))

def pair(+a: Nat,+b: Nat) -> {G.two(a <> b <> Nil{}) == G.fold(a <> b <> Nil{},U64.zero()) : U64}:
  Equal.sym(U64,G.fold(a <> b <> Nil{},U64.zero()),G.two(a <> b <> Nil{}),
    Equal.cong(U64,U64,v => U64.or(v,U64.bit(b)),U64.or(U64.zero(),U64.bit(a)),U64.bit(a),A.left_zero(U64.bit(a))))

def combine(+kind: G.Leaper,+src: Nat) -> {G.mask(kind,src) == G.fold(G.targets(kind,src),U64.zero()) : U64}:
  match kind:
    case G.Knight{}: {==}
    case G.King{}: {==}
    case G.WhitePawn{}: pair(G.target(src,1n,True{},1n,True{}),G.target(src,1n,False{},1n,True{}))
    case G.BlackPawn{}: pair(G.target(src,1n,True{},1n,False{}),G.target(src,1n,False{},1n,False{}))

def mask_packed(+kind: G.Leaper,+src: Nat) -> {G.mask(kind,src) == packed(G.targets(kind,src)) : U64}:
  Equal.trans(U64,G.mask(kind,src),G.fold(G.targets(kind,src),U64.zero()),packed(G.targets(kind,src)),combine(kind,src),fold(G.targets(kind,src)))

def get(+xs: List<&2,Nat>,+i: Nat,bound: {Nat.is_lt(i,64n) == True{} : Bool}) -> {B.get64(packed(xs),i) == S.contains(i,xs) : Bool}:
  match i:
    case 0n: word_get(32n,0n,Many{xs},0n,{==})
    case 1n: word_get(32n,0n,Many{xs},1n,{==})
    case 2n: word_get(32n,0n,Many{xs},2n,{==})
    case 3n: word_get(32n,0n,Many{xs},3n,{==})
    case 4n: word_get(32n,0n,Many{xs},4n,{==})
    case 5n: word_get(32n,0n,Many{xs},5n,{==})
    case 6n: word_get(32n,0n,Many{xs},6n,{==})
    case 7n: word_get(32n,0n,Many{xs},7n,{==})
    case 8n: word_get(32n,0n,Many{xs},8n,{==})
    case 9n: word_get(32n,0n,Many{xs},9n,{==})
    case 10n: word_get(32n,0n,Many{xs},10n,{==})
    case 11n: word_get(32n,0n,Many{xs},11n,{==})
    case 12n: word_get(32n,0n,Many{xs},12n,{==})
    case 13n: word_get(32n,0n,Many{xs},13n,{==})
    case 14n: word_get(32n,0n,Many{xs},14n,{==})
    case 15n: word_get(32n,0n,Many{xs},15n,{==})
    case 16n: word_get(32n,0n,Many{xs},16n,{==})
    case 17n: word_get(32n,0n,Many{xs},17n,{==})
    case 18n: word_get(32n,0n,Many{xs},18n,{==})
    case 19n: word_get(32n,0n,Many{xs},19n,{==})
    case 20n: word_get(32n,0n,Many{xs},20n,{==})
    case 21n: word_get(32n,0n,Many{xs},21n,{==})
    case 22n: word_get(32n,0n,Many{xs},22n,{==})
    case 23n: word_get(32n,0n,Many{xs},23n,{==})
    case 24n: word_get(32n,0n,Many{xs},24n,{==})
    case 25n: word_get(32n,0n,Many{xs},25n,{==})
    case 26n: word_get(32n,0n,Many{xs},26n,{==})
    case 27n: word_get(32n,0n,Many{xs},27n,{==})
    case 28n: word_get(32n,0n,Many{xs},28n,{==})
    case 29n: word_get(32n,0n,Many{xs},29n,{==})
    case 30n: word_get(32n,0n,Many{xs},30n,{==})
    case 31n: word_get(32n,0n,Many{xs},31n,{==})
    case 32n+j:
      match j:
        case 0n: word_get(32n,32n,Many{xs},0n,bound)
        case 1n+q: word_get(32n,32n,Many{xs},1n+q,bound)

def test(+kind: G.Leaper,+src: Nat,+dst: Nat,+bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(G.mask(kind,src),dst) == S.contains(dst,G.targets(kind,src)) : Bool}:
  Equal.trans(Bool,U64.test_bit(G.mask(kind,src),dst),U64.test_bit(packed(G.targets(kind,src)),dst),S.contains(dst,G.targets(kind,src)),
    Equal.cong(U64,Bool,m => U64.test_bit(m,dst),G.mask(kind,src),packed(G.targets(kind,src)),mask_packed(kind,src)),
    Equal.trans(Bool,U64.test_bit(packed(G.targets(kind,src)),dst),B.get64(packed(G.targets(kind,src)),dst),S.contains(dst,G.targets(kind,src)),
      B.test64(packed(G.targets(kind,src)),dst,bound),get(G.targets(kind,src),dst,bound)))
