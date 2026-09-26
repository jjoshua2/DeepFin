# Structural containment at a queried bit; no attack-table oracle or board validity.
import Base
import ../decoder/Bits.bend as Bits
import ../u64/Words.bend as Old

def head(+own: Bool,+king: Bool,+ray: Bool,e: {Bool.and(king,own) == True{} : Bool}) ->
  {Bool.and(own,Bool.or(ray,king)) == True{} : Bool}:
  match own king ray:
    case False{} False{} False{}: e
    case False{} False{} True{}: e
    case False{} True{} False{}: e
    case False{} True{} True{}: e
    case True{} False{} False{}: e
    case True{} False{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} True{} True{}: {==}

def word(+n: Nat,+own: Word(n),+king: Word(n),+rays: Word(n),+i: Nat,
  e: {Bits.get(n,Word.and(n,king,own),i) == True{} : Bool}) ->
  {Bits.get(n,Word.and(n,own,Word.or(n,rays,king)),i) == True{} : Bool}:
  match n:
    case 0n: e
    case 1n+p:
      match own king rays i:
        case WCon{oh,ot} WCon{kh,kt} WCon{rh,rt} 0n: head(oh,kh,rh,e)
        case WCon{oh,ot} WCon{kh,kt} WCon{rh,rt} 1n+j: word(p,ot,kt,rt,j,e)

def choose(+c: Bool,+ol: Word(32n),+oh: Word(32n),+kl: Word(32n),+kh: Word(32n),+rl: Word(32n),+rh: Word(32n),+i: Nat,+j: Nat,
  e: {Bool.pick(Bool,c,Bits.get(32n,Word.and(32n,kl,ol),i),Bits.get(32n,Word.and(32n,kh,oh),j)) == True{} : Bool}) ->
  {Bool.pick(Bool,c,Bits.get(32n,Word.and(32n,ol,Word.or(32n,rl,kl)),i),Bits.get(32n,Word.and(32n,oh,Word.or(32n,rh,kh)),j)) == True{} : Bool}:
  match c:
    case True{}: word(32n,ol,kl,rl,i,e)
    case False{}: word(32n,oh,kh,rh,j,e)

def get64(+own: U64,+king: U64,+rays: U64,+i: Nat,
  e: {Bits.get64(U64.and(king,own),i) == True{} : Bool}) ->
  {Bits.get64(U64.and(own,U64.or(rays,king)),i) == True{} : Bool}:
  match own king rays:
    case U64{U32{ol},U32{oh}} U64{U32{kl},U32{kh}} U64{U32{rl},U32{rh}}:
      choose(Nat.is_lt(i,32n),ol,oh,kl,kh,rl,rh,i,Nat.sub(i,32n),e)

def actual_bit(+own: U64,+king: U64,+rays: U64,+i: Nat,
  +bound: {Nat.is_lt(i,64n) == True{} : Bool},
  e: {U64.test_bit(U64.and(king,own),i) == True{} : Bool}) ->
  {U64.test_bit(U64.and(own,U64.or(rays,king)),i) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and(own,U64.or(rays,king)),i),Bits.get64(U64.and(own,U64.or(rays,king)),i),True{},
    Bits.test64(U64.and(own,U64.or(rays,king)),i,bound),
    get64(own,king,rays,i,
      Equal.trans(Bool,Bits.get64(U64.and(king,own),i),U64.test_bit(U64.and(king,own),i),True{},
        Equal.sym(Bool,U64.test_bit(U64.and(king,own),i),Bits.get64(U64.and(king,own),i),Bits.test64(U64.and(king,own),i,bound)),e)))
