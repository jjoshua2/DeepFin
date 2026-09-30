# Pointwise inclusion of owned king bits in own AND (rays OR kings).
# Structural Word proof; no enumeration of boards or ray masks.
import Base
import ../decoder/Bits.bend as B

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

def word(+n: Nat,+own: Word(n),+king: Word(n),+ray: Word(n),+i: Nat,
  e: {B.get(n,Word.and(n,king,own),i) == True{} : Bool}) ->
  {B.get(n,Word.and(n,own,Word.or(n,ray,king)),i) == True{} : Bool}:
  match n:
    case 0n: e
    case 1n+p:
      match own king ray i:
        case WCon{oh,ot} WCon{kh,kt} WCon{rh,rt} 0n: head(oh,kh,rh,e)
        case WCon{oh,ot} WCon{kh,kt} WCon{rh,rt} 1n+j: word(p,ot,kt,rt,j,e)

def choose(+ol: Word(32n),+oh: Word(32n),+kl: Word(32n),+kh: Word(32n),+rl: Word(32n),+rh: Word(32n),+i: Nat,low: Bool,
  e: {Bool.pick(Bool,low,B.get(32n,Word.and(32n,kl,ol),i),B.get(32n,Word.and(32n,kh,oh),Nat.sub(i,32n))) == True{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,Word.and(32n,ol,Word.or(32n,rl,kl)),i),B.get(32n,Word.and(32n,oh,Word.or(32n,rh,kh)),Nat.sub(i,32n))) == True{} : Bool}:
  match low:
    case True{}: word(32n,ol,kl,rl,i,e)
    case False{}: word(32n,oh,kh,rh,Nat.sub(i,32n),e)

def get64(+own: U64,+king: U64,+ray: U64,+i: Nat,
  e: {B.get64(U64.and(king,own),i) == True{} : Bool}) ->
  {B.get64(U64.and(own,U64.or(ray,king)),i) == True{} : Bool}:
  match own king ray:
    case U64{U32{ol},U32{oh}} U64{U32{kl},U32{kh}} U64{U32{rl},U32{rh}}:
      choose(ol,oh,kl,kh,rl,rh,i,Nat.is_lt(i,32n),e)

def sensitive(+own: U64,+king: U64,+ray: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  e: {U64.test_bit(U64.and(king,own),i) == True{} : Bool}) ->
  {U64.test_bit(U64.and(own,U64.or(ray,king)),i) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and(own,U64.or(ray,king)),i),B.get64(U64.and(own,U64.or(ray,king)),i),True{},
    B.test64(U64.and(own,U64.or(ray,king)),i,bound),
    get64(own,king,ray,i,Equal.trans(Bool,B.get64(U64.and(king,own),i),U64.test_bit(U64.and(king,own),i),True{},
      Equal.sym(Bool,U64.test_bit(U64.and(king,own),i),B.get64(U64.and(king,own),i),B.test64(U64.and(king,own),i,bound)),e)))
