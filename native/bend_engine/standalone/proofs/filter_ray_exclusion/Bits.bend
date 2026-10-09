import Base
import ../decoder/Bits.bend as B
import ../u64/Words.bend as Old
import ../ray_clear_disjoint/Algebra.bend as D

def or_false_left(+a: Bool,+b: Bool,e: {Bool.or(a,b) == False{} : Bool}) -> {a == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: Old.false_true({True{} == False{} : Bool},Equal.sym(Bool,True{},False{},e))

def or_false_right(+a: Bool,+b: Bool,e: {Bool.or(a,b) == False{} : Bool}) -> {b == False{} : Bool}:
  match a:
    case False{}: e
    case True{}: Old.false_true({b == False{} : Bool},Equal.sym(Bool,True{},False{},e))

def head(+own: Bool,+ray: Bool,+king: Bool,
  owned: {own == True{} : Bool},bypass: {Bool.and(own,Bool.or(ray,king)) == False{} : Bool}) ->
  {ray == False{} : Bool}:
  or_false_left(ray,king,Equal.trans(Bool,Bool.or(ray,king),Bool.and(own,Bool.or(ray,king)),False{},
    Equal.cong(Bool,Bool,o => Bool.and(o,Bool.or(ray,king)),True{},own,Equal.sym(Bool,own,True{},owned)),bypass))

def word_exclusion(+n: Nat,+own: Word(n),+ray: Word(n),+king: Word(n),+i: Nat,
  owned: {B.get(n,own,i) == True{} : Bool},
  bypass: {B.get(n,Word.and(n,own,Word.or(n,ray,king)),i) == False{} : Bool}) ->
  {B.get(n,ray,i) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match own ray king i:
        case WCon{oh,ot} WCon{rh,rt} WCon{kh,kt} 0n: head(oh,rh,kh,owned,bypass)
        case WCon{oh,ot} WCon{rh,rt} WCon{kh,kt} 1n+j: word_exclusion(p,ot,rt,kt,j,owned,bypass)

def choose_exclusion(+ol: Word(32n),+oh: Word(32n),+rl: Word(32n),+rh: Word(32n),+kl: Word(32n),+kh: Word(32n),+i: Nat,+low: Bool,
  owned: {Bool.pick(Bool,low,B.get(32n,ol,i),B.get(32n,oh,Nat.sub(i,32n))) == True{} : Bool},
  bypass: {Bool.pick(Bool,low,B.get(32n,Word.and(32n,ol,Word.or(32n,rl,kl)),i),
    B.get(32n,Word.and(32n,oh,Word.or(32n,rh,kh)),Nat.sub(i,32n))) == False{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,rl,i),B.get(32n,rh,Nat.sub(i,32n))) == False{} : Bool}:
  match low:
    case True{}: word_exclusion(32n,ol,rl,kl,i,owned,bypass)
    case False{}: word_exclusion(32n,oh,rh,kh,Nat.sub(i,32n),owned,bypass)

def get64_exclusion(+own: U64,+ray: U64,+king: U64,+i: Nat,
  owned: {B.get64(own,i) == True{} : Bool},
  bypass: {B.get64(U64.and(own,U64.or(ray,king)),i) == False{} : Bool}) ->
  {B.get64(ray,i) == False{} : Bool}:
  match own ray king:
    case U64{U32{ol},U32{oh}} U64{U32{rl},U32{rh}} U64{U32{kl},U32{kh}}:
      choose_exclusion(ol,oh,rl,rh,kl,kh,i,Nat.is_lt(i,32n),owned,bypass)

def test_exclusion(+own: U64,+ray: U64,+king: U64,+i: Nat,
  +bound: {Nat.is_lt(i,64n) == True{} : Bool},
  owned: {U64.test_bit(own,i) == True{} : Bool},
  bypass: {U64.test_bit(U64.and(own,U64.or(ray,king)),i) == False{} : Bool}) ->
  {U64.test_bit(ray,i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(ray,i),B.get64(ray,i),False{},B.test64(ray,i,bound),
    get64_exclusion(own,ray,king,i,
      Equal.trans(Bool,B.get64(own,i),U64.test_bit(own,i),True{},
        Equal.sym(Bool,U64.test_bit(own,i),B.get64(own,i),B.test64(own,i,bound)),owned),
      Equal.trans(Bool,B.get64(U64.and(own,U64.or(ray,king)),i),U64.test_bit(U64.and(own,U64.or(ray,king)),i),False{},
        Equal.sym(Bool,U64.test_bit(U64.and(own,U64.or(ray,king)),i),B.get64(U64.and(own,U64.or(ray,king)),i),
          B.test64(U64.and(own,U64.or(ray,king)),i,bound)),bypass)))

# The source bit is the FIRST AND operand, matching D.disjoint.
def word_bit_zero(+n: Nat,+w: Word(n),+i: Nat,e: {B.get(n,w,i) == False{} : Bool}) ->
  {Word.and(n,B.bit(n,i),w) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w i:
        case WCon{+h,+t} 0n:
          Equal.trans(Word(1n+p),WCon{h,Word.and(p,Word.zero(p),t)},WCon{False{},Word.and(p,Word.zero(p),t)},Word.zero(1n+p),
            Equal.cong(Bool,Word(1n+p),bit_head_value => WCon{bit_head_value,Word.and(p,Word.zero(p),t)},h,False{},e),
            Equal.cong(Word(p),Word(1n+p),bit_tail_value => WCon{False{},bit_tail_value},Word.and(p,Word.zero(p),t),Word.zero(p),D.word_left_zero(p,t)))
        case WCon{h,t} 1n+j:
          Equal.cong(Word(p),Word(1n+p),bit_tail_value => WCon{False{},bit_tail_value},
            Word.and(p,B.bit(p,j),t),Word.zero(p),word_bit_zero(p,t,j,e))

def low_zero(+lo: Word(32n),+hi: Word(32n),+i: Nat,e: {B.get(32n,lo,i) == False{} : Bool}) ->
  {U64.and(U64{U32{B.bit(32n,i)},0},U64{U32{lo},U32{hi}}) == U64.zero() : U64}:
  Equal.trans(U64,U64{U32{Word.and(32n,B.bit(32n,i),lo)},U32{Word.and(32n,Word.zero(32n),hi)}},
    U64{0,U32{Word.and(32n,Word.zero(32n),hi)}},U64.zero(),
    Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.zero(32n),hi)}},
      Word.and(32n,B.bit(32n,i),lo),Word.zero(32n),word_bit_zero(32n,lo,i,e)),
    Equal.cong(Word(32n),U64,h => U64{0,U32{h}},Word.and(32n,Word.zero(32n),hi),Word.zero(32n),D.word_left_zero(32n,hi)))

def high_zero(+lo: Word(32n),+hi: Word(32n),+i: Nat,e: {B.get(32n,hi,i) == False{} : Bool}) ->
  {U64.and(U64{0,U32{B.bit(32n,i)}},U64{U32{lo},U32{hi}}) == U64.zero() : U64}:
  Equal.trans(U64,U64{U32{Word.and(32n,Word.zero(32n),lo)},U32{Word.and(32n,B.bit(32n,i),hi)}},
    U64{0,U32{Word.and(32n,B.bit(32n,i),hi)}},U64.zero(),
    Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,B.bit(32n,i),hi)}},
      Word.and(32n,Word.zero(32n),lo),Word.zero(32n),D.word_left_zero(32n,lo)),
    Equal.cong(Word(32n),U64,h => U64{0,U32{h}},Word.and(32n,B.bit(32n,i),hi),Word.zero(32n),word_bit_zero(32n,hi,i,e)))

def choose_zero(+lo: Word(32n),+hi: Word(32n),+i: Nat,+low: Bool,
  e: {Bool.pick(Bool,low,B.get(32n,lo,i),B.get(32n,hi,Nat.sub(i,32n))) == False{} : Bool}) ->
  {U64.and(Bool.pick(U64,low,U64{U32{B.bit(32n,i)},0},U64{0,U32{B.bit(32n,Nat.sub(i,32n))}}),
    U64{U32{lo},U32{hi}}) == U64.zero() : U64}:
  match low:
    case True{}: low_zero(lo,hi,i,e)
    case False{}: high_zero(lo,hi,Nat.sub(i,32n),e)

def get64_zero(+a: U64,+i: Nat,e: {B.get64(a,i) == False{} : Bool}) ->
  {U64.and(B.mask64(i),a) == U64.zero() : U64}:
  match a:
    case U64{U32{lo},U32{hi}}: choose_zero(lo,hi,i,Nat.is_lt(i,32n),e)

def bit_zero(+a: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  e: {U64.test_bit(a,i) == False{} : Bool}) -> D.disjoint(U64.bit(i),a):
  Equal.trans(U64,U64.and(U64.bit(i),a),U64.and(B.mask64(i),a),U64.zero(),
    Equal.cong(U64,U64,chosen_mask => U64.and(chosen_mask,a),U64.bit(i),B.mask64(i),B.bit_encoding(i,bound)),
    get64_zero(a,i,Equal.trans(Bool,B.get64(a,i),U64.test_bit(a,i),False{},
      Equal.sym(Bool,U64.test_bit(a,i),B.get64(a,i),B.test64(a,i,bound)),e)))
