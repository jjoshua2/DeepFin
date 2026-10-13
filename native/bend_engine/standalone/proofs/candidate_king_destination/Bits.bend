import Base
import ./Algebra.bend as A
import ../decoder/Bits.bend as B
import ../filter_ray_exclusion/Bits.bend as Clear
import ../ordinary_king_frame/Algebra.bend as Parts
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as Selected
import ../u64/Words.bend as Old
import ../ray_clear_disjoint/Algebra.bend as D

def word_clear(+n: Nat,+a: Word(n),+k: Word(n),+i: Nat,
  +empty: {Word.and(n,a,k) == Word.zero(n) : Word(n)},set: {B.get(n,a,i) == True{} : Bool}) ->
  {B.get(n,k,i) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a k i:
        case WCon{False{},at} WCon{kh,kt} 0n:
          Old.false_true({kh == False{} : Bool},set)
        case WCon{True{},at} WCon{kh,kt} 0n:
          Equal.cong(Word(1n+p),Bool,w => Parts.head(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),empty)
        case WCon{ah,at} WCon{kh,kt} 1n+j:
          word_clear(p,at,kt,j,Equal.cong(Word(1n+p),Word(p),w => Parts.tail(p,w),Word.and(1n+p,a,k),Word.zero(1n+p),empty),set)

def choose_clear(+al: Word(32n),+ah: Word(32n),+kl: Word(32n),+kh: Word(32n),+i: Nat,low: Bool,
  el: {Word.and(32n,al,kl) == Word.zero(32n) : Word(32n)},
  eh: {Word.and(32n,ah,kh) == Word.zero(32n) : Word(32n)},
  set: {Bool.pick(Bool,low,B.get(32n,al,i),B.get(32n,ah,Nat.sub(i,32n))) == True{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,kl,i),B.get(32n,kh,Nat.sub(i,32n))) == False{} : Bool}:
  match low:
    case True{}: word_clear(32n,al,kl,i,el,set)
    case False{}: word_clear(32n,ah,kh,Nat.sub(i,32n),eh,set)

def get_clear(+a: U64,+k: U64,+i: Nat,+empty: {U64.and(a,k) == U64.zero() : U64},set: {B.get64(a,i) == True{} : Bool}) ->
  {B.get64(k,i) == False{} : Bool}:
  match a k:
    case U64{U32{al},U32{ah}} U64{U32{kl},U32{kh}}:
      choose_clear(al,ah,kl,kh,i,Nat.is_lt(i,32n),
        Equal.cong(U64,Word(32n),Parts.low,U64.and(a,k),U64.zero(),empty),
        Equal.cong(U64,Word(32n),Parts.high,U64.and(a,k),U64.zero(),empty),set)

def ctz_clear(+a: U64,+k: U64,empty: D.disjoint(a,k),+nonzero: {U64.is_zero(a) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(U64.ctz(a))),k):
  +i = U32.to_nat(U64.ctz(a))
  +bound = Ctz.actual_ctz_range(a,nonzero)
  Clear.bit_zero(k,i,bound,Equal.trans(Bool,U64.test_bit(k,i),B.get64(k,i),False{},B.test64(k,i,bound),
    get_clear(a,k,i,empty,Equal.trans(Bool,B.get64(a,i),U64.test_bit(a,i),True{},
      Equal.sym(Bool,U64.test_bit(a,i),B.get64(a,i),B.test64(a,i,bound)),Selected.actual_ctz_selects_bit(a,nonzero)))))
