import Base
import ./Words.bend as W
import ./Naturals.bend as N
import ../u64/Words.bend as Old
import ../successor/Step.bend as Step
import ../decoder/Bits.bend as B
import ../bitboard_ctz_structural/U64BitProbe.bend as Bit
import ../bitboard_ctz_structural/TestBitBridge.bend as Test
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz

def count(a: U64) -> Nat: Word.count(64n,U64.to_word(a))

def join_or(+n: Nat,+m: Nat,+a: Word(n),+b: Word(m),+c: Word(n),+d: Word(m)) ->
  {Word.join(n,m,Word.or(n,a,c),Word.or(m,b,d)) ==
    Word.or(Nat.add(n,m),Word.join(n,m,a,b),Word.join(n,m,c,d)) : Word(Nat.add(n,m))}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a c:
        case WCon{+h,+t} WCon{+k,+u}:
          Equal.cong(Word(Nat.add(p,m)),Word(Nat.add(1n+p,m)),x => WCon{Bool.or(h,k),x},
            Word.join(p,m,Word.or(p,t,u),Word.or(m,b,d)),
            Word.or(Nat.add(p,m),Word.join(p,m,t,b),Word.join(p,m,u,d)),join_or(p,m,t,b,u,d))

def join_not(+n: Nat,+m: Nat,+a: Word(n),+b: Word(m)) ->
  {Word.join(n,m,Word.not(n,a),Word.not(m,b)) == Word.not(Nat.add(n,m),Word.join(n,m,a,b)) : Word(Nat.add(n,m))}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a:
        case WCon{+h,+t}:
          Equal.cong(Word(Nat.add(p,m)),Word(Nat.add(1n+p,m)),x => WCon{Bool.not(h),x},
            Word.join(p,m,Word.not(p,t),Word.not(m,b)),Word.not(Nat.add(p,m),Word.join(p,m,t,b)),join_not(p,m,t,b))

def or_word(+a: U64,+b: U64) ->
  {U64.to_word(U64.or(a,b)) == Word.or(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}: join_or(32n,32n,al,ah,bl,bh)
def not_word(+a: U64) ->
  {U64.to_word(U64.not(a)) == Word.not(64n,U64.to_word(a)) : Word(64n)}:
  match a:
    case U64{U32{al},U32{ah}}: join_not(32n,32n,al,ah)
def and_not_word(+a: U64,+remove: U64) ->
  {U64.to_word(U64.and_not(a,remove)) == Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(remove))) : Word(64n)}:
  Equal.trans(Word(64n),U64.to_word(U64.and_not(a,remove)),
    Word.and(64n,U64.to_word(a),U64.to_word(U64.not(remove))),
    Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(remove))),Step.and_word(a,U64.not(remove)),
    Equal.cong(Word(64n),Word(64n),x => Word.and(64n,U64.to_word(a),x),U64.to_word(U64.not(remove)),Word.not(64n,U64.to_word(remove)),not_word(remove)))

def bit_self(+n: Nat,+i: Nat,bound: {Nat.is_lt(i,n) == True{} : Bool}) -> {B.get(n,B.bit(n,i),i) == True{} : Bool}:
  match n:
    case 0n: B.impossible({B.get(0n,B.bit(0n,i),i) == True{} : Bool},i,bound)
    case 1n+p:
      match i:
        case 0n: {==}
        case 1n+j: bit_self(p,j,bound)

def selector_exact(+n: Nat,+a: Word(n),+i: Nat) -> {B.get(n,a,i) == Ctz.word_bit(n,a,i) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a i:
        case WCon{h,t} 0n: {==}
        case WCon{h,t} 1n+j: selector_exact(p,t,j)
def singleton_exact(+n: Nat,+i: Nat) -> {B.bit(n,i) == Bit.word_singleton(n,i) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match i:
        case 0n: {==}
        case 1n+j:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},B.bit(p,j),Bit.word_singleton(p,j),singleton_exact(p,j))
def bit_word(+i: Nat,bound: {Nat.is_lt(i,64n) == True{} : Bool}) -> {U64.to_word(U64.bit(i)) == B.bit(64n,i) : Word(64n)}:
  Equal.trans(Word(64n),U64.to_word(U64.bit(i)),Bit.word_singleton(64n,i),B.bit(64n,i),
    Bit.bit_word_exact(i,bound),Equal.sym(Word(64n),B.bit(64n,i),Bit.word_singleton(64n,i),singleton_exact(64n,i)))

def test_word(+a: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(a,i) == B.get(64n,U64.to_word(a),i) : Bool}:
  Equal.trans(Bool,U64.test_bit(a,i),Ctz.word_bit(64n,U64.to_word(a),i),B.get(64n,U64.to_word(a),i),
    Test.actual_testbit_word(a,i,bound),Equal.sym(Bool,B.get(64n,U64.to_word(a),i),Ctz.word_bit(64n,U64.to_word(a),i),selector_exact(64n,U64.to_word(a),i)))
def owned_word(+a: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  owned: {U64.test_bit(a,i) == True{} : Bool}) -> {B.get(64n,U64.to_word(a),i) == True{} : Bool}:
  Equal.trans(Bool,B.get(64n,U64.to_word(a),i),U64.test_bit(a,i),True{},
    Equal.sym(Bool,U64.test_bit(a,i),B.get(64n,U64.to_word(a),i),test_word(a,i,bound)),owned)

def delete_le(+a: U64,+remove: U64) -> {Nat.is_le(count(U64.and_not(a,remove)),count(a)) == True{} : Bool}:
  %Equal.sym(Word(64n),U64.to_word(U64.and_not(a,remove)),Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(remove))),and_not_word(a,remove)) :
    {Nat.is_le(Word.count(64n,_),count(a)) == True{} : Bool}
  W.and_le(64n,U64.to_word(a),Word.not(64n,U64.to_word(remove)))

def paid_delete(+a: U64,+remove: U64,+src: Nat,+bound: {Nat.is_lt(src,64n) == True{} : Bool},
  owned: {U64.test_bit(a,src) == True{} : Bool},erased: {U64.test_bit(remove,src) == True{} : Bool}) ->
  {Nat.is_le(1n+count(U64.and_not(a,remove)),count(a)) == True{} : Bool}:
  %Equal.sym(Word(64n),U64.to_word(U64.and_not(a,remove)),Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(remove))),and_not_word(a,remove)) :
    {Nat.is_le(1n+Word.count(64n,_),count(a)) == True{} : Bool}
  W.owned_delete(64n,U64.to_word(a),U64.to_word(remove),src,owned_word(a,src,bound,owned),owned_word(remove,src,bound,erased))

def insert_le(+a: U64,+dst: Nat,bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {Nat.is_le(count(U64.or(a,U64.bit(dst))),1n+count(a)) == True{} : Bool}:
  %Equal.sym(Word(64n),U64.to_word(U64.or(a,U64.bit(dst))),Word.or(64n,U64.to_word(a),U64.to_word(U64.bit(dst))),or_word(a,U64.bit(dst))) :
    {Nat.is_le(Word.count(64n,_),1n+count(a)) == True{} : Bool}
  %Equal.sym(Word(64n),U64.to_word(U64.bit(dst)),B.bit(64n,dst),bit_word(dst,bound)) :
    {Nat.is_le(Word.count(64n,Word.or(64n,U64.to_word(a),_)),1n+count(a)) == True{} : Bool}
  W.insert_le(64n,U64.to_word(a),dst)

def replace_le(+a: U64,+remove: U64,+src: Nat,+dst: Nat,
  +source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},
  owned: {U64.test_bit(a,src) == True{} : Bool},erased: {U64.test_bit(remove,src) == True{} : Bool}) ->
  {Nat.is_le(count(U64.or(U64.and_not(a,remove),U64.bit(dst))),count(a)) == True{} : Bool}:
  N.transitive(count(U64.or(U64.and_not(a,remove),U64.bit(dst))),1n+count(U64.and_not(a,remove)),count(a),
    insert_le(U64.and_not(a,remove),dst,target_bound),paid_delete(a,remove,src,source_bound,owned,erased))
