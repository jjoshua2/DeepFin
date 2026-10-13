import Base
import ../color_counts/U64Count.bend as U
import ../decoder/Bits.bend as B
import ../successor/Step.bend as Step

def get_and(+n: Nat,+a: Word(n),+b: Word(n),+i: Nat) ->
  {B.get(n,Word.and(n,a,b),i) == Bool.and(B.get(n,a,i),B.get(n,b,i)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+k:
      match a b i:
        case WCon{ah,at} WCon{bh,bt} 0n: {==}
        case WCon{ah,at} WCon{bh,bt} 1n+j: get_and(k,at,bt,j)

def owned(+a: U64,+b: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  left: {U64.test_bit(a,i) == True{} : Bool},right: {U64.test_bit(b,i) == True{} : Bool}) ->
  {U64.test_bit(U64.and(a,b),i) == True{} : Bool}:
  %Equal.sym(Bool,U64.test_bit(U64.and(a,b),i),B.get(64n,U64.to_word(U64.and(a,b)),i),U.test_word(U64.and(a,b),i,bound)) :
    {_ == True{} : Bool}
  %Equal.sym(Word(64n),U64.to_word(U64.and(a,b)),Word.and(64n,U64.to_word(a),U64.to_word(b)),Step.and_word(a,b)) :
    {B.get(64n,_,i) == True{} : Bool}
  %Equal.sym(Bool,B.get(64n,Word.and(64n,U64.to_word(a),U64.to_word(b)),i),
    Bool.and(B.get(64n,U64.to_word(a),i),B.get(64n,U64.to_word(b),i)),get_and(64n,U64.to_word(a),U64.to_word(b),i)) :
    {_ == True{} : Bool}
  %Equal.sym(Bool,B.get(64n,U64.to_word(a),i),True{},U.owned_word(a,i,bound,left)) :
    {Bool.and(_,B.get(64n,U64.to_word(b),i)) == True{} : Bool}
  %Equal.sym(Bool,B.get(64n,U64.to_word(b),i),True{},U.owned_word(b,i,bound,right)) :
    {Bool.and(True{},_) == True{} : Bool}
  {==}
