import Base
import ../color_counts/U64Count.bend as U
import ../decoder/Bits.bend as B
import ../successor/Step.bend as Step
import ../pawn_counts/Intersection.bend as I

def test_and(+a: U64,+b: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool}) ->
  {U64.test_bit(U64.and(a,b),i) == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}:
  %Equal.sym(Bool,U64.test_bit(U64.and(a,b),i),B.get(64n,U64.to_word(U64.and(a,b)),i),U.test_word(U64.and(a,b),i,bound)) :
    {_ == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}
  %Equal.sym(Word(64n),U64.to_word(U64.and(a,b)),Word.and(64n,U64.to_word(a),U64.to_word(b)),Step.and_word(a,b)) :
    {B.get(64n,_,i) == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}
  %Equal.sym(Bool,B.get(64n,Word.and(64n,U64.to_word(a),U64.to_word(b)),i),Bool.and(B.get(64n,U64.to_word(a),i),B.get(64n,U64.to_word(b),i)),I.get_and(64n,U64.to_word(a),U64.to_word(b),i)) :
    {_ == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}
  %U.test_word(a,i,bound) : {Bool.and(_,B.get(64n,U64.to_word(b),i)) == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}
  %U.test_word(b,i,bound) : {Bool.and(U64.test_bit(a,i),_) == Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}
  {==}
def false_left(+b: Bool) -> {Bool.and(False{},b) == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}
def fresh(+a: U64,+b: U64,+i: Nat,bound: {Nat.is_lt(i,64n) == True{} : Bool},clear: {U64.test_bit(a,i) == False{} : Bool}) ->
  {U64.test_bit(U64.and(a,b),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and(a,b),i),Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)),False{},test_and(a,b,i,bound),
    Equal.trans(Bool,Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)),Bool.and(False{},U64.test_bit(b,i)),False{},
      Equal.cong(Bool,Bool,x => Bool.and(x,U64.test_bit(b,i)),U64.test_bit(a,i),False{},clear),false_left(U64.test_bit(b,i))))
def source(+a: U64,+b: U64,+i: Nat,+put: Bool,+side: Bool,bound: {Nat.is_lt(i,64n) == True{} : Bool},
  same_a: {U64.test_bit(a,i) == put : Bool},same_b: {U64.test_bit(b,i) == side : Bool}) ->
  {U64.test_bit(U64.and(a,b),i) == Bool.and(put,side) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and(a,b),i),Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)),Bool.and(put,side),test_and(a,b,i,bound),
    Equal.trans(Bool,Bool.and(U64.test_bit(a,i),U64.test_bit(b,i)),Bool.and(put,U64.test_bit(b,i)),Bool.and(put,side),
      Equal.cong(Bool,Bool,x => Bool.and(x,U64.test_bit(b,i)),U64.test_bit(a,i),put,same_a),
      Equal.cong(Bool,Bool,x => Bool.and(put,x),U64.test_bit(b,i),side,same_b)))
