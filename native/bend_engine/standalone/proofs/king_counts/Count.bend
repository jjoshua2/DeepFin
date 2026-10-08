import Base
import ./Words.bend as W
import ../color_counts/U64Count.bend as U
import ../decoder/Bits.bend as B
import ../candidate_king_destination/Algebra.bend as D
import ../ray/Algebra.bend as A
import ../../../legal_probe/Chess.bend as Chess

def relocate(+a: U64,+src: Nat,+dst: Nat,
  +sb: {Nat.is_lt(src,64n) == True{} : Bool},+db: {Nat.is_lt(dst,64n) == True{} : Bool},
  owned: {U64.test_bit(a,src) == True{} : Bool},fresh: {U64.test_bit(a,dst) == False{} : Bool}) ->
  {U.count(U64.or(U64.and_not(a,U64.bit(src)),U64.bit(dst))) == U.count(a) : Nat}:
  %Equal.sym(Word(64n),U64.to_word(U64.or(U64.and_not(a,U64.bit(src)),U64.bit(dst))),
    Word.or(64n,U64.to_word(U64.and_not(a,U64.bit(src))),U64.to_word(U64.bit(dst))),U.or_word(U64.and_not(a,U64.bit(src)),U64.bit(dst))) :
    {Word.count(64n,_) == U.count(a) : Nat}
  %Equal.sym(Word(64n),U64.to_word(U64.and_not(a,U64.bit(src))),
    Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(U64.bit(src)))),U.and_not_word(a,U64.bit(src))) :
    {Word.count(64n,Word.or(64n,_,U64.to_word(U64.bit(dst)))) == U.count(a) : Nat}
  %Equal.sym(Word(64n),U64.to_word(U64.bit(src)),B.bit(64n,src),U.bit_word(src,sb)) :
    {Word.count(64n,Word.or(64n,Word.and(64n,U64.to_word(a),Word.not(64n,_)),U64.to_word(U64.bit(dst)))) == U.count(a) : Nat}
  %Equal.sym(Word(64n),U64.to_word(U64.bit(dst)),B.bit(64n,dst),U.bit_word(dst,db)) :
    {Word.count(64n,Word.or(64n,Word.and(64n,U64.to_word(a),Word.not(64n,B.bit(64n,src))),_)) == U.count(a) : Nat}
  W.relocate(64n,U64.to_word(a),src,dst,db,U.owned_word(a,src,sb,owned),
    Equal.trans(Bool,B.get(64n,U64.to_word(a),dst),U64.test_bit(a,dst),False{},Equal.sym(Bool,U64.test_bit(a,dst),B.get(64n,U64.to_word(a),dst),U.test_word(a,dst,db)),fresh))

def erase_empty(+a: U64,+src: Nat,+sb: {Nat.is_lt(src,64n) == True{} : Bool},fresh: {U64.test_bit(a,src) == False{} : Bool}) ->
  {U.count(U64.and_not(a,U64.bit(src))) == U.count(a) : Nat}:
  %Equal.sym(Word(64n),U64.to_word(U64.and_not(a,U64.bit(src))),
    Word.and(64n,U64.to_word(a),Word.not(64n,U64.to_word(U64.bit(src)))),U.and_not_word(a,U64.bit(src))) :
    {Word.count(64n,_) == U.count(a) : Nat}
  %Equal.sym(Word(64n),U64.to_word(U64.bit(src)),B.bit(64n,src),U.bit_word(src,sb)) :
    {Word.count(64n,Word.and(64n,U64.to_word(a),Word.not(64n,_))) == U.count(a) : Nat}
  W.erase_empty(64n,U64.to_word(a),src,
    Equal.trans(Bool,B.get(64n,U64.to_word(a),src),U64.test_bit(a,src),False{},Equal.sym(Bool,U64.test_bit(a,src),B.get(64n,U64.to_word(a),src),U.test_word(a,src,sb)),fresh))

def update(+a: U64,+src: Nat,+dst: Nat,put: Bool,
  +sb: {Nat.is_lt(src,64n) == True{} : Bool},db: {Nat.is_lt(dst,64n) == True{} : Bool},
  same: {U64.test_bit(a,src) == put : Bool},fresh: {U64.test_bit(a,dst) == False{} : Bool}) ->
  {U.count(Chess.update_piece(a,U64.bit(src),U64.bit(dst),put)) == U.count(a) : Nat}:
  match put:
    case True{}: relocate(a,src,dst,sb,db,same,fresh)
    case False{}:
      %Equal.sym(U64,U64.or(U64.and_not(a,U64.bit(src)),U64.zero()),U64.and_not(a,U64.bit(src)),A.zero(U64.and_not(a,U64.bit(src)))) :
        {U.count(_) == U.count(a) : Nat}
      erase_empty(a,src,sb,same)

def clear_test(+a: U64,+i: Nat,clear: {U64.and(U64.bit(i),a) == U64.zero() : U64}) ->
  {U64.test_bit(a,i) == False{} : Bool}:
  Equal.cong(U64,Bool,x => Bool.not(U64.is_eq(x,U64.zero())),U64.and(a,U64.bit(i)),U64.zero(),
    Equal.trans(U64,U64.and(a,U64.bit(i)),U64.and(U64.bit(i),a),U64.zero(),D.and_commute(a,U64.bit(i)),clear))
