# Analytical singleton observations and rank-restricted coherent EP geometry.
import Base
import ./Right.bend as R
import ../layout/Domain.bend as Domain
import ../u64/Words.bend as Words
import ../bitboard_membership/AllEmittedSound.bend as All
import ../child_ep_coherence/Post.bend as Post
import ../decoder/Bits.bend as Bits
import ../color_counts/U64Count.bend as Word64
import ../bitboard_membership/IndexEquality.bend as Index
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

# Small source-derived helpers copied from the checked Geometry module.
def rank_ok(+d: U32,white: Bool) -> Bool:
  U32.is_eq(U32.div(d,8),Bool.pick(U32,white,5,2))

def rank_one_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(1,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

def ep_rank(+b: Chess.Board,+d: U32,
  below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  valid: {Pos.valid_ep(d,b) == True{} : Bool}) ->
  {rank_ok(d,U32.is_eq(Chess.get_turn(b),1)) == True{} : Bool}:
  All.bool_and_true_left(rank_ok(d,U32.is_eq(Chess.get_turn(b),1)),
    Bool.and(Bool.not(U64.test_bit(Chess.occupied(b),U32.to_nat(d))),
      Bool.and(Bool.not(U64.test_bit(Chess.occupied(b),U32.to_nat(Bool.pick(U32,U32.is_eq(Chess.get_turn(b),1),U32.add(d,8),U32.sub(d,8))))),
        Pos.occupied_by(b,U32.xor(d,8),0,U32.xor(Chess.get_turn(b),1)))),
    Equal.trans(Bool,Post.guard(d,b,Chess.get_turn(b)),Pos.valid_ep(d,b),True{},
      Equal.sym(Bool,Pos.valid_ep(d,b),Post.guard(d,b,Chess.get_turn(b)),Post.unfold_ep(d,b,below)),valid))

# The bounded Word observation is structural, avoiding closed U64 calculations.
def get_zero(+n: Nat,+q: Nat) -> {Bits.get(n,Word.zero(n),q) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match q:
        case 0n: {==}
        case 1n+j: get_zero(p,j)

def bit_get(+n: Nat,+src: Nat,+q: Nat,
  below: {Nat.is_lt(q,n) == True{} : Bool}) ->
  {Bits.get(n,Bits.bit(n,src),q) == Nat.is_eq(src,q) : Bool}:
  match n:
    case 0n: Domain.impossible({Bits.get(0n,Bits.bit(0n,src),q) == Nat.is_eq(src,q) : Bool},q,below)
    case 1n+p:
      match src q:
        case 0n 0n: {==}
        case 0n 1n+j: get_zero(p,j)
        case 1n+i 0n: {==}
        case 1n+i 1n+j: bit_get(p,i,j,below)

def singleton(+src: U32,+q: U32,+sb: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  +qb: {Nat.is_lt(U32.to_nat(q),64n) == True{} : Bool}) ->
  {U64.test_bit(U64.bit(U32.to_nat(src)),U32.to_nat(q)) == U32.is_eq(src,q) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.bit(U32.to_nat(src)),U32.to_nat(q)),
    Bits.get(64n,U64.to_word(U64.bit(U32.to_nat(src))),U32.to_nat(q)),U32.is_eq(src,q),
    Word64.test_word(U64.bit(U32.to_nat(src)),U32.to_nat(q),qb),
    Equal.trans(Bool,Bits.get(64n,U64.to_word(U64.bit(U32.to_nat(src))),U32.to_nat(q)),
      Bits.get(64n,Bits.bit(64n,U32.to_nat(src)),U32.to_nat(q)),U32.is_eq(src,q),
      Equal.cong(Word(64n),Bool,x => Bits.get(64n,x,U32.to_nat(q)),
        U64.to_word(U64.bit(U32.to_nat(src))),Bits.bit(64n,U32.to_nat(src)),Word64.bit_word(U32.to_nat(src),sb)),
      Equal.trans(Bool,Bits.get(64n,Bits.bit(64n,U32.to_nat(src)),U32.to_nat(q)),
        Nat.is_eq(U32.to_nat(src),U32.to_nat(q)),U32.is_eq(src,q),
        bit_get(64n,U32.to_nat(src),U32.to_nat(q),qb),
        Equal.sym(Bool,U32.is_eq(src,q),Nat.is_eq(U32.to_nat(src),U32.to_nat(q)),Index.u32_eq_nat_eq(src,q)))))

def home_tests(-r: R.Right,-s: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(R.king(r))) == U32.is_eq(s,R.king(r)) : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(s)),U32.to_nat(R.rook(r))) == U32.is_eq(s,R.rook(r)) : Bool}

def singleton_home_tests(+r: R.Right,+s: U32,
  +below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool}) -> home_tests(r,s):
  (singleton(s,R.king(r),below,R.king_bound(r)),singleton(s,R.rook(r),below,R.rook_bound(r)))

def all_misses(-d: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),0n) == False{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),4n) == False{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),7n) == False{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),56n) == False{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),60n) == False{} : Bool} &
  ({U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),63n) == False{} : Bool})))))

def misses(-r: R.Right,-d: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),U32.to_nat(R.king(r))) == False{} : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),U32.to_nat(R.rook(r))) == False{} : Bool}

def project_misses(+r: R.Right,-d: U32,good: all_misses(d)) -> misses(r,d):
  match r:
    case R.Wk{}:
      (a1,(e1,(h1,(a8,(e8,h8)))))=good
      (e1,h1)
    case R.Wq{}:
      (a1,(e1,(h1,(a8,(e8,h8)))))=good
      (e1,a1)
    case R.Bk{}:
      (a1,(e1,(h1,(a8,(e8,h8)))))=good
      (e8,h8)
    case R.Bq{}:
      (a1,(e1,(h1,(a8,(e8,h8)))))=good
      (e8,a8)

def rank_zero_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(0,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

def rank_three_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(3,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

def rank_four_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(4,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

def rank_six_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(6,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

def rank_seven_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(7,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Words.false_true(Goal,good)
    case True{}: Words.false_true(Goal,good)

# Only16 in-range destinations can have coherent EP rank2 or5.
# Every other rank is eliminated before any six-home U64 observation is reduced.
def victim_cases(+d: U32,+white: Bool,n: Nat,
  same: {U32.to_nat(d) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},
  +rank: {rank_ok(d,white) == True{} : Bool}) -> all_misses(d):
  match n:
    case 0n:
      +fixed: {d == 0 : U32}=Domain.injective(d,0,same)
      %Equal.sym(U32,d,0,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(0),white,
        Equal.trans(Bool,rank_ok(0,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),0,d,Equal.sym(U32,d,0,fixed)),rank))
    case 1n:
      +fixed: {d == 1 : U32}=Domain.injective(d,1,same)
      %Equal.sym(U32,d,1,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(1),white,
        Equal.trans(Bool,rank_ok(1,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),1,d,Equal.sym(U32,d,1,fixed)),rank))
    case 2n:
      +fixed: {d == 2 : U32}=Domain.injective(d,2,same)
      %Equal.sym(U32,d,2,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(2),white,
        Equal.trans(Bool,rank_ok(2,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),2,d,Equal.sym(U32,d,2,fixed)),rank))
    case 3n:
      +fixed: {d == 3 : U32}=Domain.injective(d,3,same)
      %Equal.sym(U32,d,3,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(3),white,
        Equal.trans(Bool,rank_ok(3,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),3,d,Equal.sym(U32,d,3,fixed)),rank))
    case 4n:
      +fixed: {d == 4 : U32}=Domain.injective(d,4,same)
      %Equal.sym(U32,d,4,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(4),white,
        Equal.trans(Bool,rank_ok(4,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),4,d,Equal.sym(U32,d,4,fixed)),rank))
    case 5n:
      +fixed: {d == 5 : U32}=Domain.injective(d,5,same)
      %Equal.sym(U32,d,5,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(5),white,
        Equal.trans(Bool,rank_ok(5,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),5,d,Equal.sym(U32,d,5,fixed)),rank))
    case 6n:
      +fixed: {d == 6 : U32}=Domain.injective(d,6,same)
      %Equal.sym(U32,d,6,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(6),white,
        Equal.trans(Bool,rank_ok(6,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),6,d,Equal.sym(U32,d,6,fixed)),rank))
    case 7n:
      +fixed: {d == 7 : U32}=Domain.injective(d,7,same)
      %Equal.sym(U32,d,7,fixed) : all_misses(_)
      rank_zero_impossible(all_misses(7),white,
        Equal.trans(Bool,rank_ok(7,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),7,d,Equal.sym(U32,d,7,fixed)),rank))
    case 8n:
      +fixed: {d == 8 : U32}=Domain.injective(d,8,same)
      %Equal.sym(U32,d,8,fixed) : all_misses(_)
      rank_one_impossible(all_misses(8),white,
        Equal.trans(Bool,rank_ok(8,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),8,d,Equal.sym(U32,d,8,fixed)),rank))
    case 9n:
      +fixed: {d == 9 : U32}=Domain.injective(d,9,same)
      %Equal.sym(U32,d,9,fixed) : all_misses(_)
      rank_one_impossible(all_misses(9),white,
        Equal.trans(Bool,rank_ok(9,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),9,d,Equal.sym(U32,d,9,fixed)),rank))
    case 10n:
      +fixed: {d == 10 : U32}=Domain.injective(d,10,same)
      %Equal.sym(U32,d,10,fixed) : all_misses(_)
      rank_one_impossible(all_misses(10),white,
        Equal.trans(Bool,rank_ok(10,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),10,d,Equal.sym(U32,d,10,fixed)),rank))
    case 11n:
      +fixed: {d == 11 : U32}=Domain.injective(d,11,same)
      %Equal.sym(U32,d,11,fixed) : all_misses(_)
      rank_one_impossible(all_misses(11),white,
        Equal.trans(Bool,rank_ok(11,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),11,d,Equal.sym(U32,d,11,fixed)),rank))
    case 12n:
      +fixed: {d == 12 : U32}=Domain.injective(d,12,same)
      %Equal.sym(U32,d,12,fixed) : all_misses(_)
      rank_one_impossible(all_misses(12),white,
        Equal.trans(Bool,rank_ok(12,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),12,d,Equal.sym(U32,d,12,fixed)),rank))
    case 13n:
      +fixed: {d == 13 : U32}=Domain.injective(d,13,same)
      %Equal.sym(U32,d,13,fixed) : all_misses(_)
      rank_one_impossible(all_misses(13),white,
        Equal.trans(Bool,rank_ok(13,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),13,d,Equal.sym(U32,d,13,fixed)),rank))
    case 14n:
      +fixed: {d == 14 : U32}=Domain.injective(d,14,same)
      %Equal.sym(U32,d,14,fixed) : all_misses(_)
      rank_one_impossible(all_misses(14),white,
        Equal.trans(Bool,rank_ok(14,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),14,d,Equal.sym(U32,d,14,fixed)),rank))
    case 15n:
      +fixed: {d == 15 : U32}=Domain.injective(d,15,same)
      %Equal.sym(U32,d,15,fixed) : all_misses(_)
      rank_one_impossible(all_misses(15),white,
        Equal.trans(Bool,rank_ok(15,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),15,d,Equal.sym(U32,d,15,fixed)),rank))
    case 16n:
      +fixed: {d == 16 : U32}=Domain.injective(d,16,same)
      %Equal.sym(U32,d,16,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 17n:
      +fixed: {d == 17 : U32}=Domain.injective(d,17,same)
      %Equal.sym(U32,d,17,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 18n:
      +fixed: {d == 18 : U32}=Domain.injective(d,18,same)
      %Equal.sym(U32,d,18,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 19n:
      +fixed: {d == 19 : U32}=Domain.injective(d,19,same)
      %Equal.sym(U32,d,19,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 20n:
      +fixed: {d == 20 : U32}=Domain.injective(d,20,same)
      %Equal.sym(U32,d,20,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 21n:
      +fixed: {d == 21 : U32}=Domain.injective(d,21,same)
      %Equal.sym(U32,d,21,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 22n:
      +fixed: {d == 22 : U32}=Domain.injective(d,22,same)
      %Equal.sym(U32,d,22,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 23n:
      +fixed: {d == 23 : U32}=Domain.injective(d,23,same)
      %Equal.sym(U32,d,23,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 24n:
      +fixed: {d == 24 : U32}=Domain.injective(d,24,same)
      %Equal.sym(U32,d,24,fixed) : all_misses(_)
      rank_three_impossible(all_misses(24),white,
        Equal.trans(Bool,rank_ok(24,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),24,d,Equal.sym(U32,d,24,fixed)),rank))
    case 25n:
      +fixed: {d == 25 : U32}=Domain.injective(d,25,same)
      %Equal.sym(U32,d,25,fixed) : all_misses(_)
      rank_three_impossible(all_misses(25),white,
        Equal.trans(Bool,rank_ok(25,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),25,d,Equal.sym(U32,d,25,fixed)),rank))
    case 26n:
      +fixed: {d == 26 : U32}=Domain.injective(d,26,same)
      %Equal.sym(U32,d,26,fixed) : all_misses(_)
      rank_three_impossible(all_misses(26),white,
        Equal.trans(Bool,rank_ok(26,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),26,d,Equal.sym(U32,d,26,fixed)),rank))
    case 27n:
      +fixed: {d == 27 : U32}=Domain.injective(d,27,same)
      %Equal.sym(U32,d,27,fixed) : all_misses(_)
      rank_three_impossible(all_misses(27),white,
        Equal.trans(Bool,rank_ok(27,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),27,d,Equal.sym(U32,d,27,fixed)),rank))
    case 28n:
      +fixed: {d == 28 : U32}=Domain.injective(d,28,same)
      %Equal.sym(U32,d,28,fixed) : all_misses(_)
      rank_three_impossible(all_misses(28),white,
        Equal.trans(Bool,rank_ok(28,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),28,d,Equal.sym(U32,d,28,fixed)),rank))
    case 29n:
      +fixed: {d == 29 : U32}=Domain.injective(d,29,same)
      %Equal.sym(U32,d,29,fixed) : all_misses(_)
      rank_three_impossible(all_misses(29),white,
        Equal.trans(Bool,rank_ok(29,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),29,d,Equal.sym(U32,d,29,fixed)),rank))
    case 30n:
      +fixed: {d == 30 : U32}=Domain.injective(d,30,same)
      %Equal.sym(U32,d,30,fixed) : all_misses(_)
      rank_three_impossible(all_misses(30),white,
        Equal.trans(Bool,rank_ok(30,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),30,d,Equal.sym(U32,d,30,fixed)),rank))
    case 31n:
      +fixed: {d == 31 : U32}=Domain.injective(d,31,same)
      %Equal.sym(U32,d,31,fixed) : all_misses(_)
      rank_three_impossible(all_misses(31),white,
        Equal.trans(Bool,rank_ok(31,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),31,d,Equal.sym(U32,d,31,fixed)),rank))
    case 32n:
      +fixed: {d == 32 : U32}=Domain.injective(d,32,same)
      %Equal.sym(U32,d,32,fixed) : all_misses(_)
      rank_four_impossible(all_misses(32),white,
        Equal.trans(Bool,rank_ok(32,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),32,d,Equal.sym(U32,d,32,fixed)),rank))
    case 33n:
      +fixed: {d == 33 : U32}=Domain.injective(d,33,same)
      %Equal.sym(U32,d,33,fixed) : all_misses(_)
      rank_four_impossible(all_misses(33),white,
        Equal.trans(Bool,rank_ok(33,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),33,d,Equal.sym(U32,d,33,fixed)),rank))
    case 34n:
      +fixed: {d == 34 : U32}=Domain.injective(d,34,same)
      %Equal.sym(U32,d,34,fixed) : all_misses(_)
      rank_four_impossible(all_misses(34),white,
        Equal.trans(Bool,rank_ok(34,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),34,d,Equal.sym(U32,d,34,fixed)),rank))
    case 35n:
      +fixed: {d == 35 : U32}=Domain.injective(d,35,same)
      %Equal.sym(U32,d,35,fixed) : all_misses(_)
      rank_four_impossible(all_misses(35),white,
        Equal.trans(Bool,rank_ok(35,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),35,d,Equal.sym(U32,d,35,fixed)),rank))
    case 36n:
      +fixed: {d == 36 : U32}=Domain.injective(d,36,same)
      %Equal.sym(U32,d,36,fixed) : all_misses(_)
      rank_four_impossible(all_misses(36),white,
        Equal.trans(Bool,rank_ok(36,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),36,d,Equal.sym(U32,d,36,fixed)),rank))
    case 37n:
      +fixed: {d == 37 : U32}=Domain.injective(d,37,same)
      %Equal.sym(U32,d,37,fixed) : all_misses(_)
      rank_four_impossible(all_misses(37),white,
        Equal.trans(Bool,rank_ok(37,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),37,d,Equal.sym(U32,d,37,fixed)),rank))
    case 38n:
      +fixed: {d == 38 : U32}=Domain.injective(d,38,same)
      %Equal.sym(U32,d,38,fixed) : all_misses(_)
      rank_four_impossible(all_misses(38),white,
        Equal.trans(Bool,rank_ok(38,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),38,d,Equal.sym(U32,d,38,fixed)),rank))
    case 39n:
      +fixed: {d == 39 : U32}=Domain.injective(d,39,same)
      %Equal.sym(U32,d,39,fixed) : all_misses(_)
      rank_four_impossible(all_misses(39),white,
        Equal.trans(Bool,rank_ok(39,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),39,d,Equal.sym(U32,d,39,fixed)),rank))
    case 40n:
      +fixed: {d == 40 : U32}=Domain.injective(d,40,same)
      %Equal.sym(U32,d,40,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 41n:
      +fixed: {d == 41 : U32}=Domain.injective(d,41,same)
      %Equal.sym(U32,d,41,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 42n:
      +fixed: {d == 42 : U32}=Domain.injective(d,42,same)
      %Equal.sym(U32,d,42,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 43n:
      +fixed: {d == 43 : U32}=Domain.injective(d,43,same)
      %Equal.sym(U32,d,43,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 44n:
      +fixed: {d == 44 : U32}=Domain.injective(d,44,same)
      %Equal.sym(U32,d,44,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 45n:
      +fixed: {d == 45 : U32}=Domain.injective(d,45,same)
      %Equal.sym(U32,d,45,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 46n:
      +fixed: {d == 46 : U32}=Domain.injective(d,46,same)
      %Equal.sym(U32,d,46,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 47n:
      +fixed: {d == 47 : U32}=Domain.injective(d,47,same)
      %Equal.sym(U32,d,47,fixed) : all_misses(_)
      ({==},({==},({==},({==},({==},{==})))))
    case 48n:
      +fixed: {d == 48 : U32}=Domain.injective(d,48,same)
      %Equal.sym(U32,d,48,fixed) : all_misses(_)
      rank_six_impossible(all_misses(48),white,
        Equal.trans(Bool,rank_ok(48,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),48,d,Equal.sym(U32,d,48,fixed)),rank))
    case 49n:
      +fixed: {d == 49 : U32}=Domain.injective(d,49,same)
      %Equal.sym(U32,d,49,fixed) : all_misses(_)
      rank_six_impossible(all_misses(49),white,
        Equal.trans(Bool,rank_ok(49,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),49,d,Equal.sym(U32,d,49,fixed)),rank))
    case 50n:
      +fixed: {d == 50 : U32}=Domain.injective(d,50,same)
      %Equal.sym(U32,d,50,fixed) : all_misses(_)
      rank_six_impossible(all_misses(50),white,
        Equal.trans(Bool,rank_ok(50,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),50,d,Equal.sym(U32,d,50,fixed)),rank))
    case 51n:
      +fixed: {d == 51 : U32}=Domain.injective(d,51,same)
      %Equal.sym(U32,d,51,fixed) : all_misses(_)
      rank_six_impossible(all_misses(51),white,
        Equal.trans(Bool,rank_ok(51,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),51,d,Equal.sym(U32,d,51,fixed)),rank))
    case 52n:
      +fixed: {d == 52 : U32}=Domain.injective(d,52,same)
      %Equal.sym(U32,d,52,fixed) : all_misses(_)
      rank_six_impossible(all_misses(52),white,
        Equal.trans(Bool,rank_ok(52,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),52,d,Equal.sym(U32,d,52,fixed)),rank))
    case 53n:
      +fixed: {d == 53 : U32}=Domain.injective(d,53,same)
      %Equal.sym(U32,d,53,fixed) : all_misses(_)
      rank_six_impossible(all_misses(53),white,
        Equal.trans(Bool,rank_ok(53,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),53,d,Equal.sym(U32,d,53,fixed)),rank))
    case 54n:
      +fixed: {d == 54 : U32}=Domain.injective(d,54,same)
      %Equal.sym(U32,d,54,fixed) : all_misses(_)
      rank_six_impossible(all_misses(54),white,
        Equal.trans(Bool,rank_ok(54,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),54,d,Equal.sym(U32,d,54,fixed)),rank))
    case 55n:
      +fixed: {d == 55 : U32}=Domain.injective(d,55,same)
      %Equal.sym(U32,d,55,fixed) : all_misses(_)
      rank_six_impossible(all_misses(55),white,
        Equal.trans(Bool,rank_ok(55,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),55,d,Equal.sym(U32,d,55,fixed)),rank))
    case 56n:
      +fixed: {d == 56 : U32}=Domain.injective(d,56,same)
      %Equal.sym(U32,d,56,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(56),white,
        Equal.trans(Bool,rank_ok(56,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),56,d,Equal.sym(U32,d,56,fixed)),rank))
    case 57n:
      +fixed: {d == 57 : U32}=Domain.injective(d,57,same)
      %Equal.sym(U32,d,57,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(57),white,
        Equal.trans(Bool,rank_ok(57,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),57,d,Equal.sym(U32,d,57,fixed)),rank))
    case 58n:
      +fixed: {d == 58 : U32}=Domain.injective(d,58,same)
      %Equal.sym(U32,d,58,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(58),white,
        Equal.trans(Bool,rank_ok(58,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),58,d,Equal.sym(U32,d,58,fixed)),rank))
    case 59n:
      +fixed: {d == 59 : U32}=Domain.injective(d,59,same)
      %Equal.sym(U32,d,59,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(59),white,
        Equal.trans(Bool,rank_ok(59,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),59,d,Equal.sym(U32,d,59,fixed)),rank))
    case 60n:
      +fixed: {d == 60 : U32}=Domain.injective(d,60,same)
      %Equal.sym(U32,d,60,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(60),white,
        Equal.trans(Bool,rank_ok(60,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),60,d,Equal.sym(U32,d,60,fixed)),rank))
    case 61n:
      +fixed: {d == 61 : U32}=Domain.injective(d,61,same)
      %Equal.sym(U32,d,61,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(61),white,
        Equal.trans(Bool,rank_ok(61,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),61,d,Equal.sym(U32,d,61,fixed)),rank))
    case 62n:
      +fixed: {d == 62 : U32}=Domain.injective(d,62,same)
      %Equal.sym(U32,d,62,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(62),white,
        Equal.trans(Bool,rank_ok(62,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),62,d,Equal.sym(U32,d,62,fixed)),rank))
    case 63n:
      +fixed: {d == 63 : U32}=Domain.injective(d,63,same)
      %Equal.sym(U32,d,63,fixed) : all_misses(_)
      rank_seven_impossible(all_misses(63),white,
        Equal.trans(Bool,rank_ok(63,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),63,d,Equal.sym(U32,d,63,fixed)),rank))
    case 64n+p: Domain.impossible(all_misses(d),p,below)

def victim(+r: R.Right,+d: U32,+white: Bool,
  below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  rank: {rank_ok(d,white) == True{} : Bool}) -> misses(r,d):
  project_misses(r,d,victim_cases(d,white,U32.to_nat(d),{==},below,rank))

def generated_ep_victim(+r: R.Right,+b: Chess.Board,+d: U32,
  +below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: {d == Chess.get_ep(b) : U32}) -> misses(r,d):
  victim(r,d,U32.is_eq(Chess.get_turn(b),1),below,
    ep_rank(b,d,below,Equal.trans(Bool,Pos.valid_ep(d,b),Pos.valid_ep(Chess.get_ep(b),b),True{},
      Equal.cong(U32,Bool,x => Pos.valid_ep(x,b),d,Chess.get_ep(b),tagged),coherent)))
