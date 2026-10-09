# Coherent EP victim geometry, independent of parent home occupants.
import Base
import ../layout/Domain.bend as Domain
import ../u64/Words.bend as Old
import ../bitboard_membership/AllEmittedSound.bend as All
import ../child_ep_coherence/Post.bend as Post
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def rank_ok(+d: U32,white: Bool) -> Bool:
  U32.is_eq(U32.div(d,8),Bool.pick(U32,white,5,2))

def misses(-d: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),4n) == False{} : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(U32.xor(d,8))),7n) == False{} : Bool}

def rank_one_impossible(-Goal: Type,white: Bool,
  good: {U32.is_eq(1,Bool.pick(U32,white,5,2)) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Old.false_true(Goal,good)
    case True{}: Old.false_true(Goal,good)

# Only targets 12 and 15 have victims 4 and 7; their rank is invalid for EP.
# The finite domain proof uses U32.to_nat injectivity rather than a round-trip axiom.
def victim_cases(+d: U32,+white: Bool,n: Nat,
  same: {U32.to_nat(d) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},
  +rank: {rank_ok(d,white) == True{} : Bool}) -> misses(d):
  match n:
    case 0n:
      +fixed: {d == 0 : U32}=Domain.injective(d,0,same)
      %Equal.sym(U32,d,0,fixed) : misses(_)
      ({==},{==})
    case 1n:
      +fixed: {d == 1 : U32}=Domain.injective(d,1,same)
      %Equal.sym(U32,d,1,fixed) : misses(_)
      ({==},{==})
    case 2n:
      +fixed: {d == 2 : U32}=Domain.injective(d,2,same)
      %Equal.sym(U32,d,2,fixed) : misses(_)
      ({==},{==})
    case 3n:
      +fixed: {d == 3 : U32}=Domain.injective(d,3,same)
      %Equal.sym(U32,d,3,fixed) : misses(_)
      ({==},{==})
    case 4n:
      +fixed: {d == 4 : U32}=Domain.injective(d,4,same)
      %Equal.sym(U32,d,4,fixed) : misses(_)
      ({==},{==})
    case 5n:
      +fixed: {d == 5 : U32}=Domain.injective(d,5,same)
      %Equal.sym(U32,d,5,fixed) : misses(_)
      ({==},{==})
    case 6n:
      +fixed: {d == 6 : U32}=Domain.injective(d,6,same)
      %Equal.sym(U32,d,6,fixed) : misses(_)
      ({==},{==})
    case 7n:
      +fixed: {d == 7 : U32}=Domain.injective(d,7,same)
      %Equal.sym(U32,d,7,fixed) : misses(_)
      ({==},{==})
    case 8n:
      +fixed: {d == 8 : U32}=Domain.injective(d,8,same)
      %Equal.sym(U32,d,8,fixed) : misses(_)
      ({==},{==})
    case 9n:
      +fixed: {d == 9 : U32}=Domain.injective(d,9,same)
      %Equal.sym(U32,d,9,fixed) : misses(_)
      ({==},{==})
    case 10n:
      +fixed: {d == 10 : U32}=Domain.injective(d,10,same)
      %Equal.sym(U32,d,10,fixed) : misses(_)
      ({==},{==})
    case 11n:
      +fixed: {d == 11 : U32}=Domain.injective(d,11,same)
      %Equal.sym(U32,d,11,fixed) : misses(_)
      ({==},{==})
    case 12n:
      +fixed: {d == 12 : U32}=Domain.injective(d,12,same)
      %Equal.sym(U32,d,12,fixed) : misses(_)
      rank_one_impossible(misses(12),white,
        Equal.trans(Bool,rank_ok(12,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),12,d,Equal.sym(U32,d,12,fixed)),rank))
    case 13n:
      +fixed: {d == 13 : U32}=Domain.injective(d,13,same)
      %Equal.sym(U32,d,13,fixed) : misses(_)
      ({==},{==})
    case 14n:
      +fixed: {d == 14 : U32}=Domain.injective(d,14,same)
      %Equal.sym(U32,d,14,fixed) : misses(_)
      ({==},{==})
    case 15n:
      +fixed: {d == 15 : U32}=Domain.injective(d,15,same)
      %Equal.sym(U32,d,15,fixed) : misses(_)
      rank_one_impossible(misses(15),white,
        Equal.trans(Bool,rank_ok(15,white),rank_ok(d,white),True{},
          Equal.cong(U32,Bool,x => rank_ok(x,white),15,d,Equal.sym(U32,d,15,fixed)),rank))
    case 16n:
      +fixed: {d == 16 : U32}=Domain.injective(d,16,same)
      %Equal.sym(U32,d,16,fixed) : misses(_)
      ({==},{==})
    case 17n:
      +fixed: {d == 17 : U32}=Domain.injective(d,17,same)
      %Equal.sym(U32,d,17,fixed) : misses(_)
      ({==},{==})
    case 18n:
      +fixed: {d == 18 : U32}=Domain.injective(d,18,same)
      %Equal.sym(U32,d,18,fixed) : misses(_)
      ({==},{==})
    case 19n:
      +fixed: {d == 19 : U32}=Domain.injective(d,19,same)
      %Equal.sym(U32,d,19,fixed) : misses(_)
      ({==},{==})
    case 20n:
      +fixed: {d == 20 : U32}=Domain.injective(d,20,same)
      %Equal.sym(U32,d,20,fixed) : misses(_)
      ({==},{==})
    case 21n:
      +fixed: {d == 21 : U32}=Domain.injective(d,21,same)
      %Equal.sym(U32,d,21,fixed) : misses(_)
      ({==},{==})
    case 22n:
      +fixed: {d == 22 : U32}=Domain.injective(d,22,same)
      %Equal.sym(U32,d,22,fixed) : misses(_)
      ({==},{==})
    case 23n:
      +fixed: {d == 23 : U32}=Domain.injective(d,23,same)
      %Equal.sym(U32,d,23,fixed) : misses(_)
      ({==},{==})
    case 24n:
      +fixed: {d == 24 : U32}=Domain.injective(d,24,same)
      %Equal.sym(U32,d,24,fixed) : misses(_)
      ({==},{==})
    case 25n:
      +fixed: {d == 25 : U32}=Domain.injective(d,25,same)
      %Equal.sym(U32,d,25,fixed) : misses(_)
      ({==},{==})
    case 26n:
      +fixed: {d == 26 : U32}=Domain.injective(d,26,same)
      %Equal.sym(U32,d,26,fixed) : misses(_)
      ({==},{==})
    case 27n:
      +fixed: {d == 27 : U32}=Domain.injective(d,27,same)
      %Equal.sym(U32,d,27,fixed) : misses(_)
      ({==},{==})
    case 28n:
      +fixed: {d == 28 : U32}=Domain.injective(d,28,same)
      %Equal.sym(U32,d,28,fixed) : misses(_)
      ({==},{==})
    case 29n:
      +fixed: {d == 29 : U32}=Domain.injective(d,29,same)
      %Equal.sym(U32,d,29,fixed) : misses(_)
      ({==},{==})
    case 30n:
      +fixed: {d == 30 : U32}=Domain.injective(d,30,same)
      %Equal.sym(U32,d,30,fixed) : misses(_)
      ({==},{==})
    case 31n:
      +fixed: {d == 31 : U32}=Domain.injective(d,31,same)
      %Equal.sym(U32,d,31,fixed) : misses(_)
      ({==},{==})
    case 32n:
      +fixed: {d == 32 : U32}=Domain.injective(d,32,same)
      %Equal.sym(U32,d,32,fixed) : misses(_)
      ({==},{==})
    case 33n:
      +fixed: {d == 33 : U32}=Domain.injective(d,33,same)
      %Equal.sym(U32,d,33,fixed) : misses(_)
      ({==},{==})
    case 34n:
      +fixed: {d == 34 : U32}=Domain.injective(d,34,same)
      %Equal.sym(U32,d,34,fixed) : misses(_)
      ({==},{==})
    case 35n:
      +fixed: {d == 35 : U32}=Domain.injective(d,35,same)
      %Equal.sym(U32,d,35,fixed) : misses(_)
      ({==},{==})
    case 36n:
      +fixed: {d == 36 : U32}=Domain.injective(d,36,same)
      %Equal.sym(U32,d,36,fixed) : misses(_)
      ({==},{==})
    case 37n:
      +fixed: {d == 37 : U32}=Domain.injective(d,37,same)
      %Equal.sym(U32,d,37,fixed) : misses(_)
      ({==},{==})
    case 38n:
      +fixed: {d == 38 : U32}=Domain.injective(d,38,same)
      %Equal.sym(U32,d,38,fixed) : misses(_)
      ({==},{==})
    case 39n:
      +fixed: {d == 39 : U32}=Domain.injective(d,39,same)
      %Equal.sym(U32,d,39,fixed) : misses(_)
      ({==},{==})
    case 40n:
      +fixed: {d == 40 : U32}=Domain.injective(d,40,same)
      %Equal.sym(U32,d,40,fixed) : misses(_)
      ({==},{==})
    case 41n:
      +fixed: {d == 41 : U32}=Domain.injective(d,41,same)
      %Equal.sym(U32,d,41,fixed) : misses(_)
      ({==},{==})
    case 42n:
      +fixed: {d == 42 : U32}=Domain.injective(d,42,same)
      %Equal.sym(U32,d,42,fixed) : misses(_)
      ({==},{==})
    case 43n:
      +fixed: {d == 43 : U32}=Domain.injective(d,43,same)
      %Equal.sym(U32,d,43,fixed) : misses(_)
      ({==},{==})
    case 44n:
      +fixed: {d == 44 : U32}=Domain.injective(d,44,same)
      %Equal.sym(U32,d,44,fixed) : misses(_)
      ({==},{==})
    case 45n:
      +fixed: {d == 45 : U32}=Domain.injective(d,45,same)
      %Equal.sym(U32,d,45,fixed) : misses(_)
      ({==},{==})
    case 46n:
      +fixed: {d == 46 : U32}=Domain.injective(d,46,same)
      %Equal.sym(U32,d,46,fixed) : misses(_)
      ({==},{==})
    case 47n:
      +fixed: {d == 47 : U32}=Domain.injective(d,47,same)
      %Equal.sym(U32,d,47,fixed) : misses(_)
      ({==},{==})
    case 48n:
      +fixed: {d == 48 : U32}=Domain.injective(d,48,same)
      %Equal.sym(U32,d,48,fixed) : misses(_)
      ({==},{==})
    case 49n:
      +fixed: {d == 49 : U32}=Domain.injective(d,49,same)
      %Equal.sym(U32,d,49,fixed) : misses(_)
      ({==},{==})
    case 50n:
      +fixed: {d == 50 : U32}=Domain.injective(d,50,same)
      %Equal.sym(U32,d,50,fixed) : misses(_)
      ({==},{==})
    case 51n:
      +fixed: {d == 51 : U32}=Domain.injective(d,51,same)
      %Equal.sym(U32,d,51,fixed) : misses(_)
      ({==},{==})
    case 52n:
      +fixed: {d == 52 : U32}=Domain.injective(d,52,same)
      %Equal.sym(U32,d,52,fixed) : misses(_)
      ({==},{==})
    case 53n:
      +fixed: {d == 53 : U32}=Domain.injective(d,53,same)
      %Equal.sym(U32,d,53,fixed) : misses(_)
      ({==},{==})
    case 54n:
      +fixed: {d == 54 : U32}=Domain.injective(d,54,same)
      %Equal.sym(U32,d,54,fixed) : misses(_)
      ({==},{==})
    case 55n:
      +fixed: {d == 55 : U32}=Domain.injective(d,55,same)
      %Equal.sym(U32,d,55,fixed) : misses(_)
      ({==},{==})
    case 56n:
      +fixed: {d == 56 : U32}=Domain.injective(d,56,same)
      %Equal.sym(U32,d,56,fixed) : misses(_)
      ({==},{==})
    case 57n:
      +fixed: {d == 57 : U32}=Domain.injective(d,57,same)
      %Equal.sym(U32,d,57,fixed) : misses(_)
      ({==},{==})
    case 58n:
      +fixed: {d == 58 : U32}=Domain.injective(d,58,same)
      %Equal.sym(U32,d,58,fixed) : misses(_)
      ({==},{==})
    case 59n:
      +fixed: {d == 59 : U32}=Domain.injective(d,59,same)
      %Equal.sym(U32,d,59,fixed) : misses(_)
      ({==},{==})
    case 60n:
      +fixed: {d == 60 : U32}=Domain.injective(d,60,same)
      %Equal.sym(U32,d,60,fixed) : misses(_)
      ({==},{==})
    case 61n:
      +fixed: {d == 61 : U32}=Domain.injective(d,61,same)
      %Equal.sym(U32,d,61,fixed) : misses(_)
      ({==},{==})
    case 62n:
      +fixed: {d == 62 : U32}=Domain.injective(d,62,same)
      %Equal.sym(U32,d,62,fixed) : misses(_)
      ({==},{==})
    case 63n:
      +fixed: {d == 63 : U32}=Domain.injective(d,63,same)
      %Equal.sym(U32,d,63,fixed) : misses(_)
      ({==},{==})
    case 64n+p: Domain.impossible(misses(d),p,below)

def victim(+d: U32,+white: Bool,
  below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  rank: {rank_ok(d,white) == True{} : Bool}) -> misses(d):
  victim_cases(d,white,U32.to_nat(d),{==},below,rank)

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

# The destination comes from the actual flag1 occurrence tag; no geometric premise
# about the source or any parent home piece is introduced here.
def generated_ep_victim(+b: Chess.Board,+d: U32,
  +below: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  tagged: {d == Chess.get_ep(b) : U32}) -> misses(d):
  victim(d,U32.is_eq(Chess.get_turn(b),1),below,
    ep_rank(b,d,below,Equal.trans(Bool,Pos.valid_ep(d,b),Pos.valid_ep(Chess.get_ep(b),b),True{},
      Equal.cong(U32,Bool,x => Pos.valid_ep(x,b),d,Chess.get_ep(b),tagged),coherent)))

# Bounded singleton observations for the two white kingside home squares.
def home_tests(-s: U32) -> Type:
  {U64.test_bit(U64.bit(U32.to_nat(s)),4n) == U32.is_eq(s,4) : Bool} &
  {U64.test_bit(U64.bit(U32.to_nat(s)),7n) == U32.is_eq(s,7) : Bool}

def home_cases(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},
  below: {Nat.is_lt(n,64n) == True{} : Bool}) -> home_tests(s):
  match n:
    case 0n:
      %Equal.sym(U32,s,0,Domain.injective(s,0,same)) : home_tests(_)
      ({==},{==})
    case 1n:
      %Equal.sym(U32,s,1,Domain.injective(s,1,same)) : home_tests(_)
      ({==},{==})
    case 2n:
      %Equal.sym(U32,s,2,Domain.injective(s,2,same)) : home_tests(_)
      ({==},{==})
    case 3n:
      %Equal.sym(U32,s,3,Domain.injective(s,3,same)) : home_tests(_)
      ({==},{==})
    case 4n:
      %Equal.sym(U32,s,4,Domain.injective(s,4,same)) : home_tests(_)
      ({==},{==})
    case 5n:
      %Equal.sym(U32,s,5,Domain.injective(s,5,same)) : home_tests(_)
      ({==},{==})
    case 6n:
      %Equal.sym(U32,s,6,Domain.injective(s,6,same)) : home_tests(_)
      ({==},{==})
    case 7n:
      %Equal.sym(U32,s,7,Domain.injective(s,7,same)) : home_tests(_)
      ({==},{==})
    case 8n:
      %Equal.sym(U32,s,8,Domain.injective(s,8,same)) : home_tests(_)
      ({==},{==})
    case 9n:
      %Equal.sym(U32,s,9,Domain.injective(s,9,same)) : home_tests(_)
      ({==},{==})
    case 10n:
      %Equal.sym(U32,s,10,Domain.injective(s,10,same)) : home_tests(_)
      ({==},{==})
    case 11n:
      %Equal.sym(U32,s,11,Domain.injective(s,11,same)) : home_tests(_)
      ({==},{==})
    case 12n:
      %Equal.sym(U32,s,12,Domain.injective(s,12,same)) : home_tests(_)
      ({==},{==})
    case 13n:
      %Equal.sym(U32,s,13,Domain.injective(s,13,same)) : home_tests(_)
      ({==},{==})
    case 14n:
      %Equal.sym(U32,s,14,Domain.injective(s,14,same)) : home_tests(_)
      ({==},{==})
    case 15n:
      %Equal.sym(U32,s,15,Domain.injective(s,15,same)) : home_tests(_)
      ({==},{==})
    case 16n:
      %Equal.sym(U32,s,16,Domain.injective(s,16,same)) : home_tests(_)
      ({==},{==})
    case 17n:
      %Equal.sym(U32,s,17,Domain.injective(s,17,same)) : home_tests(_)
      ({==},{==})
    case 18n:
      %Equal.sym(U32,s,18,Domain.injective(s,18,same)) : home_tests(_)
      ({==},{==})
    case 19n:
      %Equal.sym(U32,s,19,Domain.injective(s,19,same)) : home_tests(_)
      ({==},{==})
    case 20n:
      %Equal.sym(U32,s,20,Domain.injective(s,20,same)) : home_tests(_)
      ({==},{==})
    case 21n:
      %Equal.sym(U32,s,21,Domain.injective(s,21,same)) : home_tests(_)
      ({==},{==})
    case 22n:
      %Equal.sym(U32,s,22,Domain.injective(s,22,same)) : home_tests(_)
      ({==},{==})
    case 23n:
      %Equal.sym(U32,s,23,Domain.injective(s,23,same)) : home_tests(_)
      ({==},{==})
    case 24n:
      %Equal.sym(U32,s,24,Domain.injective(s,24,same)) : home_tests(_)
      ({==},{==})
    case 25n:
      %Equal.sym(U32,s,25,Domain.injective(s,25,same)) : home_tests(_)
      ({==},{==})
    case 26n:
      %Equal.sym(U32,s,26,Domain.injective(s,26,same)) : home_tests(_)
      ({==},{==})
    case 27n:
      %Equal.sym(U32,s,27,Domain.injective(s,27,same)) : home_tests(_)
      ({==},{==})
    case 28n:
      %Equal.sym(U32,s,28,Domain.injective(s,28,same)) : home_tests(_)
      ({==},{==})
    case 29n:
      %Equal.sym(U32,s,29,Domain.injective(s,29,same)) : home_tests(_)
      ({==},{==})
    case 30n:
      %Equal.sym(U32,s,30,Domain.injective(s,30,same)) : home_tests(_)
      ({==},{==})
    case 31n:
      %Equal.sym(U32,s,31,Domain.injective(s,31,same)) : home_tests(_)
      ({==},{==})
    case 32n:
      %Equal.sym(U32,s,32,Domain.injective(s,32,same)) : home_tests(_)
      ({==},{==})
    case 33n:
      %Equal.sym(U32,s,33,Domain.injective(s,33,same)) : home_tests(_)
      ({==},{==})
    case 34n:
      %Equal.sym(U32,s,34,Domain.injective(s,34,same)) : home_tests(_)
      ({==},{==})
    case 35n:
      %Equal.sym(U32,s,35,Domain.injective(s,35,same)) : home_tests(_)
      ({==},{==})
    case 36n:
      %Equal.sym(U32,s,36,Domain.injective(s,36,same)) : home_tests(_)
      ({==},{==})
    case 37n:
      %Equal.sym(U32,s,37,Domain.injective(s,37,same)) : home_tests(_)
      ({==},{==})
    case 38n:
      %Equal.sym(U32,s,38,Domain.injective(s,38,same)) : home_tests(_)
      ({==},{==})
    case 39n:
      %Equal.sym(U32,s,39,Domain.injective(s,39,same)) : home_tests(_)
      ({==},{==})
    case 40n:
      %Equal.sym(U32,s,40,Domain.injective(s,40,same)) : home_tests(_)
      ({==},{==})
    case 41n:
      %Equal.sym(U32,s,41,Domain.injective(s,41,same)) : home_tests(_)
      ({==},{==})
    case 42n:
      %Equal.sym(U32,s,42,Domain.injective(s,42,same)) : home_tests(_)
      ({==},{==})
    case 43n:
      %Equal.sym(U32,s,43,Domain.injective(s,43,same)) : home_tests(_)
      ({==},{==})
    case 44n:
      %Equal.sym(U32,s,44,Domain.injective(s,44,same)) : home_tests(_)
      ({==},{==})
    case 45n:
      %Equal.sym(U32,s,45,Domain.injective(s,45,same)) : home_tests(_)
      ({==},{==})
    case 46n:
      %Equal.sym(U32,s,46,Domain.injective(s,46,same)) : home_tests(_)
      ({==},{==})
    case 47n:
      %Equal.sym(U32,s,47,Domain.injective(s,47,same)) : home_tests(_)
      ({==},{==})
    case 48n:
      %Equal.sym(U32,s,48,Domain.injective(s,48,same)) : home_tests(_)
      ({==},{==})
    case 49n:
      %Equal.sym(U32,s,49,Domain.injective(s,49,same)) : home_tests(_)
      ({==},{==})
    case 50n:
      %Equal.sym(U32,s,50,Domain.injective(s,50,same)) : home_tests(_)
      ({==},{==})
    case 51n:
      %Equal.sym(U32,s,51,Domain.injective(s,51,same)) : home_tests(_)
      ({==},{==})
    case 52n:
      %Equal.sym(U32,s,52,Domain.injective(s,52,same)) : home_tests(_)
      ({==},{==})
    case 53n:
      %Equal.sym(U32,s,53,Domain.injective(s,53,same)) : home_tests(_)
      ({==},{==})
    case 54n:
      %Equal.sym(U32,s,54,Domain.injective(s,54,same)) : home_tests(_)
      ({==},{==})
    case 55n:
      %Equal.sym(U32,s,55,Domain.injective(s,55,same)) : home_tests(_)
      ({==},{==})
    case 56n:
      %Equal.sym(U32,s,56,Domain.injective(s,56,same)) : home_tests(_)
      ({==},{==})
    case 57n:
      %Equal.sym(U32,s,57,Domain.injective(s,57,same)) : home_tests(_)
      ({==},{==})
    case 58n:
      %Equal.sym(U32,s,58,Domain.injective(s,58,same)) : home_tests(_)
      ({==},{==})
    case 59n:
      %Equal.sym(U32,s,59,Domain.injective(s,59,same)) : home_tests(_)
      ({==},{==})
    case 60n:
      %Equal.sym(U32,s,60,Domain.injective(s,60,same)) : home_tests(_)
      ({==},{==})
    case 61n:
      %Equal.sym(U32,s,61,Domain.injective(s,61,same)) : home_tests(_)
      ({==},{==})
    case 62n:
      %Equal.sym(U32,s,62,Domain.injective(s,62,same)) : home_tests(_)
      ({==},{==})
    case 63n:
      %Equal.sym(U32,s,63,Domain.injective(s,63,same)) : home_tests(_)
      ({==},{==})
    case 64n+p: Domain.impossible(home_tests(s),p,below)

def singleton_home_tests(+s: U32,below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool}) -> home_tests(s):
  home_cases(s,U32.to_nat(s),{==},below)
