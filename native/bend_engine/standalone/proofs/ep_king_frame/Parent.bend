# Extract actual coherent-EP guard fields at the producer's in-range target.
import Base
import ../child_ep_coherence/Post.bend as Post
import ../u64/Words.bend as Old
import ../bitboard_membership/AllEmittedSound.bend as All
import ../layout/Domain.bend as Domain
import ../board/Spec.bend as Board
import ../candidate_king_destination/Targets.bend as Targets
import ./Rows.bend as Rows
import ../bitboard_membership/IndexEquality.bend as Index
import ../../../legal_probe/Chess.bend as C
import ../../Position.bend as Position

def victim(d: U32) -> U32: U32.xor(d,8)
def origin(+b: C.Board,+d: U32) -> U32:
  Bool.pick(U32,U32.is_eq(C.get_turn(b),1),U32.add(d,8),U32.sub(d,8))

def not_true(a: Bool,yes: {Bool.not(a) == True{} : Bool}) -> {a == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: Old.false_true({True{} == False{} : Bool},yes)

def fields(+b: C.Board,+d: U32,+bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},valid: {Position.valid_ep(d,b) == True{} : Bool}) ->
  {U64.test_bit(C.occupied(b),U32.to_nat(d)) == False{} : Bool} & {C.piece(b,victim(d)) == 0 : U32}:
  +rank=U32.is_eq(U32.div(d,8),Bool.pick(U32,U32.is_eq(C.get_turn(b),1),5,2))
  +empty=Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(d)))
  +clear=Bool.not(U64.test_bit(C.occupied(b),U32.to_nat(origin(b,d))))
  +occupied=Position.occupied_by(b,victim(d),0,U32.xor(C.get_turn(b),1))
  +guard=Equal.trans(Bool,Post.guard(d,b,C.get_turn(b)),Position.valid_ep(d,b),True{},Equal.sym(Bool,Position.valid_ep(d,b),Post.guard(d,b,C.get_turn(b)),Post.unfold_ep(d,b,bound)),valid)
  +rest=All.bool_and_true_right(rank,Bool.and(empty,Bool.and(clear,occupied)),guard)
  +kind=All.bool_and_true_right(U64.test_bit(C.color(b,U32.xor(C.get_turn(b),1)),U32.to_nat(victim(d))),U32.is_eq(C.piece(b,victim(d)),0),
    All.bool_and_true_right(clear,occupied,All.bool_and_true_right(empty,Bool.and(clear,occupied),rest)))
  (not_true(U64.test_bit(C.occupied(b),U32.to_nat(d)),All.bool_and_true_left(empty,Bool.and(clear,occupied),rest)),Index.u32_eq_true_same(C.piece(b,victim(d)),0,kind))

def victim_cases(+d: U32,n: Nat,same: {U32.to_nat(d) == n : Nat},bound: {Nat.is_lt(n,64n) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(victim(d)),64n) == True{} : Bool}:
  match n:
    case 0n:
      %Equal.sym(U32,d,0,Domain.injective(d,0,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 1n:
      %Equal.sym(U32,d,1,Domain.injective(d,1,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 2n:
      %Equal.sym(U32,d,2,Domain.injective(d,2,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 3n:
      %Equal.sym(U32,d,3,Domain.injective(d,3,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 4n:
      %Equal.sym(U32,d,4,Domain.injective(d,4,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 5n:
      %Equal.sym(U32,d,5,Domain.injective(d,5,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 6n:
      %Equal.sym(U32,d,6,Domain.injective(d,6,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 7n:
      %Equal.sym(U32,d,7,Domain.injective(d,7,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 8n:
      %Equal.sym(U32,d,8,Domain.injective(d,8,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 9n:
      %Equal.sym(U32,d,9,Domain.injective(d,9,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 10n:
      %Equal.sym(U32,d,10,Domain.injective(d,10,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 11n:
      %Equal.sym(U32,d,11,Domain.injective(d,11,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 12n:
      %Equal.sym(U32,d,12,Domain.injective(d,12,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 13n:
      %Equal.sym(U32,d,13,Domain.injective(d,13,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 14n:
      %Equal.sym(U32,d,14,Domain.injective(d,14,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 15n:
      %Equal.sym(U32,d,15,Domain.injective(d,15,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 16n:
      %Equal.sym(U32,d,16,Domain.injective(d,16,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 17n:
      %Equal.sym(U32,d,17,Domain.injective(d,17,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 18n:
      %Equal.sym(U32,d,18,Domain.injective(d,18,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 19n:
      %Equal.sym(U32,d,19,Domain.injective(d,19,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 20n:
      %Equal.sym(U32,d,20,Domain.injective(d,20,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 21n:
      %Equal.sym(U32,d,21,Domain.injective(d,21,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 22n:
      %Equal.sym(U32,d,22,Domain.injective(d,22,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 23n:
      %Equal.sym(U32,d,23,Domain.injective(d,23,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 24n:
      %Equal.sym(U32,d,24,Domain.injective(d,24,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 25n:
      %Equal.sym(U32,d,25,Domain.injective(d,25,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 26n:
      %Equal.sym(U32,d,26,Domain.injective(d,26,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 27n:
      %Equal.sym(U32,d,27,Domain.injective(d,27,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 28n:
      %Equal.sym(U32,d,28,Domain.injective(d,28,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 29n:
      %Equal.sym(U32,d,29,Domain.injective(d,29,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 30n:
      %Equal.sym(U32,d,30,Domain.injective(d,30,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 31n:
      %Equal.sym(U32,d,31,Domain.injective(d,31,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 32n:
      %Equal.sym(U32,d,32,Domain.injective(d,32,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 33n:
      %Equal.sym(U32,d,33,Domain.injective(d,33,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 34n:
      %Equal.sym(U32,d,34,Domain.injective(d,34,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 35n:
      %Equal.sym(U32,d,35,Domain.injective(d,35,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 36n:
      %Equal.sym(U32,d,36,Domain.injective(d,36,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 37n:
      %Equal.sym(U32,d,37,Domain.injective(d,37,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 38n:
      %Equal.sym(U32,d,38,Domain.injective(d,38,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 39n:
      %Equal.sym(U32,d,39,Domain.injective(d,39,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 40n:
      %Equal.sym(U32,d,40,Domain.injective(d,40,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 41n:
      %Equal.sym(U32,d,41,Domain.injective(d,41,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 42n:
      %Equal.sym(U32,d,42,Domain.injective(d,42,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 43n:
      %Equal.sym(U32,d,43,Domain.injective(d,43,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 44n:
      %Equal.sym(U32,d,44,Domain.injective(d,44,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 45n:
      %Equal.sym(U32,d,45,Domain.injective(d,45,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 46n:
      %Equal.sym(U32,d,46,Domain.injective(d,46,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 47n:
      %Equal.sym(U32,d,47,Domain.injective(d,47,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 48n:
      %Equal.sym(U32,d,48,Domain.injective(d,48,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 49n:
      %Equal.sym(U32,d,49,Domain.injective(d,49,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 50n:
      %Equal.sym(U32,d,50,Domain.injective(d,50,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 51n:
      %Equal.sym(U32,d,51,Domain.injective(d,51,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 52n:
      %Equal.sym(U32,d,52,Domain.injective(d,52,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 53n:
      %Equal.sym(U32,d,53,Domain.injective(d,53,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 54n:
      %Equal.sym(U32,d,54,Domain.injective(d,54,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 55n:
      %Equal.sym(U32,d,55,Domain.injective(d,55,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 56n:
      %Equal.sym(U32,d,56,Domain.injective(d,56,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 57n:
      %Equal.sym(U32,d,57,Domain.injective(d,57,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 58n:
      %Equal.sym(U32,d,58,Domain.injective(d,58,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 59n:
      %Equal.sym(U32,d,59,Domain.injective(d,59,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 60n:
      %Equal.sym(U32,d,60,Domain.injective(d,60,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 61n:
      %Equal.sym(U32,d,61,Domain.injective(d,61,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 62n:
      %Equal.sym(U32,d,62,Domain.injective(d,62,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 63n:
      %Equal.sym(U32,d,63,Domain.injective(d,63,same)) : {Nat.is_lt(U32.to_nat(victim(_)),64n) == True{} : Bool}
      {==}
    case 64n+p: Domain.impossible({Nat.is_lt(U32.to_nat(victim(d)),64n) == True{} : Bool},p,bound)

def victim_bound(+d: U32,bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(victim(d)),64n) == True{} : Bool}:
  victim_cases(d,U32.to_nat(d),{==},bound)

def clear_known(+b: C.Board,+d: U32,+good: {Board.valid(b) == True{} : Bool},bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},cert: {U64.test_bit(C.occupied(b),U32.to_nat(d)) == False{} : Bool} & {C.piece(b,victim(d)) == 0 : U32}) ->
  {U64.and(U64.bit(U32.to_nat(d)),C.get_kings(b)) == U64.zero() : U64} & {U64.and(U64.bit(U32.to_nat(victim(d))),C.get_kings(b)) == U64.zero() : U64}:
  (empty,kind)=cert
  (Targets.empty_square(b,U32.to_nat(d),good,empty),Rows.decoded_square(b,victim(d),good,victim_bound(d,bound),kind))

def clear(+b: C.Board,+d: U32,good: {Board.valid(b) == True{} : Bool},+bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},valid: {Position.valid_ep(d,b) == True{} : Bool}) ->
  {U64.and(U64.bit(U32.to_nat(d)),C.get_kings(b)) == U64.zero() : U64} & {U64.and(U64.bit(U32.to_nat(victim(d))),C.get_kings(b)) == U64.zero() : U64}:
  clear_known(b,d,good,bound,fields(b,d,bound,valid))
