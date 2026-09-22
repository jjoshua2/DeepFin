import Base
import ../prefix/Order.bend as Earlier
import ../layout/Low.bend as Low
import ../u64/Words.bend as Old

def lt_observed(+a: U32, +b: U32) ->
  {U32.is_lt(a,b) == Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Cmp,Bool,c => Cmp.is_lt(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Earlier.word(32n,x,y))

def nat_lt_le(+a: Nat, +b: Nat, +c: Nat,
  ab: {Nat.is_lt(a,b) == True{} : Bool}, bc: {Nat.is_le(b,c) == True{} : Bool}) ->
  {Nat.is_lt(a,c) == True{} : Bool}:
  match a b c:
    case 0n 0n _: Old.false_true({Nat.is_lt(0n,c) == True{} : Bool},ab)
    case 1n+p 0n _: Old.false_true({Nat.is_lt(1n+p,c) == True{} : Bool},ab)
    case 0n 1n+q 0n: bc
    case 0n 1n+q 1n+r: {==}
    case 1n+p 1n+q 0n: bc
    case 1n+p 1n+q 1n+r: nat_lt_le(p,q,r,ab,bc)

def lt_nat(+a: U32, +b: U32, e: {U32.is_lt(a,b) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),U32.is_lt(a,b),True{},
    Equal.sym(Bool,U32.is_lt(a,b),Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),lt_observed(a,b)),e)

def lt_le(+a: U32, +b: U32, +c: U32,
  ab: {U32.is_lt(a,b) == True{} : Bool}, bc: {U32.is_le(b,c) == True{} : Bool}) ->
  {U32.is_lt(a,c) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(a,c),Nat.is_lt(U32.to_nat(a),U32.to_nat(c)),True{},lt_observed(a,c),
    nat_lt_le(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),lt_nat(a,b,ab),Earlier.to_nat(b,c,bc)))

def le_lt(+a: U32, +b: U32, +c: U32,
  ab: {U32.is_le(a,b) == True{} : Bool}, bc: {U32.is_lt(b,c) == True{} : Bool}) ->
  {U32.is_lt(a,c) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(a,c),Nat.is_lt(U32.to_nat(a),U32.to_nat(c)),True{},lt_observed(a,c),
    Low.le_lt(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),Earlier.to_nat(a,b,ab),lt_nat(b,c,bc)))

def not_equal(c: Cmp, e: {Cmp.is_lt(c) == True{} : Bool}) -> {Bool.not(Cmp.is_eq(c)) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: e
    case GT{}: {==}
