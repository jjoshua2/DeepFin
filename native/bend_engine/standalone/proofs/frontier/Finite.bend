# Bounded scalar eliminators with the existing injective Nat observation.
import Base
import ./Logic.bend as L
import ../layout/Domain.bend as Domain
import ../relative/Order.bend as Order

def extend6(+n: Nat,e: {Nat.is_lt(n,6n) == True{} : Bool}) -> {Nat.is_lt(n,128n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n+p: Domain.impossible({Nat.is_lt(6n+p,128n) == True{} : Bool},p,e)

def u6(+v: U32,+bound: {Nat.is_lt(U32.to_nat(v),6n) == True{} : Bool},-T: U32 -> Type,
  rule: @+n: Nat -> {Nat.is_lt(n,6n) == True{} : Bool} -> T(U32.from_nat(n))) -> T(v):
  %Domain.recover(v,extend6(U32.to_nat(v),bound)) : T(_)
  rule(U32.to_nat(v))(bound)

def extend8(+n: Nat,e: {Nat.is_lt(n,8n) == True{} : Bool}) -> {Nat.is_lt(n,128n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: Domain.impossible({Nat.is_lt(8n+p,128n) == True{} : Bool},p,e)

def u8(+v: U32,+bound: {Nat.is_lt(U32.to_nat(v),8n) == True{} : Bool},-T: U32 -> Type,
  rule: @+n: Nat -> {Nat.is_lt(n,8n) == True{} : Bool} -> T(U32.from_nat(n))) -> T(v):
  %Domain.recover(v,extend8(U32.to_nat(v),bound)) : T(_)
  rule(U32.to_nat(v))(bound)

def extend9(+n: Nat,e: {Nat.is_lt(n,9n) == True{} : Bool}) -> {Nat.is_lt(n,128n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n+p: Domain.impossible({Nat.is_lt(9n+p,128n) == True{} : Bool},p,e)

def u9(+v: U32,+bound: {Nat.is_lt(U32.to_nat(v),9n) == True{} : Bool},-T: U32 -> Type,
  rule: @+n: Nat -> {Nat.is_lt(n,9n) == True{} : Bool} -> T(U32.from_nat(n))) -> T(v):
  %Domain.recover(v,extend9(U32.to_nat(v),bound)) : T(_)
  rule(U32.to_nat(v))(bound)

def extend57(+n: Nat,e: {Nat.is_le(n,56n) == True{} : Bool}) -> {Nat.is_lt(n,128n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n+p: L.impossible({Nat.is_lt(57n+p,128n) == True{} : Bool},e)

def u57(+v: U32,+bound: {Nat.is_le(U32.to_nat(v),56n) == True{} : Bool},-T: U32 -> Type,
  rule: @+n: Nat -> {Nat.is_le(n,56n) == True{} : Bool} -> T(U32.from_nat(n))) -> T(v):
  %Domain.recover(v,extend57(U32.to_nat(v),bound)) : T(_)
  rule(U32.to_nat(v))(bound)

def lt_nat(+a: U32,+b: U32,e: {U32.is_lt(a,b) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,Nat.is_lt(Word.to_nat(32n,x),Word.to_nat(32n,y)),U32.is_lt(U32{x},U32{y}),True{},
        Equal.sym(Bool,U32.is_lt(U32{x},U32{y}),Nat.is_lt(Word.to_nat(32n,x),Word.to_nat(32n,y)),
          Equal.cong(Cmp,Bool,c => Cmp.is_lt(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))),e)

def le_nat(+a: U32,+b: U32,e: {U32.is_le(a,b) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,Nat.is_le(Word.to_nat(32n,x),Word.to_nat(32n,y)),U32.is_le(U32{x},U32{y}),True{},
        Equal.sym(Bool,U32.is_le(U32{x},U32{y}),Nat.is_le(Word.to_nat(32n,x),Word.to_nat(32n,y)),
          Equal.cong(Cmp,Bool,c => Cmp.is_le(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))),e)
