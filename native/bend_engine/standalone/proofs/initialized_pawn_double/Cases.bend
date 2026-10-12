import Base
import ./Spec.bend as S
import ../attack_geometry/Geometry.bend as G
import ../layout/Domain.bend as D
import ../u64/Words.bend as Old

def facts(-s: U32,-w: Bool) -> Type:
  ({U64.test_bit(G.mask(S.kind(w),U32.to_nat(s)),U32.to_nat(U32.xor(s,16))) == False{} : Bool} &
   {U64.test_bit(U64.bit(U32.to_nat(S.first(s,w))),U32.to_nat(U32.xor(s,16))) == False{} : Bool}) &
   {Nat.is_lt(U32.to_nat(U32.xor(s,16)),64n) == True{} : Bool}

def finite_true(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool}) -> facts(s,True{}):
  match n:
    case 0n:
      %Equal.sym(U32,s,0,D.injective(s,0,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 1n:
      %Equal.sym(U32,s,1,D.injective(s,1,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 2n:
      %Equal.sym(U32,s,2,D.injective(s,2,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 3n:
      %Equal.sym(U32,s,3,D.injective(s,3,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 4n:
      %Equal.sym(U32,s,4,D.injective(s,4,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 5n:
      %Equal.sym(U32,s,5,D.injective(s,5,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 6n:
      %Equal.sym(U32,s,6,D.injective(s,6,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 7n:
      %Equal.sym(U32,s,7,D.injective(s,7,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 8n:
      %Equal.sym(U32,s,8,D.injective(s,8,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 9n:
      %Equal.sym(U32,s,9,D.injective(s,9,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 10n:
      %Equal.sym(U32,s,10,D.injective(s,10,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 11n:
      %Equal.sym(U32,s,11,D.injective(s,11,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 12n:
      %Equal.sym(U32,s,12,D.injective(s,12,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 13n:
      %Equal.sym(U32,s,13,D.injective(s,13,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 14n:
      %Equal.sym(U32,s,14,D.injective(s,14,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 15n:
      %Equal.sym(U32,s,15,D.injective(s,15,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 16n:
      %Equal.sym(U32,s,16,D.injective(s,16,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 17n:
      %Equal.sym(U32,s,17,D.injective(s,17,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 18n:
      %Equal.sym(U32,s,18,D.injective(s,18,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 19n:
      %Equal.sym(U32,s,19,D.injective(s,19,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 20n:
      %Equal.sym(U32,s,20,D.injective(s,20,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 21n:
      %Equal.sym(U32,s,21,D.injective(s,21,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 22n:
      %Equal.sym(U32,s,22,D.injective(s,22,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 23n:
      %Equal.sym(U32,s,23,D.injective(s,23,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 24n:
      %Equal.sym(U32,s,24,D.injective(s,24,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 25n:
      %Equal.sym(U32,s,25,D.injective(s,25,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 26n:
      %Equal.sym(U32,s,26,D.injective(s,26,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 27n:
      %Equal.sym(U32,s,27,D.injective(s,27,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 28n:
      %Equal.sym(U32,s,28,D.injective(s,28,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 29n:
      %Equal.sym(U32,s,29,D.injective(s,29,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 30n:
      %Equal.sym(U32,s,30,D.injective(s,30,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 31n:
      %Equal.sym(U32,s,31,D.injective(s,31,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 32n:
      %Equal.sym(U32,s,32,D.injective(s,32,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 33n:
      %Equal.sym(U32,s,33,D.injective(s,33,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 34n:
      %Equal.sym(U32,s,34,D.injective(s,34,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 35n:
      %Equal.sym(U32,s,35,D.injective(s,35,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 36n:
      %Equal.sym(U32,s,36,D.injective(s,36,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 37n:
      %Equal.sym(U32,s,37,D.injective(s,37,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 38n:
      %Equal.sym(U32,s,38,D.injective(s,38,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 39n:
      %Equal.sym(U32,s,39,D.injective(s,39,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 40n:
      %Equal.sym(U32,s,40,D.injective(s,40,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 41n:
      %Equal.sym(U32,s,41,D.injective(s,41,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 42n:
      %Equal.sym(U32,s,42,D.injective(s,42,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 43n:
      %Equal.sym(U32,s,43,D.injective(s,43,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 44n:
      %Equal.sym(U32,s,44,D.injective(s,44,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 45n:
      %Equal.sym(U32,s,45,D.injective(s,45,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 46n:
      %Equal.sym(U32,s,46,D.injective(s,46,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 47n:
      %Equal.sym(U32,s,47,D.injective(s,47,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 48n:
      %Equal.sym(U32,s,48,D.injective(s,48,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 49n:
      %Equal.sym(U32,s,49,D.injective(s,49,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 50n:
      %Equal.sym(U32,s,50,D.injective(s,50,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 51n:
      %Equal.sym(U32,s,51,D.injective(s,51,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 52n:
      %Equal.sym(U32,s,52,D.injective(s,52,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 53n:
      %Equal.sym(U32,s,53,D.injective(s,53,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 54n:
      %Equal.sym(U32,s,54,D.injective(s,54,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 55n:
      %Equal.sym(U32,s,55,D.injective(s,55,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 56n:
      %Equal.sym(U32,s,56,D.injective(s,56,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 57n:
      %Equal.sym(U32,s,57,D.injective(s,57,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 58n:
      %Equal.sym(U32,s,58,D.injective(s,58,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 59n:
      %Equal.sym(U32,s,59,D.injective(s,59,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 60n:
      %Equal.sym(U32,s,60,D.injective(s,60,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 61n:
      %Equal.sym(U32,s,61,D.injective(s,61,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 62n:
      %Equal.sym(U32,s,62,D.injective(s,62,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 63n:
      %Equal.sym(U32,s,63,D.injective(s,63,same)) : facts(_,True{})
      (({==},{==}),{==})
    case 64n+p: D.impossible(facts(s,True{}),p,below)

def finite_false(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool}) -> facts(s,False{}):
  match n:
    case 0n:
      %Equal.sym(U32,s,0,D.injective(s,0,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 1n:
      %Equal.sym(U32,s,1,D.injective(s,1,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 2n:
      %Equal.sym(U32,s,2,D.injective(s,2,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 3n:
      %Equal.sym(U32,s,3,D.injective(s,3,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 4n:
      %Equal.sym(U32,s,4,D.injective(s,4,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 5n:
      %Equal.sym(U32,s,5,D.injective(s,5,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 6n:
      %Equal.sym(U32,s,6,D.injective(s,6,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 7n:
      %Equal.sym(U32,s,7,D.injective(s,7,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 8n:
      %Equal.sym(U32,s,8,D.injective(s,8,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 9n:
      %Equal.sym(U32,s,9,D.injective(s,9,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 10n:
      %Equal.sym(U32,s,10,D.injective(s,10,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 11n:
      %Equal.sym(U32,s,11,D.injective(s,11,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 12n:
      %Equal.sym(U32,s,12,D.injective(s,12,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 13n:
      %Equal.sym(U32,s,13,D.injective(s,13,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 14n:
      %Equal.sym(U32,s,14,D.injective(s,14,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 15n:
      %Equal.sym(U32,s,15,D.injective(s,15,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 16n:
      %Equal.sym(U32,s,16,D.injective(s,16,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 17n:
      %Equal.sym(U32,s,17,D.injective(s,17,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 18n:
      %Equal.sym(U32,s,18,D.injective(s,18,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 19n:
      %Equal.sym(U32,s,19,D.injective(s,19,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 20n:
      %Equal.sym(U32,s,20,D.injective(s,20,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 21n:
      %Equal.sym(U32,s,21,D.injective(s,21,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 22n:
      %Equal.sym(U32,s,22,D.injective(s,22,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 23n:
      %Equal.sym(U32,s,23,D.injective(s,23,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 24n:
      %Equal.sym(U32,s,24,D.injective(s,24,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 25n:
      %Equal.sym(U32,s,25,D.injective(s,25,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 26n:
      %Equal.sym(U32,s,26,D.injective(s,26,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 27n:
      %Equal.sym(U32,s,27,D.injective(s,27,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 28n:
      %Equal.sym(U32,s,28,D.injective(s,28,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 29n:
      %Equal.sym(U32,s,29,D.injective(s,29,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 30n:
      %Equal.sym(U32,s,30,D.injective(s,30,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 31n:
      %Equal.sym(U32,s,31,D.injective(s,31,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 32n:
      %Equal.sym(U32,s,32,D.injective(s,32,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 33n:
      %Equal.sym(U32,s,33,D.injective(s,33,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 34n:
      %Equal.sym(U32,s,34,D.injective(s,34,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 35n:
      %Equal.sym(U32,s,35,D.injective(s,35,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 36n:
      %Equal.sym(U32,s,36,D.injective(s,36,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 37n:
      %Equal.sym(U32,s,37,D.injective(s,37,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 38n:
      %Equal.sym(U32,s,38,D.injective(s,38,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 39n:
      %Equal.sym(U32,s,39,D.injective(s,39,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 40n:
      %Equal.sym(U32,s,40,D.injective(s,40,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 41n:
      %Equal.sym(U32,s,41,D.injective(s,41,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 42n:
      %Equal.sym(U32,s,42,D.injective(s,42,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 43n:
      %Equal.sym(U32,s,43,D.injective(s,43,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 44n:
      %Equal.sym(U32,s,44,D.injective(s,44,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 45n:
      %Equal.sym(U32,s,45,D.injective(s,45,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 46n:
      %Equal.sym(U32,s,46,D.injective(s,46,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 47n:
      %Equal.sym(U32,s,47,D.injective(s,47,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 48n:
      %Equal.sym(U32,s,48,D.injective(s,48,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 49n:
      %Equal.sym(U32,s,49,D.injective(s,49,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 50n:
      %Equal.sym(U32,s,50,D.injective(s,50,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 51n:
      %Equal.sym(U32,s,51,D.injective(s,51,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 52n:
      %Equal.sym(U32,s,52,D.injective(s,52,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 53n:
      %Equal.sym(U32,s,53,D.injective(s,53,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 54n:
      %Equal.sym(U32,s,54,D.injective(s,54,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 55n:
      %Equal.sym(U32,s,55,D.injective(s,55,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 56n:
      %Equal.sym(U32,s,56,D.injective(s,56,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 57n:
      %Equal.sym(U32,s,57,D.injective(s,57,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 58n:
      %Equal.sym(U32,s,58,D.injective(s,58,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 59n:
      %Equal.sym(U32,s,59,D.injective(s,59,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 60n:
      %Equal.sym(U32,s,60,D.injective(s,60,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 61n:
      %Equal.sym(U32,s,61,D.injective(s,61,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 62n:
      %Equal.sym(U32,s,62,D.injective(s,62,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 63n:
      %Equal.sym(U32,s,63,D.injective(s,63,same)) : facts(_,False{})
      (({==},{==}),{==})
    case 64n+p: D.impossible(facts(s,False{}),p,below)

def finite(+s: U32,+n: Nat,w: Bool,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool}) -> facts(s,w):
  match w:
    case True{}: finite_true(s,n,same,below)
    case False{}: finite_false(s,n,same,below)

def recover(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool}) ->
  {U32.from_nat(U32.to_nat(s)) == s : U32}:
  match n:
    case 0n:
      %Equal.sym(U32,s,0,D.injective(s,0,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 1n:
      %Equal.sym(U32,s,1,D.injective(s,1,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 2n:
      %Equal.sym(U32,s,2,D.injective(s,2,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 3n:
      %Equal.sym(U32,s,3,D.injective(s,3,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 4n:
      %Equal.sym(U32,s,4,D.injective(s,4,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 5n:
      %Equal.sym(U32,s,5,D.injective(s,5,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 6n:
      %Equal.sym(U32,s,6,D.injective(s,6,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 7n:
      %Equal.sym(U32,s,7,D.injective(s,7,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 8n:
      %Equal.sym(U32,s,8,D.injective(s,8,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 9n:
      %Equal.sym(U32,s,9,D.injective(s,9,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 10n:
      %Equal.sym(U32,s,10,D.injective(s,10,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 11n:
      %Equal.sym(U32,s,11,D.injective(s,11,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 12n:
      %Equal.sym(U32,s,12,D.injective(s,12,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 13n:
      %Equal.sym(U32,s,13,D.injective(s,13,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 14n:
      %Equal.sym(U32,s,14,D.injective(s,14,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 15n:
      %Equal.sym(U32,s,15,D.injective(s,15,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 16n:
      %Equal.sym(U32,s,16,D.injective(s,16,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 17n:
      %Equal.sym(U32,s,17,D.injective(s,17,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 18n:
      %Equal.sym(U32,s,18,D.injective(s,18,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 19n:
      %Equal.sym(U32,s,19,D.injective(s,19,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 20n:
      %Equal.sym(U32,s,20,D.injective(s,20,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 21n:
      %Equal.sym(U32,s,21,D.injective(s,21,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 22n:
      %Equal.sym(U32,s,22,D.injective(s,22,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 23n:
      %Equal.sym(U32,s,23,D.injective(s,23,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 24n:
      %Equal.sym(U32,s,24,D.injective(s,24,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 25n:
      %Equal.sym(U32,s,25,D.injective(s,25,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 26n:
      %Equal.sym(U32,s,26,D.injective(s,26,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 27n:
      %Equal.sym(U32,s,27,D.injective(s,27,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 28n:
      %Equal.sym(U32,s,28,D.injective(s,28,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 29n:
      %Equal.sym(U32,s,29,D.injective(s,29,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 30n:
      %Equal.sym(U32,s,30,D.injective(s,30,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 31n:
      %Equal.sym(U32,s,31,D.injective(s,31,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 32n:
      %Equal.sym(U32,s,32,D.injective(s,32,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 33n:
      %Equal.sym(U32,s,33,D.injective(s,33,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 34n:
      %Equal.sym(U32,s,34,D.injective(s,34,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 35n:
      %Equal.sym(U32,s,35,D.injective(s,35,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 36n:
      %Equal.sym(U32,s,36,D.injective(s,36,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 37n:
      %Equal.sym(U32,s,37,D.injective(s,37,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 38n:
      %Equal.sym(U32,s,38,D.injective(s,38,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 39n:
      %Equal.sym(U32,s,39,D.injective(s,39,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 40n:
      %Equal.sym(U32,s,40,D.injective(s,40,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 41n:
      %Equal.sym(U32,s,41,D.injective(s,41,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 42n:
      %Equal.sym(U32,s,42,D.injective(s,42,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 43n:
      %Equal.sym(U32,s,43,D.injective(s,43,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 44n:
      %Equal.sym(U32,s,44,D.injective(s,44,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 45n:
      %Equal.sym(U32,s,45,D.injective(s,45,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 46n:
      %Equal.sym(U32,s,46,D.injective(s,46,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 47n:
      %Equal.sym(U32,s,47,D.injective(s,47,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 48n:
      %Equal.sym(U32,s,48,D.injective(s,48,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 49n:
      %Equal.sym(U32,s,49,D.injective(s,49,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 50n:
      %Equal.sym(U32,s,50,D.injective(s,50,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 51n:
      %Equal.sym(U32,s,51,D.injective(s,51,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 52n:
      %Equal.sym(U32,s,52,D.injective(s,52,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 53n:
      %Equal.sym(U32,s,53,D.injective(s,53,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 54n:
      %Equal.sym(U32,s,54,D.injective(s,54,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 55n:
      %Equal.sym(U32,s,55,D.injective(s,55,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 56n:
      %Equal.sym(U32,s,56,D.injective(s,56,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 57n:
      %Equal.sym(U32,s,57,D.injective(s,57,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 58n:
      %Equal.sym(U32,s,58,D.injective(s,58,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 59n:
      %Equal.sym(U32,s,59,D.injective(s,59,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 60n:
      %Equal.sym(U32,s,60,D.injective(s,60,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 61n:
      %Equal.sym(U32,s,61,D.injective(s,61,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 62n:
      %Equal.sym(U32,s,62,D.injective(s,62,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 63n:
      %Equal.sym(U32,s,63,D.injective(s,63,same)) : {U32.from_nat(U32.to_nat(_)) == _ : U32}
      {==}
    case 64n+p: D.impossible({U32.from_nat(U32.to_nat(s)) == s : U32},p,below)

def destination_true(+s: U32,n: Nat,+same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},
  start: {U32.is_eq(U32.div(s,8),S.start(True{})) == True{} : Bool}) -> {U32.xor(s,16) == S.second(s,True{}) : U32}:
  match n:
    case 0n:
      %Equal.sym(U32,s,0,D.injective(s,0,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(0,16) == S.second(0,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),0,s,
            Equal.sym(U32,s,0,D.injective(s,0,same))),start))
    case 1n:
      %Equal.sym(U32,s,1,D.injective(s,1,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(1,16) == S.second(1,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),1,s,
            Equal.sym(U32,s,1,D.injective(s,1,same))),start))
    case 2n:
      %Equal.sym(U32,s,2,D.injective(s,2,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(2,16) == S.second(2,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),2,s,
            Equal.sym(U32,s,2,D.injective(s,2,same))),start))
    case 3n:
      %Equal.sym(U32,s,3,D.injective(s,3,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(3,16) == S.second(3,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),3,s,
            Equal.sym(U32,s,3,D.injective(s,3,same))),start))
    case 4n:
      %Equal.sym(U32,s,4,D.injective(s,4,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(4,16) == S.second(4,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),4,s,
            Equal.sym(U32,s,4,D.injective(s,4,same))),start))
    case 5n:
      %Equal.sym(U32,s,5,D.injective(s,5,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(5,16) == S.second(5,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),5,s,
            Equal.sym(U32,s,5,D.injective(s,5,same))),start))
    case 6n:
      %Equal.sym(U32,s,6,D.injective(s,6,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(6,16) == S.second(6,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),6,s,
            Equal.sym(U32,s,6,D.injective(s,6,same))),start))
    case 7n:
      %Equal.sym(U32,s,7,D.injective(s,7,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(7,16) == S.second(7,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),7,s,
            Equal.sym(U32,s,7,D.injective(s,7,same))),start))
    case 8n:
      %Equal.sym(U32,s,8,D.injective(s,8,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 9n:
      %Equal.sym(U32,s,9,D.injective(s,9,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 10n:
      %Equal.sym(U32,s,10,D.injective(s,10,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 11n:
      %Equal.sym(U32,s,11,D.injective(s,11,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 12n:
      %Equal.sym(U32,s,12,D.injective(s,12,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 13n:
      %Equal.sym(U32,s,13,D.injective(s,13,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 14n:
      %Equal.sym(U32,s,14,D.injective(s,14,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 15n:
      %Equal.sym(U32,s,15,D.injective(s,15,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      {==}
    case 16n:
      %Equal.sym(U32,s,16,D.injective(s,16,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(16,16) == S.second(16,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),16,s,
            Equal.sym(U32,s,16,D.injective(s,16,same))),start))
    case 17n:
      %Equal.sym(U32,s,17,D.injective(s,17,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(17,16) == S.second(17,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),17,s,
            Equal.sym(U32,s,17,D.injective(s,17,same))),start))
    case 18n:
      %Equal.sym(U32,s,18,D.injective(s,18,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(18,16) == S.second(18,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),18,s,
            Equal.sym(U32,s,18,D.injective(s,18,same))),start))
    case 19n:
      %Equal.sym(U32,s,19,D.injective(s,19,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(19,16) == S.second(19,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),19,s,
            Equal.sym(U32,s,19,D.injective(s,19,same))),start))
    case 20n:
      %Equal.sym(U32,s,20,D.injective(s,20,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(20,16) == S.second(20,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),20,s,
            Equal.sym(U32,s,20,D.injective(s,20,same))),start))
    case 21n:
      %Equal.sym(U32,s,21,D.injective(s,21,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(21,16) == S.second(21,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),21,s,
            Equal.sym(U32,s,21,D.injective(s,21,same))),start))
    case 22n:
      %Equal.sym(U32,s,22,D.injective(s,22,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(22,16) == S.second(22,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),22,s,
            Equal.sym(U32,s,22,D.injective(s,22,same))),start))
    case 23n:
      %Equal.sym(U32,s,23,D.injective(s,23,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(23,16) == S.second(23,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),23,s,
            Equal.sym(U32,s,23,D.injective(s,23,same))),start))
    case 24n:
      %Equal.sym(U32,s,24,D.injective(s,24,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(24,16) == S.second(24,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),24,s,
            Equal.sym(U32,s,24,D.injective(s,24,same))),start))
    case 25n:
      %Equal.sym(U32,s,25,D.injective(s,25,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(25,16) == S.second(25,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),25,s,
            Equal.sym(U32,s,25,D.injective(s,25,same))),start))
    case 26n:
      %Equal.sym(U32,s,26,D.injective(s,26,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(26,16) == S.second(26,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),26,s,
            Equal.sym(U32,s,26,D.injective(s,26,same))),start))
    case 27n:
      %Equal.sym(U32,s,27,D.injective(s,27,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(27,16) == S.second(27,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),27,s,
            Equal.sym(U32,s,27,D.injective(s,27,same))),start))
    case 28n:
      %Equal.sym(U32,s,28,D.injective(s,28,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(28,16) == S.second(28,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),28,s,
            Equal.sym(U32,s,28,D.injective(s,28,same))),start))
    case 29n:
      %Equal.sym(U32,s,29,D.injective(s,29,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(29,16) == S.second(29,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),29,s,
            Equal.sym(U32,s,29,D.injective(s,29,same))),start))
    case 30n:
      %Equal.sym(U32,s,30,D.injective(s,30,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(30,16) == S.second(30,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),30,s,
            Equal.sym(U32,s,30,D.injective(s,30,same))),start))
    case 31n:
      %Equal.sym(U32,s,31,D.injective(s,31,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(31,16) == S.second(31,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),31,s,
            Equal.sym(U32,s,31,D.injective(s,31,same))),start))
    case 32n:
      %Equal.sym(U32,s,32,D.injective(s,32,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(32,16) == S.second(32,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),32,s,
            Equal.sym(U32,s,32,D.injective(s,32,same))),start))
    case 33n:
      %Equal.sym(U32,s,33,D.injective(s,33,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(33,16) == S.second(33,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),33,s,
            Equal.sym(U32,s,33,D.injective(s,33,same))),start))
    case 34n:
      %Equal.sym(U32,s,34,D.injective(s,34,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(34,16) == S.second(34,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),34,s,
            Equal.sym(U32,s,34,D.injective(s,34,same))),start))
    case 35n:
      %Equal.sym(U32,s,35,D.injective(s,35,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(35,16) == S.second(35,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),35,s,
            Equal.sym(U32,s,35,D.injective(s,35,same))),start))
    case 36n:
      %Equal.sym(U32,s,36,D.injective(s,36,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(36,16) == S.second(36,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),36,s,
            Equal.sym(U32,s,36,D.injective(s,36,same))),start))
    case 37n:
      %Equal.sym(U32,s,37,D.injective(s,37,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(37,16) == S.second(37,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),37,s,
            Equal.sym(U32,s,37,D.injective(s,37,same))),start))
    case 38n:
      %Equal.sym(U32,s,38,D.injective(s,38,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(38,16) == S.second(38,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),38,s,
            Equal.sym(U32,s,38,D.injective(s,38,same))),start))
    case 39n:
      %Equal.sym(U32,s,39,D.injective(s,39,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(39,16) == S.second(39,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),39,s,
            Equal.sym(U32,s,39,D.injective(s,39,same))),start))
    case 40n:
      %Equal.sym(U32,s,40,D.injective(s,40,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(40,16) == S.second(40,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),40,s,
            Equal.sym(U32,s,40,D.injective(s,40,same))),start))
    case 41n:
      %Equal.sym(U32,s,41,D.injective(s,41,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(41,16) == S.second(41,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),41,s,
            Equal.sym(U32,s,41,D.injective(s,41,same))),start))
    case 42n:
      %Equal.sym(U32,s,42,D.injective(s,42,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(42,16) == S.second(42,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),42,s,
            Equal.sym(U32,s,42,D.injective(s,42,same))),start))
    case 43n:
      %Equal.sym(U32,s,43,D.injective(s,43,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(43,16) == S.second(43,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),43,s,
            Equal.sym(U32,s,43,D.injective(s,43,same))),start))
    case 44n:
      %Equal.sym(U32,s,44,D.injective(s,44,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(44,16) == S.second(44,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),44,s,
            Equal.sym(U32,s,44,D.injective(s,44,same))),start))
    case 45n:
      %Equal.sym(U32,s,45,D.injective(s,45,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(45,16) == S.second(45,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),45,s,
            Equal.sym(U32,s,45,D.injective(s,45,same))),start))
    case 46n:
      %Equal.sym(U32,s,46,D.injective(s,46,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(46,16) == S.second(46,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),46,s,
            Equal.sym(U32,s,46,D.injective(s,46,same))),start))
    case 47n:
      %Equal.sym(U32,s,47,D.injective(s,47,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(47,16) == S.second(47,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),47,s,
            Equal.sym(U32,s,47,D.injective(s,47,same))),start))
    case 48n:
      %Equal.sym(U32,s,48,D.injective(s,48,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(48,16) == S.second(48,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),48,s,
            Equal.sym(U32,s,48,D.injective(s,48,same))),start))
    case 49n:
      %Equal.sym(U32,s,49,D.injective(s,49,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(49,16) == S.second(49,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),49,s,
            Equal.sym(U32,s,49,D.injective(s,49,same))),start))
    case 50n:
      %Equal.sym(U32,s,50,D.injective(s,50,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(50,16) == S.second(50,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),50,s,
            Equal.sym(U32,s,50,D.injective(s,50,same))),start))
    case 51n:
      %Equal.sym(U32,s,51,D.injective(s,51,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(51,16) == S.second(51,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),51,s,
            Equal.sym(U32,s,51,D.injective(s,51,same))),start))
    case 52n:
      %Equal.sym(U32,s,52,D.injective(s,52,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(52,16) == S.second(52,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),52,s,
            Equal.sym(U32,s,52,D.injective(s,52,same))),start))
    case 53n:
      %Equal.sym(U32,s,53,D.injective(s,53,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(53,16) == S.second(53,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),53,s,
            Equal.sym(U32,s,53,D.injective(s,53,same))),start))
    case 54n:
      %Equal.sym(U32,s,54,D.injective(s,54,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(54,16) == S.second(54,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),54,s,
            Equal.sym(U32,s,54,D.injective(s,54,same))),start))
    case 55n:
      %Equal.sym(U32,s,55,D.injective(s,55,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(55,16) == S.second(55,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),55,s,
            Equal.sym(U32,s,55,D.injective(s,55,same))),start))
    case 56n:
      %Equal.sym(U32,s,56,D.injective(s,56,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(56,16) == S.second(56,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),56,s,
            Equal.sym(U32,s,56,D.injective(s,56,same))),start))
    case 57n:
      %Equal.sym(U32,s,57,D.injective(s,57,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(57,16) == S.second(57,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),57,s,
            Equal.sym(U32,s,57,D.injective(s,57,same))),start))
    case 58n:
      %Equal.sym(U32,s,58,D.injective(s,58,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(58,16) == S.second(58,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),58,s,
            Equal.sym(U32,s,58,D.injective(s,58,same))),start))
    case 59n:
      %Equal.sym(U32,s,59,D.injective(s,59,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(59,16) == S.second(59,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),59,s,
            Equal.sym(U32,s,59,D.injective(s,59,same))),start))
    case 60n:
      %Equal.sym(U32,s,60,D.injective(s,60,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(60,16) == S.second(60,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),60,s,
            Equal.sym(U32,s,60,D.injective(s,60,same))),start))
    case 61n:
      %Equal.sym(U32,s,61,D.injective(s,61,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(61,16) == S.second(61,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),61,s,
            Equal.sym(U32,s,61,D.injective(s,61,same))),start))
    case 62n:
      %Equal.sym(U32,s,62,D.injective(s,62,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(62,16) == S.second(62,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),62,s,
            Equal.sym(U32,s,62,D.injective(s,62,same))),start))
    case 63n:
      %Equal.sym(U32,s,63,D.injective(s,63,same)) : {U32.xor(_,16) == S.second(_,True{}) : U32}
      Old.false_true({U32.xor(63,16) == S.second(63,True{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(True{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(True{})),63,s,
            Equal.sym(U32,s,63,D.injective(s,63,same))),start))
    case 64n+p: D.impossible({U32.xor(s,16) == S.second(s,True{}) : U32},p,below)

def destination_false(+s: U32,n: Nat,+same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},
  start: {U32.is_eq(U32.div(s,8),S.start(False{})) == True{} : Bool}) -> {U32.xor(s,16) == S.second(s,False{}) : U32}:
  match n:
    case 0n:
      %Equal.sym(U32,s,0,D.injective(s,0,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(0,16) == S.second(0,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),0,s,
            Equal.sym(U32,s,0,D.injective(s,0,same))),start))
    case 1n:
      %Equal.sym(U32,s,1,D.injective(s,1,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(1,16) == S.second(1,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),1,s,
            Equal.sym(U32,s,1,D.injective(s,1,same))),start))
    case 2n:
      %Equal.sym(U32,s,2,D.injective(s,2,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(2,16) == S.second(2,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),2,s,
            Equal.sym(U32,s,2,D.injective(s,2,same))),start))
    case 3n:
      %Equal.sym(U32,s,3,D.injective(s,3,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(3,16) == S.second(3,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),3,s,
            Equal.sym(U32,s,3,D.injective(s,3,same))),start))
    case 4n:
      %Equal.sym(U32,s,4,D.injective(s,4,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(4,16) == S.second(4,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),4,s,
            Equal.sym(U32,s,4,D.injective(s,4,same))),start))
    case 5n:
      %Equal.sym(U32,s,5,D.injective(s,5,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(5,16) == S.second(5,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),5,s,
            Equal.sym(U32,s,5,D.injective(s,5,same))),start))
    case 6n:
      %Equal.sym(U32,s,6,D.injective(s,6,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(6,16) == S.second(6,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),6,s,
            Equal.sym(U32,s,6,D.injective(s,6,same))),start))
    case 7n:
      %Equal.sym(U32,s,7,D.injective(s,7,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(7,16) == S.second(7,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),7,s,
            Equal.sym(U32,s,7,D.injective(s,7,same))),start))
    case 8n:
      %Equal.sym(U32,s,8,D.injective(s,8,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(8,16) == S.second(8,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),8,s,
            Equal.sym(U32,s,8,D.injective(s,8,same))),start))
    case 9n:
      %Equal.sym(U32,s,9,D.injective(s,9,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(9,16) == S.second(9,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),9,s,
            Equal.sym(U32,s,9,D.injective(s,9,same))),start))
    case 10n:
      %Equal.sym(U32,s,10,D.injective(s,10,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(10,16) == S.second(10,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),10,s,
            Equal.sym(U32,s,10,D.injective(s,10,same))),start))
    case 11n:
      %Equal.sym(U32,s,11,D.injective(s,11,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(11,16) == S.second(11,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),11,s,
            Equal.sym(U32,s,11,D.injective(s,11,same))),start))
    case 12n:
      %Equal.sym(U32,s,12,D.injective(s,12,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(12,16) == S.second(12,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),12,s,
            Equal.sym(U32,s,12,D.injective(s,12,same))),start))
    case 13n:
      %Equal.sym(U32,s,13,D.injective(s,13,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(13,16) == S.second(13,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),13,s,
            Equal.sym(U32,s,13,D.injective(s,13,same))),start))
    case 14n:
      %Equal.sym(U32,s,14,D.injective(s,14,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(14,16) == S.second(14,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),14,s,
            Equal.sym(U32,s,14,D.injective(s,14,same))),start))
    case 15n:
      %Equal.sym(U32,s,15,D.injective(s,15,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(15,16) == S.second(15,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),15,s,
            Equal.sym(U32,s,15,D.injective(s,15,same))),start))
    case 16n:
      %Equal.sym(U32,s,16,D.injective(s,16,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(16,16) == S.second(16,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),16,s,
            Equal.sym(U32,s,16,D.injective(s,16,same))),start))
    case 17n:
      %Equal.sym(U32,s,17,D.injective(s,17,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(17,16) == S.second(17,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),17,s,
            Equal.sym(U32,s,17,D.injective(s,17,same))),start))
    case 18n:
      %Equal.sym(U32,s,18,D.injective(s,18,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(18,16) == S.second(18,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),18,s,
            Equal.sym(U32,s,18,D.injective(s,18,same))),start))
    case 19n:
      %Equal.sym(U32,s,19,D.injective(s,19,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(19,16) == S.second(19,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),19,s,
            Equal.sym(U32,s,19,D.injective(s,19,same))),start))
    case 20n:
      %Equal.sym(U32,s,20,D.injective(s,20,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(20,16) == S.second(20,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),20,s,
            Equal.sym(U32,s,20,D.injective(s,20,same))),start))
    case 21n:
      %Equal.sym(U32,s,21,D.injective(s,21,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(21,16) == S.second(21,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),21,s,
            Equal.sym(U32,s,21,D.injective(s,21,same))),start))
    case 22n:
      %Equal.sym(U32,s,22,D.injective(s,22,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(22,16) == S.second(22,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),22,s,
            Equal.sym(U32,s,22,D.injective(s,22,same))),start))
    case 23n:
      %Equal.sym(U32,s,23,D.injective(s,23,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(23,16) == S.second(23,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),23,s,
            Equal.sym(U32,s,23,D.injective(s,23,same))),start))
    case 24n:
      %Equal.sym(U32,s,24,D.injective(s,24,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(24,16) == S.second(24,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),24,s,
            Equal.sym(U32,s,24,D.injective(s,24,same))),start))
    case 25n:
      %Equal.sym(U32,s,25,D.injective(s,25,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(25,16) == S.second(25,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),25,s,
            Equal.sym(U32,s,25,D.injective(s,25,same))),start))
    case 26n:
      %Equal.sym(U32,s,26,D.injective(s,26,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(26,16) == S.second(26,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),26,s,
            Equal.sym(U32,s,26,D.injective(s,26,same))),start))
    case 27n:
      %Equal.sym(U32,s,27,D.injective(s,27,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(27,16) == S.second(27,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),27,s,
            Equal.sym(U32,s,27,D.injective(s,27,same))),start))
    case 28n:
      %Equal.sym(U32,s,28,D.injective(s,28,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(28,16) == S.second(28,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),28,s,
            Equal.sym(U32,s,28,D.injective(s,28,same))),start))
    case 29n:
      %Equal.sym(U32,s,29,D.injective(s,29,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(29,16) == S.second(29,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),29,s,
            Equal.sym(U32,s,29,D.injective(s,29,same))),start))
    case 30n:
      %Equal.sym(U32,s,30,D.injective(s,30,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(30,16) == S.second(30,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),30,s,
            Equal.sym(U32,s,30,D.injective(s,30,same))),start))
    case 31n:
      %Equal.sym(U32,s,31,D.injective(s,31,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(31,16) == S.second(31,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),31,s,
            Equal.sym(U32,s,31,D.injective(s,31,same))),start))
    case 32n:
      %Equal.sym(U32,s,32,D.injective(s,32,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(32,16) == S.second(32,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),32,s,
            Equal.sym(U32,s,32,D.injective(s,32,same))),start))
    case 33n:
      %Equal.sym(U32,s,33,D.injective(s,33,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(33,16) == S.second(33,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),33,s,
            Equal.sym(U32,s,33,D.injective(s,33,same))),start))
    case 34n:
      %Equal.sym(U32,s,34,D.injective(s,34,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(34,16) == S.second(34,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),34,s,
            Equal.sym(U32,s,34,D.injective(s,34,same))),start))
    case 35n:
      %Equal.sym(U32,s,35,D.injective(s,35,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(35,16) == S.second(35,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),35,s,
            Equal.sym(U32,s,35,D.injective(s,35,same))),start))
    case 36n:
      %Equal.sym(U32,s,36,D.injective(s,36,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(36,16) == S.second(36,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),36,s,
            Equal.sym(U32,s,36,D.injective(s,36,same))),start))
    case 37n:
      %Equal.sym(U32,s,37,D.injective(s,37,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(37,16) == S.second(37,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),37,s,
            Equal.sym(U32,s,37,D.injective(s,37,same))),start))
    case 38n:
      %Equal.sym(U32,s,38,D.injective(s,38,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(38,16) == S.second(38,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),38,s,
            Equal.sym(U32,s,38,D.injective(s,38,same))),start))
    case 39n:
      %Equal.sym(U32,s,39,D.injective(s,39,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(39,16) == S.second(39,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),39,s,
            Equal.sym(U32,s,39,D.injective(s,39,same))),start))
    case 40n:
      %Equal.sym(U32,s,40,D.injective(s,40,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(40,16) == S.second(40,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),40,s,
            Equal.sym(U32,s,40,D.injective(s,40,same))),start))
    case 41n:
      %Equal.sym(U32,s,41,D.injective(s,41,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(41,16) == S.second(41,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),41,s,
            Equal.sym(U32,s,41,D.injective(s,41,same))),start))
    case 42n:
      %Equal.sym(U32,s,42,D.injective(s,42,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(42,16) == S.second(42,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),42,s,
            Equal.sym(U32,s,42,D.injective(s,42,same))),start))
    case 43n:
      %Equal.sym(U32,s,43,D.injective(s,43,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(43,16) == S.second(43,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),43,s,
            Equal.sym(U32,s,43,D.injective(s,43,same))),start))
    case 44n:
      %Equal.sym(U32,s,44,D.injective(s,44,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(44,16) == S.second(44,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),44,s,
            Equal.sym(U32,s,44,D.injective(s,44,same))),start))
    case 45n:
      %Equal.sym(U32,s,45,D.injective(s,45,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(45,16) == S.second(45,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),45,s,
            Equal.sym(U32,s,45,D.injective(s,45,same))),start))
    case 46n:
      %Equal.sym(U32,s,46,D.injective(s,46,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(46,16) == S.second(46,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),46,s,
            Equal.sym(U32,s,46,D.injective(s,46,same))),start))
    case 47n:
      %Equal.sym(U32,s,47,D.injective(s,47,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(47,16) == S.second(47,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),47,s,
            Equal.sym(U32,s,47,D.injective(s,47,same))),start))
    case 48n:
      %Equal.sym(U32,s,48,D.injective(s,48,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 49n:
      %Equal.sym(U32,s,49,D.injective(s,49,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 50n:
      %Equal.sym(U32,s,50,D.injective(s,50,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 51n:
      %Equal.sym(U32,s,51,D.injective(s,51,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 52n:
      %Equal.sym(U32,s,52,D.injective(s,52,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 53n:
      %Equal.sym(U32,s,53,D.injective(s,53,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 54n:
      %Equal.sym(U32,s,54,D.injective(s,54,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 55n:
      %Equal.sym(U32,s,55,D.injective(s,55,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      {==}
    case 56n:
      %Equal.sym(U32,s,56,D.injective(s,56,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(56,16) == S.second(56,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),56,s,
            Equal.sym(U32,s,56,D.injective(s,56,same))),start))
    case 57n:
      %Equal.sym(U32,s,57,D.injective(s,57,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(57,16) == S.second(57,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),57,s,
            Equal.sym(U32,s,57,D.injective(s,57,same))),start))
    case 58n:
      %Equal.sym(U32,s,58,D.injective(s,58,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(58,16) == S.second(58,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),58,s,
            Equal.sym(U32,s,58,D.injective(s,58,same))),start))
    case 59n:
      %Equal.sym(U32,s,59,D.injective(s,59,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(59,16) == S.second(59,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),59,s,
            Equal.sym(U32,s,59,D.injective(s,59,same))),start))
    case 60n:
      %Equal.sym(U32,s,60,D.injective(s,60,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(60,16) == S.second(60,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),60,s,
            Equal.sym(U32,s,60,D.injective(s,60,same))),start))
    case 61n:
      %Equal.sym(U32,s,61,D.injective(s,61,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(61,16) == S.second(61,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),61,s,
            Equal.sym(U32,s,61,D.injective(s,61,same))),start))
    case 62n:
      %Equal.sym(U32,s,62,D.injective(s,62,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(62,16) == S.second(62,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),62,s,
            Equal.sym(U32,s,62,D.injective(s,62,same))),start))
    case 63n:
      %Equal.sym(U32,s,63,D.injective(s,63,same)) : {U32.xor(_,16) == S.second(_,False{}) : U32}
      Old.false_true({U32.xor(63,16) == S.second(63,False{}) : U32},
        Equal.trans(Bool,False{},U32.is_eq(U32.div(s,8),S.start(False{})),True{},
          Equal.cong(U32,Bool,x => U32.is_eq(U32.div(x,8),S.start(False{})),63,s,
            Equal.sym(U32,s,63,D.injective(s,63,same))),start))
    case 64n+p: D.impossible({U32.xor(s,16) == S.second(s,False{}) : U32},p,below)

def destination(+s: U32,+n: Nat,w: Bool,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},
  start: {U32.is_eq(U32.div(s,8),S.start(w)) == True{} : Bool}) -> {U32.xor(s,16) == S.second(s,w) : U32}:
  match w:
    case True{}: destination_true(s,n,same,below,start)
    case False{}: destination_false(s,n,same,below,start)
