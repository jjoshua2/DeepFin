# Header addresses observed as mathematical values; no pair enumeration.
import Base
import ./Facts.bend as F
import ../layout/Domain.bend as D
import ../layout/Low.bend as Low
import ../prefix/Nats.bend as N
import ../relative/Order.bend as Order
import ../u64/Words.bend as Old

def cert(+k: Nat) -> Type:
  {U32.to_nat(F.offset(k)) == Nat.add(128n,k) : Nat}
  & {U32.is_lt(F.key(k),128) == True{} : Bool}
  & {U32.is_le(128,F.offset(k)) == True{} : Bool}

def facts(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool}) -> cert(k):
  match k:
    case 0n: ({==},({==},{==}))
    case 1n: ({==},({==},{==}))
    case 2n: ({==},({==},{==}))
    case 3n: ({==},({==},{==}))
    case 4n: ({==},({==},{==}))
    case 5n: ({==},({==},{==}))
    case 6n: ({==},({==},{==}))
    case 7n: ({==},({==},{==}))
    case 8n: ({==},({==},{==}))
    case 9n: ({==},({==},{==}))
    case 10n: ({==},({==},{==}))
    case 11n: ({==},({==},{==}))
    case 12n: ({==},({==},{==}))
    case 13n: ({==},({==},{==}))
    case 14n: ({==},({==},{==}))
    case 15n: ({==},({==},{==}))
    case 16n: ({==},({==},{==}))
    case 17n: ({==},({==},{==}))
    case 18n: ({==},({==},{==}))
    case 19n: ({==},({==},{==}))
    case 20n: ({==},({==},{==}))
    case 21n: ({==},({==},{==}))
    case 22n: ({==},({==},{==}))
    case 23n: ({==},({==},{==}))
    case 24n: ({==},({==},{==}))
    case 25n: ({==},({==},{==}))
    case 26n: ({==},({==},{==}))
    case 27n: ({==},({==},{==}))
    case 28n: ({==},({==},{==}))
    case 29n: ({==},({==},{==}))
    case 30n: ({==},({==},{==}))
    case 31n: ({==},({==},{==}))
    case 32n: ({==},({==},{==}))
    case 33n: ({==},({==},{==}))
    case 34n: ({==},({==},{==}))
    case 35n: ({==},({==},{==}))
    case 36n: ({==},({==},{==}))
    case 37n: ({==},({==},{==}))
    case 38n: ({==},({==},{==}))
    case 39n: ({==},({==},{==}))
    case 40n: ({==},({==},{==}))
    case 41n: ({==},({==},{==}))
    case 42n: ({==},({==},{==}))
    case 43n: ({==},({==},{==}))
    case 44n: ({==},({==},{==}))
    case 45n: ({==},({==},{==}))
    case 46n: ({==},({==},{==}))
    case 47n: ({==},({==},{==}))
    case 48n: ({==},({==},{==}))
    case 49n: ({==},({==},{==}))
    case 50n: ({==},({==},{==}))
    case 51n: ({==},({==},{==}))
    case 52n: ({==},({==},{==}))
    case 53n: ({==},({==},{==}))
    case 54n: ({==},({==},{==}))
    case 55n: ({==},({==},{==}))
    case 56n: ({==},({==},{==}))
    case 57n: ({==},({==},{==}))
    case 58n: ({==},({==},{==}))
    case 59n: ({==},({==},{==}))
    case 60n: ({==},({==},{==}))
    case 61n: ({==},({==},{==}))
    case 62n: ({==},({==},{==}))
    case 63n: ({==},({==},{==}))
    case 64n: ({==},({==},{==}))
    case 65n: ({==},({==},{==}))
    case 66n: ({==},({==},{==}))
    case 67n: ({==},({==},{==}))
    case 68n: ({==},({==},{==}))
    case 69n: ({==},({==},{==}))
    case 70n: ({==},({==},{==}))
    case 71n: ({==},({==},{==}))
    case 72n: ({==},({==},{==}))
    case 73n: ({==},({==},{==}))
    case 74n: ({==},({==},{==}))
    case 75n: ({==},({==},{==}))
    case 76n: ({==},({==},{==}))
    case 77n: ({==},({==},{==}))
    case 78n: ({==},({==},{==}))
    case 79n: ({==},({==},{==}))
    case 80n: ({==},({==},{==}))
    case 81n: ({==},({==},{==}))
    case 82n: ({==},({==},{==}))
    case 83n: ({==},({==},{==}))
    case 84n: ({==},({==},{==}))
    case 85n: ({==},({==},{==}))
    case 86n: ({==},({==},{==}))
    case 87n: ({==},({==},{==}))
    case 88n: ({==},({==},{==}))
    case 89n: ({==},({==},{==}))
    case 90n: ({==},({==},{==}))
    case 91n: ({==},({==},{==}))
    case 92n: ({==},({==},{==}))
    case 93n: ({==},({==},{==}))
    case 94n: ({==},({==},{==}))
    case 95n: ({==},({==},{==}))
    case 96n: ({==},({==},{==}))
    case 97n: ({==},({==},{==}))
    case 98n: ({==},({==},{==}))
    case 99n: ({==},({==},{==}))
    case 100n: ({==},({==},{==}))
    case 101n: ({==},({==},{==}))
    case 102n: ({==},({==},{==}))
    case 103n: ({==},({==},{==}))
    case 104n: ({==},({==},{==}))
    case 105n: ({==},({==},{==}))
    case 106n: ({==},({==},{==}))
    case 107n: ({==},({==},{==}))
    case 108n: ({==},({==},{==}))
    case 109n: ({==},({==},{==}))
    case 110n: ({==},({==},{==}))
    case 111n: ({==},({==},{==}))
    case 112n: ({==},({==},{==}))
    case 113n: ({==},({==},{==}))
    case 114n: ({==},({==},{==}))
    case 115n: ({==},({==},{==}))
    case 116n: ({==},({==},{==}))
    case 117n: ({==},({==},{==}))
    case 118n: ({==},({==},{==}))
    case 119n: ({==},({==},{==}))
    case 120n: ({==},({==},{==}))
    case 121n: ({==},({==},{==}))
    case 122n: ({==},({==},{==}))
    case 123n: ({==},({==},{==}))
    case 124n: ({==},({==},{==}))
    case 125n: ({==},({==},{==}))
    case 126n: ({==},({==},{==}))
    case 127n: ({==},({==},{==}))
    case 128n+p: D.impossible(cert(128n+p),p,valid)

def offset_value(+k: Nat,p: cert(k)) -> {U32.to_nat(F.offset(k)) == Nat.add(128n,k) : Nat}:
  (value,rest) = p
  value

def key_small(+k: Nat,p: cert(k)) -> {U32.is_lt(F.key(k),128) == True{} : Bool}:
  (value,(lo,hi)) = p
  lo

def offset_large(+k: Nat,p: cert(k)) -> {U32.is_le(128,F.offset(k)) == True{} : Bool}:
  (value,(lo,hi)) = p
  hi

def nat_ne(+i: Nat,+j: Nat,e: {Nat.is_lt(i,j) == True{} : Bool}) ->
  {Nat.is_eq(j,i) == False{} : Bool}:
  match i j:
    case 0n 0n: Equal.sym(Bool,False{},True{},e)
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: nat_ne(p,q,e)

def eq_observed(+a: U32,+b: U32) ->
  {U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))

def ne_values(+a: U32,+b: U32,+x: Nat,+y: Nat,
  ax: {U32.to_nat(a) == x : Nat},by: {U32.to_nat(b) == y : Nat},
  ne: {Nat.is_eq(x,y) == False{} : Bool}) -> {U32.is_eq(a,b) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(a,b),Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),False{},eq_observed(a,b),
    Equal.trans(Bool,Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),Nat.is_eq(x,U32.to_nat(b)),False{},
      Equal.cong(Nat,Bool,n => Nat.is_eq(n,U32.to_nat(b)),U32.to_nat(a),x,ax),
      Equal.trans(Bool,Nat.is_eq(x,U32.to_nat(b)),Nat.is_eq(x,y),False{},
        Equal.cong(Nat,Bool,n => Nat.is_eq(x,n),U32.to_nat(b),y,by),ne)))

def nat_sym(+a: Nat,+b: Nat) -> {Nat.is_eq(a,b) == Nat.is_eq(b,a) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: nat_sym(p,q)

def ne_flip(+a: U32,+b: U32,e: {U32.is_eq(a,b) == False{} : Bool}) ->
  {U32.is_eq(b,a) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(b,a),Nat.is_eq(U32.to_nat(b),U32.to_nat(a)),False{},eq_observed(b,a),
    Equal.trans(Bool,Nat.is_eq(U32.to_nat(b),U32.to_nat(a)),Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),False{},
      nat_sym(U32.to_nat(b),U32.to_nat(a)),
      Equal.trans(Bool,Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),U32.is_eq(a,b),False{},
        Equal.sym(Bool,U32.is_eq(a,b),Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),eq_observed(a,b)),e)))

def query(+j: Nat,prefix: Bool) -> U32:
  Bool.pick(U32,prefix,F.offset(j),F.key(j))

def query_bound(+j: Nat,prefix: Bool,valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {U32.is_lt(query(j,prefix),256) == True{} : Bool}:
  match prefix:
    case False{}: F.key_bound(j,F.facts(j,valid))
    case True{}: F.offset_bound(j,F.facts(j,valid))

def distinct(+j: Nat,+k: Nat,+prefix: Bool,
  +earlier: {Nat.is_lt(j,k) == True{} : Bool},+valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.is_eq(F.key(k),query(j,prefix)) == False{} : Bool}
  & {U32.is_eq(F.offset(k),query(j,prefix)) == False{} : Bool}:
  match prefix:
    case False{}:
      (ne_values(F.key(k),F.key(j),k,j,D.exact(k,valid),D.exact(j,Low.le_lt(j,k,128n,N.lt_to_le(j,k,earlier),valid)),nat_ne(j,k,earlier)),
        ne_flip(F.key(j),F.offset(k),Order.lt_ne(F.key(j),F.offset(k),
          Order.lt_le(F.key(j),128,F.offset(k),key_small(j,facts(j,Low.le_lt(j,k,128n,N.lt_to_le(j,k,earlier),valid))),offset_large(k,facts(k,valid))))))
    case True{}:
      (Order.lt_ne(F.key(k),F.offset(j),Order.lt_le(F.key(k),128,F.offset(j),key_small(k,facts(k,valid)),offset_large(j,facts(j,Low.le_lt(j,k,128n,N.lt_to_le(j,k,earlier),valid))))),
        ne_values(F.offset(k),F.offset(j),Nat.add(128n,k),Nat.add(128n,j),
          offset_value(k,facts(k,valid)),offset_value(j,facts(j,Low.le_lt(j,k,128n,N.lt_to_le(j,k,earlier),valid))),nat_ne(j,k,earlier)))
