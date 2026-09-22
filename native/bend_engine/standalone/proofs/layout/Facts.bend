# Small finite geometric facts; no normalization of the runtime ray builder here.
import Base
import ./Spec.bend as Spec
import ./Geometry.bend as Geometry
import ./Domain.bend as Domain

def facts(+n: Nat, +key: U32) -> Type:
  {Nat.is_lt(Geometry.population(n),13n) == True{} : Bool}
  & {U32.to_nat(U32.from_nat(Geometry.population(n))) == Geometry.population(n) : Nat}
  & {U32.is_le(U32.from_nat(Geometry.population(n)),Spec.max_population(key)) == True{} : Bool}
  & {U32.is_le(U32.shln(1,Geometry.population(n)),Spec.max_size(key)) == True{} : Bool}

def finite(+n: Nat, valid: {Nat.is_lt(n,128n) == True{} : Bool}) -> facts(n,U32.from_nat(n)):
  match n:
    case 0n: ({==}, ({==}, ({==}, {==})))
    case 1n: ({==}, ({==}, ({==}, {==})))
    case 2n: ({==}, ({==}, ({==}, {==})))
    case 3n: ({==}, ({==}, ({==}, {==})))
    case 4n: ({==}, ({==}, ({==}, {==})))
    case 5n: ({==}, ({==}, ({==}, {==})))
    case 6n: ({==}, ({==}, ({==}, {==})))
    case 7n: ({==}, ({==}, ({==}, {==})))
    case 8n: ({==}, ({==}, ({==}, {==})))
    case 9n: ({==}, ({==}, ({==}, {==})))
    case 10n: ({==}, ({==}, ({==}, {==})))
    case 11n: ({==}, ({==}, ({==}, {==})))
    case 12n: ({==}, ({==}, ({==}, {==})))
    case 13n: ({==}, ({==}, ({==}, {==})))
    case 14n: ({==}, ({==}, ({==}, {==})))
    case 15n: ({==}, ({==}, ({==}, {==})))
    case 16n: ({==}, ({==}, ({==}, {==})))
    case 17n: ({==}, ({==}, ({==}, {==})))
    case 18n: ({==}, ({==}, ({==}, {==})))
    case 19n: ({==}, ({==}, ({==}, {==})))
    case 20n: ({==}, ({==}, ({==}, {==})))
    case 21n: ({==}, ({==}, ({==}, {==})))
    case 22n: ({==}, ({==}, ({==}, {==})))
    case 23n: ({==}, ({==}, ({==}, {==})))
    case 24n: ({==}, ({==}, ({==}, {==})))
    case 25n: ({==}, ({==}, ({==}, {==})))
    case 26n: ({==}, ({==}, ({==}, {==})))
    case 27n: ({==}, ({==}, ({==}, {==})))
    case 28n: ({==}, ({==}, ({==}, {==})))
    case 29n: ({==}, ({==}, ({==}, {==})))
    case 30n: ({==}, ({==}, ({==}, {==})))
    case 31n: ({==}, ({==}, ({==}, {==})))
    case 32n: ({==}, ({==}, ({==}, {==})))
    case 33n: ({==}, ({==}, ({==}, {==})))
    case 34n: ({==}, ({==}, ({==}, {==})))
    case 35n: ({==}, ({==}, ({==}, {==})))
    case 36n: ({==}, ({==}, ({==}, {==})))
    case 37n: ({==}, ({==}, ({==}, {==})))
    case 38n: ({==}, ({==}, ({==}, {==})))
    case 39n: ({==}, ({==}, ({==}, {==})))
    case 40n: ({==}, ({==}, ({==}, {==})))
    case 41n: ({==}, ({==}, ({==}, {==})))
    case 42n: ({==}, ({==}, ({==}, {==})))
    case 43n: ({==}, ({==}, ({==}, {==})))
    case 44n: ({==}, ({==}, ({==}, {==})))
    case 45n: ({==}, ({==}, ({==}, {==})))
    case 46n: ({==}, ({==}, ({==}, {==})))
    case 47n: ({==}, ({==}, ({==}, {==})))
    case 48n: ({==}, ({==}, ({==}, {==})))
    case 49n: ({==}, ({==}, ({==}, {==})))
    case 50n: ({==}, ({==}, ({==}, {==})))
    case 51n: ({==}, ({==}, ({==}, {==})))
    case 52n: ({==}, ({==}, ({==}, {==})))
    case 53n: ({==}, ({==}, ({==}, {==})))
    case 54n: ({==}, ({==}, ({==}, {==})))
    case 55n: ({==}, ({==}, ({==}, {==})))
    case 56n: ({==}, ({==}, ({==}, {==})))
    case 57n: ({==}, ({==}, ({==}, {==})))
    case 58n: ({==}, ({==}, ({==}, {==})))
    case 59n: ({==}, ({==}, ({==}, {==})))
    case 60n: ({==}, ({==}, ({==}, {==})))
    case 61n: ({==}, ({==}, ({==}, {==})))
    case 62n: ({==}, ({==}, ({==}, {==})))
    case 63n: ({==}, ({==}, ({==}, {==})))
    case 64n: ({==}, ({==}, ({==}, {==})))
    case 65n: ({==}, ({==}, ({==}, {==})))
    case 66n: ({==}, ({==}, ({==}, {==})))
    case 67n: ({==}, ({==}, ({==}, {==})))
    case 68n: ({==}, ({==}, ({==}, {==})))
    case 69n: ({==}, ({==}, ({==}, {==})))
    case 70n: ({==}, ({==}, ({==}, {==})))
    case 71n: ({==}, ({==}, ({==}, {==})))
    case 72n: ({==}, ({==}, ({==}, {==})))
    case 73n: ({==}, ({==}, ({==}, {==})))
    case 74n: ({==}, ({==}, ({==}, {==})))
    case 75n: ({==}, ({==}, ({==}, {==})))
    case 76n: ({==}, ({==}, ({==}, {==})))
    case 77n: ({==}, ({==}, ({==}, {==})))
    case 78n: ({==}, ({==}, ({==}, {==})))
    case 79n: ({==}, ({==}, ({==}, {==})))
    case 80n: ({==}, ({==}, ({==}, {==})))
    case 81n: ({==}, ({==}, ({==}, {==})))
    case 82n: ({==}, ({==}, ({==}, {==})))
    case 83n: ({==}, ({==}, ({==}, {==})))
    case 84n: ({==}, ({==}, ({==}, {==})))
    case 85n: ({==}, ({==}, ({==}, {==})))
    case 86n: ({==}, ({==}, ({==}, {==})))
    case 87n: ({==}, ({==}, ({==}, {==})))
    case 88n: ({==}, ({==}, ({==}, {==})))
    case 89n: ({==}, ({==}, ({==}, {==})))
    case 90n: ({==}, ({==}, ({==}, {==})))
    case 91n: ({==}, ({==}, ({==}, {==})))
    case 92n: ({==}, ({==}, ({==}, {==})))
    case 93n: ({==}, ({==}, ({==}, {==})))
    case 94n: ({==}, ({==}, ({==}, {==})))
    case 95n: ({==}, ({==}, ({==}, {==})))
    case 96n: ({==}, ({==}, ({==}, {==})))
    case 97n: ({==}, ({==}, ({==}, {==})))
    case 98n: ({==}, ({==}, ({==}, {==})))
    case 99n: ({==}, ({==}, ({==}, {==})))
    case 100n: ({==}, ({==}, ({==}, {==})))
    case 101n: ({==}, ({==}, ({==}, {==})))
    case 102n: ({==}, ({==}, ({==}, {==})))
    case 103n: ({==}, ({==}, ({==}, {==})))
    case 104n: ({==}, ({==}, ({==}, {==})))
    case 105n: ({==}, ({==}, ({==}, {==})))
    case 106n: ({==}, ({==}, ({==}, {==})))
    case 107n: ({==}, ({==}, ({==}, {==})))
    case 108n: ({==}, ({==}, ({==}, {==})))
    case 109n: ({==}, ({==}, ({==}, {==})))
    case 110n: ({==}, ({==}, ({==}, {==})))
    case 111n: ({==}, ({==}, ({==}, {==})))
    case 112n: ({==}, ({==}, ({==}, {==})))
    case 113n: ({==}, ({==}, ({==}, {==})))
    case 114n: ({==}, ({==}, ({==}, {==})))
    case 115n: ({==}, ({==}, ({==}, {==})))
    case 116n: ({==}, ({==}, ({==}, {==})))
    case 117n: ({==}, ({==}, ({==}, {==})))
    case 118n: ({==}, ({==}, ({==}, {==})))
    case 119n: ({==}, ({==}, ({==}, {==})))
    case 120n: ({==}, ({==}, ({==}, {==})))
    case 121n: ({==}, ({==}, ({==}, {==})))
    case 122n: ({==}, ({==}, ({==}, {==})))
    case 123n: ({==}, ({==}, ({==}, {==})))
    case 124n: ({==}, ({==}, ({==}, {==})))
    case 125n: ({==}, ({==}, ({==}, {==})))
    case 126n: ({==}, ({==}, ({==}, {==})))
    case 127n: ({==}, ({==}, ({==}, {==})))
    case 128n+p: Domain.impossible(facts(128n+p,U32.from_nat(128n+p)),p,valid)

def actual(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool}) -> facts(U32.to_nat(key),key):
  %Domain.recover(key,valid) : facts(U32.to_nat(key),_)
  finite(U32.to_nat(key),valid)

def shift_facts(+n: Nat) -> Type:
  {U32.to_nat(U32.shln(1,n)) == Nat.pow(2n,n) : Nat}
  & {U32.is_gt(U32.shln(1,n),0) == True{} : Bool}
  & {U32.is_lt(U32.from_nat(n),32) == True{} : Bool}

def shift(+n: Nat, valid: {Nat.is_lt(n,13n) == True{} : Bool}) -> shift_facts(n):
  match n:
    case 0n: ({==}, ({==}, {==}))
    case 1n: ({==}, ({==}, {==}))
    case 2n: ({==}, ({==}, {==}))
    case 3n: ({==}, ({==}, {==}))
    case 4n: ({==}, ({==}, {==}))
    case 5n: ({==}, ({==}, {==}))
    case 6n: ({==}, ({==}, {==}))
    case 7n: ({==}, ({==}, {==}))
    case 8n: ({==}, ({==}, {==}))
    case 9n: ({==}, ({==}, {==}))
    case 10n: ({==}, ({==}, {==}))
    case 11n: ({==}, ({==}, {==}))
    case 12n: ({==}, ({==}, {==}))
    case 13n+p: Domain.impossible(shift_facts(13n+p),p,valid)

def small(n: Nat, key: U32, f: facts(n,key)) -> {Nat.is_lt(Geometry.population(n),13n) == True{} : Bool}:
  (a,(b,(c,d))) = f
  a

def exact(n: Nat, key: U32, f: facts(n,key)) ->
  {U32.to_nat(U32.from_nat(Geometry.population(n))) == Geometry.population(n) : Nat}:
  (a,(b,(c,d))) = f
  b

def bound(n: Nat, key: U32, f: facts(n,key)) ->
  {U32.is_le(U32.from_nat(Geometry.population(n)),Spec.max_population(key)) == True{} : Bool}:
  (a,(b,(c,d))) = f
  c

def size_bound(n: Nat, key: U32, f: facts(n,key)) ->
  {U32.is_le(U32.shln(1,Geometry.population(n)),Spec.max_size(key)) == True{} : Bool}:
  (a,(b,(c,d))) = f
  d

def power(n: Nat, f: shift_facts(n)) -> {U32.to_nat(U32.shln(1,n)) == Nat.pow(2n,n) : Nat}:
  (a,(b,c)) = f
  a

def positive(n: Nat, f: shift_facts(n)) -> {U32.is_gt(U32.shln(1,n),0) == True{} : Bool}:
  (a,(b,c)) = f
  b

def safe(n: Nat, f: shift_facts(n)) -> {U32.is_lt(U32.from_nat(n),32) == True{} : Bool}:
  (a,(b,c)) = f
  c
