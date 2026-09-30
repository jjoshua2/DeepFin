# Scalar certificates only; no bitboard answer table.
import Base
import ../layout/Domain.bend as D

def statement(+q: Nat) -> Type:
  {Nat.is_lt(q,128n) == True{} : Bool} &
  ({Nat.is_lt(Nat.add(64n,q),128n) == True{} : Bool} &
  {U32.add(64,U32.from_nat(q)) == U32.from_nat(Nat.add(64n,q)) : U32})

def checked(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> statement(q):
  match q:
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
    case 64n+p: D.impossible(statement(64n+p),p,bound)

def pick_rook(+q: Nat,cert: statement(q)) -> {Nat.is_lt(q,128n) == True{} : Bool}:
  (yes,rest) = cert
  yes

def pick_bishop(+q: Nat,cert: statement(q)) -> {Nat.is_lt(Nat.add(64n,q),128n) == True{} : Bool}:
  (a,(yes,rest)) = cert
  yes

def pick_key(+q: Nat,cert: statement(q)) ->
  {U32.add(64,U32.from_nat(q)) == U32.from_nat(Nat.add(64n,q)) : U32}:
  (a,(b,yes)) = cert
  yes

def rook(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> {Nat.is_lt(q,128n) == True{} : Bool}:
  pick_rook(q,checked(q,bound))

def bishop(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(64n,q),128n) == True{} : Bool}:
  pick_bishop(q,checked(q,bound))

def key(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool}) ->
  {U32.add(64,U32.from_nat(q)) == U32.from_nat(Nat.add(64n,q)) : U32}:
  pick_key(q,checked(q,bound))

def flip(+side: Bool) -> {U32.xor(Bool.to_u32(side),1) == Bool.to_u32(Bool.not(side)) : U32}:
  match side:
    case True{}: {==}
    case False{}: {==}
