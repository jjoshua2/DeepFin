# Scalar coordinate normalization only; no attack masks are enumerated.
import Base
import ../layout/Domain.bend as D
import ../ray/Spec.bend as Ray

def shape(+q: Nat) -> Type:
  {Nat.mod(q,64n) == q : Nat} &
  ({Nat.is_ge(q,64n) == False{} : Bool} &
  ({Nat.mod(64n+q,64n) == q : Nat} &
  {Nat.is_ge(64n+q,64n) == True{} : Bool}))

def checked(+q: Nat,bound: {Nat.is_lt(q,64n) == True{} : Bool}) -> shape(q):
  match q:
    case 0n: ({==},({==},({==},{==})))
    case 1n: ({==},({==},({==},{==})))
    case 2n: ({==},({==},({==},{==})))
    case 3n: ({==},({==},({==},{==})))
    case 4n: ({==},({==},({==},{==})))
    case 5n: ({==},({==},({==},{==})))
    case 6n: ({==},({==},({==},{==})))
    case 7n: ({==},({==},({==},{==})))
    case 8n: ({==},({==},({==},{==})))
    case 9n: ({==},({==},({==},{==})))
    case 10n: ({==},({==},({==},{==})))
    case 11n: ({==},({==},({==},{==})))
    case 12n: ({==},({==},({==},{==})))
    case 13n: ({==},({==},({==},{==})))
    case 14n: ({==},({==},({==},{==})))
    case 15n: ({==},({==},({==},{==})))
    case 16n: ({==},({==},({==},{==})))
    case 17n: ({==},({==},({==},{==})))
    case 18n: ({==},({==},({==},{==})))
    case 19n: ({==},({==},({==},{==})))
    case 20n: ({==},({==},({==},{==})))
    case 21n: ({==},({==},({==},{==})))
    case 22n: ({==},({==},({==},{==})))
    case 23n: ({==},({==},({==},{==})))
    case 24n: ({==},({==},({==},{==})))
    case 25n: ({==},({==},({==},{==})))
    case 26n: ({==},({==},({==},{==})))
    case 27n: ({==},({==},({==},{==})))
    case 28n: ({==},({==},({==},{==})))
    case 29n: ({==},({==},({==},{==})))
    case 30n: ({==},({==},({==},{==})))
    case 31n: ({==},({==},({==},{==})))
    case 32n: ({==},({==},({==},{==})))
    case 33n: ({==},({==},({==},{==})))
    case 34n: ({==},({==},({==},{==})))
    case 35n: ({==},({==},({==},{==})))
    case 36n: ({==},({==},({==},{==})))
    case 37n: ({==},({==},({==},{==})))
    case 38n: ({==},({==},({==},{==})))
    case 39n: ({==},({==},({==},{==})))
    case 40n: ({==},({==},({==},{==})))
    case 41n: ({==},({==},({==},{==})))
    case 42n: ({==},({==},({==},{==})))
    case 43n: ({==},({==},({==},{==})))
    case 44n: ({==},({==},({==},{==})))
    case 45n: ({==},({==},({==},{==})))
    case 46n: ({==},({==},({==},{==})))
    case 47n: ({==},({==},({==},{==})))
    case 48n: ({==},({==},({==},{==})))
    case 49n: ({==},({==},({==},{==})))
    case 50n: ({==},({==},({==},{==})))
    case 51n: ({==},({==},({==},{==})))
    case 52n: ({==},({==},({==},{==})))
    case 53n: ({==},({==},({==},{==})))
    case 54n: ({==},({==},({==},{==})))
    case 55n: ({==},({==},({==},{==})))
    case 56n: ({==},({==},({==},{==})))
    case 57n: ({==},({==},({==},{==})))
    case 58n: ({==},({==},({==},{==})))
    case 59n: ({==},({==},({==},{==})))
    case 60n: ({==},({==},({==},{==})))
    case 61n: ({==},({==},({==},{==})))
    case 62n: ({==},({==},({==},{==})))
    case 63n: ({==},({==},({==},{==})))
    case 64n+p: D.impossible(shape(64n+p),p,bound)

def normalize(+key: Nat,+sq: Nat,+dir: Nat,+occ: U64,
  place: {Nat.mod(key,64n) == sq : Nat},
  direction: {Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n) == dir : Nat}) ->
  {Ray.rays(key,occ) == Ray.directional(sq,dir,occ) : U64}:
  Equal.trans(U64,Ray.rays(key,occ),Ray.directional(sq,Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n),occ),Ray.directional(sq,dir,occ),
    Equal.cong(Nat,U64,x => Ray.directional(x,Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n),occ),Nat.mod(key,64n),sq,place),
    Equal.cong(Nat,U64,x => Ray.directional(sq,x,occ),Bool.pick(Nat,Nat.is_ge(key,64n),4n,0n),dir,direction))

def rook(+sq: Nat,+occ: U64,bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Ray.rays(sq,occ) == Ray.directional(sq,0n,occ) : U64}:
  (a,(b,(c,d))) = checked(sq,bound)
  normalize(sq,sq,0n,occ,a,Equal.cong(Bool,Nat,v => Bool.pick(Nat,v,4n,0n),Nat.is_ge(sq,64n),False{},b))

def bishop(+sq: Nat,+occ: U64,bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Ray.rays(64n+sq,occ) == Ray.directional(sq,4n,occ) : U64}:
  (a,(b,(c,d))) = checked(sq,bound)
  normalize(64n+sq,sq,4n,occ,c,Equal.cong(Bool,Nat,v => Bool.pick(Nat,v,4n,0n),Nat.is_ge(64n+sq,64n),True{},d))
