# Bounded address and separation certificates for four extra-mask regions.
import Base
import ./Geometry.bend as G
import ./Slots.bend as S
import ../layout/Domain.bend as Domain
import ../headers/Addresses.bend as A
import ../relative/Order.bend as Order

def cert(+kind: G.Leaper,+sq: Nat) -> Type:
  {U32.to_nat(S.index(kind,sq)) == Nat.add(S.offset(kind),sq) : Nat} &
  ({U32.is_lt(S.index(kind,sq),S.end(kind)) == True{} : Bool} &
  ({U32.is_le(S.base(kind),S.index(kind,sq)) == True{} : Bool} &
   {U32.is_lt(S.index(kind,sq),131072) == True{} : Bool}))

def fixed(+kind: G.Leaper,+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool}) -> cert(kind,sq):
  match kind:
    case G.Knight{}:
      match sq:
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
        case 64n+p: Domain.impossible(cert(G.Knight{},64n+p),p,e)
    case G.King{}:
      match sq:
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
        case 64n+p: Domain.impossible(cert(G.King{},64n+p),p,e)
    case G.WhitePawn{}:
      match sq:
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
        case 64n+p: Domain.impossible(cert(G.WhitePawn{},64n+p),p,e)
    case G.BlackPawn{}:
      match sq:
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
        case 64n+p: Domain.impossible(cert(G.BlackPawn{},64n+p),p,e)

def observed(+kind: G.Leaper,+sq: Nat,e: cert(kind,sq)) ->
  {U32.to_nat(S.index(kind,sq)) == Nat.add(S.offset(kind),sq) : Nat}:
  (v,rest) = e
  v

def upper(+kind: G.Leaper,+sq: Nat,e: cert(kind,sq)) ->
  {U32.is_lt(S.index(kind,sq),S.end(kind)) == True{} : Bool}:
  (v,(hi,rest)) = e
  hi

def lower(+kind: G.Leaper,+sq: Nat,e: cert(kind,sq)) ->
  {U32.is_le(S.base(kind),S.index(kind,sq)) == True{} : Bool}:
  (v,(hi,(lo,b))) = e
  lo

def bounded(+kind: G.Leaper,+sq: Nat,e: cert(kind,sq)) ->
  {U32.is_lt(S.index(kind,sq),131072) == True{} : Bool}:
  (v,(hi,(lo,b))) = e
  b

def add_eq(+c: Nat,+a: Nat,+b: Nat) ->
  {Nat.is_eq(Nat.add(c,a),Nat.add(c,b)) == Nat.is_eq(a,b) : Bool}:
  match c:
    case 0n: {==}
    case 1n+p: add_eq(p,a,b)

def same_kind(+kind: G.Leaper,+a: Nat,+b: Nat,
  ab: {Nat.is_lt(a,64n) == True{} : Bool},bb: {Nat.is_lt(b,64n) == True{} : Bool},
  ne: {Nat.is_eq(a,b) == False{} : Bool}) ->
  {U32.is_eq(S.index(kind,a),S.index(kind,b)) == False{} : Bool}:
  A.ne_values(S.index(kind,a),S.index(kind,b),Nat.add(S.offset(kind),a),Nat.add(S.offset(kind),b),
    observed(kind,a,fixed(kind,a,ab)),observed(kind,b,fixed(kind,b,bb)),
    Equal.trans(Bool,Nat.is_eq(Nat.add(S.offset(kind),a),Nat.add(S.offset(kind),b)),Nat.is_eq(a,b),False{},add_eq(S.offset(kind),a,b),ne))

def distinct(+ka: G.Leaper,+kb: G.Leaper,+a: Nat,+b: Nat,
  +ab: {Nat.is_lt(a,64n) == True{} : Bool},+bb: {Nat.is_lt(b,64n) == True{} : Bool},
  ne: {S.overlaps(ka,kb,a,b) == False{} : Bool}) ->
  {U32.is_eq(S.index(ka,a),S.index(kb,b)) == False{} : Bool}:
  match ka kb:
    case G.Knight{} G.Knight{}:
      same_kind(G.Knight{},a,b,ab,bb,ne)
    case G.Knight{} G.King{}:
      Order.lt_ne(S.index(G.Knight{},a),S.index(G.King{},b),Order.lt_le(S.index(G.Knight{},a),S.base(G.King{}),S.index(G.King{},b),Order.lt_le(S.index(G.Knight{},a),S.end(G.Knight{}),S.base(G.King{}),upper(G.Knight{},a,fixed(G.Knight{},a,ab)),{==}),lower(G.King{},b,fixed(G.King{},b,bb))))
    case G.Knight{} G.WhitePawn{}:
      Order.lt_ne(S.index(G.Knight{},a),S.index(G.WhitePawn{},b),Order.lt_le(S.index(G.Knight{},a),S.base(G.WhitePawn{}),S.index(G.WhitePawn{},b),Order.lt_le(S.index(G.Knight{},a),S.end(G.Knight{}),S.base(G.WhitePawn{}),upper(G.Knight{},a,fixed(G.Knight{},a,ab)),{==}),lower(G.WhitePawn{},b,fixed(G.WhitePawn{},b,bb))))
    case G.Knight{} G.BlackPawn{}:
      Order.lt_ne(S.index(G.Knight{},a),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.Knight{},a),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.Knight{},a),S.end(G.Knight{}),S.base(G.BlackPawn{}),upper(G.Knight{},a,fixed(G.Knight{},a,ab)),{==}),lower(G.BlackPawn{},b,fixed(G.BlackPawn{},b,bb))))
    case G.King{} G.Knight{}:
      A.ne_flip(S.index(G.Knight{},b),S.index(G.King{},a),Order.lt_ne(S.index(G.Knight{},b),S.index(G.King{},a),Order.lt_le(S.index(G.Knight{},b),S.base(G.King{}),S.index(G.King{},a),Order.lt_le(S.index(G.Knight{},b),S.end(G.Knight{}),S.base(G.King{}),upper(G.Knight{},b,fixed(G.Knight{},b,bb)),{==}),lower(G.King{},a,fixed(G.King{},a,ab)))))
    case G.King{} G.King{}:
      same_kind(G.King{},a,b,ab,bb,ne)
    case G.King{} G.WhitePawn{}:
      Order.lt_ne(S.index(G.King{},a),S.index(G.WhitePawn{},b),Order.lt_le(S.index(G.King{},a),S.base(G.WhitePawn{}),S.index(G.WhitePawn{},b),Order.lt_le(S.index(G.King{},a),S.end(G.King{}),S.base(G.WhitePawn{}),upper(G.King{},a,fixed(G.King{},a,ab)),{==}),lower(G.WhitePawn{},b,fixed(G.WhitePawn{},b,bb))))
    case G.King{} G.BlackPawn{}:
      Order.lt_ne(S.index(G.King{},a),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.King{},a),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.King{},a),S.end(G.King{}),S.base(G.BlackPawn{}),upper(G.King{},a,fixed(G.King{},a,ab)),{==}),lower(G.BlackPawn{},b,fixed(G.BlackPawn{},b,bb))))
    case G.WhitePawn{} G.Knight{}:
      A.ne_flip(S.index(G.Knight{},b),S.index(G.WhitePawn{},a),Order.lt_ne(S.index(G.Knight{},b),S.index(G.WhitePawn{},a),Order.lt_le(S.index(G.Knight{},b),S.base(G.WhitePawn{}),S.index(G.WhitePawn{},a),Order.lt_le(S.index(G.Knight{},b),S.end(G.Knight{}),S.base(G.WhitePawn{}),upper(G.Knight{},b,fixed(G.Knight{},b,bb)),{==}),lower(G.WhitePawn{},a,fixed(G.WhitePawn{},a,ab)))))
    case G.WhitePawn{} G.King{}:
      A.ne_flip(S.index(G.King{},b),S.index(G.WhitePawn{},a),Order.lt_ne(S.index(G.King{},b),S.index(G.WhitePawn{},a),Order.lt_le(S.index(G.King{},b),S.base(G.WhitePawn{}),S.index(G.WhitePawn{},a),Order.lt_le(S.index(G.King{},b),S.end(G.King{}),S.base(G.WhitePawn{}),upper(G.King{},b,fixed(G.King{},b,bb)),{==}),lower(G.WhitePawn{},a,fixed(G.WhitePawn{},a,ab)))))
    case G.WhitePawn{} G.WhitePawn{}:
      same_kind(G.WhitePawn{},a,b,ab,bb,ne)
    case G.WhitePawn{} G.BlackPawn{}:
      Order.lt_ne(S.index(G.WhitePawn{},a),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.WhitePawn{},a),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},b),Order.lt_le(S.index(G.WhitePawn{},a),S.end(G.WhitePawn{}),S.base(G.BlackPawn{}),upper(G.WhitePawn{},a,fixed(G.WhitePawn{},a,ab)),{==}),lower(G.BlackPawn{},b,fixed(G.BlackPawn{},b,bb))))
    case G.BlackPawn{} G.Knight{}:
      A.ne_flip(S.index(G.Knight{},b),S.index(G.BlackPawn{},a),Order.lt_ne(S.index(G.Knight{},b),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.Knight{},b),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.Knight{},b),S.end(G.Knight{}),S.base(G.BlackPawn{}),upper(G.Knight{},b,fixed(G.Knight{},b,bb)),{==}),lower(G.BlackPawn{},a,fixed(G.BlackPawn{},a,ab)))))
    case G.BlackPawn{} G.King{}:
      A.ne_flip(S.index(G.King{},b),S.index(G.BlackPawn{},a),Order.lt_ne(S.index(G.King{},b),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.King{},b),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.King{},b),S.end(G.King{}),S.base(G.BlackPawn{}),upper(G.King{},b,fixed(G.King{},b,bb)),{==}),lower(G.BlackPawn{},a,fixed(G.BlackPawn{},a,ab)))))
    case G.BlackPawn{} G.WhitePawn{}:
      A.ne_flip(S.index(G.WhitePawn{},b),S.index(G.BlackPawn{},a),Order.lt_ne(S.index(G.WhitePawn{},b),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.WhitePawn{},b),S.base(G.BlackPawn{}),S.index(G.BlackPawn{},a),Order.lt_le(S.index(G.WhitePawn{},b),S.end(G.WhitePawn{}),S.base(G.BlackPawn{}),upper(G.WhitePawn{},b,fixed(G.WhitePawn{},b,bb)),{==}),lower(G.BlackPawn{},a,fixed(G.BlackPawn{},a,ab)))))
    case G.BlackPawn{} G.BlackPawn{}:
      same_kind(G.BlackPawn{},a,b,ab,bb,ne)

def and_false(+b: Bool) -> {Bool.and(b,False{}) == False{} : Bool}:
  match b:
    case True{}: {==}
    case False{}: {==}

def other_square(+ka: G.Leaper,+kb: G.Leaper,+a: Nat,+b: Nat,
  ne: {Nat.is_eq(a,b) == False{} : Bool}) -> {S.overlaps(ka,kb,a,b) == False{} : Bool}:
  Equal.trans(Bool,S.overlaps(ka,kb,a,b),Bool.and(S.same(ka,kb),False{}),False{},
    Equal.cong(Bool,Bool,x => Bool.and(S.same(ka,kb),x),Nat.is_eq(a,b),False{},ne),and_false(S.same(ka,kb)))
