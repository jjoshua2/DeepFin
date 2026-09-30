# Scalar coordinate certificates plus structural accumulation; no host bitboards.
import Base
import ./Geometry.bend as G
import ./Slots.bend as S
import ../../Tables.bend as Tables
import ../layout/Domain.bend as Domain

def list(n: Nat,+sq: U32,+dir: U32,+king: Bool) -> List<&2,Nat>:
  match n:
    case 0n: Nil{}
    case 1n+p:
      U32.to_nat(Tables.jump(sq,Bool.pick(U32 & U32,king,Tables.delta(dir),Tables.knight(dir)))) <>
        list(p,sq,U32.inc(dir),king)

def targets(kind: G.Leaper,+sq: Nat) -> List<&2,Nat>:
  match kind:
    case G.Knight{}: list(8n,U32.from_nat(sq),0,False{})
    case G.King{}: list(8n,U32.from_nat(sq),0,True{})
    case G.WhitePawn{}:
      U32.to_nat(Tables.jump(U32.from_nat(sq),(1,1))) <> U32.to_nat(Tables.jump(U32.from_nat(sq),(4294967295,1))) <> Nil{}
    case G.BlackPawn{}:
      U32.to_nat(Tables.jump(U32.from_nat(sq),(1,4294967295))) <> U32.to_nat(Tables.jump(U32.from_nat(sq),(4294967295,4294967295))) <> Nil{}

def accumulation(+n: Nat,+sq: U32,+dir: U32,+king: Bool,+acc: U64) ->
  {Tables.leaps(n,sq,dir,king,acc) == G.fold(list(n,sq,dir,king),acc) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      accumulation(p,sq,U32.inc(dir),king,U64.or(acc,U64.bit(U32.to_nat(Tables.jump(sq,Bool.pick(U32 & U32,king,Tables.delta(dir),Tables.knight(dir)))))))

def unfold(+kind: G.Leaper,+sq: Nat) ->
  {S.value(kind,sq) == G.combine(kind,targets(kind,sq)) : U64}:
  match kind:
    case G.Knight{}: accumulation(8n,U32.from_nat(sq),0,False{},U64.zero())
    case G.King{}: accumulation(8n,U32.from_nat(sq),0,True{},U64.zero())
    case G.WhitePawn{}: {==}
    case G.BlackPawn{}: {==}

def all(+sq: Nat) -> Type:
  {targets(G.Knight{},sq) == G.targets(G.Knight{},sq) : List<&2,Nat>} &
  ({targets(G.King{},sq) == G.targets(G.King{},sq) : List<&2,Nat>} &
  ({targets(G.WhitePawn{},sq) == G.targets(G.WhitePawn{},sq) : List<&2,Nat>} &
   {targets(G.BlackPawn{},sq) == G.targets(G.BlackPawn{},sq) : List<&2,Nat>}))

def squares(+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool}) -> all(sq):
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
    case 64n+p: Domain.impossible(all(64n+p),p,e)

def choose(+kind: G.Leaper,+sq: Nat,values: all(sq)) ->
  {targets(kind,sq) == G.targets(kind,sq) : List<&2,Nat>}:
  match kind:
    case G.Knight{}:
      (n,rest) = values
      n
    case G.King{}:
      (n,(k,rest)) = values
      k
    case G.WhitePawn{}:
      (n,(k,(w,b))) = values
      w
    case G.BlackPawn{}:
      (n,(k,(w,b))) = values
      b

def actual(+kind: G.Leaper,+sq: Nat,e: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {S.value(kind,sq) == G.mask(kind,sq) : U64}:
  Equal.trans(U64,S.value(kind,sq),G.combine(kind,targets(kind,sq)),G.mask(kind,sq),unfold(kind,sq),
    Equal.cong(List<&2,Nat>,U64,ns => G.combine(kind,ns),targets(kind,sq),G.targets(kind,sq),choose(kind,sq,squares(sq,e))))
