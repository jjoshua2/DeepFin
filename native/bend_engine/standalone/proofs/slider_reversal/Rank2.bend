# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square16(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(16n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},({==},({==},({==},Unit{})))))
    case 5n: ({==},({==},Unit{}))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(16n,8n+p),p,bound)

def square17(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(17n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},({==},({==},({==},Unit{})))))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(17n,8n+p),p,bound)

def square18(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(18n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},({==},({==},({==},Unit{})))))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(18n,8n+p),p,bound)

def square19(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(19n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(19n,8n+p),p,bound)

def square20(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(20n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(20n,8n+p),p,bound)

def square21(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(21n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},({==},({==},({==},Unit{})))))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(21n,8n+p),p,bound)

def square22(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(22n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},({==},({==},Unit{})))))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(22n,8n+p),p,bound)

def square23(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(23n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},({==},({==},({==},Unit{})))))
    case 3n: ({==},({==},Unit{}))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},({==},Unit{})))))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(23n,8n+p),p,bound)
