# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square56(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(56n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(56n,8n+p),p,bound)

def square57(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(57n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 6n: Unit{}
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(57n,8n+p),p,bound)

def square58(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(58n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},({==},({==},({==},Unit{})))))
    case 6n: Unit{}
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(58n,8n+p),p,bound)

def square59(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(59n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: Unit{}
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(59n,8n+p),p,bound)

def square60(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(60n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: Unit{}
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(60n,8n+p),p,bound)

def square61(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(61n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},({==},Unit{}))
    case 6n: Unit{}
    case 7n: ({==},({==},({==},({==},({==},Unit{})))))
    case 8n+p: D.impossible(S.certificate(61n,8n+p),p,bound)

def square62(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(62n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: ({==},Unit{})
    case 6n: Unit{}
    case 7n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 8n+p: D.impossible(S.certificate(62n,8n+p),p,bound)

def square63(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(63n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: Unit{}
    case 3n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: Unit{}
    case 7n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 8n+p: D.impossible(S.certificate(63n,8n+p),p,bound)
