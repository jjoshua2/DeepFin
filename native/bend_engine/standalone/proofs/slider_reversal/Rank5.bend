# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square40(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(40n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},({==},({==},({==},Unit{})))))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(40n,8n+p),p,bound)

def square41(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(41n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},({==},({==},({==},Unit{})))))
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(41n,8n+p),p,bound)

def square42(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(42n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},({==},({==},({==},Unit{})))))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(42n,8n+p),p,bound)

def square43(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(43n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(43n,8n+p),p,bound)

def square44(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(44n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(44n,8n+p),p,bound)

def square45(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(45n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},({==},({==},({==},Unit{})))))
    case 8n+p: D.impossible(S.certificate(45n,8n+p),p,bound)

def square46(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(46n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},({==},({==},({==},Unit{})))))
    case 8n+p: D.impossible(S.certificate(46n,8n+p),p,bound)

def square47(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(47n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},Unit{}))
    case 3n: ({==},({==},({==},({==},({==},Unit{})))))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},({==},({==},({==},Unit{})))))
    case 8n+p: D.impossible(S.certificate(47n,8n+p),p,bound)
