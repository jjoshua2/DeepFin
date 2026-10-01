# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square24(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(24n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(24n,8n+p),p,bound)

def square25(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(25n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(25n,8n+p),p,bound)

def square26(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(26n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(26n,8n+p),p,bound)

def square27(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(27n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(27n,8n+p),p,bound)

def square28(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(28n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(28n,8n+p),p,bound)

def square29(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(29n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(29n,8n+p),p,bound)

def square30(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(30n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(30n,8n+p),p,bound)

def square31(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(31n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},({==},({==},Unit{}))))
    case 3n: ({==},({==},({==},Unit{})))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(31n,8n+p),p,bound)
