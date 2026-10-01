# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square32(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(32n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(32n,8n+p),p,bound)

def square33(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(33n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(33n,8n+p),p,bound)

def square34(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(34n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(34n,8n+p),p,bound)

def square35(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(35n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(35n,8n+p),p,bound)

def square36(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(36n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(36n,8n+p),p,bound)

def square37(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(37n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(37n,8n+p),p,bound)

def square38(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(38n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(38n,8n+p),p,bound)

def square39(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(39n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},({==},Unit{})))
    case 3n: ({==},({==},({==},({==},Unit{}))))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(39n,8n+p),p,bound)
