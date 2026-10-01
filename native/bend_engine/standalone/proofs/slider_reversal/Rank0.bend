# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square0(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(0n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 5n: Unit{}
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(0n,8n+p),p,bound)

def square1(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(1n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 5n: Unit{}
    case 6n: ({==},Unit{})
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(1n,8n+p),p,bound)

def square2(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(2n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},({==},({==},({==},Unit{})))))
    case 5n: Unit{}
    case 6n: ({==},({==},Unit{}))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(2n,8n+p),p,bound)

def square3(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(3n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: Unit{}
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(3n,8n+p),p,bound)

def square4(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(4n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(4n,8n+p),p,bound)

def square5(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(5n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},({==},Unit{}))
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},({==},Unit{})))))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(5n,8n+p),p,bound)

def square6(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(6n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: ({==},Unit{})
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(6n,8n+p),p,bound)

def square7(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(7n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 3n: Unit{}
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(7n,8n+p),p,bound)
