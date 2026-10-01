# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square48(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(48n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(48n,8n+p),p,bound)

def square49(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(49n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(49n,8n+p),p,bound)

def square50(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(50n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},({==},({==},({==},Unit{})))))
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},Unit{}))
    case 8n+p: D.impossible(S.certificate(50n,8n+p),p,bound)

def square51(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(51n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},({==},({==},Unit{}))))
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},({==},Unit{})))
    case 8n+p: D.impossible(S.certificate(51n,8n+p),p,bound)

def square52(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(52n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},({==},Unit{})))
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},({==},({==},Unit{}))))
    case 8n+p: D.impossible(S.certificate(52n,8n+p),p,bound)

def square53(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(53n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},({==},Unit{}))
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},({==},({==},({==},Unit{})))))
    case 8n+p: D.impossible(S.certificate(53n,8n+p),p,bound)

def square54(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(54n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 8n+p: D.impossible(S.certificate(54n,8n+p),p,bound)

def square55(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(55n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},Unit{})
    case 3n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},Unit{})
    case 7n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 8n+p: D.impossible(S.certificate(55n,8n+p),p,bound)
