# Generated coordinate-only prefix certificates; every equality is checked by Bend.
import Base
import ./Spec.bend as S
import ../layout/Domain.bend as D

def square8(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(8n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 1n: Unit{}
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 5n: ({==},Unit{})
    case 6n: Unit{}
    case 7n: Unit{}
    case 8n+p: D.impossible(S.certificate(8n,8n+p),p,bound)

def square9(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(9n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 1n: ({==},Unit{})
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 5n: ({==},Unit{})
    case 6n: ({==},Unit{})
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(9n,8n+p),p,bound)

def square10(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(10n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},({==},Unit{})))))
    case 1n: ({==},({==},Unit{}))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},({==},({==},({==},Unit{})))))
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},Unit{}))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(10n,8n+p),p,bound)

def square11(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(11n,dir):
  match dir:
    case 0n: ({==},({==},({==},({==},Unit{}))))
    case 1n: ({==},({==},({==},Unit{})))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},({==},({==},Unit{}))))
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},Unit{})))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(11n,8n+p),p,bound)

def square12(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(12n,dir):
  match dir:
    case 0n: ({==},({==},({==},Unit{})))
    case 1n: ({==},({==},({==},({==},Unit{}))))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},({==},Unit{})))
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},({==},Unit{}))))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(12n,8n+p),p,bound)

def square13(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(13n,dir):
  match dir:
    case 0n: ({==},({==},Unit{}))
    case 1n: ({==},({==},({==},({==},({==},Unit{})))))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},({==},Unit{}))
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},({==},({==},Unit{})))))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(13n,8n+p),p,bound)

def square14(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(14n,dir):
  match dir:
    case 0n: ({==},Unit{})
    case 1n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: ({==},Unit{})
    case 5n: ({==},Unit{})
    case 6n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(14n,8n+p),p,bound)

def square15(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(15n,dir):
  match dir:
    case 0n: Unit{}
    case 1n: ({==},({==},({==},({==},({==},({==},({==},Unit{})))))))
    case 2n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 3n: ({==},Unit{})
    case 4n: Unit{}
    case 5n: Unit{}
    case 6n: ({==},({==},({==},({==},({==},({==},Unit{}))))))
    case 7n: ({==},Unit{})
    case 8n+p: D.impossible(S.certificate(15n,8n+p),p,bound)
