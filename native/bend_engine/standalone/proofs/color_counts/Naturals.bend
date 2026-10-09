import Base
import ../u64/Words.bend as Old

def reflexive(+n: Nat) -> {Nat.is_le(n,n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: reflexive(p)

def transitive(+a: Nat,+b: Nat,+c: Nat,
  ab: {Nat.is_le(a,b) == True{} : Bool},bc: {Nat.is_le(b,c) == True{} : Bool}) ->
  {Nat.is_le(a,c) == True{} : Bool}:
  match a b c:
    case 0n _ 0n: {==}
    case 0n _ 1n+cp: {==}
    case 1n+ap 0n _: Old.false_true({Nat.is_le(1n+ap,c) == True{} : Bool},ab)
    case 1n+ap 1n+bp 0n: Old.false_true({Nat.is_le(1n+ap,0n) == True{} : Bool},bc)
    case 1n+ap 1n+bp 1n+cp: transitive(ap,bp,cp,ab,bc)

def drop_left(+a: Nat,+b: Nat,e: {Nat.is_le(1n+a,b) == True{} : Bool}) ->
  {Nat.is_le(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+bp: {==}
    case 1n+ap 0n: Old.false_true({Nat.is_le(1n+ap,0n) == True{} : Bool},e)
    case 1n+ap 1n+bp: drop_left(ap,bp,e)
