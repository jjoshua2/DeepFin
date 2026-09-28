# Independent count specification from lengths in bounded file/rank geometry.
# No Tables function, carry-rippler, attack table or host-generated count oracle.
import Base

def file(square: Nat) -> Nat:
  Nat.mod(square,8n)

def rank(square: Nat) -> Nat:
  Nat.div(square,8n)

def trim(distance: Nat) -> Nat:
  Nat.sub(distance,1n)

def rook(+x: Nat, +y: Nat) -> Nat:
  Nat.add(Nat.add(trim(x),trim(Nat.sub(7n,x))),
    Nat.add(trim(y),trim(Nat.sub(7n,y))))

def bishop(+x: Nat, +y: Nat) -> Nat:
  Nat.add(Nat.add(trim(Nat.min(x,y)),trim(Nat.min(x,Nat.sub(7n,y)))),
    Nat.add(trim(Nat.min(Nat.sub(7n,x),y)),trim(Nat.min(Nat.sub(7n,x),Nat.sub(7n,y)))))

def population(+key: Nat) -> Nat:
  +sq = Nat.mod(key,64n)
  Bool.pick(Nat,Nat.is_lt(key,64n),rook(file(sq),rank(sq)),bishop(file(sq),rank(sq)))

def maximum(key: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(key,64n),12n,9n)
