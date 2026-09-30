# Independent unbounded-coordinate single-step specification. No Tables imports.
import Base

def positive(+x: Nat) -> Maybe<&2,Nat>:
  Bool.pick(Maybe<&2,Nat>,Nat.is_lt(x,7n),Some{1n+x},None{})

def negative(x: Nat) -> Maybe<&2,Nat>:
  match x:
    case 0n: None{}
    case 1n+p: Some{p}

def flatten(f: Maybe<&2,Nat>,r: Maybe<&2,Nat>) -> Nat:
  match f r:
    case Some{x} Some{y}: Nat.add(x,Nat.mul(y,8n))
    case _ _: 64n

def move(dir: Nat,+f: Nat,+r: Nat) -> Nat:
  match dir:
    case 0n: flatten(positive(f),Some{r})
    case 1n: flatten(negative(f),Some{r})
    case 2n: flatten(Some{f},positive(r))
    case 3n: flatten(Some{f},negative(r))
    case 4n: flatten(positive(f),positive(r))
    case 5n: flatten(positive(f),negative(r))
    case 6n: flatten(negative(f),positive(r))
    case 7n: flatten(negative(f),negative(r))
    case _: 64n

def next(+sq: Nat,dir: Nat) -> Nat:
  move(dir,Nat.mod(sq,8n),Nat.div(sq,8n))
