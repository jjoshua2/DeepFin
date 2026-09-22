# Exact complete constructor topology; values do not enter this specification.
import Base
import ../storage/Observe.bend as O

def full(d: Nat) -> O.Shape:
  match d:
    case 0n: O.Leaf{}
    case 1n+p:
      +child = full(p)
      O.Node{child,child}
