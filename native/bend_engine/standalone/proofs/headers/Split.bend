# Decompose the exact-prefix schedule without executing or replacing table values.
import Base
import ../prefix/Spec.bend as P
import ../prefix/Nats.bend as N

def split(+p: Nat,+n: Nat,+k: Nat,a: Array<U64>) ->
  {P.scheduled(Nat.add(p,n),k,a) ==
    P.scheduled(n,Nat.add(p,k),P.scheduled(p,k,a)) : Array<U64>}:
  match p:
    case 0n: {==}
    case 1n+r:
      %N.add_suc(r,k) :
        {P.scheduled(Nat.add(r,n),1n+k,P.block(a,k)) ==
          P.scheduled(n,_,P.scheduled(r,1n+k,P.block(a,k))) : Array<U64>}
      split(r,n,1n+k,P.block(a,k))
