# Full-array equality links the independent prefix schedule to the actual loop.
import Base
import ./Spec.bend as S
import ./Nats.bend as Nats
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

# The public proof constructs every size certificate from the existing layout law.
def refine(+n: Nat, +k: Nat, a: Array<U64>, cert: S.sizes(n,k)) ->
  {Tables.tables(n,U32.from_nat(k),S.prefix(k),a) == S.scheduled(n,k,a) : Array<U64>}:
  match n:
    case 0n: {==}
    case 1n+p:
      (head,tail) = cert
      +step = Equal.cong(U32,U32,x => U32.add(S.prefix(k),x),
        Layout.size(U32.from_nat(k)),S.size(k),head)
      %Equal.sym(U32,U32.add(S.prefix(k),Layout.size(U32.from_nat(k))),S.prefix(1n+k),step) :
        {Tables.tables(p,U32.from_nat(1n+k),_,S.block(a,k)) == S.scheduled(p,1n+k,S.block(a,k)) : Array<U64>}
      refine(p,1n+k,S.block(a,k),tail)
