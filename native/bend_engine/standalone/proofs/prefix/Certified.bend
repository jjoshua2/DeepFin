# The actual population law is discharged in the imported parent PROOF.
import Base
import ./Spec.bend as S
import ./Nats.bend as Nats
import ./Size.bend as Size
import ../layout/Domain.bend as Domain
import ../layout/Spec.bend as Layout
import ../layout/LAWS.bend as LayoutLaws
import ../layout/PROOF.bend as LayoutProof

def size(+k: Nat, +valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {Layout.size(U32.from_nat(k)) == S.size(k) : U32}:
  +key = U32.from_nat(k)
  +bounded = Equal.trans(Bool,Nat.is_lt(U32.to_nat(key),128n),Nat.is_lt(k,128n),True{},
    Equal.cong(Nat,Bool,n => Nat.is_lt(n,128n),U32.to_nat(key),k,Domain.exact(k,valid)),valid)
  Size.size(k,valid,LayoutLaws.mask_population(key,bounded))

def make(+n: Nat, +k: Nat, +valid: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool}) -> S.sizes(n,k):
  match n:
    case 0n: Unit{}
    case 1n+p: (size(k,Nats.strip(p,k,128n,valid)),make(p,1n+k,Nats.next_bound(p,k,valid)))
