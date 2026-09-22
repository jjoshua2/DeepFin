# Independent geometric prefix schedule; real table operations are retained.
import Base
import ../layout/Geometry.bend as Geometry
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

# These expressions describe actual sizes through the already proved geometry.
def size(k: Nat) -> U32:
  U32.shln(1,Geometry.population(k))

def prefix(+n: Nat) -> U32:
  match n:
    case 0n: 512
    case 1n+p: U32.add(prefix(p),size(p))

# Widened observation: no huge unary machine-limit constant is normalized.
def wide(+n: Nat) -> U64:
  match n:
    case 0n: U64.from_u32(512)
    case 1n+p: U64.add(wide(p),U64.from_u32(size(p)))

def block(a: Array<U64>, +k: Nat) -> Array<U64>:
  +key = U32.from_nat(k)
  +at = prefix(k)
  +mask = Layout.mask(key)
  a = Array.set(U64,Array.set(U64,a,key,mask),U32.add(128,key),U64.from_u32(at))
  Tables.fill(U32.to_nat(Layout.size(key)),at,U32.mod(key,64),U32.is_ge(key,64),mask,U64.zero(),a)

# Explicit-offset schedule of the actual operations, not an ideal shadow array.
def scheduled(n: Nat, +k: Nat, a: Array<U64>) -> Array<U64>:
  match n:
    case 0n: a
    case 1n+p: scheduled(p,1n+k,block(a,k))

# Certificates used internally to relate the explicit and original schedules.
# The public proof constructs these; callers do not assume them.
def sizes(n: Nat, +k: Nat) -> Type:
  match n:
    case 0n: Unit
    case 1n+p: {Layout.size(U32.from_nat(k)) == size(k) : U32} & sizes(p,1n+k)
