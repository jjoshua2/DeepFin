# Independent directed path prefixes. The source and hit endpoint are excluded.
import Base
import ../attack_reversal/Spec.bend as Member
import ../ray/Rays.bend as Rays
import ../ray/Spec.bend as Ray
import ../../Tables.bend as Tables

type Slider is Data:
  Rook{}
  Bishop{}
  Queen{}

type Segment is Data:
  Missing{}
  Between{xs: List<&2,Nat>}

type Directions is Data:
  Eight{e: Segment,w: Segment,n: Segment,s: Segment,ne: Segment,se: Segment,nw: Segment,sw: Segment}

def prepend(x: Nat,v: Segment) -> Segment:
  match v:
    case Missing{}: Missing{}
    case Between{xs}: Between{x <> xs}

def cut(+dst: Nat,xs: List<&2,Nat>) -> Segment:
  match xs:
    case Nil{}: Missing{}
    case +h <> tail:
      Bool.pick(Segment,Nat.is_eq(dst,h),Between{Nil{}},prepend(h,cut(dst,tail)))

def append(xs: List<&2,Nat>,ys: List<&2,Nat>) -> List<&2,Nat>:
  match xs:
    case Nil{}: ys
    case h <> t: h <> append(t,ys)

def reverse(xs: List<&2,Nat>) -> List<&2,Nat>:
  match xs:
    case Nil{}: Nil{}
    case h <> t: append(reverse(t),h <> Nil{})

def mirror(v: Segment) -> Segment:
  match v:
    case Missing{}: Missing{}
    case Between{xs}: Between{reverse(xs)}

def clear(xs: List<&2,Nat>,+occ: U64) -> Bool:
  match xs:
    case Nil{}: True{}
    case h <> t: Bool.and(Bool.not(U64.test_bit(occ,h)),clear(t,occ))

def visible(v: Segment,occ: U64) -> Bool:
  match v:
    case Missing{}: False{}
    case Between{xs}: clear(xs,occ)

def paths(+src: Nat,+dst: Nat) -> Directions:
  Eight{cut(dst,Rays.path(7n,src,0n)),cut(dst,Rays.path(7n,src,1n)),
    cut(dst,Rays.path(7n,src,2n)),cut(dst,Rays.path(7n,src,3n)),
    cut(dst,Rays.path(7n,src,4n)),cut(dst,Rays.path(7n,src,5n)),
    cut(dst,Rays.path(7n,src,6n)),cut(dst,Rays.path(7n,src,7n))}

def opposite(v: Directions) -> Directions:
  Eight{e,w,n,s,ne,se,nw,sw} = v
  Eight{mirror(w),mirror(e),mirror(s),mirror(n),mirror(sw),mirror(nw),mirror(se),mirror(ne)}

def four(a: Bool,b: Bool,c: Bool,d: Bool) -> Bool:
  Bool.or(Bool.or(a,b),Bool.or(c,d))

def answer(kind: Slider,v: Directions,+occ: U64) -> Bool:
  match kind v:
    case Rook{} Eight{e,w,n,s,ne,se,nw,sw}: four(visible(e,occ),visible(w,occ),visible(n,occ),visible(s,occ))
    case Bishop{} Eight{e,w,n,s,ne,se,nw,sw}: four(visible(ne,occ),visible(se,occ),visible(nw,occ),visible(sw,occ))
    case Queen{} Eight{e,w,n,s,ne,se,nw,sw}:
      Bool.or(four(visible(e,occ),visible(w,occ),visible(n,occ),visible(s,occ)),
        four(visible(ne,occ),visible(se,occ),visible(nw,occ),visible(sw,occ)))

def geometry(kind: Slider,+src: Nat,+occ: U64) -> U64:
  match kind:
    case Rook{}: Ray.directional(src,0n,occ)
    case Bishop{}: Ray.directional(src,4n,occ)
    case Queen{}: U64.or(Ray.directional(src,0n,occ),Ray.directional(src,4n,occ))

def computed(kind: Slider,+src: Nat,+occ: U64) -> U64:
  match kind:
    case Rook{}: Tables.slider(U32.from_nat(src),False{},occ,False{})
    case Bishop{}: Tables.slider(U32.from_nat(src),True{},occ,False{})
    case Queen{}: U64.or(Tables.slider(U32.from_nat(src),False{},occ,False{}),Tables.slider(U32.from_nat(src),True{},occ,False{}))

# Opposite compass directions; invalid directions remain out of domain.
def reverse_dir(dir: Nat) -> Nat:
  match dir:
    case 0n: 1n
    case 1n: 0n
    case 2n: 3n
    case 3n: 2n
    case 4n: 7n
    case 5n: 6n
    case 6n: 5n
    case 7n: 4n
    case _: 8n

def view(src: Nat,dst: Nat,dir: Nat) -> Segment:
  cut(dst,Rays.path(7n,src,dir))

def found(v: Segment) -> Bool:
  match v:
    case Missing{}: False{}
    case Between{xs}: True{}

# Certificates compare only coordinate prefixes, never occupancy or desired attacks.
def neighbor(+src: Nat,+dir: Nat,+dst: Nat) -> Type:
  {view(dst,src,reverse_dir(dir)) == mirror(view(src,dst,dir)) : Segment}

def certificate(+src: Nat,+dir: Nat) -> Type:
  Member.every(dst => neighbor(src,dir,dst),Rays.path(7n,src,dir))

def edge(src: Nat,dst: Nat,dir: Nat,occ: U64) -> Bool:
  visible(view(src,dst,dir),occ)
