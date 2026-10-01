# Interpret all eight certified strict prefixes as the geometric ray masks.
import Base
import ./Spec.bend as S
import ./Bits.bend as Bit
import ./Booleans.bend as B
import ../decoder/Bits.bend as Project
import ../ray/Path.bend as Path
import ../ray/Rays.bend as Rays
import ../ray/Spec.bend as Ray

def union(+a: U64,+b: U64,+dst: Nat,+va: Bool,+vb: Bool,
  bound: {Nat.is_lt(dst,64n) == True{} : Bool},
  ea: {U64.test_bit(a,dst) == va : Bool},eb: {U64.test_bit(b,dst) == vb : Bool}) ->
  {U64.test_bit(U64.or(a,b),dst) == Bool.or(va,vb) : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.or(a,b),dst),Bool.or(U64.test_bit(a,dst),U64.test_bit(b,dst)),Bool.or(va,vb),
    Project.test64_or(a,b,dst,bound),B.or_cong(U64.test_bit(a,dst),U64.test_bit(b,dst),va,vb,ea,eb))

def four_paths(+a: List<&2,Nat>,+b: List<&2,Nat>,+c: List<&2,Nat>,+d: List<&2,Nat>,+occ: U64,+dst: Nat,
  +bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(U64.or(U64.or(Path.attack(a,occ),Path.attack(b,occ)),U64.or(Path.attack(c,occ),Path.attack(d,occ))),dst) ==
    S.four(S.visible(S.cut(dst,a),occ),S.visible(S.cut(dst,b),occ),S.visible(S.cut(dst,c),occ),S.visible(S.cut(dst,d),occ)) : Bool}:
  union(U64.or(Path.attack(a,occ),Path.attack(b,occ)),U64.or(Path.attack(c,occ),Path.attack(d,occ)),dst,
    Bool.or(S.visible(S.cut(dst,a),occ),S.visible(S.cut(dst,b),occ)),
    Bool.or(S.visible(S.cut(dst,c),occ),S.visible(S.cut(dst,d),occ)),bound,
    union(Path.attack(a,occ),Path.attack(b,occ),dst,S.visible(S.cut(dst,a),occ),S.visible(S.cut(dst,b),occ),
      bound,Bit.path(a,occ,dst,bound),Bit.path(b,occ,dst,bound)),
    union(Path.attack(c,occ),Path.attack(d,occ),dst,S.visible(S.cut(dst,c),occ),S.visible(S.cut(dst,d),occ),
      bound,Bit.path(c,occ,dst,bound),Bit.path(d,occ,dst,bound)))

def geometry(+kind: S.Slider,+src: Nat,+dst: Nat,+occ: U64,+bound: {Nat.is_lt(dst,64n) == True{} : Bool}) ->
  {U64.test_bit(S.geometry(kind,src,occ),dst) == S.answer(kind,S.paths(src,dst),occ) : Bool}:
  match kind:
    case S.Rook{}:
      four_paths(Rays.path(7n,src,0n),Rays.path(7n,src,1n),Rays.path(7n,src,2n),Rays.path(7n,src,3n),occ,dst,bound)
    case S.Bishop{}:
      four_paths(Rays.path(7n,src,4n),Rays.path(7n,src,5n),Rays.path(7n,src,6n),Rays.path(7n,src,7n),occ,dst,bound)
    case S.Queen{}:
      union(S.geometry(S.Rook{},src,occ),S.geometry(S.Bishop{},src,occ),dst,
        S.answer(S.Rook{},S.paths(src,dst),occ),S.answer(S.Bishop{},S.paths(src,dst),occ),bound,
        four_paths(Rays.path(7n,src,0n),Rays.path(7n,src,1n),Rays.path(7n,src,2n),Rays.path(7n,src,3n),occ,dst,bound),
        four_paths(Rays.path(7n,src,4n),Rays.path(7n,src,5n),Rays.path(7n,src,6n),Rays.path(7n,src,7n),occ,dst,bound))
