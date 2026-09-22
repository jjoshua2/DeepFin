import Base
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../storage/Model.bend as M
import ../u64/Words.bend as Old

def size_image(+c: O.Cells) -> {S.size(O.shape(O.pack(c))) == O.size(c) : U32}:
  match c:
    case O.Cell{x}: {==}
    case O.Branch{xs,ys}:
      Equal.cong(U32,U32,n => U32.shl(n),S.size(O.shape(O.pack(xs))),O.size(xs),size_image(xs))

def read_choice(+z: Bool, +a: O.Cells, +b: O.Cells, +n: U32, +j: U32) ->
  {M.read(Bool.pick(O.Cells,z,a,b),n,j) == Bool.pick(U64,z,M.read(a,n,j),M.read(b,n,j)) : U64}:
  match z:
    case True{}: {==}
    case False{}: {==}

def cases(+a: Bool, +b: Bool, +l: Bool, +r: Bool,
  +before_l: U64, +before_r: U64, +after_l: U64, +after_r: U64,
  left: {l == True{} : Bool} -> {after_l == before_l : U64},
  right: {r == True{} : Bool} -> {after_r == before_r : U64},
  e: {S.choice(l,r,a,b) == True{} : Bool}) ->
  {Bool.pick(U64,a,Bool.pick(U64,b,after_l,before_r),Bool.pick(U64,b,before_l,after_r))
    == Bool.pick(U64,b,before_l,before_r) : U64}:
  match a b:
    case True{} True{}: left(e)
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: right(e)

def other(+c: O.Cells, +n: U32, +i: U32, +j: U32, +v: U64,
  e: {S.apart(O.shape(O.pack(c)),n,i,j) == True{} : Bool}) ->
  {M.read(M.update(c,n,i,v),n,j) == M.read(c,n,j) : U64}:
  match c:
    case O.Cell{x}: Old.false_true({v == x : U64},e)
    case O.Branch{+xs,+ys}:
      +h = U32.shr(n)
      Equal.trans(U64,M.read(M.update(O.Branch{xs,ys},n,i,v),n,j),
        Bool.pick(U64,U32.is_lt(i,h),
          M.read(O.Branch{M.update(xs,h,i,v),ys},n,j),
          M.read(O.Branch{xs,M.update(ys,h,U32.sub(i,h),v)},n,j)),
        M.read(O.Branch{xs,ys},n,j),
        read_choice(U32.is_lt(i,h),O.Branch{M.update(xs,h,i,v),ys},O.Branch{xs,M.update(ys,h,U32.sub(i,h),v)},n,j),
        cases(U32.is_lt(i,h),U32.is_lt(j,h),
          S.apart(O.shape(O.pack(xs)),h,i,j),S.apart(O.shape(O.pack(ys)),h,U32.sub(i,h),U32.sub(j,h)),
          M.read(xs,h,j),M.read(ys,h,U32.sub(j,h)),
          M.read(M.update(xs,h,i,v),h,j),M.read(M.update(ys,h,U32.sub(i,h),v),h,U32.sub(j,h)),
          p => other(xs,h,i,j,v,p),p => other(ys,h,U32.sub(i,h),U32.sub(j,h),v,p),e))
