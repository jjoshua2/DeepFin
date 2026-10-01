import Base
import ./Spec.bend as S
import ./Found.bend as Found
import ./Booleans.bend as B
import ./Certificate.bend as C
import ../attack_reversal/Lists.bend as L
import ../ray/Rays.bend as Rays
import ../layout/Domain.bend as Domain

def inverse(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {S.reverse_dir(S.reverse_dir(dir)) == dir : Nat}:
  match dir:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: Domain.impossible({S.reverse_dir(S.reverse_dir(8n+p)) == 8n+p : Nat},p,bound)

def bounded(+dir: Nat,bound: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {Nat.is_lt(S.reverse_dir(dir),8n) == True{} : Bool}:
  match dir:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+p: Domain.impossible({Nat.is_lt(S.reverse_dir(8n+p),8n) == True{} : Bool},p,bound)

def forward(+src: Nat,+dst: Nat,+dir: Nat,+occ: U64,
  source_bound: {Nat.is_lt(src,64n) == True{} : Bool},direction_bound: {Nat.is_lt(dir,8n) == True{} : Bool},
  +e: {S.visible(S.view(src,dst,dir),occ) == True{} : Bool}) ->
  {S.visible(S.view(dst,src,S.reverse_dir(dir)),occ) == True{} : Bool}:
  Equal.trans(Bool,S.visible(S.view(dst,src,S.reverse_dir(dir)),occ),S.visible(S.mirror(S.view(src,dst,dir)),occ),True{},
    Equal.cong(S.Segment,Bool,v => S.visible(v,occ),S.view(dst,src,S.reverse_dir(dir)),S.mirror(S.view(src,dst,dir)),
      L.at(t => S.neighbor(src,dir,t),Rays.path(7n,src,dir),dst,C.all(src,dir,source_bound,direction_bound),
        Found.member(dst,Rays.path(7n,src,dir),occ,e))),
    Equal.trans(Bool,S.visible(S.mirror(S.view(src,dst,dir)),occ),S.visible(S.view(src,dst,dir),occ),True{},
      B.mirror_visible(S.view(src,dst,dir),occ),e))

def backward(+src: Nat,+dst: Nat,+dir: Nat,+occ: U64,
  target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},+direction_bound: {Nat.is_lt(dir,8n) == True{} : Bool},
  e: {S.visible(S.view(dst,src,S.reverse_dir(dir)),occ) == True{} : Bool}) ->
  {S.visible(S.view(src,dst,dir),occ) == True{} : Bool}:
  Equal.trans(Bool,S.visible(S.view(src,dst,dir),occ),S.visible(S.view(src,dst,S.reverse_dir(S.reverse_dir(dir))),occ),True{},
    Equal.cong(Nat,Bool,d => S.visible(S.view(src,dst,d),occ),dir,S.reverse_dir(S.reverse_dir(dir)),
      Equal.sym(Nat,S.reverse_dir(S.reverse_dir(dir)),dir,inverse(dir,direction_bound))),
    forward(dst,src,S.reverse_dir(dir),occ,target_bound,bounded(dir,direction_bound),e))

def bools(+src: Nat,+dst: Nat,+dir: Nat,+occ: U64,
  source_bound: {Nat.is_lt(src,64n) == True{} : Bool},target_bound: {Nat.is_lt(dst,64n) == True{} : Bool},
  +direction_bound: {Nat.is_lt(dir,8n) == True{} : Bool}) ->
  {S.visible(S.view(src,dst,dir),occ) == S.visible(S.view(dst,src,S.reverse_dir(dir)),occ) : Bool}:
  L.equivalent(S.visible(S.view(src,dst,dir),occ),S.visible(S.view(dst,src,S.reverse_dir(dir)),occ),
    e => forward(src,dst,dir,occ,source_bound,direction_bound,e),
    e => backward(src,dst,dir,occ,target_bound,direction_bound,e))
