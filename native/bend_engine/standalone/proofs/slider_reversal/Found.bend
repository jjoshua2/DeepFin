import Base
import ./Spec.bend as S
import ../attack_reversal/Spec.bend as Member
import ../attack_reversal/Lists.bend as ListProof

def choice(+same: Bool,+h: Nat,+v: S.Segment) ->
  {S.found(Bool.pick(S.Segment,same,S.Between{Nil{}},S.prepend(h,v))) == Bool.or(same,S.found(v)) : Bool}:
  match same v:
    case False{} S.Missing{}: {==}
    case False{} S.Between{xs}: {==}
    case True{} S.Missing{}: {==}
    case True{} S.Between{xs}: {==}

def cut(+dst: Nat,+xs: List<&2,Nat>) ->
  {S.found(S.cut(dst,xs)) == Member.contains(dst,xs) : Bool}:
  match xs:
    case Nil{}: {==}
    case +h <> +tail:
      Equal.trans(Bool,S.found(S.cut(dst,h <> tail)),Bool.or(Nat.is_eq(dst,h),S.found(S.cut(dst,tail))),
        Member.contains(dst,h <> tail),choice(Nat.is_eq(dst,h),h,S.cut(dst,tail)),
        Equal.cong(Bool,Bool,x => Bool.or(Nat.is_eq(dst,h),x),S.found(S.cut(dst,tail)),Member.contains(dst,tail),cut(dst,tail)))

def visible(+v: S.Segment,occ: U64,e: {S.visible(v,occ) == True{} : Bool}) ->
  {S.found(v) == True{} : Bool}:
  match v:
    case S.Missing{}: e
    case S.Between{xs}: {==}

def member(+dst: Nat,+xs: List<&2,Nat>,+occ: U64,e: {S.visible(S.cut(dst,xs),occ) == True{} : Bool}) -> Member.member(dst,xs):
  ListProof.from_true(dst,xs,
    Equal.trans(Bool,Member.contains(dst,xs),S.found(S.cut(dst,xs)),True{},
      Equal.sym(Bool,S.found(S.cut(dst,xs)),Member.contains(dst,xs),cut(dst,xs)),visible(S.cut(dst,xs),occ,e)))
