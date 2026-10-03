import Base
import ../u64/Words.bend as Old
import ../layout/Domain.bend as Domain
import ../ray/Traversal.bend as NatEq

def bool_cmp_refl(+b: Bool) -> {Bool.cmp(b,b) == EQ{} : Cmp}:
  match b:
    case False{}: {==}
    case True{}: {==}

def cmp_fin_eq_refl(+b: Bool,+c: Cmp,+e: {Cmp.is_eq(c) == True{} : Bool}) ->
  {Cmp.is_eq(Word.cmp.fin(b,b,c)) == True{} : Bool}:
  match c:
    case LT{}: Old.false_true({False{} == True{} : Bool},e)
    case EQ{}:
      Equal.cong(Cmp,Bool,x => Cmp.is_eq(x),Bool.cmp(b,b),EQ{},bool_cmp_refl(b))
    case GT{}: Old.false_true({False{} == True{} : Bool},e)

def cmp_eq_refl(+n: Nat,+w: Word(n)) -> {Cmp.is_eq(Word.cmp(n,w,w)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{b,t}: cmp_fin_eq_refl(b,Word.cmp(p,t,t),cmp_eq_refl(p,t))

def empty_word_same(+a: Word(0n),+b: Word(0n)) -> {a == b : Word(0n)}:
  match a b:
    case WNil{} WNil{}: {==}

def cmp_fin_same_implies(+bit: Bool,+c: Cmp,
  +e: {Cmp.is_eq(Word.cmp.fin(bit,bit,c)) == True{} : Bool}) ->
  {Cmp.is_eq(c) == True{} : Bool}:
  match c:
    case LT{}: Old.false_true({False{} == True{} : Bool},e)
    case EQ{}: {==}
    case GT{}: Old.false_true({False{} == True{} : Bool},e)

def cmp_fin_diff_lr_impossible(+c: Cmp,
  +e: {Cmp.is_eq(Word.cmp.fin(False{},True{},c)) == True{} : Bool}) ->
  {False{} == True{} : Bool}:
  match c:
    case LT{}: Old.false_true({False{} == True{} : Bool},e)
    case EQ{}: Old.false_true({False{} == True{} : Bool},e)
    case GT{}: Old.false_true({False{} == True{} : Bool},e)

def cmp_fin_diff_rl_impossible(+c: Cmp,
  +e: {Cmp.is_eq(Word.cmp.fin(True{},False{},c)) == True{} : Bool}) ->
  {False{} == True{} : Bool}:
  match c:
    case LT{}: Old.false_true({False{} == True{} : Bool},e)
    case EQ{}: Old.false_true({False{} == True{} : Bool},e)
    case GT{}: Old.false_true({False{} == True{} : Bool},e)

def u32_eq_refl(+a: U32) -> {U32.is_eq(a,a) == True{} : Bool}:
  match a:
    case U32{w}: cmp_eq_refl(32n,w)

def cmp_eq_true_same(+n: Nat,+a: Word(n),+b: Word(n),
  +e: {Cmp.is_eq(Word.cmp(n,a,b)) == True{} : Bool}) -> {a == b : Word(n)}:
  match n:
    case 0n: empty_word_same(a,b)
    case 1n+p:
      match a b:
        case WCon{False{},at} WCon{False{},bt}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},at,bt,
            cmp_eq_true_same(p,at,bt,cmp_fin_same_implies(False{},Word.cmp(p,at,bt),e)))
        case WCon{False{},at} WCon{True{},bt}:
          Old.false_true({WCon{False{},at} == WCon{True{},bt} : Word(1n+p)},
            cmp_fin_diff_lr_impossible(Word.cmp(p,at,bt),e))
        case WCon{True{},at} WCon{False{},bt}:
          Old.false_true({WCon{True{},at} == WCon{False{},bt} : Word(1n+p)},
            cmp_fin_diff_rl_impossible(Word.cmp(p,at,bt),e))
        case WCon{True{},at} WCon{True{},bt}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{True{},t},at,bt,
            cmp_eq_true_same(p,at,bt,cmp_fin_same_implies(True{},Word.cmp(p,at,bt),e)))

def u32_eq_true_same(+a: U32,+b: U32,+e: {U32.is_eq(a,b) == True{} : Bool}) -> {a == b : U32}:
  match a b:
    case U32{aw} U32{bw}:
      Equal.cong(Word(32n),U32,w => U32{w},aw,bw,
        cmp_eq_true_same(32n,aw,bw,e))

def u32_eq_true_to_nat(+a: U32,+b: U32,+e: {U32.is_eq(a,b) == True{} : Bool}) ->
  {U32.to_nat(a) == U32.to_nat(b) : Nat}:
  Equal.cong(U32,Nat,x => U32.to_nat(x),a,b,u32_eq_true_same(a,b,e))

def u32_same_to_eq_true(+a: U32,+b: U32,+e: {U32.to_nat(a) == U32.to_nat(b) : Nat}) ->
  {U32.is_eq(a,b) == True{} : Bool}:
  Equal.trans(Bool,U32.is_eq(a,b),U32.is_eq(b,b),True{},
    Equal.cong(U32,Bool,x => U32.is_eq(x,b),a,b,Domain.injective(a,b,e)),u32_eq_refl(b))

def nat_cmp_refl(+a: Nat) -> {Nat.cmp(a,a) == EQ{} : Cmp}:
  match a:
    case 0n: {==}
    case 1n+p: nat_cmp_refl(p)

def nat_is_eq_refl(+a: Nat) -> {Nat.is_eq(a,a) == True{} : Bool}:
  Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),Nat.cmp(a,a),EQ{},nat_cmp_refl(a))

def nat_same_to_eq_true(+a: Nat,+b: Nat,+e: {a == b : Nat}) ->
  {Nat.is_eq(a,b) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_eq(a,b),Nat.is_eq(b,b),True{},
    Equal.cong(Nat,Bool,x => Nat.is_eq(x,b),a,b,e),nat_is_eq_refl(b))

def u32_eq_nat_eq_cases(+a: U32,+b: U32,+ab: Bool,+ba: Bool,
  +eab: {U32.is_eq(a,b) == ab : Bool},
  +eba: {Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) == ba : Bool}) ->
  {U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  match ab ba:
    case True{} True{}:
      Equal.trans(Bool,U32.is_eq(a,b),True{},Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),
        eab,Equal.sym(Bool,Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),True{},eba))
    case True{} False{}:
      Old.false_true({U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool},
        Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,True{},Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),False{},
          Equal.sym(Bool,Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),True{},
            nat_same_to_eq_true(U32.to_nat(a),U32.to_nat(b),
              u32_eq_true_to_nat(a,b,eab))),
          eba)))
    case False{} True{}:
      Old.false_true({U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool},
        Equal.trans(Bool,False{},U32.is_eq(a,b),True{},
          Equal.sym(Bool,U32.is_eq(a,b),False{},eab),
          u32_same_to_eq_true(a,b,NatEq.nat_eq(U32.to_nat(a),U32.to_nat(b),eba))))
    case False{} False{}:
      Equal.trans(Bool,U32.is_eq(a,b),False{},Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),
        eab,Equal.sym(Bool,Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),False{},eba))

def u32_eq_nat_eq(+a: U32,+b: U32) ->
  {U32.is_eq(a,b) == Nat.is_eq(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  u32_eq_nat_eq_cases(a,b,U32.is_eq(a,b),Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),{==},{==})
