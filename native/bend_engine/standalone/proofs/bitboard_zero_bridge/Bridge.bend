import Base
import ../bitboard_inventory/ClearStep.bend as ClearStep

def cmp_fin_zero(+c: Cmp) -> {Word.cmp.fin(False{},False{},c) == c : Cmp}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

def cmp_fin_set_bit(+c: Cmp) ->
  {Cmp.is_eq(Word.cmp.fin(True{},False{},c)) == False{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

def cmp_zero(+n: Nat,+w: Word(n)) ->
  {Cmp.is_eq(Word.cmp(n,w,Word.zero(n))) == ClearStep.is_zero_word(n,w) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.trans(Bool,
            Cmp.is_eq(Word.cmp(1n+p,w,Word.zero(1n+p))),
            Cmp.is_eq(Word.cmp(p,t,Word.zero(p))),
            ClearStep.is_zero_word(p,t),
            Equal.cong(Cmp,Bool,c => Cmp.is_eq(c),
              Word.cmp.fin(False{},False{},Word.cmp(p,t,Word.zero(p))),
              Word.cmp(p,t,Word.zero(p)),
              cmp_fin_zero(Word.cmp(p,t,Word.zero(p)))),
            cmp_zero(p,t))
        case WCon{True{},t}:
          Equal.trans(Bool,
            Cmp.is_eq(Word.cmp(1n+p,w,Word.zero(1n+p))),
            False{},
            ClearStep.is_zero_word(1n+p,w),
            cmp_fin_set_bit(Word.cmp(p,t,Word.zero(p))),
            {==})

def join_zero(+n: Nat,+m: Nat,+a: Word(n),+b: Word(m)) ->
  {ClearStep.is_zero_word(Nat.add(n,m),Word.join(n,m,a,b)) ==
    Bool.and(ClearStep.is_zero_word(n,a),ClearStep.is_zero_word(m,b)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a:
        case WCon{False{},t}: join_zero(p,m,t,b)
        case WCon{True{},t}: {==}

def u32_is_zero_word(+w: Word(32n)) ->
  {U32.is_zero(U32{w}) == ClearStep.is_zero_word(32n,w) : Bool}:
  cmp_zero(32n,w)

def u64_is_zero_word(+a: U64) ->
  {U64.is_zero(a) == ClearStep.is_zero_word(64n,U64.to_word(a)) : Bool}:
  match a:
    case U64{U32{lo},U32{hi}}:
      Equal.trans(Bool,
        U64.is_zero(U64{U32{lo},U32{hi}}),
        Bool.and(ClearStep.is_zero_word(32n,lo),ClearStep.is_zero_word(32n,hi)),
        ClearStep.is_zero_word(64n,Word.join(32n,32n,lo,hi)),
        Equal.trans(Bool,
          Bool.and(U32.is_zero(U32{lo}),U32.is_zero(U32{hi})),
          Bool.and(ClearStep.is_zero_word(32n,lo),U32.is_zero(U32{hi})),
          Bool.and(ClearStep.is_zero_word(32n,lo),ClearStep.is_zero_word(32n,hi)),
          Equal.cong(Bool,Bool,x => Bool.and(x,U32.is_zero(U32{hi})),
            U32.is_zero(U32{lo}),ClearStep.is_zero_word(32n,lo),u32_is_zero_word(lo)),
          Equal.cong(Bool,Bool,x => Bool.and(ClearStep.is_zero_word(32n,lo),x),
            U32.is_zero(U32{hi}),ClearStep.is_zero_word(32n,hi),u32_is_zero_word(hi))),
        Equal.sym(Bool,
          ClearStep.is_zero_word(64n,Word.join(32n,32n,lo,hi)),
          Bool.and(ClearStep.is_zero_word(32n,lo),ClearStep.is_zero_word(32n,hi)),
          join_zero(32n,32n,lo,hi)))
