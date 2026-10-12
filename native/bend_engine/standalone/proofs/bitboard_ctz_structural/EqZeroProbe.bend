import Base

def and_right_false(+b: Bool) -> {Bool.and(b,False{}) == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def and_right_true(+b: Bool) -> {Bool.and(b,True{}) == b : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def fin_eq_and(+alo: U32,+blo: U32,+hi: Cmp) ->
  {Cmp.is_eq(U64.cmp.fin(alo,blo,hi)) == Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),Cmp.is_eq(hi)) : Bool}:
  match hi:
    case LT{}:
      Equal.trans(Bool,False{},Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),False{}),
        Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),Cmp.is_eq(LT{})),
        Equal.sym(Bool,Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),False{}),False{},
          and_right_false(Cmp.is_eq(U32.cmp(alo,blo)))),{==})
    case EQ{}:
      Equal.sym(Bool,Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),True{}),
        Cmp.is_eq(U32.cmp(alo,blo)),and_right_true(Cmp.is_eq(U32.cmp(alo,blo))))
    case GT{}:
      Equal.trans(Bool,False{},Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),False{}),
        Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),Cmp.is_eq(GT{})),
        Equal.sym(Bool,Bool.and(Cmp.is_eq(U32.cmp(alo,blo)),False{}),False{},
          and_right_false(Cmp.is_eq(U32.cmp(alo,blo)))),{==})

def eq_zero_is_zero(+a: U64) -> {U64.is_eq(a,U64.zero()) == U64.is_zero(a) : Bool}:
  match a:
    case U64{lo,hi}:
      Equal.trans(Bool,
        U64.is_eq(U64{lo,hi},U64.zero()),
        Cmp.is_eq(U64.cmp.fin(lo,0,U32.cmp(hi,0))),
        U64.is_zero(U64{lo,hi}),
        {==},
        Equal.trans(Bool,
          Cmp.is_eq(U64.cmp.fin(lo,0,U32.cmp(hi,0))),
          Bool.and(Cmp.is_eq(U32.cmp(lo,0)),Cmp.is_eq(U32.cmp(hi,0))),
          U64.is_zero(U64{lo,hi}),
          fin_eq_and(lo,0,U32.cmp(hi,0)),
          {==}))
