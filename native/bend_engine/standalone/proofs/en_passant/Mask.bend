# Extra captured-square removal can reuse the existing arbitrary-mask frame.
import Base
import ../move_update/Spec.bend as Move

def step(+s: Bool,+t: Bool) ->
  {Bool.or(s,Bool.or(t,False{})) == Bool.or(Bool.or(s,t),Bool.or(t,False{})) : Bool}:
  match s t:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def word(+n: Nat,+s: Word(n),+t: Word(n)) ->
  {Word.or(n,s,Word.or(n,t,Word.zero(n))) == Word.or(n,Word.or(n,s,t),Word.or(n,t,Word.zero(n))) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match s t:
        case WCon{sh,st} WCon{th,tt}:
          Equal.trans(Word(1n+p),
            WCon{Bool.or(sh,Bool.or(th,False{})),Word.or(p,st,Word.or(p,tt,Word.zero(p)))},
            WCon{Bool.or(Bool.or(sh,th),Bool.or(th,False{})),Word.or(p,st,Word.or(p,tt,Word.zero(p)))},
            WCon{Bool.or(Bool.or(sh,th),Bool.or(th,False{})),Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p)))},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,st,Word.or(p,tt,Word.zero(p)))},
              Bool.or(sh,Bool.or(th,False{})),Bool.or(Bool.or(sh,th),Bool.or(th,False{})),step(sh,th)),
            Equal.cong(Word(p),Word(1n+p),tail => WCon{Bool.or(Bool.or(sh,th),Bool.or(th,False{})),tail},
              Word.or(p,st,Word.or(p,tt,Word.zero(p))),Word.or(p,Word.or(p,st,tt),Word.or(p,tt,Word.zero(p))),word(p,st,tt)))

def raw(+s: U64,+t: U64) -> U64:
  U64.or(s,U64.or(t,U64.zero()))

def same(+s: U64,+t: U64) -> {raw(s,t) == Move.removal(s,t) : U64}:
  match s t:
    case U64{U32{sl},U32{sh}} U64{U32{tl},U32{th}}:
      Equal.trans(U64,
        U64{U32{Word.or(32n,sl,Word.or(32n,tl,Word.zero(32n)))},U32{Word.or(32n,sh,Word.or(32n,th,Word.zero(32n)))}},
        U64{U32{Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))},U32{Word.or(32n,sh,Word.or(32n,th,Word.zero(32n)))}},
        U64{U32{Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))},U32{Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n)))}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,sh,Word.or(32n,th,Word.zero(32n)))}},
          Word.or(32n,sl,Word.or(32n,tl,Word.zero(32n))),Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n))),word(32n,sl,tl)),
        Equal.cong(Word(32n),U64,h => U64{U32{Word.or(32n,Word.or(32n,sl,tl),Word.or(32n,tl,Word.zero(32n)))},U32{h}},
          Word.or(32n,sh,Word.or(32n,th,Word.zero(32n))),Word.or(32n,Word.or(32n,sh,th),Word.or(32n,th,Word.zero(32n))),word(32n,sh,th)))
