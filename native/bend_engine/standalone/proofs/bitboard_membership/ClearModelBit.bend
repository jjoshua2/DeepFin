# Structural per-index behavior of the clear model; no actual Chess generator claim.
import Base
import ../bitboard_inventory/ClearStep.bend as ClearStep
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz

def and_true_right(+b: Bool) ->
  {Bool.and(b,True{}) == b : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def eq_succ(+a: Nat,+b: Nat) ->
  {Nat.is_eq(Nat.add(1n,a),Nat.add(1n,b)) == Nat.is_eq(a,b) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: eq_succ(p,q)

def clear_model_bit(+n: Nat,+w: Word(n),+i: Nat) ->
  {Ctz.word_bit(n,ClearStep.clear_model(n,w),i) ==
    Bool.and(Ctz.word_bit(n,w,i),Bool.not(Nat.is_eq(i,Ctz.first_set(n,w)))) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          match i:
            case 0n: {==}
            case 1n+q:
              Equal.trans(Bool,
                Ctz.word_bit(p,ClearStep.clear_model(p,t),q),
                Bool.and(Ctz.word_bit(p,t,q),Bool.not(Nat.is_eq(q,Ctz.first_set(p,t)))),
                Bool.and(Ctz.word_bit(1n+p,WCon{False{},t},1n+q),
                  Bool.not(Nat.is_eq(1n+q,Ctz.first_set(1n+p,WCon{False{},t})))),
                clear_model_bit(p,t,q),
                Equal.cong(Nat,Bool,x => Bool.and(Ctz.word_bit(p,t,q),Bool.not(Nat.is_eq(q,x))),
                  Ctz.first_set(p,t),Ctz.first_set(p,t),{==}))
        case WCon{True{},t}:
          match i:
            case 0n: {==}
            case 1n+q:
              Equal.sym(Bool,
                Bool.and(Ctz.word_bit(p,t,q),True{}),Ctz.word_bit(p,t,q),
                and_true_right(Ctz.word_bit(p,t,q)))
