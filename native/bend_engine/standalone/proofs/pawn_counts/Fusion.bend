# Both planes clear the actual destination before either inserts it.
import Base
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def mask_bool(+s: Bool,+c: Bool,+t: Bool,+r: Bool) -> Bool:
  Bool.or(Bool.or(s,c),Bool.or(t,r))
def mask_word(+n: Nat,+s: Word(n),+c: Word(n),+t: Word(n),+r: Word(n)) -> Word(n):
  Word.or(n,Word.or(n,s,c),Word.or(n,t,r))
def mask(+s: U64,+c: U64,+t: U64,+r: U64) -> U64:
  U64.or(U64.or(s,c),U64.or(t,r))
def update_word(+n: Nat,+a: Word(n),+removed: Word(n),+target: Word(n),put: Bool) -> Word(n):
  Word.or(n,Word.and(n,a,Word.not(n,removed)),Bool.pick(Word(n),put,target,Word.zero(n)))
def update_bool(+a: Bool,+removed: Bool,+target: Bool,put: Bool) -> Bool:
  Bool.or(Bool.and(a,Bool.not(removed)),Bool.pick(Bool,put,target,False{}))

def target_clear(+s: Bool,+c: Bool,+t: Bool,+r: Bool) -> {Bool.and(t,Bool.not(mask_bool(s,c,t,r))) == False{} : Bool}:
  match s c t r:
    case False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{}: {==}
    case False{} True{} True{} True{}: {==}
    case True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{}: {==}
    case True{} False{} True{} True{}: {==}
    case True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{}: {==}

def fused_bool(+pawn: Bool,+color: Bool,+src: Bool,+cap: Bool,+target: Bool,+rook: Bool,+p: Bool,+c: Bool) ->
  {Bool.and(update_bool(pawn,mask_bool(src,cap,target,rook),target,p),update_bool(color,mask_bool(src,cap,target,rook),target,c)) ==
    update_bool(Bool.and(pawn,color),mask_bool(src,cap,target,rook),target,Bool.and(p,c)) : Bool}:
  match pawn color src cap target rook p c:
    case False{} False{} False{} False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} False{} False{} True{} False{} True{}: {==}
    case False{} False{} False{} False{} False{} True{} True{} False{}: {==}
    case False{} False{} False{} False{} False{} True{} True{} True{}: {==}
    case False{} False{} False{} False{} True{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} True{} False{} False{} True{}: {==}
    case False{} False{} False{} False{} True{} False{} True{} False{}: {==}
    case False{} False{} False{} False{} True{} False{} True{} True{}: {==}
    case False{} False{} False{} False{} True{} True{} False{} False{}: {==}
    case False{} False{} False{} False{} True{} True{} False{} True{}: {==}
    case False{} False{} False{} False{} True{} True{} True{} False{}: {==}
    case False{} False{} False{} False{} True{} True{} True{} True{}: {==}
    case False{} False{} False{} True{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} True{} False{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} False{} True{} False{} True{}: {==}
    case False{} False{} False{} True{} False{} True{} True{} False{}: {==}
    case False{} False{} False{} True{} False{} True{} True{} True{}: {==}
    case False{} False{} False{} True{} True{} False{} False{} False{}: {==}
    case False{} False{} False{} True{} True{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} True{} False{} True{} False{}: {==}
    case False{} False{} False{} True{} True{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} True{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{} True{} False{} True{}: {==}
    case False{} False{} False{} True{} True{} True{} True{} False{}: {==}
    case False{} False{} False{} True{} True{} True{} True{} True{}: {==}
    case False{} False{} True{} False{} False{} False{} False{} False{}: {==}
    case False{} False{} True{} False{} False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{} False{} False{} True{} False{}: {==}
    case False{} False{} True{} False{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} False{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} False{} True{} False{} False{} True{}: {==}
    case False{} False{} True{} False{} True{} False{} True{} False{}: {==}
    case False{} False{} True{} False{} True{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} True{} True{} False{} False{}: {==}
    case False{} False{} True{} False{} True{} True{} False{} True{}: {==}
    case False{} False{} True{} False{} True{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} False{} False{} False{} True{}: {==}
    case False{} False{} True{} True{} False{} False{} True{} False{}: {==}
    case False{} False{} True{} True{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} True{} False{} True{} False{} False{}: {==}
    case False{} False{} True{} True{} False{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} True{} False{} False{} True{}: {==}
    case False{} False{} True{} True{} True{} False{} True{} False{}: {==}
    case False{} False{} True{} True{} True{} False{} True{} True{}: {==}
    case False{} False{} True{} True{} True{} True{} False{} False{}: {==}
    case False{} False{} True{} True{} True{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} True{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} False{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} False{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{} False{} True{} False{} False{}: {==}
    case False{} True{} False{} False{} False{} True{} False{} True{}: {==}
    case False{} True{} False{} False{} False{} True{} True{} False{}: {==}
    case False{} True{} False{} False{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} False{} True{} True{} False{} True{}: {==}
    case False{} True{} False{} False{} True{} True{} True{} False{}: {==}
    case False{} True{} False{} False{} True{} True{} True{} True{}: {==}
    case False{} True{} False{} True{} False{} False{} False{} False{}: {==}
    case False{} True{} False{} True{} False{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} False{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} False{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} False{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} False{} True{} False{} True{}: {==}
    case False{} True{} False{} True{} False{} True{} True{} False{}: {==}
    case False{} True{} False{} True{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} True{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} True{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} True{} False{} True{} False{}: {==}
    case False{} True{} False{} True{} True{} False{} True{} True{}: {==}
    case False{} True{} False{} True{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{} True{} False{} True{}: {==}
    case False{} True{} False{} True{} True{} True{} True{} False{}: {==}
    case False{} True{} False{} True{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} False{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} False{} False{} False{} False{} True{}: {==}
    case False{} True{} True{} False{} False{} False{} True{} False{}: {==}
    case False{} True{} True{} False{} False{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} False{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} False{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} False{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} False{} True{} True{} True{}: {==}
    case False{} True{} True{} False{} True{} False{} False{} False{}: {==}
    case False{} True{} True{} False{} True{} False{} False{} True{}: {==}
    case False{} True{} True{} False{} True{} False{} True{} False{}: {==}
    case False{} True{} True{} False{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} False{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} False{} False{} False{} True{}: {==}
    case False{} True{} True{} True{} False{} False{} True{} False{}: {==}
    case False{} True{} True{} True{} False{} False{} True{} True{}: {==}
    case False{} True{} True{} True{} False{} True{} False{} False{}: {==}
    case False{} True{} True{} True{} False{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} False{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} False{} True{} True{} True{}: {==}
    case False{} True{} True{} True{} True{} False{} False{} False{}: {==}
    case False{} True{} True{} True{} True{} False{} False{} True{}: {==}
    case False{} True{} True{} True{} True{} False{} True{} False{}: {==}
    case False{} True{} True{} True{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} True{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} True{} True{} True{} False{} True{}: {==}
    case False{} True{} True{} True{} True{} True{} True{} False{}: {==}
    case False{} True{} True{} True{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} False{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} False{} False{} True{} False{} True{}: {==}
    case True{} False{} False{} False{} False{} True{} True{} False{}: {==}
    case True{} False{} False{} False{} False{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} True{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} True{} False{} False{} True{}: {==}
    case True{} False{} False{} False{} True{} False{} True{} False{}: {==}
    case True{} False{} False{} False{} True{} False{} True{} True{}: {==}
    case True{} False{} False{} False{} True{} True{} False{} False{}: {==}
    case True{} False{} False{} False{} True{} True{} False{} True{}: {==}
    case True{} False{} False{} False{} True{} True{} True{} False{}: {==}
    case True{} False{} False{} False{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} True{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} True{} False{} False{} False{} True{}: {==}
    case True{} False{} False{} True{} False{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} False{} True{} False{} True{}: {==}
    case True{} False{} False{} True{} False{} True{} True{} False{}: {==}
    case True{} False{} False{} True{} False{} True{} True{} True{}: {==}
    case True{} False{} False{} True{} True{} False{} False{} False{}: {==}
    case True{} False{} False{} True{} True{} False{} False{} True{}: {==}
    case True{} False{} False{} True{} True{} False{} True{} False{}: {==}
    case True{} False{} False{} True{} True{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} True{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} True{} True{} False{} True{}: {==}
    case True{} False{} False{} True{} True{} True{} True{} False{}: {==}
    case True{} False{} False{} True{} True{} True{} True{} True{}: {==}
    case True{} False{} True{} False{} False{} False{} False{} False{}: {==}
    case True{} False{} True{} False{} False{} False{} False{} True{}: {==}
    case True{} False{} True{} False{} False{} False{} True{} False{}: {==}
    case True{} False{} True{} False{} False{} False{} True{} True{}: {==}
    case True{} False{} True{} False{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} False{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} False{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} False{} True{} False{} False{} True{}: {==}
    case True{} False{} True{} False{} True{} False{} True{} False{}: {==}
    case True{} False{} True{} False{} True{} False{} True{} True{}: {==}
    case True{} False{} True{} False{} True{} True{} False{} False{}: {==}
    case True{} False{} True{} False{} True{} True{} False{} True{}: {==}
    case True{} False{} True{} False{} True{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} True{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} False{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} False{} False{} False{} True{}: {==}
    case True{} False{} True{} True{} False{} False{} True{} False{}: {==}
    case True{} False{} True{} True{} False{} False{} True{} True{}: {==}
    case True{} False{} True{} True{} False{} True{} False{} False{}: {==}
    case True{} False{} True{} True{} False{} True{} False{} True{}: {==}
    case True{} False{} True{} True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} True{} False{} False{} True{}: {==}
    case True{} False{} True{} True{} True{} False{} True{} False{}: {==}
    case True{} False{} True{} True{} True{} False{} True{} True{}: {==}
    case True{} False{} True{} True{} True{} True{} False{} False{}: {==}
    case True{} False{} True{} True{} True{} True{} False{} True{}: {==}
    case True{} False{} True{} True{} True{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} True{} True{} True{} True{}: {==}
    case True{} True{} False{} False{} False{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} False{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} False{} False{} True{} True{}: {==}
    case True{} True{} False{} False{} False{} True{} False{} False{}: {==}
    case True{} True{} False{} False{} False{} True{} False{} True{}: {==}
    case True{} True{} False{} False{} False{} True{} True{} False{}: {==}
    case True{} True{} False{} False{} False{} True{} True{} True{}: {==}
    case True{} True{} False{} False{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} False{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} False{} True{} True{} False{} True{}: {==}
    case True{} True{} False{} False{} True{} True{} True{} False{}: {==}
    case True{} True{} False{} False{} True{} True{} True{} True{}: {==}
    case True{} True{} False{} True{} False{} False{} False{} False{}: {==}
    case True{} True{} False{} True{} False{} False{} False{} True{}: {==}
    case True{} True{} False{} True{} False{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} False{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} False{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} False{} True{} False{} True{}: {==}
    case True{} True{} False{} True{} False{} True{} True{} False{}: {==}
    case True{} True{} False{} True{} False{} True{} True{} True{}: {==}
    case True{} True{} False{} True{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} True{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} True{} True{} False{} True{} False{}: {==}
    case True{} True{} False{} True{} True{} False{} True{} True{}: {==}
    case True{} True{} False{} True{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} True{} True{} False{} True{}: {==}
    case True{} True{} False{} True{} True{} True{} True{} False{}: {==}
    case True{} True{} False{} True{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} False{} False{} False{} False{} False{}: {==}
    case True{} True{} True{} False{} False{} False{} False{} True{}: {==}
    case True{} True{} True{} False{} False{} False{} True{} False{}: {==}
    case True{} True{} True{} False{} False{} False{} True{} True{}: {==}
    case True{} True{} True{} False{} False{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} False{} True{} False{} True{}: {==}
    case True{} True{} True{} False{} False{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} False{} True{} True{} True{}: {==}
    case True{} True{} True{} False{} True{} False{} False{} False{}: {==}
    case True{} True{} True{} False{} True{} False{} False{} True{}: {==}
    case True{} True{} True{} False{} True{} False{} True{} False{}: {==}
    case True{} True{} True{} False{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} False{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} False{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} True{} False{} False{} False{} False{}: {==}
    case True{} True{} True{} True{} False{} False{} False{} True{}: {==}
    case True{} True{} True{} True{} False{} False{} True{} False{}: {==}
    case True{} True{} True{} True{} False{} False{} True{} True{}: {==}
    case True{} True{} True{} True{} False{} True{} False{} False{}: {==}
    case True{} True{} True{} True{} False{} True{} False{} True{}: {==}
    case True{} True{} True{} True{} False{} True{} True{} False{}: {==}
    case True{} True{} True{} True{} False{} True{} True{} True{}: {==}
    case True{} True{} True{} True{} True{} False{} False{} False{}: {==}
    case True{} True{} True{} True{} True{} False{} False{} True{}: {==}
    case True{} True{} True{} True{} True{} False{} True{} False{}: {==}
    case True{} True{} True{} True{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} True{} True{} True{} False{} True{}: {==}
    case True{} True{} True{} True{} True{} True{} True{} False{}: {==}
    case True{} True{} True{} True{} True{} True{} True{} True{}: {==}

def fused_word(+n: Nat,+pawn: Word(n),+color: Word(n),+src: Word(n),+cap: Word(n),+target: Word(n),+rook: Word(n),+p: Bool,+c: Bool) ->
  {Word.and(n,update_word(n,pawn,mask_word(n,src,cap,target,rook),target,p),
    update_word(n,color,mask_word(n,src,cap,target,rook),target,c)) ==
    update_word(n,Word.and(n,pawn,color),mask_word(n,src,cap,target,rook),target,Bool.and(p,c)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+k:
      match pawn color src cap target rook:
        case WCon{ph,pt} WCon{ch,ct} WCon{sh,st} WCon{ah,at} WCon{th,tt} WCon{rh,rt}:
          match p c:
            case False{} False{}:
              Equal.trans(Word(1n+k),WCon{Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,False{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,False{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},False{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},False{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(False{},False{}))},
                Equal.cong(Bool,Word(1n+k),h => WCon{h,Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,False{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,False{})),update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},False{})),fused_bool(ph,ch,sh,ah,th,rh,False{},False{})),
                Equal.cong(Word(k),Word(1n+k),t => WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},False{})),t},Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(False{},False{})),fused_word(k,pt,ct,st,at,tt,rt,False{},False{})))
            case False{} True{}:
              Equal.trans(Word(1n+k),WCon{Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,False{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,True{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},True{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},True{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(False{},True{}))},
                Equal.cong(Bool,Word(1n+k),h => WCon{h,Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,False{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,True{})),update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},True{})),fused_bool(ph,ch,sh,ah,th,rh,False{},True{})),
                Equal.cong(Word(k),Word(1n+k),t => WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(False{},True{})),t},Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,False{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(False{},True{})),fused_word(k,pt,ct,st,at,tt,rt,False{},True{})))
            case True{} False{}:
              Equal.trans(Word(1n+k),WCon{Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,True{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,False{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},False{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},False{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(True{},False{}))},
                Equal.cong(Bool,Word(1n+k),h => WCon{h,Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{}))},Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,True{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,False{})),update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},False{})),fused_bool(ph,ch,sh,ah,th,rh,True{},False{})),
                Equal.cong(Word(k),Word(1n+k),t => WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},False{})),t},Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,False{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(True{},False{})),fused_word(k,pt,ct,st,at,tt,rt,True{},False{})))
            case True{} True{}:
              Equal.trans(Word(1n+k),WCon{Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,True{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,True{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},True{})),Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},True{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(True{},True{}))},
                Equal.cong(Bool,Word(1n+k),h => WCon{h,Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{}))},Bool.and(update_bool(ph,mask_bool(sh,ah,th,rh),th,True{}),update_bool(ch,mask_bool(sh,ah,th,rh),th,True{})),update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},True{})),fused_bool(ph,ch,sh,ah,th,rh,True{},True{})),
                Equal.cong(Word(k),Word(1n+k),t => WCon{update_bool(Bool.and(ph,ch),mask_bool(sh,ah,th,rh),th,Bool.and(True{},True{})),t},Word.and(k,update_word(k,pt,mask_word(k,st,at,tt,rt),tt,True{}),update_word(k,ct,mask_word(k,st,at,tt,rt),tt,True{})),update_word(k,Word.and(k,pt,ct),mask_word(k,st,at,tt,rt),tt,Bool.and(True{},True{})),fused_word(k,pt,ct,st,at,tt,rt,True{},True{})))

def fused(+pawn: U64,+color: U64,+src: U64,+cap: U64,+target: U64,+rook: U64,+p: Bool,+c: Bool) ->
  {U64.and(Chess.update_piece(pawn,mask(src,cap,target,rook),target,p),
    Chess.update_piece(color,mask(src,cap,target,rook),target,c)) ==
    Chess.update_piece(U64.and(pawn,color),mask(src,cap,target,rook),target,Bool.and(p,c)) : U64}:
  match pawn color src cap target rook:
    case U64{U32{pl},U32{ph}} U64{U32{cl},U32{ch}} U64{U32{sl},U32{sh}} U64{U32{al},U32{ah}} U64{U32{tl},U32{th}} U64{U32{rl},U32{rh}}:
      match p c:
        case False{} False{}:
          Equal.trans(U64,U64{U32{Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,False{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,False{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},False{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},False{}))},U32{update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(False{},False{}))}},
            Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,False{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,False{})),update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},False{})),fused_word(32n,pl,cl,sl,al,tl,rl,False{},False{})),
            Equal.cong(Word(32n),U64,h => U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},False{}))},U32{h}},Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{})),update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(False{},False{})),fused_word(32n,ph,ch,sh,ah,th,rh,False{},False{})))
        case False{} True{}:
          Equal.trans(U64,U64{U32{Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,False{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,True{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},True{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},True{}))},U32{update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(False{},True{}))}},
            Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,False{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,True{})),update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},True{})),fused_word(32n,pl,cl,sl,al,tl,rl,False{},True{})),
            Equal.cong(Word(32n),U64,h => U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(False{},True{}))},U32{h}},Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,False{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{})),update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(False{},True{})),fused_word(32n,ph,ch,sh,ah,th,rh,False{},True{})))
        case True{} False{}:
          Equal.trans(U64,U64{U32{Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,True{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,False{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},False{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},False{}))},U32{update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(True{},False{}))}},
            Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{}))}},Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,True{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,False{})),update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},False{})),fused_word(32n,pl,cl,sl,al,tl,rl,True{},False{})),
            Equal.cong(Word(32n),U64,h => U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},False{}))},U32{h}},Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,False{})),update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(True{},False{})),fused_word(32n,ph,ch,sh,ah,th,rh,True{},False{})))
        case True{} True{}:
          Equal.trans(U64,U64{U32{Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,True{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,True{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},True{}))},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},True{}))},U32{update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(True{},True{}))}},
            Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{}))}},Word.and(32n,update_word(32n,pl,mask_word(32n,sl,al,tl,rl),tl,True{}),update_word(32n,cl,mask_word(32n,sl,al,tl,rl),tl,True{})),update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},True{})),fused_word(32n,pl,cl,sl,al,tl,rl,True{},True{})),
            Equal.cong(Word(32n),U64,h => U64{U32{update_word(32n,Word.and(32n,pl,cl),mask_word(32n,sl,al,tl,rl),tl,Bool.and(True{},True{}))},U32{h}},Word.and(32n,update_word(32n,ph,mask_word(32n,sh,ah,th,rh),th,True{}),update_word(32n,ch,mask_word(32n,sh,ah,th,rh),th,True{})),update_word(32n,Word.and(32n,ph,ch),mask_word(32n,sh,ah,th,rh),th,Bool.and(True{},True{})),fused_word(32n,ph,ch,sh,ah,th,rh,True{},True{})))
