# EP bounds from the actual source-kind/xor trigger, without move geometry.
import Base
import ../bitboard_membership/IndexEquality.bend as Eq
import ../layout/Domain.bend as Domain
import ../u64/Words.bend as Old
import ../../../legal_probe/Chess.bend as Chess

def src(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f}=m
  s

def bounded(-m: Chess.Ply) -> Type:
  {Nat.is_lt(U32.to_nat(src(m)),64n) == True{} : Bool}

def ep(-sq: U32) -> Type:
  {sq == 64 : U32} | {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}

def xor_step(+p: Nat,+x: Bool,+xs: Word(p),+y: Bool,+ys: Word(p),tail: {Word.xor(p,xs,Word.xor(p,xs,ys)) == ys : Word(p)}) ->
  {Word.xor(1n+p,WCon{x,xs},Word.xor(1n+p,WCon{x,xs},WCon{y,ys})) == WCon{y,ys} : Word(1n+p)}:
  match x y:
    case False{} False{}:
      Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},Word.xor(p,xs,Word.xor(p,xs,ys)),ys,tail)
    case False{} True{}:
      Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},Word.xor(p,xs,Word.xor(p,xs,ys)),ys,tail)
    case True{} False{}:
      Equal.cong(Word(p),Word(1n+p),w => WCon{False{},w},Word.xor(p,xs,Word.xor(p,xs,ys)),ys,tail)
    case True{} True{}:
      Equal.cong(Word(p),Word(1n+p),w => WCon{True{},w},Word.xor(p,xs,Word.xor(p,xs,ys)),ys,tail)

def xor_cancel(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {Word.xor(n,a,Word.xor(n,a,b)) == b : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{+x,+xs} WCon{+y,+ys}:
          xor_step(p,x,xs,y,ys,xor_cancel(p,xs,ys))


def u32_cancel(+a: U32,+b: U32) -> {U32.xor(a,U32.xor(a,b)) == b : U32}:
  match a b:
    case U32{aw} U32{bw}:
      Equal.cong(Word(32n),U32,w => U32{w},Word.xor(32n,aw,Word.xor(32n,aw,bw)),bw,xor_cancel(32n,aw,bw))

def destination(+s: U32,+d: U32,e: {U32.is_eq(U32.xor(s,d),16) == True{} : Bool}) ->
  {d == U32.xor(s,16) : U32}:
  Equal.trans(U32,d,U32.xor(s,U32.xor(s,d)),U32.xor(s,16),
    Equal.sym(U32,U32.xor(s,U32.xor(s,d)),d,u32_cancel(s,d)),
    Equal.cong(U32,U32,x => U32.xor(s,x),U32.xor(s,d),16,Eq.u32_eq_true_same(U32.xor(s,d),16,e)))

def midpoint(+s: U32) -> U32:
  U32.div(U32.add(s,U32.xor(s,16)),2)

# Sixty-four bounded source cases; never enumerate source/destination pairs.
def midpoint_cases(+s: U32,n: Nat,
  same: {U32.to_nat(s) == n : Nat},
  below: {Nat.is_lt(n,64n) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(midpoint(s)),64n) == True{} : Bool}:
  match n:
    case 0n:
      %Equal.sym(U32,s,0,Domain.injective(s,0,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 1n:
      %Equal.sym(U32,s,1,Domain.injective(s,1,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 2n:
      %Equal.sym(U32,s,2,Domain.injective(s,2,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 3n:
      %Equal.sym(U32,s,3,Domain.injective(s,3,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 4n:
      %Equal.sym(U32,s,4,Domain.injective(s,4,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 5n:
      %Equal.sym(U32,s,5,Domain.injective(s,5,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 6n:
      %Equal.sym(U32,s,6,Domain.injective(s,6,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 7n:
      %Equal.sym(U32,s,7,Domain.injective(s,7,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 8n:
      %Equal.sym(U32,s,8,Domain.injective(s,8,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 9n:
      %Equal.sym(U32,s,9,Domain.injective(s,9,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 10n:
      %Equal.sym(U32,s,10,Domain.injective(s,10,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 11n:
      %Equal.sym(U32,s,11,Domain.injective(s,11,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 12n:
      %Equal.sym(U32,s,12,Domain.injective(s,12,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 13n:
      %Equal.sym(U32,s,13,Domain.injective(s,13,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 14n:
      %Equal.sym(U32,s,14,Domain.injective(s,14,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 15n:
      %Equal.sym(U32,s,15,Domain.injective(s,15,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 16n:
      %Equal.sym(U32,s,16,Domain.injective(s,16,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 17n:
      %Equal.sym(U32,s,17,Domain.injective(s,17,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 18n:
      %Equal.sym(U32,s,18,Domain.injective(s,18,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 19n:
      %Equal.sym(U32,s,19,Domain.injective(s,19,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 20n:
      %Equal.sym(U32,s,20,Domain.injective(s,20,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 21n:
      %Equal.sym(U32,s,21,Domain.injective(s,21,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 22n:
      %Equal.sym(U32,s,22,Domain.injective(s,22,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 23n:
      %Equal.sym(U32,s,23,Domain.injective(s,23,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 24n:
      %Equal.sym(U32,s,24,Domain.injective(s,24,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 25n:
      %Equal.sym(U32,s,25,Domain.injective(s,25,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 26n:
      %Equal.sym(U32,s,26,Domain.injective(s,26,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 27n:
      %Equal.sym(U32,s,27,Domain.injective(s,27,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 28n:
      %Equal.sym(U32,s,28,Domain.injective(s,28,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 29n:
      %Equal.sym(U32,s,29,Domain.injective(s,29,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 30n:
      %Equal.sym(U32,s,30,Domain.injective(s,30,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 31n:
      %Equal.sym(U32,s,31,Domain.injective(s,31,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 32n:
      %Equal.sym(U32,s,32,Domain.injective(s,32,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 33n:
      %Equal.sym(U32,s,33,Domain.injective(s,33,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 34n:
      %Equal.sym(U32,s,34,Domain.injective(s,34,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 35n:
      %Equal.sym(U32,s,35,Domain.injective(s,35,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 36n:
      %Equal.sym(U32,s,36,Domain.injective(s,36,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 37n:
      %Equal.sym(U32,s,37,Domain.injective(s,37,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 38n:
      %Equal.sym(U32,s,38,Domain.injective(s,38,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 39n:
      %Equal.sym(U32,s,39,Domain.injective(s,39,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 40n:
      %Equal.sym(U32,s,40,Domain.injective(s,40,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 41n:
      %Equal.sym(U32,s,41,Domain.injective(s,41,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 42n:
      %Equal.sym(U32,s,42,Domain.injective(s,42,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 43n:
      %Equal.sym(U32,s,43,Domain.injective(s,43,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 44n:
      %Equal.sym(U32,s,44,Domain.injective(s,44,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 45n:
      %Equal.sym(U32,s,45,Domain.injective(s,45,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 46n:
      %Equal.sym(U32,s,46,Domain.injective(s,46,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 47n:
      %Equal.sym(U32,s,47,Domain.injective(s,47,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 48n:
      %Equal.sym(U32,s,48,Domain.injective(s,48,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 49n:
      %Equal.sym(U32,s,49,Domain.injective(s,49,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 50n:
      %Equal.sym(U32,s,50,Domain.injective(s,50,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 51n:
      %Equal.sym(U32,s,51,Domain.injective(s,51,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 52n:
      %Equal.sym(U32,s,52,Domain.injective(s,52,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 53n:
      %Equal.sym(U32,s,53,Domain.injective(s,53,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 54n:
      %Equal.sym(U32,s,54,Domain.injective(s,54,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 55n:
      %Equal.sym(U32,s,55,Domain.injective(s,55,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 56n:
      %Equal.sym(U32,s,56,Domain.injective(s,56,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 57n:
      %Equal.sym(U32,s,57,Domain.injective(s,57,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 58n:
      %Equal.sym(U32,s,58,Domain.injective(s,58,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 59n:
      %Equal.sym(U32,s,59,Domain.injective(s,59,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 60n:
      %Equal.sym(U32,s,60,Domain.injective(s,60,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 61n:
      %Equal.sym(U32,s,61,Domain.injective(s,61,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 62n:
      %Equal.sym(U32,s,62,Domain.injective(s,62,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 63n:
      %Equal.sym(U32,s,63,Domain.injective(s,63,same)) : {Nat.is_lt(U32.to_nat(midpoint(_)),64n) == True{} : Bool}
      {==}
    case 64n+p:
      Domain.impossible({Nat.is_lt(U32.to_nat(midpoint(s)),64n) == True{} : Bool},p,below)

def average(+s: U32,+d: U32,
  below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  trigger: {U32.is_eq(U32.xor(s,d),16) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.div(U32.add(s,d),2)),64n) == True{} : Bool}:
  %Equal.sym(U32,d,U32.xor(s,16),destination(s,d,trigger)) :
    {Nat.is_lt(U32.to_nat(U32.div(U32.add(s,_),2)),64n) == True{} : Bool}
  midpoint_cases(s,U32.to_nat(s),{==},below)

def generated_classified(+s: U32,+d: U32,pawn: Bool,double: Bool,
  below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},
  trigger: {U32.is_eq(U32.xor(s,d),16) == double : Bool}) ->
  ep(Bool.pick(U32,Bool.and(pawn,double),U32.div(U32.add(s,d),2),64)):
  match pawn double:
    case False{} _: Inl{{==}}
    case True{} False{}: Inl{{==}}
    case True{} True{}: Inr{average(s,d,below,trigger)}

def generated(+s: U32,+d: U32,kind: U32,
  below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool}) ->
  ep(Bool.pick(U32,Bool.and(U32.is_zero(kind),U32.is_eq(U32.xor(s,d),16)),U32.div(U32.add(s,d),2),64)):
  generated_classified(s,d,U32.is_zero(kind),U32.is_eq(U32.xor(s,d),16),below,{==})

def dst(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f}=m
  d


def unfold(+b: Chess.Board,+m: Chess.Ply) ->
  {Chess.get_ep(Chess.make_move(b,m)) ==
    Bool.pick(U32,Bool.and(U32.is_zero(Chess.piece(b,src(m))),U32.is_eq(U32.xor(src(m),dst(m)),16)),
      U32.div(U32.add(src(m),dst(m)),2),64) : U32}:
  match b m:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,old_ep} Chess.Ply{s,d,promo,flag}: {==}


def post(+b: Chess.Board,+m: Chess.Ply,below: bounded(m)) -> ep(Chess.get_ep(Chess.make_move(b,m))):
  %Equal.sym(U32,Chess.get_ep(Chess.make_move(b,m)),
    Bool.pick(U32,Bool.and(U32.is_zero(Chess.piece(b,src(m))),U32.is_eq(U32.xor(src(m),dst(m)),16)),
      U32.div(U32.add(src(m),dst(m)),2),64),unfold(b,m)) : ep(_)
  generated(src(m),dst(m),Chess.piece(b,src(m)),below)
