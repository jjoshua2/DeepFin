# Arithmetic reduction is linked to the independent per-square witness scan.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Bits.bend as Bits

def prefix(+b: Chess.Board,pa: U64,na: U64,ka: U64,ra: U64) -> U64:
  U64.or(U64.or(U64.or(U64.and(pa,Chess.get_pawns(b)),U64.and(na,Chess.get_knights(b))),
    U64.and(ka,Chess.get_kings(b))),U64.and(ra,U64.or(Chess.get_rooks(b),Chess.get_queens(b))))

def hits(+b: Chess.Board,m: S.Masks) -> U64:
  S.Masks{pa,na,ka,ra,ba} = m
  U64.or(prefix(b,pa,na,ka,ra),U64.and(ba,U64.or(Chess.get_bishops(b),Chess.get_queens(b))))

def both(+a: Bool,+b: Bool,+c: Bool,+d: Bool,
  e: {a == c : Bool},f: {b == d : Bool}) -> {Bool.or(a,b) == Bool.or(c,d) : Bool}:
  Equal.trans(Bool,Bool.or(a,b),Bool.or(c,b),Bool.or(c,d),
    Equal.cong(Bool,Bool,x => Bool.or(x,b),a,c,e),
    Equal.cong(Bool,Bool,x => Bool.or(c,x),b,d,f))

def limbs(+board: Chess.Board,+own: U64,+m: S.Masks) ->
  {Bool.not(U64.is_zero(U64.and(hits(board,m),own))) == S.observe(board,own,m) : Bool}:
  match board own m:
    case Chess.Board{U64{U32{+pl},U32{+ph}},U64{U32{+nl},U32{+nh}},U64{U32{+bl},U32{+bh}},U64{U32{+rl},U32{+rh}},U64{U32{+ql},U32{+qh}},U64{U32{+kl},U32{+kh}},white,black,turn,rights,ep} U64{U32{+ownl},U32{+ownh}} S.Masks{U64{U32{+pal},U32{+pah}},U64{U32{+nal},U32{+nah}},U64{U32{+kal},U32{+kah}},U64{U32{+ral},U32{+rah}},U64{U32{+bal},U32{+bah}}}:
      Equal.trans(Bool,
        Bool.not(Bool.and(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),Word.zero(32n))),Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),Word.zero(32n))))),
        Bool.or(Bool.not(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),Word.zero(32n)))),Bool.not(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),Word.zero(32n))))),
        Bool.or(S.scan(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),S.scan(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah)),
        Bits.de_morgan(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),Word.zero(32n))),Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),Word.zero(32n)))),
        both(Bool.not(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),Word.zero(32n)))),Bool.not(Cmp.is_eq(Word.cmp(32n,Bits.fold(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),Word.zero(32n)))),S.scan(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),S.scan(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),
          Bits.reflected(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),Bits.reflected(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah)))

def selected(+b: Chess.Board,+w: Bool) ->
  {Chess.select_u64(w,Chess.get_white(b),Chess.get_black(b)) == S.select(b,w) : U64}:
  match w:
    case True{}: {==}
    case False{}: {==}

def decision(+b: Chess.Board,+by: U32,+m: S.Masks) ->
  {Bool.not(U64.is_zero(U64.and(hits(b,m),Chess.color(b,by)))) == S.present(b,by,m) : Bool}:
  Equal.trans(Bool,
    Bool.not(U64.is_zero(U64.and(hits(b,m),Chess.color(b,by)))),
    S.observe(b,Chess.color(b,by),m),S.present(b,by,m),
    limbs(b,Chess.color(b,by),m),
    Equal.cong(U64,Bool,x => S.observe(b,x,m),Chess.color(b,by),S.select(b,U32.is_eq(by,1)),selected(b,U32.is_eq(by,1))))
