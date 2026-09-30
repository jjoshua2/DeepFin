# Independent per-square witness relation over supplied reverse-attack masks.
# No priority decoder or production attack accumulator defines this relation.
import Base
import ../../../legal_probe/Chess.bend as Chess

type Masks is Data:
  Masks{pawn: U64,knight: U64,king: U64,rook: U64,bishop: U64}

def row(+own: Bool, +p: Bool, +n: Bool, +b: Bool, +r: Bool, +q: Bool, +k: Bool, +pa: Bool, +na: Bool, +ka: Bool, +ra: Bool, +ba: Bool) -> Bool:
  Bool.and(Bool.or(Bool.or(Bool.or(Bool.or(Bool.and(pa,p),Bool.and(na,n)),Bool.and(ka,k)),Bool.and(ra,Bool.or(r,q))),Bool.and(ba,Bool.or(b,q))),own)

def scan(+d: Nat,own: Word(d),p: Word(d),n: Word(d),b: Word(d),r: Word(d),q: Word(d),k: Word(d),pa: Word(d),na: Word(d),ka: Word(d),ra: Word(d),ba: Word(d)) -> Bool:
  match d:
    case 0n: False{}
    case 1n+t:
      match own p n b r q k pa na ka ra ba:
        case WCon{ownh,ownt} WCon{ph,pt} WCon{nh,nt} WCon{bh,bt} WCon{rh,rt} WCon{qh,qt} WCon{kh,kt} WCon{pah,pat} WCon{nah,nat} WCon{kah,kat} WCon{rah,rat} WCon{bah,bat}:
          Bool.or(row(ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah),scan(t,ownt,pt,nt,bt,rt,qt,kt,pat,nat,kat,rat,bat))

def select(board: Chess.Board,w: Bool) -> U64:
  match w:
    case True{}: Chess.get_white(board)
    case False{}: Chess.get_black(board)

def observe(board: Chess.Board,own: U64,m: Masks) -> Bool:
  match board own m:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},white,black,turn,rights,ep} U64{U32{ownl},U32{ownh}} Masks{U64{U32{pal},U32{pah}},U64{U32{nal},U32{nah}},U64{U32{kal},U32{kah}},U64{U32{ral},U32{rah}},U64{U32{bal},U32{bah}}}:
      Bool.or(scan(32n,ownl,pl,nl,bl,rl,ql,kl,pal,nal,kal,ral,bal),scan(32n,ownh,ph,nh,bh,rh,qh,kh,pah,nah,kah,rah,bah))

def present(+board: Chess.Board,by: U32,m: Masks) -> Bool:
  observe(board,select(board,U32.is_eq(by,1)),m)

def masks(pa: U64,na: U64,ka: U64,ra: U64,ba: U64) -> Masks:
  Masks{pa,na,ka,ra,ba}
