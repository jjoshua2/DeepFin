# Independent per-square representation invariant, not chess legality.
import Base
import ../../../legal_probe/Chess.bend as Chess

type Piece is Data:
  Pawn{}
  Knight{}
  Bishop{}
  Rook{}
  Queen{}
  King{}

def tag(p: Piece) -> U32:
  match p:
    case Pawn{}: 0
    case Knight{}: 1
    case Bishop{}: 2
    case Rook{}: 3
    case Queen{}: 4
    case King{}: 5

# One Boolean from each of the eight bitboards at a single square.
type Row is Data:
  Row{p: Bool,n: Bool,b: Bool,r: Bool,q: Bool,k: Bool,w: Bool,bl: Bool}

# Exactly empty, or exactly one kind and one color. No decoder fallback.
def row_valid(row: Row) -> Bool:
  match row:
    case Row{False{},False{},False{},False{},False{},False{},False{},False{}}: True{}
    case Row{True{},False{},False{},False{},False{},False{},False{},True{}}: True{}
    case Row{True{},False{},False{},False{},False{},False{},True{},False{}}: True{}
    case Row{False{},True{},False{},False{},False{},False{},False{},True{}}: True{}
    case Row{False{},True{},False{},False{},False{},False{},True{},False{}}: True{}
    case Row{False{},False{},True{},False{},False{},False{},False{},True{}}: True{}
    case Row{False{},False{},True{},False{},False{},False{},True{},False{}}: True{}
    case Row{False{},False{},False{},True{},False{},False{},False{},True{}}: True{}
    case Row{False{},False{},False{},True{},False{},False{},True{},False{}}: True{}
    case Row{False{},False{},False{},False{},True{},False{},False{},True{}}: True{}
    case Row{False{},False{},False{},False{},True{},False{},True{},False{}}: True{}
    case Row{False{},False{},False{},False{},False{},True{},False{},True{}}: True{}
    case Row{False{},False{},False{},False{},False{},True{},True{},False{}}: True{}
    case _: False{}

def row_fresh(row: Row,bit: Bool) -> Bool:
  Row{p,n,b,r,q,k,w,bl} = row
  Bool.not(Bool.and(bit,Bool.or(w,bl)))

def row_add(row: Row,+piece: Piece,+white: Bool,+bit: Bool) -> Row:
  Row{p,n,b,r,q,k,w,bl} = row
  Row{Bool.or(p,Bool.and(U32.is_eq(tag(piece),0),bit)),Bool.or(n,Bool.and(U32.is_eq(tag(piece),1),bit)),Bool.or(b,Bool.and(U32.is_eq(tag(piece),2),bit)),Bool.or(r,Bool.and(U32.is_eq(tag(piece),3),bit)),Bool.or(q,Bool.and(U32.is_eq(tag(piece),4),bit)),Bool.or(k,Bool.and(U32.is_eq(tag(piece),5),bit)),Bool.or(w,Bool.and(white,bit)),Bool.or(bl,Bool.and(Bool.not(white),bit))}

def words(+width: Nat,p: Word(width),n: Word(width),b: Word(width),r: Word(width),q: Word(width),k: Word(width),w: Word(width),bl: Word(width)) -> Bool:
  match width:
    case 0n: True{}
    case 1n+t:
      match p n b r q k w bl:
        case WCon{ph,pt} WCon{nh,nt} WCon{bh,bt} WCon{rh,rt} WCon{qh,qt} WCon{kh,kt} WCon{wh,wt} WCon{blh,blt}:
          Bool.and(row_valid(Row{ph,nh,bh,rh,qh,kh,wh,blh}),words(t,pt,nt,bt,rt,qt,kt,wt,blt))

def fresh_words(+width: Nat,w: Word(width),bl: Word(width),bit: Word(width)) -> Bool:
  match width:
    case 0n: True{}
    case 1n+t:
      match w bl bit:
        case WCon{wh,wt} WCon{bh,bt} WCon{ih,it}:
          Bool.and(Bool.not(Bool.and(ih,Bool.or(wh,bh))),fresh_words(t,wt,bt,it))

def valid(b: Chess.Board) -> Bool:
  match b:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},U64{U32{wl},U32{wh}},U64{U32{bll},U32{blh}},turn,rights,ep}:
      Bool.and(words(32n,pl,nl,bl,rl,ql,kl,wl,bll),words(32n,ph,nh,bh,rh,qh,kh,wh,blh))

def fresh_mask(b: Chess.Board,mask: U64) -> Bool:
  match b mask:
    case Chess.Board{p,n,bi,r,q,k,U64{U32{wl},U32{wh}},U64{U32{bl},U32{bh}},turn,rights,ep} U64{U32{ml},U32{mh}}:
      Bool.and(fresh_words(32n,wl,bl,ml),fresh_words(32n,wh,bh,mh))

def fresh(b: Chess.Board,sq: U32) -> Bool:
  fresh_mask(b,U64.bit(U32.to_nat(sq)))

def metadata(+b: Chess.Board) -> U32 & (U32 & U32):
  (Chess.get_turn(b),(Chess.get_rights(b),Chess.get_ep(b)))

# Exact bitboard update specification; no engine decisions are computed here.
def insert_mask(b: Chess.Board,+mask: U64,+piece: Piece,+white: Bool) -> Chess.Board:
  Chess.Board{p,n,b,r,q,k,w,bl,turn,rights,ep} = b
  Chess.Board{U64.or(p,Chess.select_u64(U32.is_eq(tag(piece),0),mask,U64.zero())),U64.or(n,Chess.select_u64(U32.is_eq(tag(piece),1),mask,U64.zero())),U64.or(b,Chess.select_u64(U32.is_eq(tag(piece),2),mask,U64.zero())),U64.or(r,Chess.select_u64(U32.is_eq(tag(piece),3),mask,U64.zero())),U64.or(q,Chess.select_u64(U32.is_eq(tag(piece),4),mask,U64.zero())),U64.or(k,Chess.select_u64(U32.is_eq(tag(piece),5),mask,U64.zero())),
    U64.or(w,Chess.select_u64(white,mask,U64.zero())),
    U64.or(bl,Chess.select_u64(white,U64.zero(),mask)),turn,rights,ep}
