# Lossless abstract-board interface, not a new engine representation or legal-position model.
import Base
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../../../legal_probe/Chess.bend as Chess

type Planes<-width: Nat> is Data:
  Planes{p: Word(width),n: Word(width),b: Word(width),r: Word(width),q: Word(width),k: Word(width),w: Word(width),bl: Word(width)}

type Snapshot is Data:
  Snapshot{low: List<&2,D.Cell>,high: List<&2,D.Cell>,turn: U32,rights: U32,ep: U32}

def valid(+width: Nat,cols: Planes<width>) -> Bool:
  Planes{p,n,b,r,q,k,w,bl} = cols
  B.words(width,p,n,b,r,q,k,w,bl)

def cons(+width: Nat,row: B.Row,cols: Planes<width>) -> Planes<1n+width>:
  B.Row{p,n,b,r,q,k,w,bl} = row
  Planes{pt,nt,bt,rt,qt,kt,wt,blt} = cols
  Planes{WCon{p,pt},WCon{n,nt},WCon{b,bt},WCon{r,rt},WCon{q,qt},WCon{k,kt},WCon{w,wt},WCon{bl,blt}}

def cells(+width: Nat,cols: Planes<width>) -> List<&2,D.Cell>:
  match width:
    case 0n: Nil{}
    case 1n+t:
      match cols:
        case Planes{WCon{p,pt},WCon{n,nt},WCon{b,bt},WCon{r,rt},WCon{q,qt},WCon{k,kt},WCon{w,wt},WCon{bl,blt}}:
          D.classify(B.Row{p,n,b,r,q,k,w,bl}) <> cells(t,Planes{pt,nt,bt,rt,qt,kt,wt,blt})

# Total on malformed lists; the proved inverse concerns actual consistent boards.
def columns(+width: Nat,xs: List<&2,D.Cell>) -> Planes<width>:
  match width:
    case 0n: Planes{WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{},WNil{}}
    case 1n+t:
      match xs:
        case Nil{}: cons(t,D.encode(D.Empty{}),columns(t,Nil{}))
        case head <> tail: cons(t,D.encode(head),columns(t,tail))

def observe(board: Chess.Board) -> Snapshot:
  match board:
    case Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},U64{U32{wl},U32{wh}},U64{U32{bll},U32{blh}},turn,rights,ep}:
      Snapshot{cells(32n,Planes{pl,nl,bl,rl,ql,kl,wl,bll}),cells(32n,Planes{ph,nh,bh,rh,qh,kh,wh,blh}),turn,rights,ep}

def join(low: Planes<32n>,high: Planes<32n>,turn: U32,rights: U32,ep: U32) -> Chess.Board:
  Planes{pl,nl,bl,rl,ql,kl,wl,bll} = low
  Planes{ph,nh,bh,rh,qh,kh,wh,blh} = high
  Chess.Board{U64{U32{pl},U32{ph}},U64{U32{nl},U32{nh}},U64{U32{bl},U32{bh}},U64{U32{rl},U32{rh}},U64{U32{ql},U32{qh}},U64{U32{kl},U32{kh}},U64{U32{wl},U32{wh}},U64{U32{bll},U32{blh}},turn,rights,ep}

def restore(s: Snapshot) -> Chess.Board:
  Snapshot{low,high,turn,rights,ep} = s
  join(columns(32n,low),columns(32n,high),turn,rights,ep)

def length(xs: List<&2,D.Cell>) -> Nat:
  match xs:
    case Nil{}: 0n
    case head <> tail: 1n+length(tail)

def lengths(s: Snapshot) -> Nat & Nat:
  Snapshot{low,high,turn,rights,ep} = s
  (length(low),length(high))

def metadata(s: Snapshot) -> U32 & (U32 & U32):
  Snapshot{low,high,turn,rights,ep} = s
  (turn,(rights,ep))

# Abstract square index uses the board convention: low limb first, bit zero first.
def at(xs: List<&2,D.Cell>,i: Nat) -> D.Cell:
  match xs i:
    case Nil{} _: D.Empty{}
    case head <> tail 0n: head
    case head <> tail 1n+j: at(tail,j)

def square(s: Snapshot,+sq: U32) -> D.Cell:
  Snapshot{low,high,turn,rights,ep} = s
  Bool.pick(D.Cell,Nat.is_lt(U32.to_nat(sq),32n),at(low,U32.to_nat(sq)),at(high,Nat.sub(U32.to_nat(sq),32n)))
