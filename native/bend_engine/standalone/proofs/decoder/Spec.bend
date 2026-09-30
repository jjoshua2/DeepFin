# Independent abstract square classification; no optimized decoder is used.
import Base
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess
import ./Bits.bend as Bits

type Cell is Data:
  Empty{}
  Occupant{kind: U32,white: Bool}
  Invalid{}

def classify(row: B.Row) -> Cell:
  match row:
    case B.Row{False{},False{},False{},False{},False{},False{},False{},False{}}: Empty{}
    case B.Row{True{},False{},False{},False{},False{},False{},False{},True{}}: Occupant{0,False{}}
    case B.Row{True{},False{},False{},False{},False{},False{},True{},False{}}: Occupant{0,True{}}
    case B.Row{False{},True{},False{},False{},False{},False{},False{},True{}}: Occupant{1,False{}}
    case B.Row{False{},True{},False{},False{},False{},False{},True{},False{}}: Occupant{1,True{}}
    case B.Row{False{},False{},True{},False{},False{},False{},False{},True{}}: Occupant{2,False{}}
    case B.Row{False{},False{},True{},False{},False{},False{},True{},False{}}: Occupant{2,True{}}
    case B.Row{False{},False{},False{},True{},False{},False{},False{},True{}}: Occupant{3,False{}}
    case B.Row{False{},False{},False{},True{},False{},False{},True{},False{}}: Occupant{3,True{}}
    case B.Row{False{},False{},False{},False{},True{},False{},False{},True{}}: Occupant{4,False{}}
    case B.Row{False{},False{},False{},False{},True{},False{},True{},False{}}: Occupant{4,True{}}
    case B.Row{False{},False{},False{},False{},False{},True{},False{},True{}}: Occupant{5,False{}}
    case B.Row{False{},False{},False{},False{},False{},True{},True{},False{}}: Occupant{5,True{}}
    case _: Invalid{}

def encode(cell: Cell) -> B.Row:
  match cell:
    case Empty{}: B.Row{False{},False{},False{},False{},False{},False{},False{},False{}}
    case Occupant{+kind,+white}:
      B.Row{U32.is_eq(kind,0),U32.is_eq(kind,1),U32.is_eq(kind,2),U32.is_eq(kind,3),U32.is_eq(kind,4),U32.is_eq(kind,5),white,Bool.not(white)}
    case Invalid{}: B.Row{True{},True{},True{},True{},True{},True{},True{},True{}}

def row(+board: Chess.Board,+sq: U32) -> B.Row:
  B.Row{U64.test_bit(Chess.get_pawns(board),U32.to_nat(sq)),U64.test_bit(Chess.get_knights(board),U32.to_nat(sq)),U64.test_bit(Chess.get_bishops(board),U32.to_nat(sq)),U64.test_bit(Chess.get_rooks(board),U32.to_nat(sq)),U64.test_bit(Chess.get_queens(board),U32.to_nat(sq)),U64.test_bit(Chess.get_kings(board),U32.to_nat(sq)),U64.test_bit(Chess.get_white(board),U32.to_nat(sq)),U64.test_bit(Chess.get_black(board),U32.to_nat(sq))}

def structural(+board: Chess.Board,+sq: U32) -> B.Row:
  B.Row{Bits.get64(Chess.get_pawns(board),U32.to_nat(sq)),Bits.get64(Chess.get_knights(board),U32.to_nat(sq)),Bits.get64(Chess.get_bishops(board),U32.to_nat(sq)),Bits.get64(Chess.get_rooks(board),U32.to_nat(sq)),Bits.get64(Chess.get_queens(board),U32.to_nat(sq)),Bits.get64(Chess.get_kings(board),U32.to_nat(sq)),Bits.get64(Chess.get_white(board),U32.to_nat(sq)),Bits.get64(Chess.get_black(board),U32.to_nat(sq))}

def square(b: Chess.Board,sq: U32) -> Cell:
  classify(row(b,sq))

# Actual query wrapper is used only in the proof interface and probe, not production.
# The king fallback is not interpreted as a piece on an empty square.
def decoded(+b: Chess.Board,+sq: U32) -> Cell:
  Bool.pick(Cell,U64.test_bit(Chess.occupied(b),U32.to_nat(sq)),
    Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},Empty{})

# This row helper is only an implementation bridge, not the independent classifier.
def selected_tag(row: B.Row) -> U32:
  B.Row{p,n,b,r,q,k,w,bl} = row
  Bool.pick(U32,p,0,Bool.pick(U32,n,1,Bool.pick(U32,b,2,Bool.pick(U32,r,3,Bool.pick(U32,q,4,5)))))

def occupied(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  Bool.or(w,bl)

def white(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  w

def local_decode(+row: B.Row) -> Cell:
  Bool.pick(Cell,occupied(row),Occupant{selected_tag(row),white(row)},Empty{})

def word_row(+width: Nat,p: Word(width),n: Word(width),b: Word(width),r: Word(width),q: Word(width),k: Word(width),w: Word(width),bl: Word(width),+i: Nat) -> B.Row:
  B.Row{Bits.get(width,p,i),Bits.get(width,n,i),Bits.get(width,b,i),Bits.get(width,r,i),Bits.get(width,q,i),Bits.get(width,k,i),Bits.get(width,w,i),Bits.get(width,bl,i)}

def mix(+choice: Bool,a: B.Row,b: B.Row) -> B.Row:
  B.Row{ap,an,ab,ar,aq,ak,aw,abl} = a
  B.Row{bp,bn,bb,br,bq,bk,bw,bbl} = b
  B.Row{Bool.pick(Bool,choice,ap,bp),Bool.pick(Bool,choice,an,bn),Bool.pick(Bool,choice,ab,bb),Bool.pick(Bool,choice,ar,br),Bool.pick(Bool,choice,aq,bq),Bool.pick(Bool,choice,ak,bk),Bool.pick(Bool,choice,aw,bw),Bool.pick(Bool,choice,abl,bbl)}
