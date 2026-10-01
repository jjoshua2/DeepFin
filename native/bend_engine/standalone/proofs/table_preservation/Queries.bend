# Actual reads/callbacks preserve any source array, without content assumptions.
import Base
import ./Core.bend as C
import ../storage/Observe.bend as O
import ../storage/Read.bend as Read
import ../../../legal_probe/Chess.bend as Chess
import ../../../bitboard_probe/Sliders.bend as Sliders

def word_projection(r: Array<U64> & U64) ->
  {C.table(U64,r) == O.storage(r) : Array<U64>}:
  (a,v) = r
  {==}

def read(table: Array<U64>,+q: U32) ->
  {C.table(U64,Array.get(U64,table,q)) == table : Array<U64>}:
  C.apply(U64,a => Array.get(U64,a,q),table,+c =>
    Equal.trans(Array<U64>,C.table(U64,Array.get(U64,O.pack(c),q)),
      O.storage(Array.get(U64,O.pack(c),q)),O.pack(c),
      word_projection(Array.get(U64,O.pack(c),q)),Read.read(c,q)))

def offset(+index: U32,r: Array<U64> & U64) ->
  {C.table(U64,Chess.slide_offset(index,r)) == C.table(U64,r) : Array<U64>}:
  (table,+at) = r
  read(table,U32.add(U64.low(at),index))

def mask(+sq: U32,+occ: U64,r: Array<U64> & U64) ->
  {C.table(U64,Chess.slide_mask(sq,occ,r)) == C.table(U64,r) : Array<U64>}:
  (table,+bits) = r
  C.apply(U64,a => Chess.slide_mask(sq,occ,(a,bits)),table,+c =>
    Equal.trans(Array<U64>,
      C.table(U64,Chess.slide_mask(sq,occ,(O.pack(c),bits))),
      C.table(U64,Array.get(U64,O.pack(c),U32.add(128,sq))),O.pack(c),
      offset(Sliders.pext_index(occ,bits),Array.get(U64,O.pack(c),U32.add(128,sq))),
      read(O.pack(c),U32.add(128,sq))))

def slide(table: Array<U64>,+sq: U32,+occ: U64) ->
  {C.table(U64,Chess.slide(table,sq,occ)) == table : Array<U64>}:
  C.apply(U64,a => Chess.slide(a,sq,occ),table,+c =>
    Equal.trans(Array<U64>,C.table(U64,Chess.slide(O.pack(c),sq,occ)),
      C.table(U64,Array.get(U64,O.pack(c),sq)),O.pack(c),
      mask(sq,occ,Array.get(U64,O.pack(c),sq)),read(O.pack(c),sq)))

def or_result(+a: U64,r: Array<U64> & U64) ->
  {C.table(U64,Chess.or_result(a,r)) == C.table(U64,r) : Array<U64>}:
  (table,b) = r
  {==}

def queen(+sq: U32,+occ: U64,r: Array<U64> & U64) ->
  {C.table(U64,Chess.queen_after(sq,occ,r)) == C.table(U64,r) : Array<U64>}:
  (table,+rook) = r
  C.apply(U64,a => Chess.queen_after(sq,occ,(a,rook)),table,+c =>
    Equal.trans(Array<U64>,C.table(U64,Chess.queen_after(sq,occ,(O.pack(c),rook))),
      C.table(U64,Chess.slide(O.pack(c),U32.add(64,sq),occ)),O.pack(c),
      or_result(rook,Chess.slide(O.pack(c),U32.add(64,sq),occ)),
      slide(O.pack(c),U32.add(64,sq),occ)))

# Eliminate all 32 Word constructors, including WNil, before case analysis.
# Scalar catch-all patterns leave impossible dependent Word tails unresolved.
def attack_bits(+b0: Bool,+b1: Bool,+b2: Bool,+b3: Bool,+b4: Bool,+b5: Bool,+b6: Bool,+b7: Bool,+b8: Bool,+b9: Bool,+b10: Bool,+b11: Bool,+b12: Bool,+b13: Bool,+b14: Bool,+b15: Bool,+b16: Bool,+b17: Bool,+b18: Bool,+b19: Bool,+b20: Bool,+b21: Bool,+b22: Bool,+b23: Bool,+b24: Bool,+b25: Bool,+b26: Bool,+b27: Bool,+b28: Bool,+b29: Bool,+b30: Bool,+b31: Bool,+c: O.Cells,+sq: U32,+side: U32,+occ: U64) ->
  {C.table(U64,Chess.attack(U32{WCon{b0,WCon{b1,WCon{b2,WCon{b3,WCon{b4,WCon{b5,WCon{b6,WCon{b7,WCon{b8,WCon{b9,WCon{b10,WCon{b11,WCon{b12,WCon{b13,WCon{b14,WCon{b15,WCon{b16,WCon{b17,WCon{b18,WCon{b19,WCon{b20,WCon{b21,WCon{b22,WCon{b23,WCon{b24,WCon{b25,WCon{b26,WCon{b27,WCon{b28,WCon{b29,WCon{b30,WCon{b31,WNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},O.pack(c),sq,side,occ)) == O.pack(c) : Array<U64>}:
  match b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31:
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      read(O.pack(c),U32.add(sq,Bool.pick(U32,U32.is_eq(side,1),384,448)))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      Equal.trans(Array<U64>,C.table(U64,Chess.attack(4,O.pack(c),sq,side,occ)),
        C.table(U64,Chess.slide(O.pack(c),sq,occ)),O.pack(c),
        queen(sq,occ,Chess.slide(O.pack(c),sq,occ)),slide(O.pack(c),sq,occ))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} False{} True{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      slide(O.pack(c),U32.add(64,sq),occ)
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case False{} True{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      read(O.pack(c),U32.add(256,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      slide(O.pack(c),sq,occ)
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))
    case True{} True{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _:
      read(O.pack(c),U32.add(320,sq))

def attack_model(kind: U32,+c: O.Cells,+sq: U32,+side: U32,+occ: U64) ->
  {C.table(U64,Chess.attack(kind,O.pack(c),sq,side,occ)) == O.pack(c) : Array<U64>}:
  match kind:
    case U32{WCon{+b0,WCon{+b1,WCon{+b2,WCon{+b3,WCon{+b4,WCon{+b5,WCon{+b6,WCon{+b7,WCon{+b8,WCon{+b9,WCon{+b10,WCon{+b11,WCon{+b12,WCon{+b13,WCon{+b14,WCon{+b15,WCon{+b16,WCon{+b17,WCon{+b18,WCon{+b19,WCon{+b20,WCon{+b21,WCon{+b22,WCon{+b23,WCon{+b24,WCon{+b25,WCon{+b26,WCon{+b27,WCon{+b28,WCon{+b29,WCon{+b30,WCon{+b31,WNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
      attack_bits(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,c,sq,side,occ)

def attack(+kind: U32,table: Array<U64>,+sq: U32,+side: U32,+occ: U64) ->
  {C.table(U64,Chess.attack(kind,table,sq,side,occ)) == table : Array<U64>}:
  C.apply(U64,a => Chess.attack(kind,a,sq,side,occ),table,c => attack_model(kind,c,sq,side,occ))
