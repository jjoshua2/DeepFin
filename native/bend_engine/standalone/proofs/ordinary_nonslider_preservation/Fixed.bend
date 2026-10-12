import Base
import ../ordinary_slider_preservation/Check.bend as C
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

# These are actual whole lookup pairs, for arbitrary table shape/content and U32
# square/side. The reverse pawn query flips the attacking side as in attacked.
def pawn_query(+c: O.Cells,+sq: U32,+by: U32,+occ: U64) ->
  {Chess.attack(0,O.pack(c),sq,U32.xor(by,1),occ) ==
    (O.pack(c),C.pawn(c,sq,by)) : Array<U64> & U64}:
  C.get_pair(c,C.pawn_index(sq,by))
def knight_query(+c: O.Cells,+sq: U32,+by: U32,+occ: U64) ->
  {Chess.attack(1,O.pack(c),sq,by,occ) ==
    (O.pack(c),C.knight(c,sq)) : Array<U64> & U64}:
  C.get_pair(c,U32.add(256,sq))
def king_query(+c: O.Cells,+sq: U32,+by: U32,+occ: U64) ->
  {Chess.attack(5,O.pack(c),sq,by,occ) ==
    (O.pack(c),C.king(c,sq)) : Array<U64> & U64}:
  C.get_pair(c,U32.add(320,sq))

# Old/new occupancy equality is derived from the implementation, not an assumed
# geometric or builder-produced fixed table.
def pawn_unchanged(+c: O.Cells,+sq: U32,+by: U32,+old_occ: U64,+new_occ: U64) ->
  {Chess.attack(0,O.pack(c),sq,U32.xor(by,1),old_occ) ==
    Chess.attack(0,O.pack(c),sq,U32.xor(by,1),new_occ) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Chess.attack(0,O.pack(c),sq,U32.xor(by,1),old_occ),
    (O.pack(c),C.pawn(c,sq,by)),Chess.attack(0,O.pack(c),sq,U32.xor(by,1),new_occ),pawn_query(c,sq,by,old_occ),
    Equal.sym(Array<U64> & U64,Chess.attack(0,O.pack(c),sq,U32.xor(by,1),new_occ),(O.pack(c),C.pawn(c,sq,by)),pawn_query(c,sq,by,new_occ)))
def knight_unchanged(+c: O.Cells,+sq: U32,+by: U32,+old_occ: U64,+new_occ: U64) ->
  {Chess.attack(1,O.pack(c),sq,by,old_occ) == Chess.attack(1,O.pack(c),sq,by,new_occ) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Chess.attack(1,O.pack(c),sq,by,old_occ),
    (O.pack(c),C.knight(c,sq)),Chess.attack(1,O.pack(c),sq,by,new_occ),knight_query(c,sq,by,old_occ),
    Equal.sym(Array<U64> & U64,Chess.attack(1,O.pack(c),sq,by,new_occ),(O.pack(c),C.knight(c,sq)),knight_query(c,sq,by,new_occ)))
def king_unchanged(+c: O.Cells,+sq: U32,+by: U32,+old_occ: U64,+new_occ: U64) ->
  {Chess.attack(5,O.pack(c),sq,by,old_occ) == Chess.attack(5,O.pack(c),sq,by,new_occ) : Array<U64> & U64}:
  Equal.trans(Array<U64> & U64,Chess.attack(5,O.pack(c),sq,by,old_occ),
    (O.pack(c),C.king(c,sq)),Chess.attack(5,O.pack(c),sq,by,new_occ),king_query(c,sq,by,old_occ),
    Equal.sym(Array<U64> & U64,Chess.attack(5,O.pack(c),sq,by,new_occ),(O.pack(c),C.king(c,sq)),king_query(c,sq,by,new_occ)))

# The new consumer uses these exact production callbacks, retaining the raw
# non-slider prefix (which can contain friendly hits).
def callbacks(+c: O.Cells,+b: Chess.Board,+sq: U32,+by: U32) ->
  {Chess.attacked(O.pack(c),b,sq,by) ==
    Chess.attacked_rook(b,sq,by,C.prefix(c,b,sq,by),Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))) : Array<U64> & Bool}:
  %Equal.sym(Array<U64> & U64,Chess.attack(0,O.pack(c),sq,U32.xor(by,1),Chess.occupied(b)),
    (O.pack(c),C.pawn(c,sq,by)),pawn_query(c,sq,by,Chess.occupied(b))) :
    {Chess.attacked_pawn(b,sq,by,_) ==
      Chess.attacked_rook(b,sq,by,C.prefix(c,b,sq,by),Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(1,O.pack(c),sq,by,Chess.occupied(b)),
    (O.pack(c),C.knight(c,sq)),knight_query(c,sq,by,Chess.occupied(b))) :
    {Chess.attacked_knight(b,sq,by,U64.and(C.pawn(c,sq,by),Chess.get_pawns(b)),_) ==
      Chess.attacked_rook(b,sq,by,C.prefix(c,b,sq,by),Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))) : Array<U64> & Bool}
  %Equal.sym(Array<U64> & U64,Chess.attack(5,O.pack(c),sq,by,Chess.occupied(b)),
    (O.pack(c),C.king(c,sq)),king_query(c,sq,by,Chess.occupied(b))) :
    {Chess.attacked_king(b,sq,by,U64.or(U64.and(C.pawn(c,sq,by),Chess.get_pawns(b)),U64.and(C.knight(c,sq),Chess.get_knights(b))),_) ==
      Chess.attacked_rook(b,sq,by,C.prefix(c,b,sq,by),Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))) : Array<U64> & Bool}
  {==}
