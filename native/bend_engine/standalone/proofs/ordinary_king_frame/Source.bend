import Base
import ./Algebra.bend as A
import ../u64/Words.bend as Old
import ../board/Spec.bend as B
import ../decoder/Spec.bend as Row
import ../decoder/Projection.bend as Projection
import ../filter_ray_exclusion/Bits.bend as Bits
import ../castle_safety/Spec.bend as Sensitive
import ../candidate_source_provenance/Source.bend as N
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../ray_clear_disjoint/Algebra.bend as D
import ../../../legal_probe/Chess.bend as Chess

def king(row: B.Row) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  k
def owned(row: B.Row,white: Bool) -> Bool:
  B.Row{p,n,b,r,q,k,w,bl} = row
  Bool.pick(Bool,white,w,bl)

def empty_color(-Goal: Type,white: Bool,e: {Bool.pick(Bool,white,False{},False{}) == True{} : Bool}) -> Goal:
  match white:
    case False{}: Old.false_true(Goal,e)
    case True{}: Old.false_true(Goal,e)

# An owned consistent row cannot take the decoder's empty-square king fallback.
def row_nonking(+row: B.Row,+white: Bool,
  good: {B.row_valid(row) == True{} : Bool},
  owner: {owned(row,white) == True{} : Bool},
  clear: {king(row) == False{} : Bool}) ->
  {U32.is_eq(Row.selected_tag(row),5) == False{} : Bool}:
  match row:
    case B.Row{True{},n,b,r,q,k,w,bl}: {==}
    case B.Row{False{},True{},b,r,q,k,w,bl}: {==}
    case B.Row{False{},False{},True{},r,q,k,w,bl}: {==}
    case B.Row{False{},False{},False{},True{},q,k,w,bl}: {==}
    case B.Row{False{},False{},False{},False{},True{},k,w,bl}: {==}
    case B.Row{False{},False{},False{},False{},False{},True{},w,bl}:
      Old.false_true({U32.is_eq(5,5) == False{} : Bool},Equal.sym(Bool,True{},False{},clear))
    case B.Row{False{},False{},False{},False{},False{},False{},False{},False{}}:
      empty_color({U32.is_eq(5,5) == False{} : Bool},white,owner)
    case B.Row{False{},False{},False{},False{},False{},False{},False{},True{}}:
      Old.false_true({U32.is_eq(5,5) == False{} : Bool},good)
    case B.Row{False{},False{},False{},False{},False{},False{},True{},False{}}:
      Old.false_true({U32.is_eq(5,5) == False{} : Bool},good)
    case B.Row{False{},False{},False{},False{},False{},False{},True{},True{}}:
      Old.false_true({U32.is_eq(5,5) == False{} : Bool},good)

def select_test(+white: Bool,+a: U64,+b: U64,+i: Nat) ->
  {U64.test_bit(Chess.select_u64(white,a,b),i) ==
    Bool.pick(Bool,white,U64.test_bit(a,i),U64.test_bit(b,i)) : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def nonking(+b: Chess.Board,+src: U32,+bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  good: {B.valid(b) == True{} : Bool},
  owner: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  clear: {U64.test_bit(Chess.get_kings(b),U32.to_nat(src)) == False{} : Bool}) ->
  {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}:
  row_nonking(Row.row(b,src),U32.is_eq(Chess.get_turn(b),1),Projection.valid(b,src,good,bound),
    Equal.trans(Bool,owned(Row.row(b,src),U32.is_eq(Chess.get_turn(b),1)),
      U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)),True{},
      Equal.sym(Bool,U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)),
        owned(Row.row(b,src),U32.is_eq(Chess.get_turn(b),1)),
        select_test(U32.is_eq(Chess.get_turn(b),1),Chess.get_white(b),Chess.get_black(b),U32.to_nat(src))),owner),clear)

# Reuse the checked exclusion bridge with the ray/king OR operands exchanged.
def king_clear(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  +bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},
  owner: {U64.test_bit(Chess.color(b,Chess.get_turn(b)),U32.to_nat(src)) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  {U64.test_bit(Chess.get_kings(b),U32.to_nat(src)) == False{} : Bool}:
  Bits.test_exclusion(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b),rays,U32.to_nat(src),bound,owner,
    Equal.trans(Bool,U64.test_bit(U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(Chess.get_kings(b),rays)),U32.to_nat(src)),
      U64.test_bit(Sensitive.sensitive(b,rays),U32.to_nat(src)),False{},
      Equal.cong(U64,Bool,mask => U64.test_bit(U64.and(Chess.color(b,Chess.get_turn(b)),mask),U32.to_nat(src)),
        U64.or(Chess.get_kings(b),rays),U64.or(rays,Chess.get_kings(b)),A.or_commute(Chess.get_kings(b),rays)),
      Bits.or_false_right(U32.is_eq(0,1),U64.test_bit(Sensitive.sensitive(b,rays),U32.to_nat(src)),bypass)))

def facts_known(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  cert: N.source(b,Chess.Ply{src,dst,0,0}),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) &
  {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}:
  (+bound,+owner) = cert
  +clear = king_clear(b,rays,src,dst,bound,owner,bypass)
  (Bits.bit_zero(Chess.get_kings(b),U32.to_nat(src),bound,clear),nonking(b,src,bound,good,owner,clear))

def facts(generation_table: Array<U64>,+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,
  here: E.member(Chess.Ply{src,dst,0,0},E.moves(C.candidates(generation_table,b))),
  good: {B.valid(b) == True{} : Bool},
  bypass: {Chess.filter_requires(Sensitive.sensitive(b,rays),Chess.Ply{src,dst,0,0}) == False{} : Bool}) ->
  D.disjoint(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) &
  {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}:
  facts_known(b,rays,src,dst,N.member(generation_table,b,Chess.Ply{src,dst,0,0},here),good,bypass)
