# Join actual prefilter storage, castling provenance, and accepted check decisions.
import Base
import ./Spec.bend as S
import ./Pair.bend as Pair
import ./Accept.bend as Accept
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Chain.bend as Chain
import ../table_preservation/Core.bend as T
import ../table_preservation/Generation.bend as Generation
import ../table_preservation/Targets.bend as Targets
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../../legal_probe/Chess.bend as Chess

def candidates(table: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  Chess.castle_side(b,False{},Chess.castle_side(b,True{},C.candidates(table,b)))

def candidate_table(+c: O.Cells,+b: Chess.Board) ->
  {T.table(List<&2,Chess.Ply>,candidates(O.pack(c),b)) == O.pack(c) : Array<U64>}:
  +own = Chess.color(b,Chess.get_turn(b))
  +keys = Chess.bit_squares(64n,U64.is_zero(own),own,Nil{})
  Equal.trans(Array<U64>,T.table(List<&2,Chess.Ply>,candidates(O.pack(c),b)),
    T.table(List<&2,Chess.Ply>,C.candidates(O.pack(c),b)),O.pack(c),
    Generation.wings(b,C.candidates(O.pack(c),b)),Targets.scan(keys,b,(O.pack(c),Nil{})))

def list_pair(-a: Array<U64>,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {T.table(List<&2,Chess.Ply>,r) == a : Array<U64>}) ->
  {r == (a,E.moves(r)) : Array<U64> & List<&2,Chess.Ply>}:
  (t,+xs) = r
  %kept : {(t,xs) == (_,xs) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def prepared_equal(+c: O.Cells,+b: Chess.Board) ->
  {Chess.legal_moves(O.pack(c),b) ==
    Chess.filter_prepare(b,(O.pack(c),E.moves(candidates(O.pack(c),b)))) : Array<U64> & List<&2,Chess.Ply>}:
  %list_pair(O.pack(c),candidates(O.pack(c),b),candidate_table(c,b)) :
    {Chess.legal_moves(O.pack(c),b) == Chess.filter_prepare(b,_) : Array<U64> & List<&2,Chess.Ply>}
  {==}

def split_member(-Goal: Type,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,here: E.member(m,xs),
  next: E.member(m,xs) -> E.member(m,xs) -> Goal) -> Goal:
  match xs:
    case Nil{}: Empty.absurd(Goal,here)
    case Con{h,rest}:
      match here:
        case Inl{+same}: next(Inl{same})(Inl{same})
        case Inr{tail}:
          split_member(Goal,rest,m,tail,one => two => next(Inr{one})(Inr{two}))

def known(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))),tag: {C.flag(m) == 2 : U32}) -> S.safe(c,b,m):
  split_member(S.safe(c,b,m),E.moves(Chess.legal_moves(O.pack(c),b)),m,here,origin => accepted =>
    Accept.prepared(c,b,E.moves(candidates(O.pack(c),b)),m,
      Chain.legal_member(O.pack(c),b,m,origin,tag),
      Accept.transport(Chess.legal_moves(O.pack(c),b),
        Chess.filter_prepare(b,(O.pack(c),E.moves(candidates(O.pack(c),b)))),m,prepared_equal(c,b),accepted)))

def lift(-a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  E.member(m,E.moves(Chess.legal_moves(a,b))) -> {C.flag(m) == 2 : U32} ->
  {S.answer(Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : E.member(m,E.moves(Chess.legal_moves(_,b))) -> {C.flag(m) == 2 : U32} ->
    {S.answer(Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}
  %same : E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))) -> {C.flag(m) == 2 : U32} ->
    {S.answer(Chess.in_check(_,Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}
  here => tag => known(c,b,m,here,tag)

def actual(a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 2 : U32}) ->
  {S.answer(Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}:
  lift(a,b,m,R.reify(a))(here)(tag)
