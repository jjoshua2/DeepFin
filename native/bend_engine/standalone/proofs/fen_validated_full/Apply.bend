# Replace the complete ordered legal pair through the actual find/apply consumer.
import Base
import ./Spec.bend as S
import ./Turn.bend as Turn
import ./Replay.bend as Replay
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../initialized_fast_full/consumer.bend as Prior
import ../board/Spec.bend as B
import ../validated_root_kings/Kings.bend as Kings
import ../accepted_castle/Generated.bend as Actual
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def full_pair(+c: O.Cells,+b: Chess.Board) ->
  {S.full_moves(O.pack(c),b) ==
    Chess.filter_legal(E.moves(Actual.candidates(O.pack(c),b)),b,(O.pack(c),Nil{})) :
    Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(Array<U64> & List<&2,Chess.Ply>,Actual.candidates(O.pack(c),b),
    (O.pack(c),E.moves(Actual.candidates(O.pack(c),b))),
    Actual.list_pair(O.pack(c),Actual.candidates(O.pack(c),b),Actual.candidate_table(c,b))) :
    {S.full_prepare(b,_) ==
      Chess.filter_legal(E.moves(Actual.candidates(O.pack(c),b)),b,(O.pack(c),Nil{})) :
      Array<U64> & List<&2,Chess.Ply>}
  {==}

def legal(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,
  initialized: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  canonical: Turn.canonical(Chess.get_turn(b)),valid: {Pos.valid(b) == True{} : Bool}) ->
  {Chess.legal_moves(O.pack(c),b) == S.full_moves(O.pack(c),b) :
    Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.legal_moves(O.pack(c),b),
    Chess.filter_legal(E.moves(Actual.candidates(O.pack(c),b)),b,(O.pack(c),Nil{})),
    S.full_moves(O.pack(c),b),
    Prior.use_legal_full(O.pack(c),d,seed,n,extra,b,U32.is_eq(Chess.get_turn(b),1),
      initialized,depth,tables,full,good,Kings.side_nonempty(b,Chess.get_turn(b),valid),canonical),
    Equal.sym(Array<U64> & List<&2,Chess.Ply>,S.full_moves(O.pack(c),b),
      Chess.filter_legal(E.moves(Actual.candidates(O.pack(c),b)),b,(O.pack(c),Nil{})),full_pair(c,b)))

def one(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,
  initialized: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  accepted: S.accepted(maybe)) ->
  {Protocol.apply_one(text,maybe,O.pack(c)) == S.full_one(text,maybe,O.pack(c)) :
    Array<U64> & Maybe<&2,Pos.Game>}:
  match maybe:
    case None{}: {==}
    case Some{+g}:
      (good,(canonical,valid))=accepted
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & Maybe<&2,Pos.Game>,
        pair => Protocol.move_found(g,text,pair),
        Chess.legal_moves(O.pack(c),Pos.board(g)),S.full_moves(O.pack(c),Pos.board(g)),
        legal(c,d,seed,n,extra,Pos.board(g),initialized,depth,tables,full,good,canonical,valid))

# Initialization is transported using the exact returned table, before application.
def result(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+text: String,
  r: Array<U64> & Maybe<&2,Pos.Game>,
  initialized: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  checked: S.root(c,r)) ->
  {S.one_after(text,r) == S.full_after(text,r) : Array<U64> & Maybe<&2,Pos.Game>}:
  (a,+maybe)=r
  (same,accepted)=checked
  %Equal.sym(Array<U64>,a,O.pack(c),same) :
    {S.one_after(text,(_,maybe)) == S.full_after(text,(_,maybe)) :
    Array<U64> & Maybe<&2,Pos.Game>}
  one(c,d,seed,n,extra,maybe,text,initialized,depth,tables,full,accepted)

# Actual moves destructures its input pair before executing the one command.
def singleton(+text: String,r: Array<U64> & Maybe<&2,Pos.Game>,
  proof: {S.one_after(text,r) == S.full_after(text,r) : Array<U64> & Maybe<&2,Pos.Game>}) ->
  {Protocol.moves(Con{text,Nil{}},r) == S.full_after(text,r) : Array<U64> & Maybe<&2,Pos.Game>}:
  Replay.singleton(pair => S.full_after(text,pair),text,r,proof)
