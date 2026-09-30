import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../castling/Route.bend as R
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

# Every law is used on its full symbolic public domain.
def extension(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>) ->
  S.growth(S.guard(b,ks),S.move(b,ks),tail,Chess.castle_side(b,ks,(table,tail))):
  Laws.castle_side_guarded_extension(b,ks,table,tail)

def guard(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {S.guard(b,ks) == True{} : Bool}:
  Laws.new_castle_member_has_guard(b,ks,table,tail,m,here,absent)

def route(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {m == S.move(b,ks) : Chess.Ply}:
  Laws.new_castle_member_matches_route(b,ks,table,tail,m,here,absent)

def fresh(+b: Chess.Board,+ks: Bool,e: {S.guard(b,ks) == True{} : Bool}) ->
  {B.fresh_mask(b,R.rook(S.route(b,ks))) == True{} : Bool}:
  Laws.producer_guard_supplies_rook_freshness(b,ks,e)

def preserve(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},here: S.member(m,S.moves(Chess.castle_side(b,ks,(table,tail)))),absent: S.member(m,tail) -> Empty) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  Laws.new_castle_member_preserves_representation(b,ks,table,tail,m,good,here,absent)

def reject(+b: Chess.Board,+ks: Bool,table: Array<U64>,+tail: List<&2,Chess.Ply>,e: {S.guard(b,ks) == False{} : Bool}) ->
  {Chess.castle_side(b,ks,(table,tail)) == (table,tail) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.rejected_guard_preserves_pair(b,ks,table,tail,e)

# Small arbitrary zero table demonstrates satisfiable source premises only.
# It is NOT represented as a correct chess-attack table.
def board() -> Chess.Board:
  Position.metadata(Position.put(5,True{},4,Position.put(3,True{},7,Position.empty())),1,1,64)

def emitted_nonempty() ->
  {S.moves(Chess.castle_side(board(),True{},(Array.new(U64,0n,U64.zero()),Nil{}))) == Con{Chess.Ply{4,6,0,2},Nil{}} : List<&2,Chess.Ply>}:
  {==}

def no_member(m: Chess.Ply,p: S.member(m,Nil{})) -> Empty:
  match p:

def example_guard() -> {S.guard(board(),True{}) == True{} : Bool}:
  Laws.new_castle_member_has_guard(board(),True{},Array.new(U64,0n,U64.zero()),Nil{},Chess.Ply{4,6,0,2},Inl{{==}},p => no_member(Chess.Ply{4,6,0,2},p))

def example_preserves() -> {B.valid(Chess.make_move(board(),Chess.Ply{4,6,0,2})) == True{} : Bool}:
  Laws.new_castle_member_preserves_representation(board(),True{},Array.new(U64,0n,U64.zero()),Nil{},Chess.Ply{4,6,0,2},{==},Inl{{==}},p => no_member(Chess.Ply{4,6,0,2},p))

# A caller can supply an old move with no guard. It must not be certified as new.
def inherited_tail_is_not_certified() ->
  {Chess.castle_side(Position.empty(),True{},(Array.new(U64,0n,U64.zero()),Con{Chess.Ply{4,6,0,2},Nil{}})) ==
    (Array.new(U64,0n,U64.zero()),Con{Chess.Ply{4,6,0,2},Nil{}}) : Array<U64> & List<&2,Chess.Ply>}:
  {==}

def inherited_guard_false() -> {S.guard(Position.empty(),True{}) == False{} : Bool}:
  {==}
