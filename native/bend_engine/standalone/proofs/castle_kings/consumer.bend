import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def use_mid(+b: Chess.Board,+white: Bool,+ks: Bool,good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}) ->
  {S.kings(S.mid(b,white,ks),white) == U64.bit(U32.to_nat(S.transit(white,ks))) : U64}:
  Laws.transit_preserves_single_moving_king(b,white,ks,good,turn,one)

def use_final(+b: Chess.Board,+white: Bool,+ks: Bool,good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {S.kings(S.final(b,white,ks),white) == U64.bit(U32.to_nat(S.dst(white,ks))) : U64}:
  Laws.castling_preserves_single_moving_king(b,white,ks,good,turn,one,guard)

def use_flip(+b: Chess.Board,+move: Chess.Ply,+white: Bool,turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Chess.get_turn(Chess.make_move(b,move)) == Bool.to_u32(Bool.not(white)) : U32}:
  Laws.move_flips_valid_side(b,move,white,turn)

def use_check(table: Array<U64>,+b: Chess.Board,+white: Bool,+ks: Bool,+stage: S.Stage,good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.in_check(table,S.stage_board(b,white,ks,stage),Bool.to_u32(white)) == Chess.attacked(table,S.stage_board(b,white,ks,stage),S.stage_square(white,ks,stage),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}:
  Laws.castling_stage_check_targets_moving_king(table,b,white,ks,stage,good,turn,one,guard)

# A nonempty satisfiable board for every side/wing, plus arbitrary attack tables.
def example(+white: Bool,+ks: Bool) -> Chess.Board:
  Position.metadata(Position.put(5,white,S.src(white),Position.put(3,white,U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,7,0)),Position.put(5,Bool.not(white),S.src(Bool.not(white)),Position.empty()))),Bool.to_u32(white),15,64)

def example_good(+white: Bool,+ks: Bool) -> {B.valid(example(white,ks)) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def example_one(+white: Bool,+ks: Bool) -> {S.kings(example(white,ks),white) == U64.bit(U32.to_nat(S.src(white))) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def example_guard(+white: Bool,+ks: Bool) -> {E.guard(example(white,ks),ks) == True{} : Bool}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def concrete_check(table: Array<U64>,+white: Bool,+ks: Bool,+stage: S.Stage) ->
  {Chess.in_check(table,S.stage_board(example(white,ks),white,ks,stage),Bool.to_u32(white)) == Chess.attacked(table,S.stage_board(example(white,ks),white,ks,stage),S.stage_square(white,ks,stage),Bool.to_u32(Bool.not(white))) : Array<U64> & Bool}:
  Laws.castling_stage_check_targets_moving_king(table,example(white,ks),white,ks,stage,example_good(white,ks),{==},example_one(white,ks),example_guard(white,ks))

# A foreign king on the rook landing square becomes an extra owned king on a raw castle.
def blocked() -> Chess.Board:
  Position.put(5,False{},5,example(True{},True{}))

def blocked_is_consistent() -> {B.valid(blocked()) == True{} : Bool}: {==}
def blocked_starts_single() -> {S.kings(blocked(),True{}) == U64.from_u32(16) : U64}: {==}
def blocked_guard_fails() -> {E.guard(blocked(),True{}) == False{} : Bool}: {==}
def blocked_creates_two_selected_kings() -> {S.kings(S.final(blocked(),True{},True{}),True{}) == U64.from_u32(96) : U64}: {==}
def blocked_not_single_destination() -> {U64.is_eq(S.kings(S.final(blocked(),True{},True{}),True{}),U64.from_u32(64)) == False{} : Bool}: {==}

# XOR is not a sanitizer for arbitrary raw side metadata.
def invalid_side_stays_invalid() ->
  {Chess.get_turn(Chess.make_move(Position.metadata(example(True{},True{}),3,15,64),Chess.Ply{4,6,0,2})) == 2 : U32}: {==}
