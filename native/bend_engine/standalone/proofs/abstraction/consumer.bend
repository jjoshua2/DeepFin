import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../board/LAWS.bend as BL
import ../board/PROOF.bend as BP
import ../decoder/Spec.bend as D
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def arbitrary_board(+b: Chess.Board,good: {B.valid(b) == True{} : Bool}) ->
  {S.restore(S.observe(b)) == b : Chess.Board}:
  Laws.whole_board_roundtrip(b,good)

def arbitrary_size(+b: Chess.Board) -> {S.lengths(S.observe(b)) == (32n,32n) : Nat & Nat}:
  Laws.snapshot_has_64_squares(b)

def distinguish_boards(+a: Chess.Board,+b: Chess.Board,ga: {B.valid(a) == True{} : Bool},gb: {B.valid(b) == True{} : Bool},
  same: {S.observe(a) == S.observe(b) : S.Snapshot}) -> {a == b : Chess.Board}:
  Laws.equal_snapshots_identify_board(a,b,ga,gb,same)

def empty_square(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  absent: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == False{} : Bool}) ->
  {D.row(b,sq) == D.encode(D.Empty{}) : B.Row}:
  Laws.empty_square_has_empty_observation(b,sq,good,bound,absent)

def inserted_board_roundtrip() ->
  {S.restore(S.observe(Position.put(4,True{},63,Position.empty()))) == Position.put(4,True{},63,Position.empty()) : Chess.Board}:
  Laws.whole_board_roundtrip(Position.put(4,True{},63,Position.empty()),
    BL.fresh_put_preserves_representation(Position.empty(),63,B.Queen{},True{},BL.empty_representation(),{==}))

def empty_roundtrip() -> {S.restore(S.observe(Position.empty())) == Position.empty() : Chess.Board}:
  Laws.whole_board_roundtrip(Position.empty(),BL.empty_representation())

def metadata_not_validated() ->
  {S.metadata(S.observe(Position.metadata(Position.empty(),4294967295,4294967295,4294967295))) == (4294967295,(4294967295,4294967295)) : U32 & (U32 & U32)}:
  {==}

def square_order(+b: Chess.Board,+sq: U32,bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.square(S.observe(b),sq) == D.square(b,sq) : D.Cell}:
  Laws.snapshot_square_correspondence(b,sq,bound)
