import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../board/LAWS.bend as BoardLaws
import ../board/PROOF.bend as BoardProof
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

# The actual scalar decoder recovers all eight square bits, not merely a tag range.
def reconstruct_occupied_row(+b: Chess.Board,+sq: U32,+good: {B.valid(b) == True{} : Bool},
  +bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  present: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == True{} : Bool}) ->
  {S.encode(S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))}) == S.row(b,sq) : B.Row}:
  Equal.trans(B.Row,S.encode(S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))}),S.encode(S.square(b,sq)),S.row(b,sq),
    Equal.cong(S.Cell,B.Row,x => S.encode(x),S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},S.square(b,sq),
      Equal.sym(S.Cell,S.square(b,sq),S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},Laws.occupied_decoder_matches_square(b,sq,good,bound,present))),
    Laws.abstract_square_roundtrip(b,sq,good,bound))

def every_square(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.decoded(b,sq) == S.square(b,sq) : S.Cell}:
  Laws.guarded_decoder_matches_square(b,sq,good,bound)

def projection(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {B.row_valid(S.row(b,sq)) == True{} : Bool}:
  Laws.square_projection_valid(b,sq,good,bound)

# Empty is not interpreted as a king despite the optimized decoder's fallback.
def empty_is_empty() -> {S.square(Position.empty(),0) == S.Empty{} : S.Cell}: {==}
def raw_empty_fallback() -> {Chess.piece(Position.empty(),0) == 5 : U32}: {==}
def empty_not_present() -> {U64.test_bit(Chess.occupied(Position.empty()),0n) == False{} : Bool}: {==}

# Non-vacuous real insertion at both limbs and their boundaries, both colors.
def pawn_at_zero() -> {S.square(Position.put(0,True{},0,Position.empty()),0) == S.Occupant{0,True{}} : S.Cell}: {==}
def knight_at_31() -> {S.square(Position.put(1,False{},31,Position.empty()),31) == S.Occupant{1,False{}} : S.Cell}: {==}
def bishop_at_32() -> {S.square(Position.put(2,True{},32,Position.empty()),32) == S.Occupant{2,True{}} : S.Cell}: {==}
def king_at_63() -> {S.square(Position.put(5,False{},63,Position.empty()),63) == S.Occupant{5,False{}} : S.Cell}: {==}

# Initial board supplies the global invariant rather than assuming a decoder answer.
def initial_square(+sq: U32,bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.decoded(Position.board(Position.start()),sq) == S.square(Position.board(Position.start()),sq) : S.Cell}:
  Laws.guarded_decoder_matches_square(Position.board(Position.start()),sq,BoardLaws.initial_representation(),bound)

# Color-only occupancy is malformed and does not certify the king fallback.
def malformed_occupancy() -> {S.classify(B.Row{False{},False{},False{},False{},False{},False{},True{},False{}}) == S.Invalid{} : S.Cell}: {==}
