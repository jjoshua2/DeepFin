# Actual parser producer and both public consumers, without a row premise.
import Base
import ./Rows.bend as Rows
import ./consumer.bend as Consumer
import ../fen_validated_full/Spec.bend as S
import ../fen_validated_full/Root.bend as Root
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos

def parsed_rows(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  Root.rows(Pos.fen(pieces,turn,castle,target,half,full)):
  Rows.fen(pieces,turn,castle,target,half,full)

def actual_fen_step(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.step(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  Consumer.use_fen_step(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full)

def actual_fen_singleton(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.singleton(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  Consumer.use_fen_singleton(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full)

def accepted(out: Maybe<&2,Pos.Game>) -> Bool:
  match out:
    case None{}: False{}
    case Some{g}: True{}

# Nonzero piece rows and non-default metadata are included.
def mixed_accepted() ->
  {accepted(Pos.fen("P7/8/8/8/8/8/8/7k","b","Kq","a3","1000000","1000000")) == True{} : Bool}: {==}
def mixed_rows() ->
  Root.rows(Pos.fen("P7/8/8/8/8/8/8/7k","b","Kq","a3","1000000","1000000")):
  Rows.fen("P7/8/8/8/8/8/8/7k","b","Kq","a3","1000000","1000000")

# Parser success is separate from chess validity: the empty board parses.
def empty_accepted() ->
  {accepted(Pos.fen("8/8/8/8/8/8/8/8","w","-","-","0","1")) == True{} : Bool}: {==}
def empty_rows() ->
  Root.rows(Pos.fen("8/8/8/8/8/8/8/8","w","-","-","0","1")):
  Rows.fen("8/8/8/8/8/8/8/8","w","-","-","0","1")

def invalid_placement() ->
  {Pos.fen("x","w","-","-","0","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_side() ->
  {Pos.fen("8/8/8/8/8/8/8/8","x","-","-","0","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_rights() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","KK","-","0","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_ep() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","i3","0","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_half_number() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","-","x","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_half_bound() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","-","1000001","1") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_full_number() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","-","0","x") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_full_positive() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","-","0","0") == None{} : Maybe<&2,Pos.Game>}: {==}
def invalid_full_bound() ->
  {Pos.fen("8/8/8/8/8/8/8/8","w","-","-","0","1000001") == None{} : Maybe<&2,Pos.Game>}: {==}
