# Only real Position layout/FEN operations execute here, never proof helpers.
import Base
import ../../Position.bend as Position
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def show_board(b: Chess.Board) -> String:
  Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} = b
  show(p) ++ " " ++ show(n) ++ " " ++ show(bi) ++ " " ++ show(r) ++ " " ++ show(q) ++ " " ++ show(k) ++ " " ++ show(w) ++ " " ++ show(bl) ++ " " ++ U32.show(turn) ++ " " ++ U32.show(rights) ++ " " ++ U32.show(ep)

def emit_game(r: Maybe<&2,Position.Game>) -> IO(Unit):
  match r:
    case None{}: IO.print("F 0")
    case Some{Position.Game{b,half,full,hist}}:
      IO.print("F 1 " ++ show_board(b) ++ " " ++ U32.show(half) ++ " " ++ U32.show(full) ++ " " ++ U32.show(U32.from_nat(List.length(&2,Position.Past,hist))))

def checked_fen(ok: Bool,pieces: String,turn: String,castle: String,target: String,half: String,full: String) -> IO(Unit):
  match ok:
    case True{}: emit_game(Position.fen(pieces,turn,castle,target,half,full))
    case False{}: IO.die(Unit,2,"invalid fen-result FEN domain")

def fen_rows(fuel: Nat,xs: List<String>) -> IO(Unit):
  match fuel xs:
    case _ Nil{}: IO.pure(Unit,Unit{})
    case 1n+p +pieces <> +turn <> +castle <> +target <> +half <> +full <> tail:
      do IO<Unit>:
        checked_fen(Nat.is_le(Nat.add(String.length(pieces),Nat.add(String.length(turn),Nat.add(String.length(castle),Nat.add(String.length(target),Nat.add(String.length(half),String.length(full)))))),1024n),pieces,turn,castle,target,half,full)
        fen_rows(p,tail)
    case _ _: IO.die(Unit,2,"invalid fen-result FEN tuple or budget")

def command(xs: List<String>) -> IO(Unit):
  match xs:
    case "fen" <> tail: fen_rows(64n,tail)
    case _: IO.die(Unit,2,"invalid fen-result operation")

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    command(xs)
