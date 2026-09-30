import Base
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../../../legal_probe/Chess.bend as Chess

def roundtrip(+c: S.Choice) -> {S.from_tag(S.tag(c)) == Some{c} : Maybe<&2,S.Choice>}:
  match c:
    case S.Knight{}: {==}
    case S.Bishop{}: {==}
    case S.Rook{}: {==}
    case S.Queen{}: {==}

def nonzero(+c: S.Choice) -> {U32.is_zero(S.tag(c)) == False{} : Bool}:
  match c:
    case S.Knight{}: {==}
    case S.Bishop{}: {==}
    case S.Rook{}: {==}
    case S.Queen{}: {==}

def selected(+kind: U32,+c: S.Choice) ->
  {Bool.pick(U32,U32.is_zero(S.tag(c)),kind,S.tag(c)) == S.tag(c) : U32}:
  match c:
    case S.Knight{}: {==}
    case S.Bishop{}: {==}
    case S.Rook{}: {==}
    case S.Queen{}: {==}

def generated(+src: U32,+dst: U32,flag: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.put_move(True{},src,dst,flag,tail) == List.append(&2,Chess.Ply,S.choices(src,dst),tail) : List<&2,Chess.Ply>}:
  {==}
