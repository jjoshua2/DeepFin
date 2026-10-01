import Base
import ./Spec.bend as S
import ../ray/Slider.bend as Actual

def computed(+kind: S.Slider,+src: Nat,+occ: U64,+bound: {Nat.is_lt(src,64n) == True{} : Bool}) ->
  {S.computed(kind,src,occ) == S.geometry(kind,src,occ) : U64}:
  match kind:
    case S.Rook{}: Actual.square(src,False{},occ,bound)
    case S.Bishop{}: Actual.square(src,True{},occ,bound)
    case S.Queen{}:
      Actual.combine(S.computed(S.Rook{},src,occ),S.geometry(S.Rook{},src,occ),
        S.computed(S.Bishop{},src,occ),S.geometry(S.Bishop{},src,occ),
        Actual.square(src,False{},occ,bound),Actual.square(src,True{},occ,bound))
