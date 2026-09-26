# Small compiled observer test; no neural package or mutable search arena.
import Base
import ./NeuralWork.bend as Work

def checked(ok: Bool) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.die(Unit, 2, "neural work budget regression")

def main() -> IO(Unit):
  r = Work.start(1000n, 2, 1000, True{})
  r = Work.selected(r, 3n)
  r = Work.returned(r, True{}, 0, 1, 4n, 5n, 6n, 7n)
  # A physically evaluated result rejected by resume must not count as useful.
  +r = Work.returned(r, True{}, 1, 1, 4n, 5n, 6n, 7n)
  do IO<Unit>:
    checked(Work.exhausted(r))
    IO.print(Work.render(2000n, 1, 4, r))
    IO.print(Work.render(0n, 4, 0, Work.returned(Work.start(0n, 0, 0, False{}), False{}, 3, 4, 0n, 0n, 0n, 0n)))
