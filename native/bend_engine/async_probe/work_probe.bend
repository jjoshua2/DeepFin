# Concrete accounting transitions used by the full application's waiting/draining
# states, with known clocks. No evaluator, tree or foreign request is required.
import Base
import ../standalone/NeuralWork.bend as Work

def require(ok: Bool) -> IO(Unit):
  match ok:
    case True{}: IO.pure(Unit, Unit{})
    case False{}: IO.die(Unit, 2, "async admission budget was not consumed")

def main() -> IO(Unit):
  r = Work.identified(Work.start(1000n, 2, 0, True{}), 7)
  r = Work.returned(Work.submitted(r, 3n), True{}, 0, 1, 0n, 0n, 4n, 5n)
  +r = Work.submitted(r, 6n)
  +cancelled = Work.cancel_pending(r)
  do IO<Unit>:
    require(Work.exhausted(r))
    IO.print(Work.render(1100n, 1, 2, cancelled))
    IO.print(Work.retirement(1200n, 1, 2, Work.retired(cancelled, True{})))
    IO.print(Work.retirement(1200n, 1, 2, Work.retired(cancelled, False{})))
