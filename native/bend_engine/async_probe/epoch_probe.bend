import Base
import ../standalone/EngineRuntime.bend as Runtime
import ../standalone/NativeEvaluation.bend as Native

def last(r: Runtime.Cache & U32) -> IO(Unit):
  (cache, epoch) = r
  do IO<Unit>:
    IO.print("epoch " ++ U32.show(epoch))
    r : Runtime.Cache & U32 <- Runtime.next(cache)
    IO.die(Unit, 3, "epoch unexpectedly recycled")

def main() -> IO(Unit):
  do IO<Unit>:
    r : Runtime.Cache & U32 <- Runtime.next(Runtime.Cache{Native.Disabled{}, Runtime.Control{4294967294, False{}, Runtime.NoRetirement{}}})
    last(r)
