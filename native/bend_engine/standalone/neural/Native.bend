# Native infrastructure boundary: load a fixed package, transfer dense floats,
# run its forward and return RAW compact-policy/WDL logits. No chess or softmax.
import Base

def Native.open(path: String) -> IO(U32):
  import "./call.c"

def Native.spec() -> IO(List<&2, U32>):
  import "./call.c"

def Native.forward(sequence: U32, values: List<&2, F32>) -> IO(List<&2, F32>):
  import "./call.c"

def Native.close() -> IO(Unit):
  import "./call.c"

def input_read(acc: List<&2, F32>, r: Array<F32> & F32) -> Array<F32> & List<&2, F32>:
  (a, x) = r
  (a, Con{x, acc})

def input_words(fuel: Nat, +i: U32, r: Array<F32> & List<&2, F32>) -> List<&2, F32>:
  match fuel:
    case 0n:
      (a, acc) = r
      List.reverse(&2, F32, acc)
    case 1n+p:
      (a, acc) = r
      input_words(p, U32.inc(i), input_read(acc, Array.get(F32, a, i)))
