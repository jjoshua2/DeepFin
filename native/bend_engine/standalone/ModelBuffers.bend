# Linear F32 buffers cross one synchronous foreign effect, never lists or pointers
# owned by the caller after dispatch. Shape count is separate from array capacity.
import Base

type Buffers is Type:
  Buffers{input: Array<F32>, output: Array<F32>, count: U32}

def Native.open() -> IO(U32):
  import "./model_call.c"

def Native.diagnostics() -> IO(Bool):
  import "./model_call.c"

def Native.run(count: U32, input: Array<F32>, output: Array<F32>) -> IO(Buffers):
  import "./model_call.c"
