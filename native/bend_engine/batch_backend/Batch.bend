# Synchronous raw tensor boundary only. Search/acceptance and row gathering are
# NOT implemented here. One owner holds all mutable storage through completion.
import Base

type Capabilities is Data:
  Capabilities{profile: U32, batch: U32, channels: U32, input_log2: U32, output_log2: U32}

type Result is Type:
  Result{input: Array<F32>, output: Array<F32>, real_rows: U32, physical_rows: U32, count: U32}

def Batch.open() -> IO(Capabilities):
  import "./batch_call.c"

def Batch.run(rows: U32, input: Array<F32>, output: Array<F32>) -> IO(Result):
  import "./batch_call.c"
